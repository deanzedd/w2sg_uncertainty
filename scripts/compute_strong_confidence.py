#!/usr/bin/env python3
"""
compute_strong_confidence.py — Pre-compute C_strong for D_l samples.

Computes the strong model's implicit reward signal for each D_l sample
using the Phase 1 DPO model (π_Phase1) and the SFT reference (π_SFT).

Algorithm per sample (x, y_w, y_l):
    r_strong(x, y) = β · log[π_Phase1(y|x) / π_SFT(y|x)]
    C_strong(i)    = 2 · (σ(r_strong(x,y_w) − r_strong(x,y_l)) − 0.5)  ∈ [-1, 1]

Writes updated pseudo_labeled.jsonl with:
    - confidence_weight = max(C_strong, 0.0) by default
                          OR raw C_strong (if --allow_negative is set)
    - strong_confidence = raw C_strong ∈ [-1, 1] (always stored for analysis)

Usage:
    python scripts/compute_strong_confidence.py \\
        --config configs/mwdpo_2phase_hh_rlhf.yaml \\
        --phase1_model_path outputs/.../strong_model_phase1 \\
        --sft_model_path outputs/.../sft_strong \\
        --d_low_path outputs/.../weak_labels/d_low/pseudo_labeled.jsonl \\
        --output_path outputs/.../weak_labels/d_low_scored/pseudo_labeled.jsonl

    # Allow negative confidence_weight (full C_strong, not clipped at 0)
    python scripts/compute_strong_confidence.py \\
        --config configs/mwdpo_2phase_hh_rlhf.yaml \\
        --phase1_model_path outputs/.../strong_model_phase1 \\
        --sft_model_path outputs/.../sft_strong \\
        --d_low_path outputs/.../weak_labels/d_low/pseudo_labeled.jsonl \\
        --output_path outputs/.../weak_labels/d_low_scored/pseudo_labeled.jsonl \\
        --allow_negative

    # Debug (process only first 64 samples)
    python scripts/compute_strong_confidence.py \\
        --config configs/mwdpo_2phase_hh_rlhf.yaml ... --debug
"""

import argparse
import json
import logging
import math
import os
import sys
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_config, setup_logging, set_seed

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute C_strong for D_l using π_Phase1 vs π_SFT implicit reward."
    )
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument(
        "--phase1_model_path", type=str, required=True,
        help="Path to Phase 1 DPO checkpoint (LoRA adapter dir or merged model dir).",
    )
    parser.add_argument(
        "--sft_model_path", type=str, required=True,
        help="Path to SFT checkpoint (π_SFT — used as reference model).",
    )
    parser.add_argument(
        "--d_low_path", type=str, required=True,
        help="Path to D_l pseudo_labeled.jsonl (output of label_multi_weak.py).",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help=(
            "Where to write D_l with C_strong scores. "
            "Defaults to <d_low_dir>/d_low_scored/pseudo_labeled.jsonl."
        ),
    )
    parser.add_argument(
        "--beta", type=float, default=None,
        help="DPO β for implicit reward (default: read from config training.beta or 0.5).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for inference (default: 8).",
    )
    parser.add_argument(
        "--max_length", type=int, default=None,
        help="Max sequence length for tokenization (default: from config).",
    )
    parser.add_argument(
        "--allow_negative",
        action="store_true",
        default=False,
        help=(
            "If set, confidence_weight = raw C_strong ∈ [-1, 1] (supports negative weights). "
            "Default (unset): confidence_weight = max(C_strong, 0.0) (clipped at zero). "
            "The raw C_strong is always stored as 'strong_confidence' regardless."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Process only first 64 samples (fast smoke test).",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization helpers
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_preference_sample(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a (prompt, response) pair for log-prob computation.

    Returns:
        input_ids:  (seq_len,)   — full prompt + response tokens
        labels:     (seq_len,)   — -100 for prompt tokens, token id for response tokens
    """
    prompt_enc = tokenizer(prompt, add_special_tokens=False)
    response_enc = tokenizer(response, add_special_tokens=False)

    prompt_ids = prompt_enc["input_ids"]
    response_ids = response_enc["input_ids"]

    # Truncate from left if too long — keep full response, truncate prompt
    total_len = len(prompt_ids) + len(response_ids)
    if total_len > max_length:
        keep_prompt = max(0, max_length - len(response_ids))
        prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels":    torch.tensor(labels,    dtype=torch.long),
    }


def collate_batch(samples: List[Dict[str, torch.Tensor]], pad_id: int):
    """Left-pad a list of tokenized samples to the same length."""
    max_len = max(s["input_ids"].shape[0] for s in samples)
    input_ids_list, attention_mask_list, labels_list = [], [], []

    for s in samples:
        L = s["input_ids"].shape[0]
        pad_len = max_len - L
        # Left-pad input_ids and labels; attention mask 0 for padding
        input_ids_list.append(
            torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long), s["input_ids"]])
        )
        attention_mask_list.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(L, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long), s["labels"]])
        )

    return {
        "input_ids":      torch.stack(input_ids_list),       # (B, L)
        "attention_mask": torch.stack(attention_mask_list),   # (B, L)
        "labels":         torch.stack(labels_list),           # (B, L)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Log-prob computation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_response_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute sum of response-token log-probs for a batch.

    Args:
        model:          causal LM (frozen, eval mode)
        input_ids:      (B, L)
        attention_mask: (B, L)
        labels:         (B, L) — -100 for prompt tokens; response token ids otherwise

    Returns:
        (B,) — sum of log-probs over response tokens for each sample
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # (B, L, V)

    # Shift: predict token t+1 from state at t
    shift_logits = logits[:, :-1, :].contiguous()  # (B, L-1, V)
    shift_labels = labels[:, 1:].clone()            # (B, L-1)

    # Mask: only compute log-prob where label != -100 (i.e., response tokens)
    response_mask = (shift_labels != -100)          # (B, L-1)

    # Replace -100 with 0 for gather (won't affect masked positions)
    shift_labels_clamped = shift_labels.clone()
    shift_labels_clamped[~response_mask] = 0

    log_probs = F.log_softmax(shift_logits, dim=-1)                         # (B, L-1, V)
    per_token_logps = log_probs.gather(2, shift_labels_clamped.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    # Zero out non-response positions
    per_token_logps = per_token_logps * response_mask.float()

    # Warn if any sample has zero response tokens
    zero_resp = (response_mask.sum(dim=-1) == 0)
    if zero_resp.any():
        logger.warning(
            f"{zero_resp.sum().item()} sample(s) have no response tokens after tokenization. "
            "Their log-prob will be 0. Check tokenization / max_length settings."
        )

    return per_token_logps.sum(dim=-1)  # (B,) — summed log-prob per sample


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_path: str,
    sft_base_path: str,
    dtype: torch.dtype,
    device_map,
    merge_lora: bool = True,
) -> AutoModelForCausalLM:
    """
    Load a model from a checkpoint directory, handling LoRA adapters.

    If the checkpoint is a LoRA adapter (contains adapter_config.json),
    we load the SFT base model and apply the LoRA adapter, then optionally
    merge and unload for clean inference.

    Args:
        checkpoint_path: Path to the DPO Phase 1 checkpoint directory.
        sft_base_path:   Path to the SFT model (used as LoRA base).
        dtype:           Torch dtype for loading.
        device_map:      Device map for from_pretrained.
        merge_lora:      If True, merge LoRA into base weights and unload adapter.
                         Recommended for inference — avoids PEFT overhead.

    Returns:
        Loaded (and optionally merged) model in eval mode with requires_grad=False.
    """
    is_lora = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))

    if is_lora:
        logger.info(
            f"Detected LoRA adapter at {checkpoint_path}. "
            f"Loading base from {sft_base_path} + adapter."
        )
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft is required for LoRA loading. pip install peft")

        load_kwargs = {"torch_dtype": dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        base = AutoModelForCausalLM.from_pretrained(sft_base_path, **load_kwargs)
        if device_map is None and torch.cuda.is_available():
            base = base.to("cuda")

        model = PeftModel.from_pretrained(base, checkpoint_path)

        if merge_lora:
            logger.info("Merging LoRA adapter into base weights...")
            model = model.merge_and_unload()
            logger.info("LoRA merge complete.")
    else:
        # Full model checkpoint (already merged or non-LoRA)
        logger.info(f"Loading full model checkpoint from {checkpoint_path}.")
        load_kwargs = {"torch_dtype": dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **load_kwargs)
        if device_map is None and torch.cuda.is_available():
            model = model.to("cuda")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# ─────────────────────────────────────────────────────────────────────────────
# C_strong computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_c_strong_for_dataset(
    phase1_model,
    sft_model,
    tokenizer,
    samples: List[Dict],
    beta: float,
    max_length: int,
    batch_size: int,
    device: torch.device,
    allow_negative: bool,
) -> List[Dict]:
    """
    Compute C_strong for all samples and return updated samples with scores.

    Updates each sample dict with:
        strong_confidence  = raw C_strong ∈ [-1, 1]
        confidence_weight  = max(C_strong, 0.0)  [or raw C_strong if allow_negative]

    Args:
        allow_negative: If True, confidence_weight = C_strong (can be < 0).
                        If False (default), confidence_weight = max(C_strong, 0.0).
    """
    logger.info(
        f"Computing C_strong for {len(samples)} samples "
        f"(β={beta}, max_len={max_length}, batch={batch_size}, "
        f"allow_negative={allow_negative})"
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    updated = []
    total_batches = math.ceil(len(samples) / batch_size)

    for batch_idx in range(total_batches):
        batch_samples = samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        n = len(batch_samples)

        # Tokenize chosen and rejected for each sample
        chosen_toks   = [tokenize_preference_sample(tokenizer, s["prompt"], s["chosen"],   max_length) for s in batch_samples]
        rejected_toks = [tokenize_preference_sample(tokenizer, s["prompt"], s["rejected"], max_length) for s in batch_samples]

        # Collate into batched tensors
        chosen_batch   = collate_batch(chosen_toks,   pad_id)
        rejected_batch = collate_batch(rejected_toks, pad_id)

        # ── Phase 1 model log-probs ────────────────────────────────────────
        logp_p1_chosen   = compute_response_log_probs(phase1_model, **chosen_batch,   device=device)
        logp_p1_rejected = compute_response_log_probs(phase1_model, **rejected_batch, device=device)

        # ── SFT model log-probs ────────────────────────────────────────────
        logp_sft_chosen   = compute_response_log_probs(sft_model, **chosen_batch,   device=device)
        logp_sft_rejected = compute_response_log_probs(sft_model, **rejected_batch, device=device)

        # ── Implicit rewards r = β·(log π_Phase1 − log π_SFT) ─────────────
        r_chosen   = beta * (logp_p1_chosen   - logp_sft_chosen)
        r_rejected = beta * (logp_p1_rejected - logp_sft_rejected)

        # ── C_strong = 2·(σ(r_chosen − r_rejected) − 0.5) ∈ [-1, 1] ──────
        c_strong = 2.0 * (torch.sigmoid(r_chosen - r_rejected) - 0.5)  # (n,)

        # ── Confidence weight: clipped at 0 (default) or raw ──────────────
        if allow_negative:
            confidence_weight = c_strong
        else:
            confidence_weight = torch.clamp(c_strong, min=0.0)

        c_strong_list   = c_strong.cpu().tolist()
        conf_weight_list = confidence_weight.cpu().tolist()

        for i, sample in enumerate(batch_samples):
            new_sample = dict(sample)
            new_sample["strong_confidence"]  = c_strong_list[i]      # raw, always stored
            new_sample["confidence_weight"]  = conf_weight_list[i]   # used by CWPOTrainer
            updated.append(new_sample)

        if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
            logger.info(
                f"  Processed {min((batch_idx+1)*batch_size, len(samples))}/{len(samples)} samples "
                f"| mean C_strong so far: {sum(s['strong_confidence'] for s in updated)/len(updated):.4f}"
            )

    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Analysis / summary
# ─────────────────────────────────────────────────────────────────────────────

def print_c_strong_analysis(samples: List[Dict]) -> None:
    """Print distribution statistics for C_strong and confidence_weight."""
    c_vals  = [s["strong_confidence"] for s in samples]
    w_vals  = [s["confidence_weight"]  for s in samples]
    n = len(c_vals)

    n_positive = sum(1 for c in c_vals if c > 0)
    n_zero     = sum(1 for c in c_vals if c == 0.0)
    n_negative = sum(1 for c in c_vals if c < 0)

    logger.info("=" * 60)
    logger.info("C_STRONG DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"  Total samples     : {n:,}")
    logger.info(f"  C_strong > 0      : {n_positive:,}  ({100*n_positive/n:.1f}%)  ← consensus (train)")
    logger.info(f"  C_strong = 0      : {n_zero:,}  ({100*n_zero/n:.1f}%)")
    logger.info(f"  C_strong < 0      : {n_negative:,}  ({100*n_negative/n:.1f}%)  ← debate (skip/down-weight)")
    logger.info(f"  Mean C_strong     : {sum(c_vals)/n:.4f}")
    logger.info(f"  Std C_strong      : {(sum((c - sum(c_vals)/n)**2 for c in c_vals)/n)**0.5:.4f}")
    logger.info(f"  Min C_strong      : {min(c_vals):.4f}")
    logger.info(f"  Max C_strong      : {max(c_vals):.4f}")
    logger.info(f"  Mean conf_weight  : {sum(w_vals)/n:.4f}")

    # Correctness breakdown if original label available
    if any("label" in s or "is_correct" in s for s in samples[:5]):
        logger.info("  (Ground truth label comparison not available from pseudo_labeled.jsonl)")

    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    if args.debug:
        cfg.use_wandb = False

    setup_logging(cfg)
    set_seed(cfg.seed)

    # ── Resolve beta ─────────────────────────────────────────────────────────
    if args.beta is not None:
        beta = args.beta
    else:
        # Try reading from training config, then phase2_training, then default
        beta = (
            cfg.get("phase2_training", {}).get("beta", None)
            or cfg.get("training", {}).get("beta", 0.5)
        )
        beta = float(beta)
    logger.info(f"Using β = {beta}")

    # ── Resolve max_length ────────────────────────────────────────────────────
    max_length = args.max_length or cfg.get("max_length", 512)
    logger.info(f"Max sequence length: {max_length}")

    # ── Resolve output path ───────────────────────────────────────────────────
    if args.output_path:
        output_path = args.output_path
    else:
        d_low_dir = os.path.dirname(args.d_low_path)
        output_path = os.path.join(
            os.path.dirname(d_low_dir), "d_low_scored", "pseudo_labeled.jsonl"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Output will be written to: {output_path}")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load D_l samples ─────────────────────────────────────────────────────
    logger.info(f"Loading D_l from: {args.d_low_path}")
    samples = []
    with open(args.d_low_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples):,} D_l samples.")

    if args.debug:
        samples = samples[:64]
        logger.info(f"[DEBUG] Truncated to {len(samples)} samples.")

    # ── Load tokenizer from SFT model ────────────────────────────────────────
    logger.info(f"Loading tokenizer from: {args.sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Determine device_map ─────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # For inference only: use device_map=auto if multi-GPU, else None (single GPU)
    device_map = "auto" if n_gpus > 1 else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # ── Load SFT model (π_SFT, reference) ────────────────────────────────────
    logger.info("Loading SFT model (π_SFT — reference)...")
    sft_load_kwargs = {"torch_dtype": dtype}
    if device_map is not None:
        sft_load_kwargs["device_map"] = device_map
    sft_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, **sft_load_kwargs)
    if device_map is None and torch.cuda.is_available():
        sft_model = sft_model.to(device)
    sft_model.eval()
    for p in sft_model.parameters():
        p.requires_grad = False
    logger.info("SFT model loaded.")

    # ── Load Phase 1 model (π_Phase1) ────────────────────────────────────────
    logger.info("Loading Phase 1 model (π_Phase1)...")
    phase1_model = load_model_from_checkpoint(
        checkpoint_path=args.phase1_model_path,
        sft_base_path=args.sft_model_path,
        dtype=dtype,
        device_map=device_map,
        merge_lora=True,  # Always merge for clean inference
    )
    logger.info("Phase 1 model loaded.")

    # ── Compute C_strong ─────────────────────────────────────────────────────
    logger.info(
        f"allow_negative={args.allow_negative}: "
        f"confidence_weight = {'C_strong (raw, can be negative)' if args.allow_negative else 'max(C_strong, 0.0)'}"
    )
    updated_samples = compute_c_strong_for_dataset(
        phase1_model=phase1_model,
        sft_model=sft_model,
        tokenizer=tokenizer,
        samples=samples,
        beta=beta,
        max_length=max_length,
        batch_size=args.batch_size,
        device=device,
        allow_negative=args.allow_negative,
    )

    # ── Analysis ──────────────────────────────────────────────────────────────
    print_c_strong_analysis(updated_samples)

    # ── Write output ──────────────────────────────────────────────────────────
    logger.info(f"Writing {len(updated_samples):,} scored samples to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in updated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Done.")


if __name__ == "__main__":
    main()
