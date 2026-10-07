#!/usr/bin/env python3
"""
compute_strong_confidence.py — Pre-compute confidence weights for D_l samples.

Computes the strong model's implicit reward signal for each D_l sample
using the Phase 1 DPO model (π_Phase1) and the SFT reference (π_SFT),
then combines with the original weak ensemble confidence.

Algorithm per sample (x, y_w, y_l):
    r_strong(x, y)  = β · log[π_Phase1(y|x) / π_SFT(y|x)]
    C_strong(i)     = 2 · (σ(r_strong(x,y_w) − r_strong(x,y_l)) − 0.5)  ∈ [-1, 1]
    C_weak(i)       = original confidence_weight from D_l labeling ∈ [0.5, 1]
    C_weak_norm(i)  = 2 · (C_weak(i) − 0.5)  ∈ [0, 1]  (rescaled to match C_strong scale)

Supported weighting modes (--weighting_mode):
    strong_only       : confidence_weight = max(C_strong, 0)              [default]
    strong_raw        : confidence_weight = C_strong  ∈ [-1, 1]
    linear_combined   : confidence_weight = α·C_weak_norm + (1-α)·max(C_strong, 0)
    multiplicative    : confidence_weight = C_weak_norm · max(C_strong, 0)
    strong_gated_weak : confidence_weight = C_weak_norm if C_strong > 0 else 0

Writes updated pseudo_labeled.jsonl with:
    - weak_confidence  = original C_weak from D_l labeling (ALWAYS preserved)
    - strong_confidence = raw C_strong ∈ [-1, 1]          (ALWAYS stored)
    - confidence_weight = computed weight per selected mode (used by CWPOTrainer)

Usage:
    python scripts/compute_strong_confidence.py \\
        --config configs/mwdpo_2phase_hh_rlhf.yaml \\
        --phase1_model_path outputs/.../strong_model_phase1 \\
        --sft_model_path outputs/.../sft_strong \\
        --d_low_path outputs/.../weak_labels/d_low/pseudo_labeled.jsonl \\
        --output_path outputs/.../weak_labels/d_low_scored/pseudo_labeled.jsonl

    # Combined: linear with α=0.3 (70% strong, 30% weak)
    ... --weighting_mode linear_combined --alpha 0.3

    # Combined: multiplicative (both must agree)
    ... --weighting_mode multiplicative

    # Combined: strong-gated weak (strong as binary referee)
    ... --weighting_mode strong_gated_weak

    # Backward compat: raw C_strong (equivalent to old --allow_negative)
    ... --weighting_mode strong_raw

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
            "[DEPRECATED — use --weighting_mode strong_raw] "
            "If set, confidence_weight = raw C_strong ∈ [-1, 1] (backward compat)."
        ),
    )
    parser.add_argument(
        "--weighting_mode",
        type=str,
        default=None,
        choices=["strong_only", "strong_raw", "linear_combined", "multiplicative", "strong_gated_weak"],
        help=(
            "Weighting mode for confidence_weight. "
            "  strong_only       : max(C_strong, 0)  [default if not set]"
            "  strong_raw        : C_strong ∈ [-1,1] (same as legacy --allow_negative)"
            "  linear_combined   : α·C_weak_norm + (1-α)·max(C_strong,0)"
            "  multiplicative    : C_weak_norm · max(C_strong,0)"
            "  strong_gated_weak : C_weak_norm if C_strong>0 else 0"
            "C_weak_norm = 2·(C_weak - 0.5) rescales weak confidence [0.5,1]→[0,1]. "
            "If not set, reads from config strong_confidence.weighting_mode; "
            "falls back to 'strong_raw' if allow_negative=true, else 'strong_only'."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Mixing coefficient for linear_combined mode: "
            "confidence_weight = α·C_weak_norm + (1-α)·max(C_strong,0). "
            "α=0 → pure strong_only, α=1 → pure weak. "
            "Default: read from config strong_confidence.alpha, or 0.3."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Process only first 64 samples (fast smoke test).",
    )
    parser.add_argument(
        "--phase2_data_mode",
        type=str,
        default=None,
        choices=["d_low", "d_weak_asymmetric"],
        help=(
            "After scoring D_l, resolve Phase 2d training jsonl. "
            "d_low (default): stop at d_low_scored. "
            "d_weak_asymmetric: also merge D_h (w=1) ∪ scored D_l → d_weak_scored. "
            "Default: config phase2_training.phase2_data_mode, else d_low."
        ),
    )
    parser.add_argument(
        "--d_high_path",
        type=str,
        default=None,
        help="D_h jsonl (required when phase2_data_mode=d_weak_asymmetric).",
    )
    parser.add_argument(
        "--d_weak_scored_path",
        type=str,
        default=None,
        help=(
            "Output path for ACE union jsonl when phase2_data_mode=d_weak_asymmetric. "
            "Default: config strong_confidence.d_weak_scored_path or "
            "<label_base>/d_weak_scored/pseudo_labeled.jsonl."
        ),
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

# Valid weighting modes
WEIGHTING_MODES = [
    "strong_only",        # max(C_strong, 0)                          — only strong signal
    "strong_raw",         # C_strong ∈ [-1,1]                         — raw strong (legacy allow_negative)
    "linear_combined",    # α·C_weak_norm + (1-α)·max(C_strong,0)    — from proposal
    "multiplicative",     # C_weak_norm · max(C_strong,0)             — conservative consensus
    "strong_gated_weak",  # C_weak_norm if C_strong>0 else 0          — strong as binary gate
]


def compute_c_strong_for_dataset(
    phase1_model,
    sft_model,
    tokenizer,
    samples: List[Dict],
    beta: float,
    max_length: int,
    batch_size: int,
    device: torch.device,
    weighting_mode: str = "strong_only",
    alpha: float = 0.3,
) -> List[Dict]:
    """
    Compute C_strong for all samples and return updated samples with scores.

    Always preserves original C_weak in 'weak_confidence' field BEFORE overwriting
    'confidence_weight'. This ensures C_weak is never lost regardless of mode.

    Updates each sample dict with:
        weak_confidence   = original confidence_weight from D_l labeling ∈ [0.5, 1]
        strong_confidence = raw C_strong ∈ [-1, 1]
        confidence_weight = computed weight per weighting_mode (used by CWPOTrainer)

    Weighting modes:
        strong_only       : max(C_strong, 0)
        strong_raw        : C_strong  ∈ [-1, 1]
        linear_combined   : α·C_weak_norm + (1-α)·max(C_strong,0)
        multiplicative    : C_weak_norm · max(C_strong,0)
        strong_gated_weak : C_weak_norm if C_strong > 0 else 0

    where C_weak_norm = 2·(C_weak - 0.5) ∈ [0, 1]  (rescales [0.5,1] → [0,1])

    Args:
        weighting_mode: One of WEIGHTING_MODES. Default: 'strong_only'.
        alpha:          Mixing coefficient for linear_combined. Default: 0.3
                        (α=0.3 → 30% weak + 70% strong).
    """
    if weighting_mode not in WEIGHTING_MODES:
        raise ValueError(
            f"Unknown weighting_mode='{weighting_mode}'. Choose from: {WEIGHTING_MODES}"
        )

    logger.info(
        f"Computing C_strong for {len(samples)} samples "
        f"(β={beta}, max_len={max_length}, batch={batch_size}, "
        f"weighting_mode='{weighting_mode}'"
        + (f", α={alpha}" if weighting_mode == "linear_combined" else "")
        + ")"
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    updated = []
    total_batches = math.ceil(len(samples) / batch_size)

    for batch_idx in range(total_batches):
        batch_samples = samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        n = len(batch_samples)

        # ── Preserve original C_weak BEFORE any overwrite ─────────────────
        # c_weak is σ(S_w - S_l) ∈ [0.5, 1] from the D_l labeling step.
        # We save it as 'weak_confidence' so it is never lost.
        # Fallback: if the field is missing (legacy files), use 0.5 (neutral).
        c_weak_list = [
            float(s.get("p_ensemble", s.get("confidence_weight", 0.5)))
            for s in batch_samples
        ]
        # Normalize C_weak: [0.5, 1] → [0, 1]
        c_weak_norm = torch.tensor(
            [2.0 * (w - 0.5) for w in c_weak_list], dtype=torch.float32
        )  # (n,)

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
        # Keep on CPU so mixing with C_weak_norm (CPU) is device-safe.
        c_strong = (2.0 * (torch.sigmoid(r_chosen - r_rejected) - 0.5)).detach().cpu()
        c_strong_pos = torch.clamp(c_strong, min=0.0)  # max(C_strong, 0) ∈ [0,1]
        c_weak_norm = c_weak_norm.to(dtype=c_strong.dtype)  # already CPU

        # ── Apply weighting mode ──────────────────────────────────────────
        if weighting_mode == "strong_only":
            # max(C_strong, 0) — only strong signal, ignore weak
            confidence_weight = c_strong_pos

        elif weighting_mode == "strong_raw":
            # Raw C_strong ∈ [-1, 1] — allow negative weights
            confidence_weight = c_strong

        elif weighting_mode == "linear_combined":
            # α·C_weak_norm + (1-α)·max(C_strong,0)  ∈ [0, 1]
            # α=0.3: 30% weak + 70% strong (trust strong model more)
            confidence_weight = alpha * c_weak_norm + (1.0 - alpha) * c_strong_pos

        elif weighting_mode == "multiplicative":
            # C_weak_norm · max(C_strong,0)  ∈ [0, 1]
            # Both must be positive and confident for non-trivial weight
            confidence_weight = c_weak_norm * c_strong_pos

        elif weighting_mode == "strong_gated_weak":
            # C_weak_norm if C_strong > 0, else 0
            # Strong acts as binary gate; weak determines magnitude
            gate = (c_strong > 0).float()  # 1.0 if strong agrees, 0.0 if not
            confidence_weight = c_weak_norm * gate

        else:
            # Unreachable due to earlier validation, but defensive fallback
            confidence_weight = c_strong_pos

        c_strong_list    = c_strong.cpu().tolist()
        conf_weight_list = confidence_weight.cpu().tolist()

        for i, sample in enumerate(batch_samples):
            new_sample = dict(sample)
            # Always preserve original C_weak before overwriting confidence_weight
            new_sample["weak_confidence"]    = c_weak_list[i]        # original C_weak ∈ [0.5,1]
            new_sample["strong_confidence"]  = c_strong_list[i]      # raw C_strong, always stored
            new_sample["confidence_weight"]  = conf_weight_list[i]   # used by CWPOTrainer
            updated.append(new_sample)

        if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
            logger.info(
                f"  Processed {min((batch_idx+1)*batch_size, len(samples))}/{len(samples)} samples "
                f"| mean C_strong so far: {sum(s['strong_confidence'] for s in updated)/len(updated):.4f}"
            )

    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 training jsonl (legacy D_l-only vs ACE D_weak asymmetric)
# Shared implementation lives in src.utils.phase2_train_data (imported by pipeline too).
# ─────────────────────────────────────────────────────────────────────────────

from src.utils.phase2_train_data import (  # noqa: E402
    PHASE2_DATA_MODES,
    build_phase2_train_jsonl,
)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis / summary
# ─────────────────────────────────────────────────────────────────────────────

def print_c_strong_analysis(samples: List[Dict]) -> None:
    """Print distribution statistics for C_strong, C_weak, and confidence_weight."""
    c_vals  = [s["strong_confidence"] for s in samples]
    w_vals  = [s["confidence_weight"]  for s in samples]
    n = len(c_vals)

    n_positive = sum(1 for c in c_vals if c > 0)
    n_zero     = sum(1 for c in c_vals if c == 0.0)
    n_negative = sum(1 for c in c_vals if c < 0)

    logger.info("=" * 60)
    logger.info("CONFIDENCE WEIGHT DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"  Total samples     : {n:,}")
    logger.info(f"  C_strong > 0      : {n_positive:,}  ({100*n_positive/n:.1f}%)  ← strong agrees with weak label")
    logger.info(f"  C_strong = 0      : {n_zero:,}  ({100*n_zero/n:.1f}%)")
    logger.info(f"  C_strong < 0      : {n_negative:,}  ({100*n_negative/n:.1f}%)  ← strong disagrees")
    logger.info(f"  Mean C_strong     : {sum(c_vals)/n:.4f}")
    logger.info(f"  Std C_strong      : {(sum((c - sum(c_vals)/n)**2 for c in c_vals)/n)**0.5:.4f}")
    logger.info(f"  Min C_strong      : {min(c_vals):.4f}")
    logger.info(f"  Max C_strong      : {max(c_vals):.4f}")

    # C_weak stats (if preserved)
    if samples and "weak_confidence" in samples[0]:
        wk_vals = [s["weak_confidence"] for s in samples]
        wk_norm = [2.0 * (v - 0.5) for v in wk_vals]
        logger.info(f"  Mean C_weak       : {sum(wk_vals)/n:.4f}  (raw ∈ [0.5,1])")
        logger.info(f"  Mean C_weak_norm  : {sum(wk_norm)/n:.4f}  (rescaled ∈ [0,1])")
        logger.info(f"  Std C_weak_norm   : {(sum((v - sum(wk_norm)/n)**2 for v in wk_norm)/n)**0.5:.4f}")
    else:
        logger.info("  C_weak            : not available in this dataset")

    logger.info(f"  Mean conf_weight  : {sum(w_vals)/n:.4f}  ← final weight used by CWPOTrainer")
    n_nonzero = sum(1 for w in w_vals if w > 0)
    logger.info(f"  Non-zero weights  : {n_nonzero:,}  ({100*n_nonzero/n:.1f}%)  ← effective training samples")

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

    # ── Resolve weighting_mode (CLI > config > backward-compat allow_negative) ─
    sc_cfg = cfg.get("strong_confidence", {})
    if args.weighting_mode is not None:
        weighting_mode = args.weighting_mode
    elif sc_cfg.get("weighting_mode", None) is not None:
        weighting_mode = str(sc_cfg["weighting_mode"])
    elif sc_cfg.get("allow_negative", False) or args.allow_negative:
        # Backward compat: allow_negative=true → strong_raw
        weighting_mode = "strong_raw"
        logger.warning(
            "[DEPRECATED] allow_negative detected. "
            "Please migrate to: strong_confidence.weighting_mode: strong_raw"
        )
    else:
        weighting_mode = "strong_only"

    if weighting_mode not in WEIGHTING_MODES:
        raise ValueError(
            f"Invalid weighting_mode='{weighting_mode}'. Choose from: {WEIGHTING_MODES}"
        )

    # ── Resolve alpha (for linear_combined) ────────────────────────────────────
    if args.alpha is not None:
        alpha = float(args.alpha)
    else:
        alpha = float(sc_cfg.get("alpha", 0.3))  # default 0.3: 30% weak + 70% strong

    logger.info(
        f"Weighting mode: '{weighting_mode}'"
        + (f" (α={alpha}" + ")" if weighting_mode == "linear_combined" else "")
    )

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

    # ── Compute C_strong + apply weighting mode ───────────────────────────────
    updated_samples = compute_c_strong_for_dataset(
        phase1_model=phase1_model,
        sft_model=sft_model,
        tokenizer=tokenizer,
        samples=samples,
        beta=beta,
        max_length=max_length,
        batch_size=args.batch_size,
        device=device,
        weighting_mode=weighting_mode,
        alpha=alpha,
    )

    # ── Analysis ──────────────────────────────────────────────────────────────
    print_c_strong_analysis(updated_samples)

    # ── Write output ──────────────────────────────────────────────────────────
    logger.info(f"Writing {len(updated_samples):,} scored samples to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in updated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Done writing scored D_l.")

    # ── Optional ACE merge (CLI only when --phase2_data_mode is explicit) ─────
    # Pipeline always calls build_phase2_train_jsonl itself after scoring, so we
    # must NOT auto-merge from config here (would merge full D_h under --debug).
    if args.phase2_data_mode is not None:
        phase2_data_mode = args.phase2_data_mode
        if phase2_data_mode == "d_weak_asymmetric":
            d_high_path = args.d_high_path
            if not d_high_path:
                d_low_dir = os.path.dirname(os.path.abspath(args.d_low_path))
                label_base = os.path.dirname(d_low_dir)
                d_high_path = os.path.join(label_base, "d_high", "pseudo_labeled.jsonl")

            if args.d_weak_scored_path:
                d_weak_scored_path = args.d_weak_scored_path
            else:
                d_weak_scored_path = sc_cfg.get("d_weak_scored_path", None)
                if not d_weak_scored_path:
                    d_low_dir = os.path.dirname(os.path.abspath(args.d_low_path))
                    label_base = os.path.dirname(d_low_dir)
                    d_weak_scored_path = os.path.join(
                        label_base, "d_weak_scored", "pseudo_labeled.jsonl"
                    )

            build_phase2_train_jsonl(
                phase2_data_mode=phase2_data_mode,
                d_high_path=d_high_path,
                d_low_scored_path=output_path,
                d_weak_scored_path=d_weak_scored_path,
            )
        else:
            logger.info(
                f"[Phase2 data] CLI mode={phase2_data_mode} → scored D_l only (no ACE merge)."
            )
    else:
        logger.info(
            "[Phase2 data] scored D_l written; ACE merge (if any) is handled by the pipeline."
        )

if __name__ == "__main__":
    main()
