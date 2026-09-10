#!/usr/bin/env python3
"""
Phase 3: Train strong model on pseudo-labeled D̂.

Methods:
  - wdpo:              Standard DPO on D̂ (weak-labeled, uniform weights)
  - cwpo:              Confidence-weighted DPO (CW-DPO) on D̂
  - baseline_dpo:      Standard DPO on D_l only (human-labeled, no weak labels)
  - mwdpo_2phase:      Phase 2 only — Debate-Weighted DPO on D_l using C_strong
  - mwdpo_bc_2phase:   Phase 2 only — same as mwdpo_2phase but after multi-head Phase 1

Stage 4 of the W2SG pipeline:
  1. Strong Model SFT was already done in Phase 1a (train_sft.py)
  2. This script does the preference optimization step on top of the SFT checkpoint.
  3. For 2-phase methods, this script is called TWICE:
       - First call  (Phase 1):  method=mwdpo_2phase, --pseudo_labels=D_h → Standard DPO
       - Second call (Phase 2):  method=mwdpo_2phase, --pseudo_labels=D_l_scored,
                                  --phase1_model_path=π_Phase1 → Debate-Weighted DPO

Usage:
    # WDPO
    python scripts/train_strong.py --config configs/wdpo_hh_rlhf.yaml \\
        --pseudo_labels outputs/wdpo/hh_rlhf/weak_labels/pseudo_labeled.jsonl \\
        --sft_model_path outputs/wdpo/hh_rlhf/sft_strong

    # CWPO
    python scripts/train_strong.py --config configs/cwpo_hh_rlhf.yaml --pseudo_labels outputs/cwpo/hh_rlhf/weak_labels/pseudo_labeled.jsonl --sft_model_path outputs/cwpo/hh_rlhf/sft_strong

    # Baseline DPO (no weak labels needed)
    python scripts/train_strong.py --config configs/baseline_dpo_hh_rlhf.yaml \\
        --sft_model_path outputs/baseline_dpo/hh_rlhf/sft_strong

    python scripts/train_strong.py --config configs/mwdpo_bc_hh_rlhf.yaml --pseudo_labels outputs/mwdpo_bc/hh_rlhf/Qwen2.5-1.5B/seed42/weak_labels/d_high/pseudo_labeled.jsonl --sft_model_path outputs/mwdpo_bc/hh_rlhf/Qwen2.5-1.5B/seed42/sft_strong
    python scripts/train_strong.py --config configs/super_multi_dpo_hh_rlhf.yaml --pseudo_labels outputs/super_multi_dpo/hh_rlhf/Qwen2.5-1.5B/seed42/weak_labels/d_high/pseudo_labeled.jsonl --sft_model_path outputs/super_multi_dpo/hh_rlhf/Qwen2.5-1.5B/seed42/sft_strong

    # 2-Phase: Phase 2 Debate-Weighted DPO on D_l
    python scripts/train_strong.py --config configs/mwdpo_2phase_hh_rlhf.yaml \\
        --sft_model_path outputs/.../sft_strong \\
        --phase1_model_path outputs/.../strong_model_phase1 \\
        --pseudo_labels outputs/.../weak_labels/d_low_scored/pseudo_labeled.jsonl

    # Debug
    python scripts/train_strong.py --config configs/wdpo_hh_rlhf.yaml \\
        --pseudo_labels path/to/labels.jsonl --sft_model_path path/to/sft \\
        --debug --max_steps 10
"""

import argparse
import logging
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.data import get_dataset
from src.models import get_model_wrapper
from src.trainers.wdpo_trainer import WDPODataset, WDPOTrainer, build_wdpo_training_args
from src.trainers.cwpo_trainer import CWPOTrainer, build_cwpo_dataset, build_cwpo_training_args
from src.trainers.dpo_trainer import BaselineDPODataset, BaselineDPOTrainer, build_baseline_dpo_args
from src.weak_labeler.base_labeler import BaseWeakLabeler
from src.utils import load_config, print_config, set_seed, setup_logging, init_wandb, finish_wandb

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train strong model (Phase 3)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pseudo_labels", type=str, default=None,
                        help="Path to pseudo_labeled.jsonl from label_weak.py (WDPO/CWPO only)")
    parser.add_argument("--sft_model_path", type=str, default=None,
                        help="Path to SFT checkpoint to initialize strong model from")
    parser.add_argument(
        "--phase1_model_path", type=str, default=None,
        help=(
            "[2-phase methods only] Path to Phase 1 DPO checkpoint (π_Phase1). "
            "Used as the starting point for Phase 2 training. "
            "If the checkpoint contains a LoRA adapter (adapter_config.json), "
            "it will be merged into the SFT base model weights before Phase 2 LoRA is applied. "
            "Required when method is mwdpo_2phase or mwdpo_bc_2phase."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--resume_dpo_checkpoint", type=str, default=None,
        help="Path to a DPO checkpoint directory to resume strong model training from "
             "(e.g. outputs/baseline_dpo/hh_rlhf/strong_model/checkpoint-20200). "
             "When set, model weights are loaded from this checkpoint (not --sft_model_path). "
             "HF Trainer also restores optimizer/scheduler state and skips completed steps. "
             "Works for all methods: baseline_dpo, wdpo, cwpo.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    if args.debug:
        cfg.use_wandb = False
        # Cheap smoke defaults for Phase A / Phase B trainers.
        _dbg_steps = args.max_steps if args.max_steps is not None else 10
        cfg.training.max_steps = _dbg_steps
        if cfg.get("phase2_training") is not None:
            cfg.phase2_training.max_steps = _dbg_steps

    setup_logging(cfg)
    print_config(cfg)
    set_seed(cfg.seed)
    init_wandb(cfg, tags=["strong_training", cfg.method, cfg.dataset_name])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    method = cfg.get("method", "wdpo")

    # ── Load strong model (from SFT checkpoint or base model) ──────────────
    # NOTE: When resuming a LoRA DPO checkpoint, we still load the base model
    # and tokenizer from sft_model_path / cfg.strong_model_name, NOT from the
    # resume checkpoint.  This is because LoRA checkpoints saved by PEFT only
    # contain the adapter weights (adapter_model.safetensors) and no tokenizer
    # files — loading AutoTokenizer from a bare adapter directory fails.
    #
    # The correct LoRA resume flow:
    #   1. Load base model + tokenizer from sft_model_path (has all HF files)
    #   2. _wrap_lora() applies a fresh LoRA config on top
    #   3. trainer.train(resume_from_checkpoint=...) tells HF Trainer to:
    #        a. Re-load the adapter weights from the checkpoint directory
    #        b. Restore optimizer / scheduler state
    #        c. Skip the steps already completed
    model_name = args.sft_model_path or cfg.strong_model_name
    if args.resume_dpo_checkpoint:
        logger.info(
            f"Resume mode: base model loaded from '{model_name}', "
            f"adapter + optimizer state will be restored from checkpoint: "
            f"{args.resume_dpo_checkpoint}"
        )
    else:
        logger.info(f"Loading strong model: {model_name}")
    wrapper = get_model_wrapper(model_name, cfg)
    ref_model = wrapper.get_ref_model()

    # ── Dispatch to training method ──────────────────────────────────────
    if method == "wdpo":
        _train_wdpo(cfg, wrapper, ref_model, args.pseudo_labels, args.resume_dpo_checkpoint)
    elif method == "cwpo":
        _train_cwpo(cfg, wrapper, ref_model, args.pseudo_labels, args.resume_dpo_checkpoint)
    elif method in ("mwdpo", "mwdpo_bootstrap_calibration", "super_multi_dpo"):
        _train_mwdpo(cfg, wrapper, ref_model, args.pseudo_labels, args.resume_dpo_checkpoint)
    elif method in ("mwdpo_2phase", "mwdpo_bc_2phase"):
        # Determine whether this is a Phase 1 call (D_h labels) or Phase 2 call (D_l_scored).
        # Heuristic: if --phase1_model_path is provided → Phase 2 (Debate-Weighted DPO on D_l).
        #            Otherwise → Phase 1 (Standard DPO on D_h, identical to mwdpo).
        if args.phase1_model_path:
            logger.info(f"[{method}] Phase 2: Debate-Weighted DPO on D_l.")
            _train_mwdpo_phase2(
                cfg,
                sft_model_path=args.sft_model_path or model_name,
                phase1_model_path=args.phase1_model_path,
                d_low_scored_path=args.pseudo_labels,
                resume_from_checkpoint=args.resume_dpo_checkpoint,
            )
        else:
            logger.info(f"[{method}] Phase 1: Standard DPO on D_h (same as mwdpo).")
            _train_mwdpo(cfg, wrapper, ref_model, args.pseudo_labels, args.resume_dpo_checkpoint)
    elif method == "baseline_dpo":
        _train_baseline_dpo(cfg, wrapper, ref_model, args.resume_dpo_checkpoint)
    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Choose: wdpo, cwpo, mwdpo, mwdpo_bootstrap_calibration, super_multi_dpo, "
            f"baseline_dpo, mwdpo_2phase, mwdpo_bc_2phase"
        )

    finish_wandb()
    logger.info("Strong model training complete!")


# ──────────────────────────────────────────────────────────────────────────── #

def _train_wdpo(cfg, wrapper, ref_model, pseudo_labels_path: str,
                resume_from_checkpoint: str = None):
    """
    Phase 3 WDPO: Standard DPO on D_weak.

    D_weak = pseudo-labeled data từ Phase 2 (implicit reward scoring D_u).
    Reference model = π_θ^SFT (frozen copy of the SFT-on-D_weak checkpoint).
    """
    if not pseudo_labels_path:
        raise ValueError("--pseudo_labels is required for WDPO training.")

    logger.info(f"Loading D_weak from: {pseudo_labels_path}")
    pseudo_labeled = BaseWeakLabeler.load(pseudo_labels_path)
    logger.info(f"D_weak size: {len(pseudo_labeled)}")

    # Build HF Dataset (raw text only; TRL handles tokenization)
    train_dataset = WDPODataset(
        pseudo_labeled, wrapper.tokenizer,
        max_length=cfg.get("max_length", 512),
        max_prompt_length=cfg.get("max_prompt_length", 256),
    )

    args = build_wdpo_training_args(cfg)
    trainer = WDPOTrainer(
        model=wrapper.model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset.to_hf(),
        processing_class=wrapper.tokenizer,
    )
    if resume_from_checkpoint:
        logger.info(f"Resuming WDPO DPO from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    logger.info(f"WDPO strong model saved to {args.output_dir}")


def _train_cwpo(cfg, wrapper, ref_model, pseudo_labels_path: str,
                resume_from_checkpoint: str = None):
    """
    Phase 3 CWPO: Confidence-Weighted DPO (CW-DPO) on D_weak.

    D_weak = pseudo-labeled data với confidence weights C từ Phase 2.
    C(x,y+,y-) = 2·(σ(πw(x,y+) − πw(x,y-)) − 0.5)
    Reference model = π_θ^SFT (frozen copy of the SFT-on-D_weak checkpoint).

    Confidence weights C are stored in pseudo_labels_path as `confidence_weight`.
    """
    if not pseudo_labels_path:
        raise ValueError("--pseudo_labels is required for CWPO training.")

    logger.info(f"Loading D_weak (with confidence weights) from: {pseudo_labels_path}")
    pseudo_labeled = BaseWeakLabeler.load(pseudo_labels_path)
    logger.info(f"D_weak size: {len(pseudo_labeled)}")

    # Log confidence weight statistics
    conf_weights = [s.get("confidence_weight", 1.0) for s in pseudo_labeled]
    logger.info(
        f"Confidence weight stats — "
        f"min: {min(conf_weights):.4f}, "
        f"max: {max(conf_weights):.4f}, "
        f"mean: {sum(conf_weights)/len(conf_weights):.4f}"
    )

    # Build HF Dataset with raw text + confidence_weight
    # TRL tokenizes prompt/chosen/rejected internally; confidence_weight passes through
    # because remove_unused_columns=False is set in build_cwpo_training_args
    train_dataset = build_cwpo_dataset(pseudo_labeled)

    args = build_cwpo_training_args(cfg)

    logger.info(
        f"CW-DPO training: lr={args.learning_rate}, beta={args.beta}, "
        f"epochs={args.num_train_epochs}, batch={args.per_device_train_batch_size}"
        f"×{args.gradient_accumulation_steps}={args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )

    trainer = CWPOTrainer(
        model=wrapper.model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset,
        processing_class=wrapper.tokenizer,
    )
    if resume_from_checkpoint:
        logger.info(f"Resuming CWPO DPO from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    logger.info(f"CWPO strong model saved to {args.output_dir}")


def _train_mwdpo(cfg, wrapper, ref_model, pseudo_labels_path: str,
                 resume_from_checkpoint: str = None):
    """
    Phase 1 MWDPO: Standard DPO on Phase-1 train labels.

    Path comes from pipeline phase1_data_mode:
      d_high         → agreement-filtered subset
      all_unlabeled  → full D_u with multi-weak ensemble preferences
    Reference model = π_θ^SFT (frozen copy of the matching SFT checkpoint).

    Note: Phase 1 uses STANDARD DPO (not confidence-weighted). The
    confidence_weight field in the data is stored but not used here.
    (Phase 2 will use it with C_combined weighting.)
    """
    if not pseudo_labels_path:
        raise ValueError(
            "--pseudo_labels is required for MWDPO training. "
            "Pass Phase-1 labels from label_multi_weak.py / label_bootstrap_calibration.py "
            "(d_high/ or combined pseudo_labeled.jsonl)."
        )

    logger.info(f"[MWDPO Phase 1] Loading Phase-1 train labels from: {pseudo_labels_path}")
    pseudo_labeled = BaseWeakLabeler.load(pseudo_labels_path)
    logger.info(f"Phase-1 train size: {len(pseudo_labeled)}")

    # Log label-set statistics (in_d_high useful when loading combined D_u)
    conf_weights = [s.get("confidence_weight", 1.0) for s in pseudo_labeled]
    in_d_high = sum(1 for s in pseudo_labeled if s.get("in_d_high", True))
    logger.info(
        f"[MWDPO Phase 1] label stats — "
        f"min_conf: {min(conf_weights):.4f}, "
        f"max_conf: {max(conf_weights):.4f}, "
        f"mean_conf: {sum(conf_weights)/len(conf_weights):.4f}, "
        f"in_d_high: {in_d_high}/{len(pseudo_labeled)}"
    )

    # Standard DPO — reuses WDPOTrainer (identical to standard DPO)
    train_dataset = WDPODataset(
        pseudo_labeled, wrapper.tokenizer,
        max_length=cfg.get("max_length", 512),
        max_prompt_length=cfg.get("max_prompt_length", 256),
    )

    # Reuse WDPO training args (standard DPO)
    args = build_wdpo_training_args(cfg)
    trainer = WDPOTrainer(
        model=wrapper.model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset.to_hf(),
        processing_class=wrapper.tokenizer,
    )
    if resume_from_checkpoint:
        logger.info(f"Resuming MWDPO Phase 1 DPO from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    logger.info(f"[MWDPO Phase 1] Strong model saved to {args.output_dir}")


def _train_mwdpo_phase2(
    cfg,
    sft_model_path: str,
    phase1_model_path: str,
    d_low_scored_path: str,
    resume_from_checkpoint: str = None,
):
    """
    Phase 2: Debate-Weighted DPO on D_l (Residual Training).

    D_l samples must have `confidence_weight` pre-computed by
    compute_strong_confidence.py:
        confidence_weight = max(C_strong, 0.0)  [or raw C_strong if allow_negative]
        C_strong = 2·(σ(β·(r_Phase1(y_w) − r_Phase1(y_l))) − 0.5)  ∈ [-1, 1]

    Training setup:
        π_init  = π_Phase1  (merged into base weights, fresh LoRA applied for Phase 2)
        π_ref   = π_SFT     (frozen reference — same SFT used in Phase 1)
        data    = D_l with C_strong confidence weights
        trainer = CWPOTrainer (confidence-weighted DPO, reusing existing infrastructure)

    Key hyperparameter differences from Phase 1 (in phase2_training config section):
        learning_rate:  2e-6 (vs 5e-6 Phase 1) — conservative on noisy D_l
        beta:           0.7  (vs 0.5 Phase 1)   — stronger KL regularization
        max_grad_norm:  0.5  (vs 1.0 Phase 1)   — tight clipping for stability
        num_epochs:     2    (vs 5   Phase 1)   — fewer epochs on noisy data

    Args:
        cfg:                   OmegaConf config (must have phase2_training section)
        sft_model_path:        Path to π_SFT checkpoint (frozen reference).
        phase1_model_path:     Path to π_Phase1 checkpoint (LoRA adapter or merged).
        d_low_scored_path:     Path to D_l pseudo_labeled.jsonl with C_strong scores.
        resume_from_checkpoint: Optional Phase 2 DPO checkpoint to resume from.
    """
    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.models.base_model import BaseModelWrapper

    if not d_low_scored_path:
        raise ValueError(
            "--pseudo_labels (D_l scored path) is required for Phase 2 training. "
            "Run compute_strong_confidence.py first to generate D_l with C_strong weights."
        )
    if not phase1_model_path:
        raise ValueError(
            "--phase1_model_path is required for Phase 2 training. "
            "Pass the Phase 1 DPO checkpoint directory."
        )

    # ── Load Phase 2 training jsonl (D_l scored, or ACE D_weak asymmetric) ──
    logger.info(f"[Phase 2] Loading Phase 2 labels from: {d_low_scored_path}")
    pseudo_labeled = BaseWeakLabeler.load(d_low_scored_path)
    logger.info(f"[Phase 2] Dataset size: {len(pseudo_labeled)}")

    # Fail fast: every sample must have a finite confidence_weight
    missing_w = [i for i, s in enumerate(pseudo_labeled) if "confidence_weight" not in s]
    if missing_w:
        raise ValueError(
            f"[Phase 2] {len(missing_w)} samples missing confidence_weight "
            f"(first idx={missing_w[0]}) in {d_low_scored_path}"
        )
    conf_weights = []
    for i, s in enumerate(pseudo_labeled):
        try:
            w = float(s["confidence_weight"])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"[Phase 2] Non-numeric confidence_weight at idx={i}: {e}"
            ) from e
        if not math.isfinite(w):
            raise ValueError(
                f"[Phase 2] Non-finite confidence_weight at idx={i}: {w}"
            )
        conf_weights.append(w)

    strong_confs = [s.get("strong_confidence", None) for s in pseudo_labeled]
    n_nonzero    = sum(1 for w in conf_weights if w > 0)
    allow_neg    = any(w < 0 for w in conf_weights)
    logger.info(
        f"[Phase 2] Confidence weight stats — "
        f"min: {min(conf_weights):.4f}, max: {max(conf_weights):.4f}, "
        f"mean: {sum(conf_weights)/len(conf_weights):.4f}, "
        f"non-zero (contributing to training): {n_nonzero}/{len(conf_weights)} "
        f"({100*n_nonzero/len(conf_weights):.1f}%)"
    )

    # Asymmetric ACE split logging / invariants via in_d_high
    n_high = sum(1 for s in pseudo_labeled if s.get("in_d_high", False))
    n_low = len(pseudo_labeled) - n_high
    phase2_data_mode = str(
        cfg.get("phase2_training", {}).get("phase2_data_mode", "d_low")
    )
    if n_high > 0:
        w_high = [
            float(s["confidence_weight"])
            for s in pseudo_labeled if s.get("in_d_high", False)
        ]
        w_low = [
            float(s["confidence_weight"])
            for s in pseudo_labeled if not s.get("in_d_high", False)
        ]
        mean_high = sum(w_high) / len(w_high) if w_high else float("nan")
        mean_low = sum(w_low) / len(w_low) if w_low else float("nan")
        logger.info(
            f"[Phase 2] in_d_high split — n_high={n_high}, n_low={n_low}, "
            f"mean(w|high)={mean_high:.6f}, mean(w|low)={mean_low:.6f}"
        )
        if phase2_data_mode == "d_weak_asymmetric":
            bad = [
                i for i, s in enumerate(pseudo_labeled)
                if s.get("in_d_high", False)
                and abs(float(s["confidence_weight"]) - 1.0) > 1e-6
            ]
            if bad:
                raise ValueError(
                    f"[Phase 2] phase2_data_mode=d_weak_asymmetric requires "
                    f"confidence_weight==1.0 on all in_d_high samples; "
                    f"found {len(bad)} violations (first idx={bad[0]})."
                )
            logger.info(
                "[Phase 2] ACE asymmetric invariant OK: all in_d_high have w=1.0"
            )
    elif phase2_data_mode == "d_weak_asymmetric":
        raise ValueError(
            "[Phase 2] phase2_data_mode=d_weak_asymmetric but no in_d_high=True "
            f"samples in {d_low_scored_path}. Re-run ACE merge "
            "(build_phase2_train_jsonl / pipeline Phase 2c→resolve)."
        )

    if strong_confs[0] is not None:
        sc_vals = [v for v in strong_confs if v is not None]
        logger.info(
            f"[Phase 2] C_strong stats — "
            f"mean: {sum(sc_vals)/len(sc_vals):.4f}, "
            f"n_positive: {sum(1 for v in sc_vals if v>0)} "
            f"({100*sum(1 for v in sc_vals if v>0)/len(sc_vals):.1f}%), "
            f"n_negative: {sum(1 for v in sc_vals if v<0)} "
            f"({100*sum(1 for v in sc_vals if v<0)/len(sc_vals):.1f}%)"
        )
    if allow_neg:
        logger.info(
            "[Phase 2] Negative confidence weights detected (allow_negative=true). "
            "CWPOTrainer will down-weight/reverse these samples."
        )

    # ── Resolve dtype and device_map ─────────────────────────────────────────
    dtype = BaseModelWrapper._resolve_dtype(cfg)
    device_map = BaseModelWrapper._resolve_device_map(cfg)

    # ── Load π_Phase1 and merge LoRA into base weights ───────────────────────
    # We merge so that Phase 2 starts with clean π_Phase1 weights (no adapter stacking).
    # Then a fresh LoRA adapter is applied for Phase 2 training.
    logger.info(f"[Phase 2] Loading Phase 1 model from: {phase1_model_path}")
    is_lora_checkpoint = os.path.exists(
        os.path.join(phase1_model_path, "adapter_config.json")
    )

    if is_lora_checkpoint:
        logger.info(
            "[Phase 2] Detected LoRA adapter in Phase 1 checkpoint. "
            "Loading SFT base + adapter, then merging for clean Phase 2 init."
        )
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft is required. pip install peft")

        load_kwargs = {"torch_dtype": dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        # Load base SFT model
        base_model = AutoModelForCausalLM.from_pretrained(sft_model_path, **load_kwargs)
        if device_map is None and _torch.cuda.is_available():
            base_model = base_model.to("cuda")

        # Apply Phase 1 LoRA adapter
        phase1_model = PeftModel.from_pretrained(base_model, phase1_model_path)

        # Merge Phase 1 LoRA into base weights → clean full model
        logger.info("[Phase 2] Merging Phase 1 LoRA into base weights...")
        phase1_merged = phase1_model.merge_and_unload()
        logger.info("[Phase 2] Merge complete.")
    else:
        # Already a full (non-LoRA) checkpoint
        logger.info("[Phase 2] Loading full Phase 1 model checkpoint (no LoRA detected).")
        load_kwargs = {"torch_dtype": dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        phase1_merged = AutoModelForCausalLM.from_pretrained(phase1_model_path, **load_kwargs)
        if device_map is None and _torch.cuda.is_available():
            phase1_merged = phase1_merged.to("cuda")

    # Enable gradients (merge_and_unload may leave requires_grad=False)
    phase1_merged.requires_grad_(True)

    # ── Apply fresh LoRA for Phase 2 training ────────────────────────────────
    if cfg.get("use_lora", False):
        logger.info("[Phase 2] Applying fresh LoRA adapter for Phase 2 training.")
        phase1_merged = BaseModelWrapper._wrap_lora(phase1_merged, cfg)

    # ── Load π_SFT as frozen reference model ─────────────────────────────────
    logger.info(f"[Phase 2] Loading SFT reference model from: {sft_model_path}")
    ref_kwargs = {"torch_dtype": dtype}
    if device_map is not None:
        ref_kwargs["device_map"] = device_map
    ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path, **ref_kwargs)
    if device_map is None and _torch.cuda.is_available():
        ref_model = ref_model.to("cuda")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    logger.info("[Phase 2] SFT reference model loaded and frozen.")

    # ── Build D_l dataset with C_strong confidence weights ───────────────────
    train_dataset = build_cwpo_dataset(pseudo_labeled)

    # ── Build Phase 2 training args (reads from phase2_training section) ──────
    args = _build_phase2_training_args(cfg)
    logger.info(
        f"[Phase 2] CW-DPO Phase 2: lr={args.learning_rate}, "
        f"beta={args.beta}, epochs={args.num_train_epochs}, "
        f"batch={args.per_device_train_batch_size}×{args.gradient_accumulation_steps}"
        f"={args.per_device_train_batch_size * args.gradient_accumulation_steps}, "
        f"max_grad_norm={args.max_grad_norm}"
    )

    # ── Train with CWPOTrainer ────────────────────────────────────────────────
    # Reuse existing CWPOTrainer — it reads `confidence_weight` from the dataset
    # and applies weighted DPO loss. No changes needed to the trainer itself.
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    trainer = CWPOTrainer(
        model=phase1_merged,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    if resume_from_checkpoint:
        logger.info(f"[Phase 2] Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    logger.info(f"[Phase 2] Debate-Weighted DPO complete. Model saved to {args.output_dir}")


def _build_phase2_training_args(cfg) -> "DPOConfig":
    """
    Build DPOConfig for Phase 2 Debate-Weighted DPO on D_l.

    Reads from cfg.phase2_training section (NOT cfg.training, which is Phase 1).
    Falls back to conservative defaults if phase2_training is missing.
    """
    import torch as _torch
    from trl import DPOConfig
    from src.trainers.sft_trainer import _detect_precision

    # Prefer phase2_training section; fall back to training section with adjusted defaults
    p2_cfg = cfg.get("phase2_training", cfg.get("training", {}))

    _explicit_dm = cfg.get("device_map", None)
    use_device_map = (
        (_explicit_dm is not None)
        or cfg.get("use_lora", False)
        or _torch.cuda.is_available()
    )

    fp16, bf16 = _detect_precision(p2_cfg)

    return DPOConfig(
        output_dir=p2_cfg.get(
            "output_dir",
            cfg.get("training", {}).get("output_dir", "outputs/phase2").replace(
                "phase1", "phase2"
            ),
        ),
        num_train_epochs=p2_cfg.get("num_train_epochs", 2),
        max_steps=int(p2_cfg["max_steps"]) if p2_cfg.get("max_steps", None) is not None else -1,
        per_device_train_batch_size=p2_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=p2_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=p2_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(p2_cfg.get("learning_rate", 2e-6)),    # Conservative: 2e-6
        lr_scheduler_type=p2_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=p2_cfg.get("warmup_steps", 50),
        warmup_ratio=0.0,   # always 0 when warmup_steps is set; avoids HF Trainer conflict
        weight_decay=p2_cfg.get("weight_decay", 0.05),
        optim=p2_cfg.get("optim", "paged_adamw_32bit"),
        logging_steps=p2_cfg.get("logging_steps", 10),
        save_steps=p2_cfg.get("save_steps", 5000),
        eval_steps=p2_cfg.get("eval_steps", 5000),
        fp16=fp16,
        bf16=bf16,
        beta=float(p2_cfg.get("beta", 0.7)),           # Conservative: 0.7 (vs Phase 1: 0.5)
        max_grad_norm=p2_cfg.get("max_grad_norm", 0.5), # Tight: 0.5 (vs Phase 1: 1.0)
        remove_unused_columns=False,   # CRITICAL: preserve confidence_weight column
        gradient_checkpointing=p2_cfg.get("gradient_checkpointing", True),
        ddp_find_unused_parameters=False if use_device_map else None,
        report_to="wandb" if cfg.get("use_wandb", True) else "none",
        run_name=cfg.get("wandb_run_name", None),
    )


def _train_baseline_dpo(cfg, wrapper, ref_model, resume_from_checkpoint: str = None):
    """
    Baseline: Standard DPO on full dataset D (toàn bộ D, không tách D_l/D_u).

    Theo spec: Baseline DPO dùng toàn bộ D (labeled_ratio=1.0) cho cả SFT và DPO.
    """
    logger.info("Baseline DPO: loading full dataset D (labeled_ratio=1.0)...")
    train_ds = get_dataset(
        cfg.dataset_name,
        split="train",
        labeled_ratio=1.0,          # toàn bộ D, không tách labeled/unlabeled
        seed=cfg.seed,
        cache_dir=cfg.get("cache_dir"),
    )
    # Dùng toàn bộ dataset (không split)
    all_ds, _ = train_ds.get_labeled_unlabeled_split()
    logger.info(f"Full dataset D size: {len(all_ds)}")

    samples = list(all_ds)
    train_dataset = BaselineDPODataset(
        samples, wrapper.tokenizer,
        max_length=cfg.get("max_length", 512),
        max_prompt_length=cfg.get("max_prompt_length", 256),
    )

    args = build_baseline_dpo_args(cfg)
    trainer = BaselineDPOTrainer(
        model=wrapper.model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset.to_hf(),
        processing_class=wrapper.tokenizer,
    )
    if resume_from_checkpoint:
        logger.info(f"Resuming Baseline DPO from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    logger.info(f"Baseline DPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
