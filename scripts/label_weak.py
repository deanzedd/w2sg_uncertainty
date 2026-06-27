#!/usr/bin/env python3
"""
Phase 2: Weak Labeling — generate pseudo-labeled dataset D̂.

WDPO (Option A): uses DPO implicit reward
    r_w(x,y) = β·(log π_w^*(y|x) − log π_w^SFT(y|x))
    where π_w^* is the DPO-trained weak model and π_w^SFT is its SFT checkpoint.

CWPO (Option B): uses scalar reward model score πw(x,y) → confidence C

Usage:
    # WDPO labeling (requires trained weak model from train_weak_model.py)
    python scripts/label_weak.py --config configs/wdpo_hh_rlhf.yaml \\
        --weak_model_path outputs/wdpo/hh_rlhf/weak_model_dpo \\
        --weak_ref_path outputs/wdpo/hh_rlhf/weak_model_sft
    python scripts/label_weak.py --config configs/wdpo_hh_rlhf.yaml --debug --max_samples 100

    # CWPO labeling (requires reward model checkpoint)
    python scripts/label_weak.py --config configs/cwpo_hh_rlhf.yaml \\
        --reward_model_path outputs/cwpo/hh_rlhf/reward_model/checkpoint-final/model.pt
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf

from src.data import get_dataset
from src.models import get_model_wrapper
from src.models.reward_model import load_reward_model_and_tokenizer, ScalarRewardModel
from src.weak_labeler import DPORewardLabeler, ConfidenceLabeler
from src.utils import load_config, print_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Weak preference labeling (Phase 2)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reward_model_path", type=str, default=None,
                        help="Path to trained reward model .pt (CWPO only)")
    parser.add_argument("--weak_model_path", type=str, default=None,
                        help="Path to DPO-trained weak model π_w^* (WDPO). "
                             "If None, uses weak_model_dpo.output_dir from config.")
    parser.add_argument("--weak_ref_path", type=str, default=None,
                        help="Path to SFT weak model π_w^SFT used as DPO reference (WDPO). "
                             "If None, attempts to read from <weak_model_path>/weak_sft_ref_path.txt")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    if args.debug:
        cfg.use_wandb = False

    setup_logging(cfg)
    print_config(cfg)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    method = cfg.get("method", "wdpo")

    # ── Load unlabeled data D_u ──────────────────────────────────────────
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    train_ds = get_dataset(
        cfg.dataset_name,
        split="train",
        labeled_ratio=cfg.labeled_ratio,
        seed=cfg.seed,
        cache_dir=cfg.get("cache_dir"),
    )
    _, unlabeled_ds = train_ds.get_labeled_unlabeled_split()
    max_samples = args.max_samples
    logger.info(f"D_u size: {len(unlabeled_ds)}")

    # ── Build labeler ────────────────────────────────────────────────────
    if method == "wdpo":
        labeler = _build_wdpo_labeler(cfg, args.weak_model_path, args.weak_ref_path, device)
    elif method == "cwpo":
        labeler = _build_cwpo_labeler(cfg, args.reward_model_path, device)
    else:
        raise ValueError(f"Unknown method '{method}' for weak labeling. Use 'wdpo' or 'cwpo'.")

    # ── Label D_u ────────────────────────────────────────────────────────
    pseudo_labeled = labeler.label_dataset(unlabeled_ds, max_samples=max_samples)

    # ── Save D̂ ──────────────────────────────────────────────────────────
    output_dir = cfg.get("weak_label_output_dir", f"outputs/{method}/weak_labels")
    output_path = os.path.join(output_dir, "pseudo_labeled.jsonl")
    labeler.save(pseudo_labeled, output_path)
    logger.info(f"Pseudo-labeled D̂ saved to {output_path} ({len(pseudo_labeled)} samples)")


def _build_wdpo_labeler(cfg, weak_model_path: str, weak_ref_path: str, device: str) -> DPORewardLabeler:
    """
    Build WDPO labeler.

    Loads:
      - π_w^*   (DPO-trained weak model) from weak_model_path
      - π_w^SFT (SFT reference model)    from weak_ref_path

    The implicit reward is: r_w(x,y) = β·(log π_w^*(y|x) - log π_w^SFT(y|x))

    If weak_model_path is None, falls back to cfg.weak_model_dpo.output_dir.
    If weak_ref_path is None, reads it from <weak_model_path>/weak_sft_ref_path.txt.
    """
    # Resolve DPO-trained weak model path (π_w^*)
    if not weak_model_path:
        weak_model_path = cfg.get("weak_model_dpo", {}).get(
            "output_dir", "outputs/weak_model/dpo"
        )
    if not os.path.exists(weak_model_path):
        raise FileNotFoundError(
            f"WDPO weak model not found at: {weak_model_path}\n"
            f"Run scripts/train_weak_model.py first."
        )
    logger.info(f"Loading WDPO weak model π_w^*: {weak_model_path}")

    # Resolve SFT reference path (π_w^SFT)
    if not weak_ref_path:
        ref_hint_file = os.path.join(weak_model_path, "weak_sft_ref_path.txt")
        if os.path.exists(ref_hint_file):
            with open(ref_hint_file) as f:
                weak_ref_path = f.read().strip()
            logger.info(f"Read SFT ref path from hint file: {weak_ref_path}")
        else:
            # Fall back to the SFT output dir in config
            weak_ref_path = cfg.get("weak_model_sft", {}).get(
                "output_dir", "outputs/weak_model/sft"
            )
            logger.warning(
                f"No --weak_ref_path provided and no hint file found. "
                f"Falling back to: {weak_ref_path}"
            )

    if not os.path.exists(weak_ref_path):
        raise FileNotFoundError(
            f"WDPO SFT reference model not found at: {weak_ref_path}\n"
            f"Run scripts/train_weak_model.py first."
        )
    logger.info(f"Loading WDPO reference model π_w^SFT: {weak_ref_path}")

    # Load both models using the weak model's architecture
    # IMPORTANT: must disable use_lora and device_map for weak model.
    # The CLI flags (use_lora=True, strong_model_name=Qwen/...) are for the
    # strong model only. Weak model (OPT-125m) was saved WITHOUT LoRA.
    weak_cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "strong_model_name": cfg.weak_model_name,
        "use_lora": False,
        "device_map": None,
    }))

    # Load π_w^* (DPO-trained)
    weak_wrapper = get_model_wrapper(weak_model_path, weak_cfg)

    # Load π_w^SFT as the reference (frozen)
    ref_wrapper = get_model_wrapper(weak_ref_path, weak_cfg)
    ref_model = ref_wrapper.model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return DPORewardLabeler(
        weak_model=weak_wrapper.model,
        ref_model=ref_model,
        tokenizer=weak_wrapper.tokenizer,
        beta=float(cfg.get("beta", 0.1)),
        max_length=cfg.get("max_length", 512),
        device=device,
    )


def _build_cwpo_labeler(cfg, reward_model_path: str, device: str) -> ConfidenceLabeler:
    """Build CWPO labeler: load trained scalar reward model.

    R2/RT2 fix: supports two checkpoint formats:
      - Checkpoint directory (new):  contains metadata.json + model.pt + tokenizer files
      - Legacy .pt file path:        state_dict only, falls back to backbone from config
    """
    import os
    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float32

    # Resolve reward_model_path: may be a directory or a .pt file
    resolved_checkpoint_dir = None
    resolved_pt_file = None

    if reward_model_path:
        if os.path.isdir(reward_model_path):
            # New format: checkpoint directory (has metadata.json + model.pt + tokenizer)
            resolved_checkpoint_dir = reward_model_path
        elif os.path.isfile(reward_model_path) and reward_model_path.endswith(".pt"):
            # Legacy format: bare .pt state dict
            resolved_pt_file = reward_model_path
    else:
        # Auto-detect from config: prefer directory over .pt
        rm_output = cfg.get("reward_model", {}).get("output_dir", "")
        default_ckpt_dir = os.path.join(rm_output, "checkpoint-final")
        default_pt = os.path.join(default_ckpt_dir, "model.pt")
        if os.path.isdir(default_ckpt_dir) and os.path.exists(default_pt):
            resolved_checkpoint_dir = default_ckpt_dir
        elif os.path.exists(default_pt):
            resolved_pt_file = default_pt

    if resolved_checkpoint_dir is not None:
        # Mode B: load full checkpoint (new format with metadata.json)
        logger.info(f"Loading CWPO reward model from checkpoint dir: {resolved_checkpoint_dir}")
        reward_model, tokenizer = load_reward_model_and_tokenizer(
            cfg.weak_model_name,        # fallback backbone if metadata.json missing
            cache_dir=cfg.get("cache_dir"),
            dtype=dtype,
            checkpoint_path=resolved_checkpoint_dir,
        )
    elif resolved_pt_file is not None:
        # Mode A legacy: load backbone fresh, then apply state dict
        logger.info(f"Loading CWPO reward model backbone: {cfg.weak_model_name}")
        logger.info(f"Applying legacy state dict from: {resolved_pt_file}")
        reward_model, tokenizer = load_reward_model_and_tokenizer(
            cfg.weak_model_name,
            cache_dir=cfg.get("cache_dir"),
            dtype=dtype,
        )
        state_dict = torch.load(resolved_pt_file, map_location="cpu")
        reward_model.load_state_dict(state_dict)
    else:
        logger.warning(
            "No --reward_model_path provided and no checkpoint found! "
            "Using untrained reward model. Run train_reward_model.py first."
        )
        reward_model, tokenizer = load_reward_model_and_tokenizer(
            cfg.weak_model_name,
            cache_dir=cfg.get("cache_dir"),
            dtype=dtype,
        )

    return ConfidenceLabeler(
        reward_model=reward_model,
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 512),
        device=device,
    )


if __name__ == "__main__":
    main()
