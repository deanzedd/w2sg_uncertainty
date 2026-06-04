#!/usr/bin/env python3
"""
Phase 2: Weak Labeling — generate pseudo-labeled dataset D̂.

WDPO: uses DPO implicit reward  r_w(x,y) = β·(log π_w − log π_ref)
CWPO: uses scalar reward model  score πw(x,y) → confidence C

Usage:
    # WDPO labeling
    python scripts/label_weak.py --config configs/wdpo_hh_rlhf.yaml
    python scripts/label_weak.py --config configs/wdpo_hh_rlhf.yaml --debug --max_samples 100

    # CWPO labeling (requires reward model checkpoint)
    python scripts/label_weak.py --config configs/cwpo_hh_rlhf.yaml \
        --reward_model_path outputs/cwpo/hh_rlhf/reward_model/checkpoint-final/model.pt
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

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
    max_samples = args.max_samples if args.debug else args.max_samples
    logger.info(f"D_u size: {len(unlabeled_ds)}")

    # ── Build labeler ────────────────────────────────────────────────────
    if method == "wdpo":
        labeler = _build_wdpo_labeler(cfg, device)
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


def _build_wdpo_labeler(cfg, device: str) -> DPORewardLabeler:
    """Build WDPO labeler: load weak model + reference model."""
    logger.info(f"Loading weak model for WDPO: {cfg.weak_model_name}")
    weak_wrapper = get_model_wrapper(cfg.weak_model_name, cfg)
    ref_model = weak_wrapper.get_ref_model()

    return DPORewardLabeler(
        weak_model=weak_wrapper.model,
        ref_model=ref_model,
        tokenizer=weak_wrapper.tokenizer,
        beta=float(cfg.get("beta", 0.1)),
        max_length=cfg.get("max_length", 512),
        device=device,
    )


def _build_cwpo_labeler(cfg, reward_model_path: str, device: str) -> ConfidenceLabeler:
    """Build CWPO labeler: load trained scalar reward model."""
    import torch

    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float32
    logger.info(f"Loading CWPO reward model backbone: {cfg.weak_model_name}")
    reward_model, tokenizer = load_reward_model_and_tokenizer(
        cfg.weak_model_name,
        cache_dir=cfg.get("cache_dir"),
        dtype=dtype,
    )

    if reward_model_path:
        logger.info(f"Loading trained reward model weights from: {reward_model_path}")
        state_dict = torch.load(reward_model_path, map_location="cpu")
        reward_model.load_state_dict(state_dict)
    else:
        logger.warning(
            "No --reward_model_path provided! Using untrained reward model. "
            "Run train_reward_model.py first."
        )

    return ConfidenceLabeler(
        reward_model=reward_model,
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 512),
        device=device,
    )


if __name__ == "__main__":
    main()
