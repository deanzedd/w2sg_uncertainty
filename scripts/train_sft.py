#!/usr/bin/env python3
"""
Phase 1a: Supervised Fine-Tuning (SFT) for strong model.

For WDPO:  SFT strong model on D_l (chosen responses)
For CWPO:  SFT strong model on D_l (chosen responses)
For baseline DPO: same SFT

Usage:
    python scripts/train_sft.py --config configs/wdpo_hh_rlhf.yaml
    python scripts/train_sft.py --config configs/cwpo_hh_rlhf.yaml
    python scripts/train_sft.py --config configs/cwpo_hh_rlhf.yaml --debug --max_samples 100
    # Override specific config values:
    python scripts/train_sft.py --config configs/wdpo_hh_rlhf.yaml \
        strong_model_name=facebook/opt-2.7b training.output_dir=outputs/opt2.7b/sft
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataset
from src.models import get_model_wrapper
from src.trainers.sft_trainer import build_sft_args, run_sft
from src.utils import load_config, print_config, set_seed, setup_logging, init_wandb

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for strong model")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--debug", action="store_true", help="Debug mode with limited data")
    parser.add_argument("--max_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("overrides", nargs="*", help="Config overrides: key=value")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config, args.overrides)
    if args.debug:
        cfg.training.num_train_epochs = 1
        cfg.training.max_steps = 10
        cfg.sft.num_train_epochs = 1
        cfg.sft.save_steps = 5
        cfg.use_wandb = False

    setup_logging(cfg)
    print_config(cfg)
    set_seed(cfg.seed)
    init_wandb(cfg, tags=["sft", cfg.dataset_name, cfg.strong_model_name])

    # ── Load dataset ─────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    train_ds = get_dataset(
        cfg.dataset_name,
        split="train",
        labeled_ratio=cfg.labeled_ratio,
        seed=cfg.seed,
        max_samples=args.max_samples if args.debug else None,
        cache_dir=cfg.get("cache_dir"),
    )

    # SFT uses D_l only (labeled split)
    labeled_ds, _ = train_ds.get_labeled_unlabeled_split()
    logger.info(f"D_l size: {len(labeled_ds)}")

    # Test split for eval
    eval_ds = get_dataset(
        cfg.dataset_name,
        split="test",
        labeled_ratio=1.0,  # all test data is "labeled"
        cache_dir=cfg.get("cache_dir"),
    )

    # ── Load model ───────────────────────────────────────────────────────
    logger.info(f"Loading strong model: {cfg.strong_model_name}")
    wrapper = get_model_wrapper(cfg.strong_model_name, cfg)

    # ── SFT training ─────────────────────────────────────────────────────
    sft_args = build_sft_args(cfg, role="strong")
    trainer = run_sft(
        model=wrapper.model,
        tokenizer=wrapper.tokenizer,
        train_dataset=labeled_ds,
        args=sft_args,
        eval_dataset=eval_ds,
    )

    logger.info("SFT complete!")


if __name__ == "__main__":
    main()
