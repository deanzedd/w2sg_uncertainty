#!/usr/bin/env python3
"""
Phase 1b: Train CWPO Reward Model (weak annotator).

Trains the scalar reward model on D_l using Bradley-Terry loss.
Architecture: pretrained backbone (OPT-125M or Qwen2.5-0.5B)
              + scalar output layer (bypasses LM head).

Usage:
    python scripts/train_reward_model.py --config configs/cwpo_hh_rlhf.yaml
    python scripts/train_reward_model.py --config configs/cwpo_hh_rlhf.yaml --debug
    # Use Qwen2.5-0.5B as weak model:
    python scripts/train_reward_model.py --config configs/cwpo_hh_rlhf.yaml \
        weak_model_name=Qwen/Qwen2.5-0.5B
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataset
from src.models.reward_model import load_reward_model_and_tokenizer
from src.trainers.reward_model_trainer import RewardModelTrainer
from src.utils import load_config, print_config, set_seed, setup_logging, init_wandb

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CWPO weak reward model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    if args.debug:
        cfg.reward_model.num_train_epochs = 1
        cfg.reward_model.save_steps = 5
        cfg.use_wandb = False

    setup_logging(cfg)
    print_config(cfg)
    set_seed(cfg.seed)
    init_wandb(cfg, tags=["reward_model", cfg.dataset_name, cfg.weak_model_name])

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    labeled_ds, _ = train_ds.get_labeled_unlabeled_split()
    logger.info(f"Reward model training on D_l: {len(labeled_ds)} samples")

    eval_ds = get_dataset(
        cfg.dataset_name,
        split="test",
        labeled_ratio=1.0,
        cache_dir=cfg.get("cache_dir"),
    )

    # ── Load reward model ────────────────────────────────────────────────
    logger.info(f"Loading weak backbone: {cfg.weak_model_name}")
    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float32
    reward_model, tokenizer = load_reward_model_and_tokenizer(
        cfg.weak_model_name,
        cache_dir=cfg.get("cache_dir"),
        dtype=dtype,
    )

    # ── Train ────────────────────────────────────────────────────────────
    trainer = RewardModelTrainer(
        model=reward_model,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
    )
    trainer.train(train_dataset=labeled_ds, eval_dataset=eval_ds)
    logger.info("Reward model training complete!")


if __name__ == "__main__":
    main()
