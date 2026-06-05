#!/usr/bin/env python3
"""
Phase 3: Train strong model on pseudo-labeled D̂.

Methods:
  - wdpo:         Standard DPO on D̂ (weak-labeled, uniform weights)
  - cwpo:         Confidence-weighted DPO on D̂
  - baseline_dpo: Standard DPO on D_l only (human-labeled, no weak)

Usage:
    # WDPO
    python scripts/train_strong.py --config configs/wdpo_hh_rlhf.yaml \
        --pseudo_labels outputs/wdpo/hh_rlhf/weak_labels/pseudo_labeled.jsonl \
        --sft_model_path outputs/wdpo/hh_rlhf/sft_strong

    # CWPO
    python scripts/train_strong.py --config configs/cwpo_hh_rlhf.yaml \
        --pseudo_labels outputs/cwpo/hh_rlhf/weak_labels/pseudo_labeled.jsonl \
        --sft_model_path outputs/cwpo/hh_rlhf/sft_strong

    # Baseline DPO (no weak labels needed)
    python scripts/train_strong.py --config configs/baseline_dpo_hh_rlhf.yaml \
        --sft_model_path outputs/baseline_dpo/hh_rlhf/sft_strong

    # Debug
    python scripts/train_strong.py --config configs/wdpo_hh_rlhf.yaml \
        --pseudo_labels path/to/labels.jsonl --sft_model_path path/to/sft \
        --debug --max_steps 10
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from trl import DPOTrainer

from src.data import get_dataset
from src.models import get_model_wrapper
from src.trainers.wdpo_trainer import WDPODataset, build_wdpo_training_args
from src.trainers.cwpo_trainer import CWPODataset, CWPOTrainer, build_cwpo_training_args
from src.trainers.dpo_trainer import BaselineDPODataset, build_baseline_dpo_args
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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    if args.debug:
        cfg.use_wandb = False
        if args.max_steps:
            cfg.training.max_steps = args.max_steps

    setup_logging(cfg)
    print_config(cfg)
    set_seed(cfg.seed)
    init_wandb(cfg, tags=["strong_training", cfg.method, cfg.dataset_name])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    method = cfg.get("method", "wdpo")

    # ── Load strong model ────────────────────────────────────────────────
    model_name = args.sft_model_path or cfg.strong_model_name
    logger.info(f"Loading strong model: {model_name}")
    wrapper = get_model_wrapper(model_name, cfg)
    ref_model = wrapper.get_ref_model().to(device)

    # ── Dispatch to training method ──────────────────────────────────────
    if method == "wdpo":
        _train_wdpo(cfg, wrapper, ref_model, args.pseudo_labels)
    elif method == "cwpo":
        _train_cwpo(cfg, wrapper, ref_model, args.pseudo_labels)
    elif method == "baseline_dpo":
        _train_baseline_dpo(cfg, wrapper, ref_model)
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose: wdpo, cwpo, baseline_dpo")

    finish_wandb()
    logger.info("Strong model training complete!")


# ──────────────────────────────────────────────────────────────────────────── #

def _train_wdpo(cfg, wrapper, ref_model, pseudo_labels_path: str):
    """Train strong model with WDPO (standard DPO on D̂)."""
    if not pseudo_labels_path:
        raise ValueError("--pseudo_labels is required for WDPO training.")

    logger.info(f"Loading pseudo-labeled D̂ from: {pseudo_labels_path}")
    pseudo_labeled = BaseWeakLabeler.load(pseudo_labels_path)
    logger.info(f"D̂ size: {len(pseudo_labeled)}")

    train_dataset = WDPODataset(
        pseudo_labeled, wrapper.tokenizer,
        max_length=cfg.get("max_length", 512),
        max_prompt_length=cfg.get("max_prompt_length", 256),
    )

    args = build_wdpo_training_args(cfg)
    trainer = DPOTrainer(
        model=wrapper.model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset.to_hf(),  # trl 1.5.x needs HF Dataset
        processing_class=wrapper.tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info(f"WDPO model saved to {args.output_dir}")


def _train_cwpo(cfg, wrapper, ref_model, pseudo_labels_path: str):
    """Train strong model with CWPO (confidence-weighted DPO on D̂)."""
    if not pseudo_labels_path:
        raise ValueError("--pseudo_labels is required for CWPO training.")

    logger.info(f"Loading pseudo-labeled D̂ from: {pseudo_labels_path}")
    pseudo_labeled = BaseWeakLabeler.load(pseudo_labels_path)
    logger.info(f"D̂ size: {len(pseudo_labeled)}")

    train_dataset = CWPODataset(
        pseudo_labeled, wrapper.tokenizer,
        max_length=cfg.get("max_length", 512),
        max_prompt_length=cfg.get("max_prompt_length", 256),
    )

    args = build_cwpo_training_args(cfg)
    trainer = CWPOTrainer(
        model=wrapper.model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=wrapper.tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info(f"CWPO model saved to {args.output_dir}")


def _train_baseline_dpo(cfg, wrapper, ref_model):
    """Train strong model with baseline DPO on D_l only."""
    logger.info("Baseline DPO: loading D_l...")
    train_ds = get_dataset(
        cfg.dataset_name,
        split="train",
        labeled_ratio=cfg.labeled_ratio,
        seed=cfg.seed,
        cache_dir=cfg.get("cache_dir"),
    )
    labeled_ds, _ = train_ds.get_labeled_unlabeled_split()
    logger.info(f"D_l size: {len(labeled_ds)}")

    samples = list(labeled_ds)
    train_dataset = BaselineDPODataset(
        samples, wrapper.tokenizer,
        max_length=cfg.get("max_length", 512),
        max_prompt_length=cfg.get("max_prompt_length", 256),
    )

    args = build_baseline_dpo_args(cfg)
    trainer = DPOTrainer(
        model=wrapper.model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset.to_hf(),  # trl 1.5.x needs HF Dataset
        processing_class=wrapper.tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info(f"Baseline DPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
