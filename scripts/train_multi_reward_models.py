#!/usr/bin/env python3
"""
Phase 1a (MWDPO): Train k scalar reward models on D_l for Multi-Weak ensemble.

Each model uses the same backbone and HPs but a different random seed,
providing diversity in the ensemble.

The models are trained SEQUENTIALLY to accommodate single-server or
two-server setups where only one training job runs at a time.

For two-server setups: run with --model_idx to train individual models
on separate servers/GPUs, then merge results automatically.

Saves each model to: {output_dir}/model_{i}/checkpoint-final/

Usage:
    # Train ALL k models sequentially (single server):
    python scripts/train_multi_reward_models.py --config configs/mwdpo_hh_rlhf.yaml

    # Train ONLY model i (for multi-server parallel training):
    python scripts/train_multi_reward_models.py --config configs/mwdpo_hh_rlhf.yaml --model_idx 0
    python scripts/train_multi_reward_models.py --config configs/mwdpo_hh_rlhf.yaml --model_idx 1
    python scripts/train_multi_reward_models.py --config configs/mwdpo_hh_rlhf.yaml --model_idx 2

    # Debug (fast, small data):
    python scripts/train_multi_reward_models.py --config configs/mwdpo_hh_rlhf.yaml --debug
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf

from src.data import get_dataset
from src.models.reward_model import load_reward_model_and_tokenizer
from src.trainers.reward_model_trainer import RewardModelTrainer
from src.utils import load_config, print_config, set_seed, setup_logging, init_wandb, finish_wandb

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train k reward models for Multi-Weak ensemble (MWDPO Phase 1a)"
    )
    parser.add_argument("--config", required=True, help="Path to mwdpo_*.yaml config")
    parser.add_argument(
        "--model_idx", type=int, default=None,
        help=(
            "Index of the specific model to train (0-indexed). "
            "If not set, trains ALL k models sequentially. "
            "Use this flag to run individual models on separate servers."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode: 1 epoch, small data")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--resume_reward_checkpoint", type=str, default=None,
        help=(
            "Path to resume a reward model training from checkpoint. "
            "Only valid when --model_idx is specified."
        ),
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides: key=value")
    return parser.parse_args()


def train_single_reward_model(
    base_cfg,
    model_idx: int,
    seed: int,
    output_dir: str,
    debug: bool = False,
    max_samples: int = None,
    resume_from_checkpoint: str = None,
) -> None:
    """
    Train a single reward model with a given seed and save to output_dir.

    Args:
        base_cfg:               merged OmegaConf config
        model_idx:              index i (for logging)
        seed:                   random seed for this model
        output_dir:             directory to save this model's checkpoint-final/
        debug:                  reduce epochs for fast testing
        max_samples:            limit D_l samples (debug)
        resume_from_checkpoint: path to checkpoint dir to resume from
    """
    # Create a per-model config override (different seed, different output_dir)
    model_cfg = OmegaConf.merge(
        base_cfg,
        OmegaConf.create({
            "seed": seed,
            "reward_model": {
                "output_dir": output_dir,
                "num_train_epochs": 1 if debug else base_cfg.reward_model.get("num_train_epochs", 5),
            },
        })
    )

    logger.info("=" * 60)
    logger.info(f"Training reward model {model_idx} | seed={seed} | output={output_dir}")
    logger.info("=" * 60)

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load D_l ──────────────────────────────────────────────────────────
    train_ds = get_dataset(
        model_cfg.dataset_name,
        split="train",
        labeled_ratio=model_cfg.labeled_ratio,
        seed=model_cfg.seed,
        max_samples=max_samples if debug else None,
        cache_dir=model_cfg.get("cache_dir"),
    )
    labeled_ds, _ = train_ds.get_labeled_unlabeled_split()
    logger.info(f"D_l size: {len(labeled_ds)}")

    eval_ds = get_dataset(
        model_cfg.dataset_name,
        split="test",
        labeled_ratio=1.0,
        cache_dir=model_cfg.get("cache_dir"),
    )

    # ── Load reward model backbone ─────────────────────────────────────────
    logger.info(f"Loading backbone: {model_cfg.weak_model_name}")
    dtype = torch.bfloat16 if model_cfg.get("bf16", True) else torch.float32
    reward_model, tokenizer = load_reward_model_and_tokenizer(
        model_cfg.weak_model_name,
        cache_dir=model_cfg.get("cache_dir"),
        dtype=dtype,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    trainer = RewardModelTrainer(
        model=reward_model,
        tokenizer=tokenizer,
        cfg=model_cfg,
        device=device,
        backbone_name=model_cfg.weak_model_name,
    )
    trainer.train(
        train_dataset=labeled_ds,
        eval_dataset=eval_ds,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    logger.info(f"Reward model {model_idx} saved to: {output_dir}/checkpoint-final")


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    # Validate method
    if cfg.get("method", "") != "mwdpo":
        logger.warning(
            f"Config method='{cfg.get('method')}' — expected 'mwdpo'. Proceeding anyway."
        )

    if args.debug:
        cfg.use_wandb = False

    setup_logging(cfg)
    print_config(cfg)

    # ── Read multi-weak ensemble config ──────────────────────────────────
    mw_cfg = cfg.get("multi_weak", {})
    num_models = mw_cfg.get("num_models", 3)
    seeds = list(mw_cfg.get("seeds", [42, 123, 456]))

    if len(seeds) != num_models:
        raise ValueError(
            f"multi_weak.seeds has {len(seeds)} entries but num_models={num_models}. "
            "They must match."
        )

    # Base output directory for all reward models
    rm_base_dir = mw_cfg.get(
        "output_dir",
        cfg.get("reward_model", {}).get("output_dir", "outputs/mwdpo/reward_models")
    )

    # ── Determine which models to train ──────────────────────────────────
    if args.model_idx is not None:
        if not (0 <= args.model_idx < num_models):
            raise ValueError(
                f"--model_idx {args.model_idx} is out of range [0, {num_models-1}]."
            )
        model_indices = [args.model_idx]
        logger.info(
            f"[Single-model mode] Training model {args.model_idx} of {num_models}."
        )
    else:
        model_indices = list(range(num_models))
        logger.info(
            f"[Sequential mode] Training all {num_models} reward models one after another."
        )

    # ── Train ─────────────────────────────────────────────────────────────
    for idx in model_indices:
        seed = seeds[idx]
        output_dir = os.path.join(rm_base_dir, f"model_{idx}")

        # Check if already trained
        checkpoint_dir = os.path.join(output_dir, "checkpoint-final")
        if os.path.exists(checkpoint_dir) and os.path.exists(
            os.path.join(checkpoint_dir, "model.pt")
        ):
            logger.info(
                f"Reward model {idx} already exists at {checkpoint_dir}. Skipping. "
                f"(Delete the directory to retrain.)"
            )
            continue

        # WandB run for each model (append model index to run name)
        base_run_name = cfg.get("wandb_run_name", None) or "mwdpo-rm"
        cfg.wandb_run_name = f"{base_run_name}-model{idx}"
        init_wandb(
            cfg,
            tags=["reward_model", f"model_{idx}", cfg.dataset_name, cfg.weak_model_name],
        )

        resume_ckpt = args.resume_reward_checkpoint if args.model_idx == idx else None

        train_single_reward_model(
            base_cfg=cfg,
            model_idx=idx,
            seed=seed,
            output_dir=output_dir,
            debug=args.debug,
            max_samples=args.max_samples,
            resume_from_checkpoint=resume_ckpt,
        )
        finish_wandb()

    logger.info("=" * 60)
    logger.info(f"All target reward models trained. Models in: {rm_base_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
