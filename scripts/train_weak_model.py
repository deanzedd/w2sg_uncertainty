#!/usr/bin/env python3
"""
Phase 1 (WDPO): Train Weak Model — SFT + DPO on D_l.

This implements WDPO Option A:
  Step 1: SFT the weak model on (x, y+) from D_labeled → π_w^SFT
  Step 2: DPO train π_w^SFT on D_labeled using standard DPO loss → π_w^*

The resulting π_w^* is used in Phase 2 to score D_unlabeled via implicit reward:
    r_w(x, y) = β · (log π_w^*(y|x) − log π_w^SFT(y|x))

Usage:
    python scripts/train_weak_model.py --config configs/wdpo_hh_rlhf.yaml
    python scripts/train_weak_model.py --config configs/wdpo_hh_rlhf.yaml --debug
    python scripts/train_weak_model.py --config configs/wdpo_hh_rlhf.yaml \\
        strong_model_name=Qwen/Qwen2.5-7B use_lora=true lora_r=16 lora_alpha=32 \\
        sft.gradient_checkpointing=true
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import DictConfig, OmegaConf
from trl import DPOConfig, DPOTrainer

from src.data import get_dataset
from src.models import get_model_wrapper
from src.trainers.sft_trainer import build_sft_args, run_sft, _to_hf_dataset, _detect_precision
from src.utils import load_config, print_config, set_seed, setup_logging, init_wandb, finish_wandb

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train WDPO weak model (SFT + DPO on D_l)")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--debug", action="store_true", help="Debug mode: limited data, no wandb")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_sft", action="store_true",
                        help="Skip weak model SFT (assumes already done)")
    parser.add_argument("--weak_sft_path", type=str, default=None,
                        help="Pre-trained weak SFT model path (to skip SFT step)")
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides: key=value")
    return parser.parse_args()


def _build_weak_sft_config(cfg: DictConfig) -> DictConfig:
    """
    Build a temporary config where sft section = weak_model_sft values.
    Allows reusing build_sft_args() with the weak model's SFT hyperparams.
    """
    weak_sft = cfg.get("weak_model_sft", {})
    # Patch the sft key temporarily
    override = {
        "sft": OmegaConf.to_container(weak_sft, resolve=True),
    }
    return OmegaConf.merge(cfg, OmegaConf.create(override))


def _build_weak_dpo_config(cfg: DictConfig) -> DPOConfig:
    """Build DPOConfig for weak model DPO training from weak_model_dpo section."""
    dpo_cfg = cfg.get("weak_model_dpo", cfg.get("training", {}))
    # weak_cfg passed here already has use_lora=False, device_map=None
    fp16, bf16 = _detect_precision(cfg)

    return DPOConfig(
        output_dir=dpo_cfg.get("output_dir", "outputs/weak_model/dpo"),
        num_train_epochs=dpo_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=dpo_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=dpo_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(dpo_cfg.get("learning_rate", 5e-5)),
        lr_scheduler_type=dpo_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=dpo_cfg.get("warmup_ratio", 0.1),
        weight_decay=dpo_cfg.get("weight_decay", 0.0),
        logging_steps=dpo_cfg.get("logging_steps", 10),
        save_steps=dpo_cfg.get("save_steps", 500),
        eval_steps=dpo_cfg.get("eval_steps", 500),
        fp16=fp16,
        bf16=bf16,
        beta=float(dpo_cfg.get("beta", 0.1)),          # 0.1 for standard WDPO DPO
        max_grad_norm=dpo_cfg.get("max_grad_norm", 1.0),
        remove_unused_columns=False,
        gradient_checkpointing=dpo_cfg.get("gradient_checkpointing", False),
        report_to="wandb" if cfg.get("use_wandb", True) else "none",
        run_name=cfg.get("wandb_run_name", None),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    if args.debug:
        cfg.use_wandb = False
        cfg.weak_model_sft.num_train_epochs = 1
        cfg.weak_model_sft.save_steps = 5
        cfg.weak_model_dpo.num_train_epochs = 1
        cfg.weak_model_dpo.save_steps = 5

    setup_logging(cfg)
    print_config(cfg)
    set_seed(cfg.seed)
    init_wandb(cfg, tags=["weak_model", "wdpo", cfg.dataset_name, cfg.weak_model_name])

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
    logger.info(f"D_l size: {len(labeled_ds)} (used for weak model SFT + DPO)")

    eval_ds = get_dataset(
        cfg.dataset_name,
        split="test",
        labeled_ratio=1.0,
        cache_dir=cfg.get("cache_dir"),
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: SFT the weak model on (x, y+) from D_l → π_w^SFT
    # ─────────────────────────────────────────────────────────────────────
    weak_sft_output = (
        args.weak_sft_path
        or cfg.weak_model_sft.get("output_dir", "outputs/weak_model/sft")
    )

    if not args.skip_sft and not args.weak_sft_path:
        logger.info("=" * 60)
        logger.info("WEAK MODEL STEP 1: SFT on D_l  →  π_w^SFT")
        logger.info("=" * 60)
        logger.info(f"Loading weak model: {cfg.weak_model_name}")

        # Build a temporary cfg with sft = weak_model_sft params
        sft_cfg = _build_weak_sft_config(cfg)

        # IMPORTANT: weak model must always use use_lora=False
        # The --use_lora flag in CLI is intended only for the large strong model.
        # OPT-125m doesn't need LoRA, and saving a LoRA adapter here would
        # break the labeling phase (which loads it as a plain HF model).
        weak_cfg = OmegaConf.merge(cfg, OmegaConf.create({
            "strong_model_name": cfg.weak_model_name,
            "use_lora": False,
            "device_map": None,   # weak model always fits on a single GPU
        }))
        weak_wrapper = get_model_wrapper(cfg.weak_model_name, weak_cfg)

        # Override sft output_dir with weak model paths
        sft_args = build_sft_args(sft_cfg, role="weak")

        sft_trainer = run_sft(
            model=weak_wrapper.model,
            tokenizer=weak_wrapper.tokenizer,
            train_dataset=labeled_ds,
            args=sft_args,
            eval_dataset=eval_ds,
        )
        logger.info(f"Weak model SFT (π_w^SFT) saved to: {sft_args.output_dir}")
        weak_sft_output = sft_args.output_dir

        # ── Free SFT model from GPU memory before loading DPO models ──────
        # Without this, SFT model + DPO policy + DPO ref all coexist in VRAM
        # causing OOM on the next forward pass (especially with bf16→fp32 conversion).
        import gc
        del sft_trainer, weak_wrapper
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("SFT model freed from GPU memory.")
    else:
        logger.info(f"Skipping weak model SFT. Using: {weak_sft_output}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: DPO train π_w^SFT on D_l → π_w^* (fine-tuned weak model)
    # ─────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("WEAK MODEL STEP 2: DPO on D_l  →  π_w^*")
    logger.info("=" * 60)
    logger.info(f"Loading SFT-ed weak model from: {weak_sft_output}")

    # Load the SFT checkpoint as the policy model.
    # Must use use_lora=False: weak model checkpoint was saved without LoRA.
    # Must use device_map=None: OPT-125M fits on a single GPU; using "auto" here
    # triggers accelerate's multi-GPU wrapping which calls convert_to_fp32 on
    # ref_model outputs → OOM even when device_map was passed via CLI as "auto".
    weak_dpo_cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "strong_model_name": cfg.weak_model_name,
        "use_lora": False,
        "device_map": None,   # single-GPU: avoids accelerate fp32-cast OOM
    }))
    policy_wrapper = get_model_wrapper(weak_sft_output, weak_dpo_cfg)

    # Reference model = frozen deep copy of π_w^SFT.
    # Ensure it lives on the same device as the policy model so DPOTrainer
    # does not trigger cross-device tensor moves (another OOM source).
    ref_model = policy_wrapper.get_ref_model()
    if hasattr(policy_wrapper.model, "device"):
        ref_model = ref_model.to(policy_wrapper.model.device)

    # Convert D_l to HF Dataset format for DPOTrainer
    import datasets as hf_datasets
    labeled_records = [labeled_ds[i] for i in range(len(labeled_ds))]
    hf_labeled = hf_datasets.Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in labeled_records
    ])

    # FIX: pass weak_dpo_cfg (use_lora=False, device_map=None), NOT cfg.
    # Passing cfg here was the root bug: cfg has use_lora=True from CLI, which
    # triggers _resolve_device_map() → device_map="auto" inside DPOConfig,
    # causing accelerate to wrap outputs with convert_to_fp32 → OOM.
    dpo_args = _build_weak_dpo_config(weak_dpo_cfg)

    logger.info(f"Training weak DPO model → output: {dpo_args.output_dir}")
    trainer = DPOTrainer(
        model=policy_wrapper.model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=hf_labeled,
        processing_class=policy_wrapper.tokenizer,
    )
    trainer.train()
    trainer.save_model(dpo_args.output_dir)

    # Also save the SFT reference path alongside for labeling phase
    ref_path_file = os.path.join(dpo_args.output_dir, "weak_sft_ref_path.txt")
    with open(ref_path_file, "w") as f:
        f.write(weak_sft_output)
    logger.info(f"SFT reference path saved to: {ref_path_file}")
    logger.info(f"Weak model π_w^* saved to: {dpo_args.output_dir}")

    finish_wandb()
    logger.info("Weak model training (SFT + DPO) complete!")


if __name__ == "__main__":
    main()
