#!/usr/bin/env python3
"""
Super_multi_dpo — 2-Phase reward model training script.

Phase 1 (Warmup):
    Train MultiHeadRewardModel (backbone M + K linear heads) jointly on D_l.
    Backbone M warms up in the multi-head context for {phase1_epochs} epochs.
    Uses MultiHeadRewardTrainer with freeze_backbone=False, use_bootstrap_phase1=False/true.

Phase 2 (Ensemble):
    Freeze M. Attach K independent LoRA adapters (random init → diversity source).
    Train each (LoRA_k + head_k) independently on D_l for {phase2_epochs} epochs.
    Uses SuperMultiRewardTrainer with train_mode from config.
    Output: K LoRARewardModel checkpoints in output_dir/model_{0..K-1}/checkpoint-final/

The K LoRA models are ScalarRewardModel-compatible and consumed by label_multi_weak.py.

Config section (super_multi):
    num_models:              3         # K
    phase1_epochs:           3         # epochs for warmup
    phase1_output_dir:       ...       # where to save Phase 1 checkpoint
    phase2_epochs:           2         # epochs for LoRA training
    phase2_learning_rate:    2e-4      # LR for LoRA + heads
    reward_lora_r:           8
    reward_lora_alpha:       16
    reward_lora_dropout:     0.05
    reward_lora_target_modules: null   # PEFT auto-detect
    use_bootstrap_phase1:    false     # bootstrap masking in Phase 1 (warmup joint training)
    use_bootstrap_phase2:    false     # bootstrap masking in Phase 2 (LoRA per-model training)
    # Note: use_bootstrap_phase1 and use_bootstrap_phase2 are independent —
    #   any combination is valid: both false, both true, or either one true.
    train_mode:              sequential  # "sequential" or "parallel"
    agreement_mode:          unanimous
    confidence_threshold:    0.8
    output_dir:              ...       # base dir for K LoRA models

Usage:
    # Full pipeline (Phase 1 + Phase 2):
    python scripts/train_super_multi_reward_model.py \\
        --config configs/super_multi_dpo_hh_rlhf.yaml

    # Phase 2 only (supply Phase 1 checkpoint):
    python scripts/train_super_multi_reward_model.py \\
        --config configs/super_multi_dpo_hh_rlhf.yaml \\
        --phase1_checkpoint outputs/.../phase1_warmup/checkpoint-final

    # Debug (1 epoch, small data):
    python scripts/train_super_multi_reward_model.py \\
        --config configs/super_multi_dpo_hh_rlhf.yaml --debug
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.data import get_dataset
from src.models.multi_head_reward_model import (
    MultiHeadRewardModel,
    load_multi_head_reward_model,
)
from src.trainers.multi_head_reward_trainer import MultiHeadRewardTrainer
from src.trainers.super_multi_reward_trainer import SuperMultiRewardTrainer
from src.utils import (
    finish_wandb,
    init_wandb,
    load_config,
    print_config,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Super_multi_dpo: 2-Phase reward model training"
    )
    parser.add_argument("--config", required=True, help="Path to super_multi_dpo_*.yaml")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 1 epoch, small data")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit D_l samples (debug)")
    parser.add_argument(
        "--phase1_checkpoint", type=str, default=None,
        help=(
            "Path to Phase 1 MultiHeadRewardModel checkpoint-final/ directory. "
            "If provided, Phase 1 training is skipped and Phase 2 starts immediately."
        ),
    )
    parser.add_argument(
        "--skip_phase1", action="store_true",
        help="Skip Phase 1 (alias for --phase1_checkpoint with auto-detected path)."
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides: key=value")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    if args.debug:
        cfg.use_wandb = False

    setup_logging(cfg)
    print_config(cfg)
    set_seed(cfg.seed)

    method = cfg.get("method", "")
    if method != "super_multi_dpo":
        logger.warning(
            f"Config method='{method}' — expected 'super_multi_dpo'. Proceeding anyway."
        )

    # ── Read super_multi config section ───────────────────────────────────
    sm_cfg = cfg.get("super_multi", {})
    num_models            = int(sm_cfg.get("num_models", 3))
    phase1_epochs         = int(sm_cfg.get("phase1_epochs", 3))
    phase1_output_dir     = sm_cfg.get("phase1_output_dir", None)
    phase2_epochs         = int(sm_cfg.get("phase2_epochs", 2))
    phase2_lr             = float(sm_cfg.get("phase2_learning_rate", 2e-4))
    lora_r                = int(sm_cfg.get("reward_lora_r", 8))
    lora_alpha            = int(sm_cfg.get("reward_lora_alpha", 16))
    lora_dropout          = float(sm_cfg.get("reward_lora_dropout", 0.05))
    lora_target_modules   = sm_cfg.get("reward_lora_target_modules", None)
    if lora_target_modules is not None:
        lora_target_modules = list(lora_target_modules)
    use_bootstrap_phase1  = bool(sm_cfg.get("use_bootstrap_phase1", False))
    use_bootstrap_phase2  = bool(sm_cfg.get("use_bootstrap_phase2", False))
    train_mode            = str(sm_cfg.get("train_mode", "sequential"))
    rm_output_dir         = sm_cfg.get(
        "output_dir",
        cfg.get("reward_model", {}).get("output_dir", "outputs/super_multi_dpo/reward_models")
    )

    # Debug overrides
    if args.debug:
        phase1_epochs = 1
        phase2_epochs = 1

    # Derive Phase 1 checkpoint path
    if phase1_output_dir is None:
        base_rm = cfg.get("reward_model", {}).get("output_dir", "outputs/reward_model")
        phase1_output_dir = os.path.join(base_rm, "phase1_warmup")
    phase1_ckpt = args.phase1_checkpoint or os.path.join(phase1_output_dir, "checkpoint-final")

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    dtype      = torch.bfloat16 if cfg.get("bf16", True) else torch.float32
    backbone_name = cfg.weak_model_name
    logger.info(f"Backbone: {backbone_name} | device: {device} | dtype: {dtype}")
    logger.info(f"K = {num_models} | train_mode = {train_mode}")
    logger.info(f"  use_bootstrap_phase1 = {use_bootstrap_phase1}")
    logger.info(f"  use_bootstrap_phase2 = {use_bootstrap_phase2}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        backbone_name,
        cache_dir=cfg.get("cache_dir"),
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Load datasets ─────────────────────────────────────────────────────
    max_s = args.max_samples if args.debug else None
    train_ds = get_dataset(
        cfg.dataset_name,
        split="train",
        labeled_ratio=cfg.labeled_ratio,
        seed=cfg.seed,
        max_samples=max_s,
        cache_dir=cfg.get("cache_dir"),
    )
    labeled_ds, _ = train_ds.get_labeled_unlabeled_split()
    logger.info(f"D_l size: {len(labeled_ds)}")

    eval_ds = get_dataset(
        cfg.dataset_name,
        split="test",
        labeled_ratio=1.0,
        cache_dir=cfg.get("cache_dir"),
    )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1 — Warmup (skip if checkpoint already exists or --skip_phase1)
    # ══════════════════════════════════════════════════════════════════════
    phase1_model_pt = os.path.join(phase1_ckpt, "model.pt")
    skip_phase1 = (
        args.skip_phase1
        or args.phase1_checkpoint is not None
        or os.path.exists(phase1_model_pt)
    )

    if skip_phase1:
        logger.info("═" * 60)
        logger.info(f"Skipping Phase 1 — loading from: {phase1_ckpt}")
        logger.info("═" * 60)
        if not os.path.exists(phase1_model_pt):
            raise FileNotFoundError(
                f"Phase 1 checkpoint not found: {phase1_model_pt}\n"
                f"Run without --skip_phase1 to train Phase 1 first."
            )
    else:
        logger.info("═" * 60)
        logger.info(
            f"Phase 1 (Warmup, {phase1_epochs} epochs): "
            f"Train MultiHeadRewardModel (backbone + {num_models} linear heads) jointly"
        )
        logger.info("═" * 60)

        # Override reward_model epochs for Phase 1
        phase1_cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({
                "reward_model": {
                    "num_train_epochs": phase1_epochs,
                    "output_dir": phase1_output_dir,
                }
            }),
        )

        phase1_model = MultiHeadRewardModel(
            backbone_name=backbone_name,
            num_heads=num_models,
            cache_dir=cfg.get("cache_dir"),
            dtype=dtype,
            head_type="linear",   # Phase 1 always uses linear heads
            freeze_backbone=False,
        )

        init_wandb(
            cfg,
            tags=["reward_model", "super_multi_dpo", "phase1", backbone_name, cfg.dataset_name],
        )

        trainer1 = MultiHeadRewardTrainer(
            model=phase1_model,
            tokenizer=tokenizer,
            cfg=phase1_cfg,
            device=device,
            backbone_name=backbone_name,
            use_bootstrap=use_bootstrap_phase1,
            freeze_backbone=False,
        )
        trainer1.train(labeled_ds, eval_ds)
        finish_wandb()

        logger.info(f"Phase 1 complete. Checkpoint: {phase1_output_dir}/checkpoint-final")
        phase1_ckpt = os.path.join(phase1_output_dir, "checkpoint-final")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2 — LoRA Ensemble
    # ══════════════════════════════════════════════════════════════════════
    logger.info("═" * 60)
    logger.info(
        f"Phase 2 (Ensemble, {phase2_epochs} epochs): "
        f"Freeze backbone → attach {num_models} LoRA adapters → train independently"
    )
    logger.info("═" * 60)

    # Load Phase 1 MultiHeadRewardModel
    logger.info(f"Loading Phase 1 model from: {phase1_ckpt}")
    phase1_model, _ = load_multi_head_reward_model(
        checkpoint_path=phase1_ckpt,
        backbone_name=backbone_name,
        cache_dir=cfg.get("cache_dir"),
        dtype=dtype,
    )

    # Override reward_model output_dir for Phase 2
    phase2_cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create({"reward_model": {"output_dir": rm_output_dir}}),
    )

    init_wandb(
        cfg,
        tags=["reward_model", "super_multi_dpo", "phase2", backbone_name, cfg.dataset_name,
              f"lora_r{lora_r}", train_mode],
    )

    trainer2 = SuperMultiRewardTrainer(
        phase1_model=phase1_model,
        tokenizer=tokenizer,
        cfg=phase2_cfg,
        device=device,
        backbone_name=backbone_name,
        use_bootstrap=use_bootstrap_phase2,
        train_mode=train_mode,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        phase2_epochs=phase2_epochs,
        phase2_lr=phase2_lr,
    )
    trainer2.train(labeled_ds, eval_ds)
    finish_wandb()

    logger.info("=" * 60)
    logger.info(
        f"Super_multi_dpo training complete. "
        f"{num_models} LoRA models saved to: {rm_output_dir}/model_{{0..{num_models-1}}}/checkpoint-final/"
    )
    logger.info(
        f"Next step: run scripts/label_multi_weak.py --config {args.config}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
