#!/usr/bin/env python3
"""
Train a MultiHeadRewardModel for MWDPO_bootstrap_calibration (Plan 1a / 2-Phase).

Supports two operating modes:

  Mode A — Original (backbone + K heads trained jointly):
    python scripts/train_bootstrap_reward_model.py \\
        --config configs/mwdpo_bc_hh_rlhf.yaml

  Mode B — 2-Phase (pretrain backbone first, then freeze + train K heads):
    Automatic (config-driven):
      bootstrap_calibration.run_phase1_pretrain: true
      python scripts/train_bootstrap_reward_model.py \\
          --config configs/mwdpo_bc_hh_rlhf.yaml

    Manual (supply existing Phase 1 checkpoint):
      python scripts/train_bootstrap_reward_model.py \\
          --config configs/mwdpo_bc_hh_rlhf.yaml \\
          --backbone_checkpoint outputs/.../phase1_backbone/checkpoint-final

Config key: method = mwdpo_bootstrap_calibration
Required config section:
    bootstrap_calibration:
      num_heads: 3
      use_bootstrap: true          # decorrelate heads via bootstrap batch masks
      head_type: mlp               # "linear" (default) or "mlp" (2-layer MLP head)
      mlp_hidden: null             # None = hidden_size // 4
      head_dropout: 0.4            # dropout in MLPRewardHead (0.0 = disabled)
      run_phase1_pretrain: true    # true = run Phase 1 automatically
      phase1_output_dir: null      # null = auto-derive from reward_model.output_dir

Ablations (runtime overrides):
  # Disable bootstrap (all heads see full batch, only init + dropout differ):
  bootstrap_calibration.use_bootstrap=false
  # Disable MLP head (linear head, backward compat):
  bootstrap_calibration.head_type=linear
  # Disable Phase 1 pretrain (original mode):
  bootstrap_calibration.run_phase1_pretrain=false
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf

from src.data import get_dataset
from src.models.multi_head_reward_model import MultiHeadRewardModel
from src.trainers.multi_head_reward_trainer import MultiHeadRewardTrainer
from src.utils import load_config, print_config, set_seed, setup_logging, init_wandb, finish_wandb
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Phase 1 helper                                                              #
# --------------------------------------------------------------------------- #

def _run_phase1_pretrain(
    cfg,
    output_dir: str,
    labeled_ds,
    eval_ds,
    device: str,
    dtype: torch.dtype,
    backbone_name: str,
) -> None:
    """
    Phase 1: Train ScalarRewardModel (backbone + 1 linear head) on D_l.

    Only the backbone weights are used in Phase 2; the scalar_head is discarded.
    Reuses RewardModelTrainer to avoid duplicating training logic.
    The reward_model.output_dir is overridden to output_dir so Phase 1 and
    Phase 2 checkpoints are stored separately.

    Args:
        cfg:          full experiment DictConfig (reward_model HPs are read from it)
        output_dir:   where to save Phase 1 checkpoint (checkpoint-final/)
        labeled_ds:   D_l training dataset
        eval_ds:      evaluation dataset
        device:       "cuda" or "cpu"
        dtype:        torch dtype
        backbone_name: HF model ID
    """
    from src.models.reward_model import load_reward_model_and_tokenizer
    from src.trainers.reward_model_trainer import RewardModelTrainer

    logger.info(f"[Phase 1] Training backbone (ScalarRewardModel) → {output_dir}")

    phase1_cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create({"reward_model": {"output_dir": output_dir}})
    )

    rm, tok = load_reward_model_and_tokenizer(
        backbone_name,
        cache_dir=cfg.get("cache_dir"),
        dtype=dtype,
    )
    trainer = RewardModelTrainer(
        model=rm,
        tokenizer=tok,
        cfg=phase1_cfg,
        device=device,
        backbone_name=backbone_name,
    )
    trainer.train(train_dataset=labeled_ds, eval_dataset=eval_ds)
    logger.info(f"[Phase 1] Backbone saved to: {output_dir}/checkpoint-final")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MultiHeadRewardModel for MWDPO_bootstrap_calibration"
    )
    parser.add_argument("--config", required=True, help="Path to mwdpo_bc_*.yaml config")
    parser.add_argument("--debug", action="store_true", help="Debug: 1 epoch, small data")
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit D_l samples (debug)"
    )
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None,
        help="Path to checkpoint directory to resume Phase 2 training from"
    )
    parser.add_argument(
        "--backbone_checkpoint", type=str, default=None,
        help=(
            "Path to Phase 1 ScalarRewardModel checkpoint-final directory. "
            "If provided, backbone weights are loaded from here and frozen (Phase 2 mode). "
            "If not provided, checks config run_phase1_pretrain flag. "
            "If neither, trains backbone + heads jointly (original Mode A)."
        ),
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

    # ── Read bootstrap_calibration config ─────────────────────────────────
    bc_cfg = cfg.get("bootstrap_calibration", {})
    num_heads     = int(bc_cfg.get("num_heads", 3))
    use_bootstrap = bool(bc_cfg.get("use_bootstrap", True))
    head_type     = str(bc_cfg.get("head_type", "linear"))
    mlp_hidden    = bc_cfg.get("mlp_hidden", None)
    if mlp_hidden is not None:
        mlp_hidden = int(mlp_hidden)
    head_dropout  = float(bc_cfg.get("head_dropout", 0.0))
    run_phase1    = bool(bc_cfg.get("run_phase1_pretrain", False))

    logger.info(
        f"[BC] num_heads={num_heads}, use_bootstrap={use_bootstrap}, "
        f"head_type={head_type}, mlp_hidden={mlp_hidden}, "
        f"head_dropout={head_dropout}, run_phase1_pretrain={run_phase1}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if cfg.get("bf16", True) else torch.float32

    # ── Load D_l ──────────────────────────────────────────────────────────
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
    logger.info(f"D_l size (train): {len(labeled_ds)}")

    eval_ds = get_dataset(
        cfg.dataset_name,
        split="test",
        labeled_ratio=1.0,
        cache_dir=cfg.get("cache_dir"),
    )

    # ── Load tokenizer ────────────────────────────────────────────────────
    backbone_name = cfg.weak_model_name
    logger.info(f"Backbone: {backbone_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        backbone_name,
        cache_dir=cfg.get("cache_dir"),
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Determine Phase 1 / Phase 2 mode ──────────────────────────────────
    backbone_ckpt   = args.backbone_checkpoint   # None if not provided via CLI
    freeze_backbone = backbone_ckpt is not None  # CLI backbone_checkpoint → Phase 2

    if run_phase1 and not freeze_backbone:
        # Config-driven Phase 1: auto-derive output_dir and run pretrain
        rm_out_dir = cfg.reward_model.get("output_dir", "outputs/reward_model")
        phase1_out = bc_cfg.get("phase1_output_dir", None) or os.path.join(
            rm_out_dir, "phase1_backbone"
        )
        phase1_ckpt = os.path.join(phase1_out, "checkpoint-final")

        if os.path.exists(os.path.join(phase1_ckpt, "model.pt")):
            logger.info(
                f"[2-Phase] Phase 1 checkpoint already exists: {phase1_ckpt}. "
                "Skipping Phase 1 training."
            )
        else:
            if args.debug:
                # Debug: limit Phase 1 to 1 epoch as well
                cfg = OmegaConf.merge(
                    cfg,
                    OmegaConf.create({"reward_model": {"num_train_epochs": 1}})
                )
            _run_phase1_pretrain(
                cfg, phase1_out, labeled_ds, eval_ds, device, dtype, backbone_name
            )

        backbone_ckpt   = phase1_ckpt
        freeze_backbone = True

    # ── Build MultiHeadRewardModel ────────────────────────────────────────
    if freeze_backbone:
        logger.info(f"[2-Phase] Phase 2: loading backbone from {backbone_ckpt}, freezing it.")
    else:
        logger.info("[Mode A] Training backbone + K heads jointly (original mode).")

    # Override num_epochs for debug
    if args.debug:
        cfg = OmegaConf.merge(
            cfg, OmegaConf.create({"reward_model": {"num_train_epochs": 1}})
        )

    model = MultiHeadRewardModel(
        backbone_name=backbone_name,
        num_heads=num_heads,
        cache_dir=cfg.get("cache_dir"),
        dtype=dtype,
        head_type=head_type,
        mlp_hidden=mlp_hidden,
        head_dropout=head_dropout,
        freeze_backbone=freeze_backbone,
        backbone_checkpoint=backbone_ckpt,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    tags = ["reward_model", "multi_head", backbone_name, cfg.dataset_name]
    if freeze_backbone:
        tags += ["2phase", "frozen_backbone", head_type]
    if use_bootstrap:
        tags.append("bootstrap")
    init_wandb(cfg, tags=tags)

    trainer = MultiHeadRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        backbone_name=backbone_name,
        use_bootstrap=use_bootstrap,
        freeze_backbone=freeze_backbone,
    )
    trainer.train(
        train_dataset=labeled_ds,
        eval_dataset=eval_ds,
        resume_from_checkpoint=args.resume_checkpoint,
    )
    finish_wandb()

    output_dir = cfg.reward_model.get("output_dir", "outputs/reward_model")
    logger.info(f"[BC] MultiHeadRewardModel saved to: {output_dir}/checkpoint-final")
    logger.info(
        f"[BC] Next step: run scripts/label_bootstrap_calibration.py "
        f"--config {args.config}"
    )


if __name__ == "__main__":
    main()
