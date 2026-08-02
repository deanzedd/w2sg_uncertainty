#!/usr/bin/env python3
"""
Phase 1b (MWDPO_bootstrap_calibration): Label D_u → D_h / D_l using
BootstrapCalibrationLabeler (1a multi-head + 2a temperature calibration).

Workflow:
    1. Load trained MultiHeadRewardModel (from train_bootstrap_reward_model.py)
    2. [Optional 2a] Calibrate per-head temperatures T_k on D_l val split
       (minimizes NLL on ground-truth labeled pairs)
    3. Score D_u with calibrated ensemble → compute p_ensemble, confidence C
    4. Filter: D_h = unanimous heads + optional confidence threshold
    5. Save D_h and D_l as pseudo_labeled.jsonl (same format as label_multi_weak.py)

Config key: method = mwdpo_bootstrap_calibration
Required config section:
    bootstrap_calibration:
      num_heads: 3
      use_bootstrap: true          # only relevant for train step, logged here for traceability
      use_calibration: true        # 2a — if false, T_k = 1.0 (ablation)
      cal_val_ratio: 0.1           # fraction of D_l used for calibration
      agreement_mode: unanimous
      confidence_threshold: 0.8

Usage:
    # Standard run (reads checkpoint from config reward_model.output_dir):
    python scripts/label_bootstrap_calibration.py \\
        --config configs/mwdpo_bc_hh_rlhf.yaml

    # With explicit MultiHeadRewardModel checkpoint path:
    python scripts/label_bootstrap_calibration.py \\
        --config configs/mwdpo_bc_hh_rlhf.yaml \\
        --checkpoint_dir outputs/mwdpo_bc/hh_rlhf/.../reward_model/checkpoint-final

    # Debug mode (fast, small data, 200 samples):
    python scripts/label_bootstrap_calibration.py \\
        --config configs/mwdpo_bc_hh_rlhf.yaml --debug --max_samples 200

    # Ablation — disable calibration (T_k = 1.0, raw margins):
    python scripts/label_bootstrap_calibration.py \\
        --config configs/mwdpo_bc_hh_rlhf.yaml \\
        bootstrap_calibration.use_calibration=false

    # Override agreement mode to unanimous_with_threshold:
    python scripts/label_bootstrap_calibration.py \\
        --config configs/mwdpo_bc_hh_rlhf.yaml \\
        bootstrap_calibration.agreement_mode=unanimous_with_threshold \\
        bootstrap_calibration.confidence_threshold=0.7

    # Use larger calibration val split (20% of D_l):
    python scripts/label_bootstrap_calibration.py \\
        --config configs/mwdpo_bc_hh_rlhf.yaml \\
        bootstrap_calibration.cal_val_ratio=0.2
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.data import get_dataset
from src.models.multi_head_reward_model import load_multi_head_reward_model
from src.weak_labeler import BootstrapCalibrationLabeler
from src.utils import load_config, print_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap-calibration labeling of D_u → D_h / D_l"
    )
    parser.add_argument("--config", required=True, help="Path to mwdpo_bc_*.yaml config")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help=(
            "Explicit path to MultiHeadRewardModel checkpoint-final directory. "
            "If not given, reads from config: reward_model.output_dir/checkpoint-final"
        ),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit D_u samples for labeling (debug)")
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
    dtype  = torch.bfloat16 if cfg.get("bf16", True) else torch.float32

    # ── Read bootstrap_calibration config ─────────────────────────────────
    bc_cfg            = cfg.get("bootstrap_calibration", {})
    use_calibration   = bool(bc_cfg.get("use_calibration", True))
    cal_val_ratio     = float(bc_cfg.get("cal_val_ratio", 0.1))
    agreement_mode    = str(bc_cfg.get("agreement_mode", "unanimous"))
    conf_threshold    = float(bc_cfg.get("confidence_threshold", 0.8))

    cal_str = "WITH temperature calibration (2a)" if use_calibration else "NO calibration (ablation)"
    logger.info(f"[BC Labeling] {cal_str}")

    # ── Resolve checkpoint path ───────────────────────────────────────────
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        rm_output_dir = cfg.reward_model.get(
            "output_dir", "outputs/mwdpo_bootstrap_calibration/reward_model"
        )
        checkpoint_dir = os.path.join(rm_output_dir, "checkpoint-final")

    if not os.path.exists(os.path.join(checkpoint_dir, "model.pt")):
        raise FileNotFoundError(
            f"MultiHeadRewardModel not found at: {checkpoint_dir}\n"
            f"Expected: {os.path.join(checkpoint_dir, 'model.pt')}\n"
            f"Run scripts/train_bootstrap_reward_model.py first."
        )

    logger.info(f"Loading MultiHeadRewardModel from: {checkpoint_dir}")

    # ── Load model + tokenizer ────────────────────────────────────────────
    model, tokenizer = load_multi_head_reward_model(
        checkpoint_path=checkpoint_dir,
        backbone_name=cfg.weak_model_name,
        cache_dir=cfg.get("cache_dir"),
        dtype=dtype,
    )
    logger.info(f"Loaded K={model.num_heads} heads.")

    # ── Load D_l (for calibration split) ─────────────────────────────────
    train_ds = get_dataset(
        cfg.dataset_name,
        split="train",
        labeled_ratio=cfg.labeled_ratio,
        seed=cfg.seed,
        cache_dir=cfg.get("cache_dir"),
    )
    labeled_ds, unlabeled_ds = train_ds.get_labeled_unlabeled_split()
    logger.info(f"D_l size: {len(labeled_ds)} | D_u size: {len(unlabeled_ds)}")

    # Create calibration val split from D_l
    # We use the LAST cal_val_ratio fraction to avoid overlap with training data
    cal_size = max(1, int(len(labeled_ds) * cal_val_ratio))
    cal_samples = list(labeled_ds)[-cal_size:]
    logger.info(f"Calibration val split: {cal_size} samples ({cal_val_ratio*100:.0f}% of D_l)")

    # ── Build BootstrapCalibrationLabeler ─────────────────────────────────
    labeler = BootstrapCalibrationLabeler(
        model=model,
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 512),
        device=device,
        batch_size=16,
        use_calibration=use_calibration,
        agreement_mode=agreement_mode,
        confidence_threshold=conf_threshold,
    )

    # ── Step 2a: Calibrate temperatures on D_l val split ─────────────────
    if use_calibration:
        # Wrap cal_samples as a simple list (already in dict format)
        labeler.calibrate(
            cal_dataset=cal_samples,
            max_cal_samples=args.max_samples if args.debug else None,
        )
    else:
        logger.info("[BC] Skipping calibration (use_calibration=False). T_k = 1.0.")

    # ── Step 1b: Label D_u → D_h / D_l ──────────────────────────────────
    max_samples = args.max_samples if args.debug else None
    logger.info(
        f"Labeling D_u (agreement_mode='{agreement_mode}', "
        f"threshold={conf_threshold})..."
    )
    d_high, d_low = labeler.label_and_filter_dataset(
        unlabeled_ds, max_samples=max_samples
    )

    # ── Save D_h and D_l ─────────────────────────────────────────────────
    output_base = cfg.get(
        "multi_weak_label_output_dir",
        cfg.get("weak_label_output_dir", "outputs/mwdpo_bootstrap_calibration/weak_labels"),
    )

    d_high_path = os.path.join(output_base, "d_high", "pseudo_labeled.jsonl")
    d_low_path  = os.path.join(output_base, "d_low",  "pseudo_labeled.jsonl")

    labeler.save(d_high, d_high_path)
    labeler.save(d_low,  d_low_path)

    logger.info(f"D_h saved to: {d_high_path} ({len(d_high)} samples)")
    logger.info(f"D_l saved to: {d_low_path}  ({len(d_low)}  samples)")

    # ── Summary ───────────────────────────────────────────────────────────
    total = len(d_high) + len(d_low)
    logger.info("=" * 60)
    logger.info("Bootstrap Calibration Labeling Summary")
    logger.info("=" * 60)
    logger.info(f"  Backbone:         {cfg.weak_model_name}")
    logger.info(f"  K heads:          {model.num_heads}")
    logger.info(f"  use_calibration:  {use_calibration}")
    logger.info(f"  Temperatures:     {[f'{t:.3f}' for t in labeler.temperatures]}")
    logger.info(f"  D_u total:        {total}")
    logger.info(f"  D_h (agree):      {len(d_high)} ({100*len(d_high)/max(1,total):.1f}%)")
    logger.info(f"  D_l (disagr):     {len(d_low)}  ({100*len(d_low)/max(1,total):.1f}%)")
    if d_high:
        avg_conf = sum(s["confidence_weight"] for s in d_high) / len(d_high)
        logger.info(f"  Avg conf (D_h):   {avg_conf:.4f}")
    logger.info("=" * 60)

    # Combined file for compatibility with train_sft.py / train_strong.py
    combined_path = os.path.join(output_base, "pseudo_labeled.jsonl")
    labeler.save(d_high + d_low, combined_path)
    logger.info(f"Combined D_h+D_l saved to: {combined_path}")


if __name__ == "__main__":
    main()
