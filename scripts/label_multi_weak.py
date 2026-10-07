#!/usr/bin/env python3
"""
Phase 1b (MWDPO): Multi-Weak ensemble labeling of D_u → D_h and D_l.

Loads k trained scalar reward models, scores all samples in D_u,
and partitions them into:
  - D_h: samples where all k models UNANIMOUSLY agree on the preference ranking
          (and optionally mean confidence > threshold)
  - D_l: remaining samples (disagreement or low confidence)

Both D_h and D_l are saved as pseudo_labeled.jsonl files for downstream use.

Usage:
    # Standard run (loads reward models automatically from config paths):
    python scripts/label_multi_weak.py --config configs/mwdpo_hh_rlhf.yaml

    # With explicit reward model paths:
    python scripts/label_multi_weak.py --config configs/mwdpo_hh_rlhf.yaml \\
        --reward_model_dirs \\
            outputs/mwdpo/hh_rlhf/reward_models/model_0 \\
            outputs/mwdpo/hh_rlhf/reward_models/model_1 \\
            outputs/mwdpo/hh_rlhf/reward_models/model_2

    # Debug (fast, small subset):
    python scripts/label_multi_weak.py --config configs/mwdpo_hh_rlhf.yaml \\
        --debug --max_samples 100
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch

from src.data import get_dataset
from src.models.reward_model import load_reward_model_and_tokenizer
from src.models.multi_head_reward_model import load_lora_reward_model
from src.weak_labeler import MultiWeakLabeler
from src.weak_labeler.base_labeler import BaseWeakLabeler
from src.utils import load_config, print_config, set_seed, setup_logging, generate_analysis_txt

logger = logging.getLogger(__name__)


def _load_single_reward_model(rm_dir: str, backbone_name: str, cache_dir, dtype):
    """
    Auto-detect and load one reward model from a checkpoint directory.

    If metadata.json says model_type == 'lora_reward_model':
        → use load_lora_reward_model() (Super_multi_dpo LoRA model)
    Otherwise:
        → use load_reward_model_and_tokenizer() (ScalarRewardModel / MWDPO)

    Returns:
        (model, tokenizer)
    """
    metadata_path = os.path.join(rm_dir, "metadata.json")
    is_lora = False
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        is_lora = meta.get("model_type") == "lora_reward_model"

    if is_lora:
        logger.info(f"  [LoRA] Detected LoRA reward model at {rm_dir}")
        return load_lora_reward_model(rm_dir, cache_dir=cache_dir, dtype=dtype)
    else:
        return load_reward_model_and_tokenizer(
            backbone_name,
            cache_dir=cache_dir,
            dtype=dtype,
            checkpoint_path=rm_dir,
        )



def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Weak labeling and D_h/D_l split (MWDPO Phase 1b)"
    )
    parser.add_argument("--config", required=True, help="Path to mwdpo_*.yaml config")
    parser.add_argument(
        "--reward_model_dirs", nargs="+", default=None,
        help=(
            "Explicit paths to k reward model checkpoint dirs (checkpoint-final/ of each). "
            "If not provided, automatically reads from config: "
            "multi_weak.output_dir/model_{0..k-1}/checkpoint-final"
        ),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of D_u samples to label (debug)")
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

    # ── Read multi-weak / super_multi config ───────────────────────────────
    method = cfg.get("method", "mwdpo")
    if method == "super_multi_dpo":
        # Super_multi_dpo stores ensemble config under super_multi:
        ens_cfg = cfg.get("super_multi", {})
        num_models = int(ens_cfg.get("num_models", 3))
        agreement_mode = str(ens_cfg.get("agreement_mode", "unanimous"))
        confidence_threshold = float(ens_cfg.get("confidence_threshold", 0.8))
        rm_base_dir = ens_cfg.get(
            "output_dir",
            cfg.get("reward_model", {}).get("output_dir", "outputs/super_multi_dpo/reward_models")
        )
    else:
        # MWDPO / default: reads from multi_weak:
        mw_cfg = cfg.get("multi_weak", {})
        num_models = mw_cfg.get("num_models", 3)
        agreement_mode = mw_cfg.get("agreement_mode", "unanimous")
        confidence_threshold = float(mw_cfg.get("confidence_threshold", 0.8))
        rm_base_dir = mw_cfg.get(
            "output_dir",
            cfg.get("reward_model", {}).get("output_dir", "outputs/mwdpo/reward_models")
        )

    # ── Resolve reward model paths ─────────────────────────────────────────
    if args.reward_model_dirs:
        if len(args.reward_model_dirs) != num_models:
            raise ValueError(
                f"--reward_model_dirs has {len(args.reward_model_dirs)} paths "
                f"but num_models={num_models}."
            )
        rm_dirs = args.reward_model_dirs
    else:
        rm_dirs = [
            os.path.join(rm_base_dir, f"model_{i}", "checkpoint-final")
            for i in range(num_models)
        ]

    # Validate all reward model dirs exist
    for i, rm_dir in enumerate(rm_dirs):
        # LoRA models use adapter_config.json instead of model.pt
        has_model_pt  = os.path.exists(os.path.join(rm_dir, "model.pt"))
        has_lora      = os.path.exists(os.path.join(rm_dir, "adapter_config.json"))
        if not has_model_pt and not has_lora:
            raise FileNotFoundError(
                f"Reward model {i} not found at: {rm_dir}\n"
                f"Expected model.pt (ScalarRewardModel) or adapter_config.json (LoRA model).\n"
                f"Run scripts/train_multi_reward_models.py or "
                f"scripts/train_super_multi_reward_model.py first."
            )

    logger.info(f"Loading {num_models} reward models from:")
    for i, p in enumerate(rm_dirs):
        logger.info(f"  model_{i}: {p}")

    # ── Load all reward models (auto-detect ScalarRewardModel vs LoRA) ────
    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float32
    reward_models = []
    tokenizer = None  # shared tokenizer (same backbone)

    for i, rm_dir in enumerate(rm_dirs):
        logger.info(f"Loading reward model {i} from {rm_dir}...")
        rm, tok = _load_single_reward_model(
            rm_dir,
            backbone_name=cfg.weak_model_name,
            cache_dir=cfg.get("cache_dir"),
            dtype=dtype,
        )
        reward_models.append(rm)
        if tokenizer is None:
            tokenizer = tok  # all models share the same tokenizer


    # ── Load D_u ──────────────────────────────────────────────────────────
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

    # Keep a reference to the original D_u samples (used later for proxy accuracy)
    original_samples = list(unlabeled_ds)
    if max_samples is not None:
        original_samples = original_samples[:max_samples]

    # ── Build MultiWeakLabeler ────────────────────────────────────────────
    labeler = MultiWeakLabeler(
        reward_models=reward_models,
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 512),
        device=device,
        batch_size=16,
        agreement_mode=agreement_mode,
        confidence_threshold=confidence_threshold,
    )

    # ── Label and filter ──────────────────────────────────────────────────
    logger.info(
        f"Filtering D_u into D_h/D_l "
        f"(agreement_mode='{agreement_mode}', threshold={confidence_threshold})..."
    )
    d_high, d_low = labeler.label_and_filter_dataset(unlabeled_ds, max_samples=max_samples)

    # ── Save D_h and D_l ─────────────────────────────────────────────────
    output_base = cfg.get("multi_weak_label_output_dir",
                           cfg.get("weak_label_output_dir", "outputs/mwdpo/weak_labels"))

    d_high_path = os.path.join(output_base, "d_high", "pseudo_labeled.jsonl")
    d_low_path  = os.path.join(output_base, "d_low",  "pseudo_labeled.jsonl")

    labeler.save(d_high, d_high_path)
    labeler.save(d_low, d_low_path)

    logger.info(f"D_h saved to: {d_high_path} ({len(d_high)} samples)")
    logger.info(f"D_l saved to: {d_low_path}  ({len(d_low)} samples)")

    # ── Summary stats ─────────────────────────────────────────────────────
    total = len(d_high) + len(d_low)
    logger.info("=" * 60)
    logger.info("Multi-Weak Labeling Summary")
    logger.info("=" * 60)
    logger.info(f"  D_u total:   {total}")
    logger.info(f"  D_h (agree): {len(d_high)} ({100*len(d_high)/max(1,total):.1f}%)")
    logger.info(f"  D_l (disagr): {len(d_low)} ({100*len(d_low)/max(1,total):.1f}%)")
    if d_high:
        avg_conf = sum(s["confidence_weight"] for s in d_high) / len(d_high)
        logger.info(f"  Avg conf (D_h): {avg_conf:.4f}")
    logger.info("=" * 60)

    # Also write a combined file for compatibility with existing train_sft.py / train_strong.py
    combined_path = os.path.join(output_base, "pseudo_labeled.jsonl")
    labeler.save(d_high + d_low, combined_path)
    logger.info(f"Combined D_h+D_l saved to: {combined_path}")

    # ── Write analysis.txt ────────────────────────────────────────────────
    mw_cfg = cfg.get("multi_weak", {}) or cfg.get("super_multi", {}) or {}
    num_models = int(mw_cfg.get("num_models", len(rm_dirs)))
    analysis_path = generate_analysis_txt(
        output_dir=output_base,
        cfg=cfg,
        method=method,
        d_high=d_high,
        d_low=d_low,
        original_samples=original_samples,
        config_path=args.config,
        num_models=num_models,
    )
    logger.info(f"Analysis report saved to: {analysis_path}")


if __name__ == "__main__":
    main()
