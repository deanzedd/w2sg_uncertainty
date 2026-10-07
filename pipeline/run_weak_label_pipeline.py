#!/usr/bin/env python3
"""
Weak-labeling-only pipeline orchestrator.

Runs only the phases needed to produce weak labels (D_h / D_l or D_weak)
from a config, then prints a detailed data analysis report.

Does NOT run SFT-strong, DPO-strong, or evaluation.

Supported methods
-----------------
  mwdpo                        → Phase 1a (train k reward models)
                               → Phase 1b (label_multi_weak.py → D_h + D_l)
                               → prints analysis + writes analysis.txt

  super_multi_dpo              → Phase 1a+b (train_super_multi_reward_model.py)
                               → Phase 1c  (label_multi_weak.py → D_h + D_l)
                               → prints analysis + writes analysis.txt

  cwpo                         → Phase 1b (train_reward_model.py)
                               → Phase 2  (label_weak.py → D_weak)
                               → prints analysis + writes analysis.txt

  wdpo                         → Phase 1b+1c (train_weak_model.py)
                               → Phase 2     (label_weak.py → D_weak)
                               → prints analysis + writes analysis.txt

  mwdpo_bootstrap_calibration  → Phase 1a (train_bootstrap_reward_model.py)
                               → Phase 1b (label_bootstrap_calibration.py)
                               → prints analysis + writes analysis.txt

Usage examples
--------------
  # Full run (train reward models + label):
  python pipeline/run_weak_label_pipeline.py --config configs/mwdpo_hh_rlhf.yaml

  # Debug mode (fast, small data):
  python pipeline/run_weak_label_pipeline.py --config configs/mwdpo_hh_rlhf.yaml --debug

  # Skip reward-model training (already trained), run labeling only:
  python pipeline/run_weak_label_pipeline.py --config configs/mwdpo_hh_rlhf.yaml \\
      --skip_reward_model

  # Skip both training and labeling — only (re-)generate the analysis report:
  python pipeline/run_weak_label_pipeline.py --config configs/mwdpo_hh_rlhf.yaml \\
      --skip_reward_model --skip_labeling

  # CWPO with explicit reward model path:
  python pipeline/run_weak_label_pipeline.py --config configs/cwpo_hh_rlhf.yaml \\
      --reward_model_path outputs/cwpo/hh_rlhf/.../reward_model/checkpoint-final

  # WDPO with explicit weak model path:
  python pipeline/run_weak_label_pipeline.py --config configs/wdpo_hh_rlhf.yaml \\
      --weak_model_path outputs/wdpo/hh_rlhf/.../weak_model_dpo \\
      --weak_ref_path  outputs/wdpo/hh_rlhf/.../weak_model_sft

  # Individual reward model for MWDPO (multi-server):
  python pipeline/run_weak_label_pipeline.py --config configs/mwdpo_hh_rlhf.yaml \\
      --model_idx 0   # trains only model 0; repeat for 1, 2, ...

  # Override config values inline:
  python pipeline/run_weak_label_pipeline.py --config configs/mwdpo_hh_rlhf.yaml \\
      multi_weak.num_models=2 seed=99
"""

import argparse
import json
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataset
from src.utils import load_config, print_config, setup_logging, generate_analysis_txt

logger = logging.getLogger(__name__)

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Weak-labeling-only pipeline — trains weak/reward models "
            "and generates weak labels with detailed analytics. "
            "Does NOT run SFT-strong, DPO-strong, or evaluation."
        )
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (e.g. configs/mwdpo_hh_rlhf.yaml)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: 1 epoch, small data, no WandB",
    )

    # ── Phase skips ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--skip_reward_model", action="store_true",
        help=(
            "Skip reward-model / weak-model training. "
            "Use when the model is already trained."
        ),
    )
    parser.add_argument(
        "--skip_weak_model", action="store_true",
        help="Alias for --skip_reward_model (convenience for WDPO users).",
    )
    parser.add_argument(
        "--skip_labeling", action="store_true",
        help=(
            "Skip the labeling step. "
            "Useful to re-generate the analysis report from existing .jsonl files."
        ),
    )

    # ── Pre-computed paths ───────────────────────────────────────────────────
    parser.add_argument(
        "--reward_model_path", type=str, default=None,
        help=(
            "Pre-trained reward model path. "
            "For CWPO: checkpoint-final dir or .pt file. "
            "For MWDPO_BC: MultiHeadRewardModel checkpoint-final dir."
        ),
    )
    parser.add_argument(
        "--weak_model_path", type=str, default=None,
        help="Pre-trained DPO weak model π_w^* path (WDPO only).",
    )
    parser.add_argument(
        "--weak_ref_path", type=str, default=None,
        help="Pre-trained SFT weak model π_w^SFT path used as DPO ref (WDPO only).",
    )

    # ── MWDPO multi-server: train one model at a time ────────────────────────
    parser.add_argument(
        "--model_idx", type=int, default=None,
        help=(
            "Train only the i-th reward model (0-indexed, MWDPO / super_multi_dpo). "
            "When set, skips labeling after training — run without this flag "
            "to run the full labeling step once all models are trained."
        ),
    )

    # ── Analysis-only mode ───────────────────────────────────────────────────
    parser.add_argument(
        "--analysis_only", action="store_true",
        help=(
            "Skip training and labeling; only regenerate analysis.txt from "
            "existing pseudo_labeled.jsonl files on disk."
        ),
    )

    parser.add_argument("overrides", nargs="*", help="Config overrides: key=value")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ─────────────────────────────────────────────────────────────────────────────

def run_script(script_name: str, *extra_args):
    """Run a script in the scripts/ directory and wait for completion."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    cmd = [sys.executable, script_path] + list(extra_args)
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Post-labeling analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(cfg, method: str, label_base_dir: str, config_path: str):
    """
    Load pseudo-labeled .jsonl files from disk and regenerate analysis.txt.

    This is called both after labeling and in --analysis_only / --skip_labeling
    mode so the report reflects the current files on disk.
    """
    logger.info("=" * 60)
    logger.info("POST-LABELING ANALYSIS")
    logger.info("=" * 60)

    # ── Reload original D_u from dataset ────────────────────────────────────
    original_samples = None
    try:
        train_ds = get_dataset(
            cfg.dataset_name,
            split="train",
            labeled_ratio=cfg.labeled_ratio,
            seed=cfg.seed,
            cache_dir=cfg.get("cache_dir"),
        )
        _, unlabeled_ds = train_ds.get_labeled_unlabeled_split()
        original_samples = list(unlabeled_ds)
        logger.info(f"Loaded D_u ({len(original_samples)} samples) for proxy accuracy.")
    except Exception as e:
        logger.warning(f"Could not load original D_u for proxy accuracy: {e}")

    # ── Determine paths and load pseudo-labeled files ────────────────────────
    has_split = method in ("mwdpo", "super_multi_dpo", "mwdpo_bootstrap_calibration")

    if has_split:
        d_high_path = os.path.join(label_base_dir, "d_high", "pseudo_labeled.jsonl")
        d_low_path  = os.path.join(label_base_dir, "d_low",  "pseudo_labeled.jsonl")

        d_high, d_low = [], []
        if os.path.exists(d_high_path):
            d_high = _load_jsonl(d_high_path)
            logger.info(f"Loaded D_h: {len(d_high)} samples from {d_high_path}")
        else:
            logger.warning(f"D_h file not found: {d_high_path}")

        if os.path.exists(d_low_path):
            d_low = _load_jsonl(d_low_path)
            logger.info(f"Loaded D_l: {len(d_low)} samples from {d_low_path}")
        else:
            logger.warning(f"D_l file not found: {d_low_path}")

        if not d_high and not d_low:
            logger.error(
                "Neither D_h nor D_l files found. "
                "Run labeling first (remove --skip_labeling / --analysis_only)."
            )
            return

        # Ensemble config
        mw_cfg = cfg.get("multi_weak", {}) or cfg.get("super_multi", {}) or {}
        num_models = int(mw_cfg.get("num_models", 1))

        analysis_path = generate_analysis_txt(
            output_dir=label_base_dir,
            cfg=cfg,
            method=method,
            d_high=d_high,
            d_low=d_low,
            original_samples=original_samples,
            config_path=config_path,
            num_models=num_models,
        )

        # ── Print summary to stdout ──────────────────────────────────────────
        _print_split_summary(d_high, d_low, original_samples)

    else:
        # cwpo / wdpo
        d_weak_path = os.path.join(label_base_dir, "pseudo_labeled.jsonl")

        if not os.path.exists(d_weak_path):
            logger.error(
                f"D_weak file not found: {d_weak_path}. "
                "Run labeling first (remove --skip_labeling / --analysis_only)."
            )
            return

        d_all = _load_jsonl(d_weak_path)
        logger.info(f"Loaded D_weak: {len(d_all)} samples from {d_weak_path}")

        analysis_path = generate_analysis_txt(
            output_dir=label_base_dir,
            cfg=cfg,
            method=method,
            d_all=d_all,
            original_samples=original_samples,
            config_path=config_path,
            num_models=1,
        )

        # ── Print summary to stdout ──────────────────────────────────────────
        _print_flat_summary(d_all, original_samples)

    logger.info(f"Full analysis written to: {analysis_path}")
    logger.info("")
    logger.info("Tip: view the full report with:  cat " + analysis_path)


# ─────────────────────────────────────────────────────────────────────────────
# Stdout summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_split_summary(d_high, d_low, original_samples):
    total = len(d_high) + len(d_low)
    logger.info("─" * 60)
    logger.info("QUICK SUMMARY")
    logger.info("─" * 60)
    logger.info(f"  D_u total  : {total:,}")
    logger.info(f"  D_h (agree): {len(d_high):,}  ({_pct(len(d_high), total)})")
    logger.info(f"  D_l (disagr): {len(d_low):,}  ({_pct(len(d_low), total)})")

    if d_high:
        avg_conf_h = sum(s.get("confidence_weight", 0) for s in d_high) / len(d_high)
        logger.info(f"  D_h avg confidence : {avg_conf_h:.4f}")
    if d_low:
        avg_conf_l = sum(s.get("confidence_weight", 0) for s in d_low) / len(d_low)
        logger.info(f"  D_l avg confidence : {avg_conf_l:.4f}")

    if original_samples:
        all_pseudo = list(d_high) + list(d_low)
        orig = list(original_samples)[: len(all_pseudo)]
        if len(orig) == len(all_pseudo):
            correct = sum(1 for p, o in zip(all_pseudo, orig)
                          if p.get("chosen") == o.get("chosen"))
            logger.info(
                f"  Proxy accuracy     : {correct/max(1,len(all_pseudo))*100:.1f}%  "
                f"({correct:,}/{len(all_pseudo):,})"
            )
    logger.info("─" * 60)


def _print_flat_summary(d_all, original_samples):
    total = len(d_all)
    logger.info("─" * 60)
    logger.info("QUICK SUMMARY")
    logger.info("─" * 60)
    logger.info(f"  D_weak total : {total:,}")

    if d_all:
        avg_conf = sum(s.get("confidence_weight", 0) for s in d_all) / total
        logger.info(f"  Avg confidence : {avg_conf:.4f}")

    if original_samples:
        orig = list(original_samples)[:total]
        if len(orig) == total:
            correct = sum(1 for p, o in zip(d_all, orig)
                          if p.get("chosen") == o.get("chosen"))
            flipped = total - correct
            logger.info(
                f"  Proxy accuracy : {correct/max(1,total)*100:.1f}%  "
                f"({correct:,}/{total:,})"
            )
            logger.info(
                f"  Label flip rate: {flipped/max(1,total)*100:.1f}%  "
                f"({flipped:,}/{total:,})"
            )
    logger.info("─" * 60)


def _load_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _pct(num, denom):
    if denom == 0:
        return "0.0%"
    return f"{100 * num / denom:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg)

    method = cfg.get("method", "mwdpo")
    debug_flag = ["--debug"] if args.debug else []

    # Normalize skip flags
    skip_model = args.skip_reward_model or args.skip_weak_model or args.analysis_only
    skip_labeling = args.skip_labeling or args.analysis_only

    logger.info("=" * 60)
    logger.info(f"WEAK LABEL PIPELINE  |  method={method}")
    logger.info(f"  Config  : {args.config}")
    logger.info(f"  Dataset : {cfg.dataset_name}  |  seed={cfg.seed}  "
                f"|  labeled_ratio={cfg.labeled_ratio}")
    logger.info(f"  Skip model training : {skip_model}")
    logger.info(f"  Skip labeling       : {skip_labeling}")
    logger.info(f"  Analysis only       : {args.analysis_only}")
    logger.info("=" * 60)

    # ════════════════════════════════════════════════════════════════════════
    # MWDPO — Multi-Weak Agreement DPO
    # ════════════════════════════════════════════════════════════════════════
    if method == "mwdpo":
        mw_cfg = cfg.get("multi_weak", {})
        rm_base_dir = mw_cfg.get(
            "output_dir",
            cfg.get("reward_model", {}).get("output_dir", "outputs/mwdpo/reward_models"),
        )
        label_base_dir = cfg.get(
            "multi_weak_label_output_dir",
            cfg.get("weak_label_output_dir", "outputs/mwdpo/weak_labels"),
        )

        # ── Phase 1a: Train k reward models ─────────────────────────────────
        if not skip_model:
            logger.info("═" * 60)
            logger.info("PHASE 1a: MWDPO — Train k Reward Models on D_l")
            logger.info("═" * 60)
            extra = []
            if args.model_idx is not None:
                extra += ["--model_idx", str(args.model_idx)]
            run_script("train_multi_reward_models.py", "--config", args.config,
                       *extra, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping reward model training.")

        # If --model_idx was set, we only trained one model — do not label yet
        if args.model_idx is not None and not skip_model:
            logger.info(
                f"Trained model {args.model_idx} only. "
                "Re-run without --model_idx once all models are ready to run labeling."
            )
            return

        # ── Phase 1b: Multi-weak labeling ───────────────────────────────────
        if not skip_labeling:
            logger.info("═" * 60)
            logger.info("PHASE 1b: MWDPO — Multi-Weak Labeling D_u → D_h + D_l")
            logger.info("═" * 60)
            run_script("label_multi_weak.py", "--config", args.config,
                       *debug_flag, *args.overrides)
        else:
            logger.info("Skipping labeling — will regenerate analysis from existing files.")

        # ── Analysis ────────────────────────────────────────────────────────
        run_analysis(cfg, method, label_base_dir, args.config)

    # ════════════════════════════════════════════════════════════════════════
    # SUPER_MULTI_DPO — 2-Phase LoRA Ensemble DPO
    # ════════════════════════════════════════════════════════════════════════
    elif method == "super_multi_dpo":
        sm_cfg = cfg.get("super_multi", {})
        label_base_dir = cfg.get(
            "multi_weak_label_output_dir",
            cfg.get("weak_label_output_dir", "outputs/super_multi_dpo/weak_labels"),
        )

        # ── Phase 1a+b: Train super-multi reward model ───────────────────────
        if not skip_model:
            logger.info("═" * 60)
            logger.info("PHASE 1a+b: super_multi_dpo — 2-Phase Reward Model Training")
            logger.info("═" * 60)
            run_script("train_super_multi_reward_model.py", "--config", args.config,
                       *debug_flag, *args.overrides)
        else:
            logger.info("Skipping reward model training.")

        # ── Phase 1c: Multi-weak labeling ────────────────────────────────────
        if not skip_labeling:
            logger.info("═" * 60)
            logger.info("PHASE 1c: super_multi_dpo — Multi-Weak Labeling D_u → D_h + D_l")
            logger.info("═" * 60)
            run_script("label_multi_weak.py", "--config", args.config,
                       *debug_flag, *args.overrides)
        else:
            logger.info("Skipping labeling — will regenerate analysis from existing files.")

        # ── Analysis ────────────────────────────────────────────────────────
        run_analysis(cfg, method, label_base_dir, args.config)

    # ════════════════════════════════════════════════════════════════════════
    # CWPO — Confidence-Weighted DPO
    # ════════════════════════════════════════════════════════════════════════
    elif method == "cwpo":
        label_dir = cfg.get("weak_label_output_dir", "outputs/cwpo/weak_labels")
        reward_model_path = args.reward_model_path or os.path.join(
            cfg.get("reward_model", {}).get("output_dir", f"outputs/cwpo/reward_model"),
            "checkpoint-final",
        )

        # ── Phase 1b: Train scalar reward model ─────────────────────────────
        if not skip_model:
            logger.info("═" * 60)
            logger.info("PHASE 1b: CWPO — Train Scalar Reward Model on D_l (Bradley-Terry)")
            logger.info("═" * 60)
            run_script("train_reward_model.py", "--config", args.config,
                       *debug_flag, *args.overrides)
        else:
            logger.info("Skipping reward model training.")

        # ── Phase 2: Confidence labeling ─────────────────────────────────────
        if not skip_labeling:
            logger.info("═" * 60)
            logger.info("PHASE 2: CWPO — Confidence Labeling D_u → D_weak")
            logger.info("═" * 60)
            extra = []
            if args.reward_model_path and os.path.exists(args.reward_model_path):
                extra += ["--reward_model_path", args.reward_model_path]
            elif os.path.exists(reward_model_path):
                extra += ["--reward_model_path", reward_model_path]
            run_script("label_weak.py", "--config", args.config,
                       *extra, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping labeling — will regenerate analysis from existing files.")

        # ── Analysis ────────────────────────────────────────────────────────
        run_analysis(cfg, method, label_dir, args.config)

    # ════════════════════════════════════════════════════════════════════════
    # WDPO — Weak-to-Strong DPO (implicit reward)
    # ════════════════════════════════════════════════════════════════════════
    elif method == "wdpo":
        label_dir = cfg.get("weak_label_output_dir", "outputs/wdpo/weak_labels")
        weak_model_path = (
            args.weak_model_path
            or cfg.get("weak_model_dpo", {}).get("output_dir", "outputs/wdpo/weak_model_dpo")
        )
        weak_ref_path = (
            args.weak_ref_path
            or cfg.get("weak_model_sft", {}).get("output_dir", "outputs/wdpo/weak_model_sft")
        )

        # ── Phase 1b+1c: Train weak model (SFT → DPO) ───────────────────────
        if not skip_model:
            logger.info("═" * 60)
            logger.info("PHASE 1b+1c: WDPO — Train Weak Model on D_l (SFT → DPO → π_w^*)")
            logger.info("═" * 60)
            run_script("train_weak_model.py", "--config", args.config,
                       *debug_flag, *args.overrides)
        else:
            logger.info("Skipping weak model training.")

        # ── Phase 2: Weak labeling ────────────────────────────────────────────
        if not skip_labeling:
            logger.info("═" * 60)
            logger.info("PHASE 2: WDPO — Weak Labeling D_u → D_weak (implicit reward scoring)")
            logger.info("═" * 60)
            extra = []
            if os.path.exists(weak_model_path):
                extra += ["--weak_model_path", weak_model_path]
            if os.path.exists(weak_ref_path):
                extra += ["--weak_ref_path", weak_ref_path]
            run_script("label_weak.py", "--config", args.config,
                       *extra, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping labeling — will regenerate analysis from existing files.")

        # ── Analysis ────────────────────────────────────────────────────────
        run_analysis(cfg, method, label_dir, args.config)

    # ════════════════════════════════════════════════════════════════════════
    # MWDPO_bootstrap_calibration
    # ════════════════════════════════════════════════════════════════════════
    elif method == "mwdpo_bootstrap_calibration":
        bc_cfg = cfg.get("bootstrap_calibration", {})
        bc_rm_dir = bc_cfg.get(
            "output_dir",
            cfg.get("reward_model", {}).get(
                "output_dir", "outputs/mwdpo_bootstrap_calibration/reward_model"
            ),
        )
        label_base_dir = cfg.get(
            "multi_weak_label_output_dir",
            cfg.get("weak_label_output_dir",
                    "outputs/mwdpo_bootstrap_calibration/weak_labels"),
        )

        # ── Phase 1a: Train MultiHeadRewardModel ─────────────────────────────
        if not skip_model:
            checkpoint_dir = args.reward_model_path or os.path.join(
                bc_rm_dir, "checkpoint-final"
            )
            if os.path.exists(os.path.join(checkpoint_dir, "model.pt")):
                logger.info(
                    f"MultiHeadRewardModel already exists at {checkpoint_dir}. Skipping. "
                    "(Delete checkpoint-final to retrain.)"
                )
            else:
                logger.info("═" * 60)
                logger.info("PHASE 1a: MWDPO_BC — Train MultiHeadRewardModel on D_l")
                logger.info("═" * 60)
                run_script("train_bootstrap_reward_model.py", "--config", args.config,
                           *debug_flag, *args.overrides)
        else:
            logger.info("Skipping reward model training.")

        # ── Phase 1b: Calibrate + Label ──────────────────────────────────────
        if not skip_labeling:
            logger.info("═" * 60)
            logger.info("PHASE 1b: MWDPO_BC — Calibrate T_k + Label D_u → D_h + D_l")
            logger.info("═" * 60)
            extra = []
            if args.reward_model_path:
                extra += ["--checkpoint_dir", args.reward_model_path]
            run_script("label_bootstrap_calibration.py", "--config", args.config,
                       *extra, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping labeling — will regenerate analysis from existing files.")

        # ── Analysis ────────────────────────────────────────────────────────
        run_analysis(cfg, method, label_base_dir, args.config)

    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            "Supported: mwdpo, super_multi_dpo, cwpo, wdpo, mwdpo_bootstrap_calibration"
        )

    logger.info("=" * 60)
    logger.info(f"Weak-label pipeline complete for method={method}!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
