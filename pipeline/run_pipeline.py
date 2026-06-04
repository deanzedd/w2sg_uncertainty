#!/usr/bin/env python3
"""
End-to-end pipeline orchestrator.

Chains all phases automatically based on the `method` field in config:

  WDPO pipeline:
    Phase 1: SFT strong model on D_l
    Phase 2: WDPO weak labeling of D_u → D̂
    Phase 3: DPO strong model on D̂
    Phase 4: Evaluation (GRA)

  CWPO pipeline:
    Phase 1a: SFT strong model on D_l
    Phase 1b: Train scalar reward model (weak annotator) on D_l
    Phase 2: CWPO confidence labeling of D_u → D̂ with confidence weights
    Phase 3: Confidence-weighted DPO strong model on D̂
    Phase 4: Evaluation (GRA)

  Baseline DPO pipeline:
    Phase 1: SFT strong model on D_l
    Phase 3: Standard DPO strong model on D_l (no labeling step)
    Phase 4: Evaluation (GRA)

Usage:
    python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml
    python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf.yaml
    python pipeline/run_pipeline.py --config configs/baseline_dpo_hh_rlhf.yaml

    # Debug mode (small data, fast)
    python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml --debug

    # Skip phases already completed
    python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf.yaml \
        --skip_sft --pseudo_labels outputs/wdpo/hh_rlhf/weak_labels/pseudo_labeled.jsonl
"""

import argparse
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import load_config, print_config, setup_logging

logger = logging.getLogger(__name__)

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


def parse_args():
    parser = argparse.ArgumentParser(description="Run full WDPO/CWPO pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_sft", action="store_true", help="Skip SFT phase")
    parser.add_argument("--skip_reward_model", action="store_true",
                        help="Skip reward model training (CWPO only)")
    parser.add_argument("--skip_labeling", action="store_true",
                        help="Skip weak labeling phase")
    parser.add_argument("--pseudo_labels", type=str, default=None,
                        help="Pre-computed pseudo labels path (skips labeling)")
    parser.add_argument("--sft_model_path", type=str, default=None,
                        help="Pre-trained SFT model path (skips SFT)")
    parser.add_argument("--reward_model_path", type=str, default=None,
                        help="Pre-trained reward model path (CWPO, skips training)")
    parser.add_argument("--run_gpt4", action="store_true",
                        help="Run GPT-4 win rate in evaluation")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def run_script(script_name: str, *extra_args):
    """Run a script in the scripts/ directory."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    cmd = [sys.executable, script_path] + list(extra_args)
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg)

    method = cfg.get("method", "wdpo")
    debug_flag = ["--debug"] if args.debug else []
    output_dir = cfg.training.get("output_dir", f"outputs/{method}")

    # ── Derive default paths ─────────────────────────────────────────────
    sft_model_path = args.sft_model_path or cfg.sft.get("output_dir", f"outputs/{method}/sft")
    weak_labels_path = args.pseudo_labels or os.path.join(
        cfg.get("weak_label_output_dir", f"outputs/{method}/weak_labels"),
        "pseudo_labeled.jsonl",
    )
    reward_model_path = args.reward_model_path or os.path.join(
        cfg.reward_model.get("output_dir", f"outputs/{method}/reward_model"),
        "checkpoint-final", "model.pt",
    )

    logger.info(f"Starting pipeline for method: {method}")
    logger.info(f"Dataset: {cfg.dataset_name} | Labeled ratio: {cfg.labeled_ratio}")

    # ══ Phase 1: SFT ════════════════════════════════════════════════════
    if not args.skip_sft:
        logger.info("═" * 50)
        logger.info("PHASE 1: SFT — Strong Model")
        logger.info("═" * 50)
        run_script("train_sft.py", "--config", args.config, *debug_flag, *args.overrides)
    else:
        logger.info("Skipping SFT (--skip_sft)")

    # ══ Phase 1b (CWPO only): Reward Model Training ══════════════════════
    if method == "cwpo" and not args.skip_reward_model and not args.reward_model_path:
        logger.info("═" * 50)
        logger.info("PHASE 1b: CWPO — Train Weak Reward Model")
        logger.info("═" * 50)
        run_script("train_reward_model.py", "--config", args.config, *debug_flag, *args.overrides)

    # ══ Phase 2: Weak Labeling ═══════════════════════════════════════════
    if method in ("wdpo", "cwpo") and not args.skip_labeling and not args.pseudo_labels:
        logger.info("═" * 50)
        logger.info(f"PHASE 2: {method.upper()} — Weak Labeling")
        logger.info("═" * 50)
        extra = []
        if method == "cwpo" and os.path.exists(reward_model_path):
            extra = ["--reward_model_path", reward_model_path]
        run_script("label_weak.py", "--config", args.config, *extra, *debug_flag, *args.overrides)

    # ══ Phase 3: Strong Model Training ══════════════════════════════════
    logger.info("═" * 50)
    logger.info(f"PHASE 3: {method.upper()} — Strong Model Training")
    logger.info("═" * 50)
    extra_args = ["--sft_model_path", sft_model_path]
    if method in ("wdpo", "cwpo"):
        extra_args += ["--pseudo_labels", weak_labels_path]
    run_script("train_strong.py", "--config", args.config, *extra_args, *debug_flag, *args.overrides)

    # ══ Phase 4: Evaluation ══════════════════════════════════════════════
    logger.info("═" * 50)
    logger.info("PHASE 4: Evaluation")
    logger.info("═" * 50)
    eval_args = [
        "--aligned_model_path", output_dir,
        "--sft_model_path", sft_model_path,
    ]
    if args.run_gpt4:
        eval_args.append("--run_gpt4")
    if method in ("wdpo", "cwpo") and os.path.exists(weak_labels_path):
        eval_args += ["--pseudo_labels", weak_labels_path]
    run_script("evaluate.py", "--config", args.config, *eval_args, *args.overrides)

    logger.info("═" * 50)
    logger.info(f"Pipeline complete for method={method}!")
    logger.info("═" * 50)


if __name__ == "__main__":
    main()
