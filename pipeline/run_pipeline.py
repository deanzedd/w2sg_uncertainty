#!/usr/bin/env python3
"""
End-to-end pipeline orchestrator.

Tập dữ liệu gốc D được tách thành:
  - D_l : dữ liệu có nhãn gốc (labeled, tỉ lệ = labeled_ratio)
  - D_u : dữ liệu đã bỏ nhãn (unlabeled, phần còn lại)

Chains all phases automatically based on the `method` field in config:

  WDPO pipeline (Option A — Traditional WDPO):
    Phase 1b: SFT weak model on D_l          → π_w^SFT   (WDPO only)
    Phase 1c: DPO weak model on D_l          → π_w^*     (WDPO only)
    Phase 2:  WDPO weak labeling of D_u      → D_weak    (implicit reward scoring)
    Phase 2b: SFT strong model on D_weak     → π_θ^SFT
    Phase 3:  DPO strong model on D_weak     → π_θ^DPO
    Phase 4:  Evaluation (GRA)

  CWPO pipeline (Option B — Recommended):
    Phase 1b: Train scalar reward model on D_l via Bradley-Terry loss
    Phase 2:  CWPO confidence labeling of D_u → D_weak   (with confidence weights C)
    Phase 2b: SFT strong model on D_weak     → π_θ^SFT
    Phase 3:  CW-DPO strong model on D_weak  → π_θ^CW-DPO
    Phase 4:  Evaluation (GRA)

  Baseline DPO pipeline:
    Phase 1a: SFT strong model on D (toàn bộ dataset)
    Phase 3:  Standard DPO strong model on D
    Phase 4:  Evaluation (GRA)

Usage:
    python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml
    python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf.yaml
    python pipeline/run_pipeline.py --config configs/baseline_dpo_hh_rlhf.yaml

    # With Qwen2.5-7B strong model (multi-GPU, LoRA):
    conda activate w2sg_uncer
    cd w2sg_uncertainty
    python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf.yaml \\
        strong_model_name=Qwen/Qwen2.5-7B \\
        use_lora=true lora_r=16 lora_alpha=32 \\
        sft.gradient_checkpointing=true

    # Debug mode (small data, fast)
    python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml --debug

    # Skip phases already completed
    python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf.yaml \\
        --skip_weak_model \\
        --pseudo_labels outputs/wdpo/hh_rlhf/weak_labels/pseudo_labeled.jsonl \\
        --sft_model_path outputs/wdpo/hh_rlhf/sft_strong
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
    # ── Phase skips ──────────────────────────────────────────────────────
    parser.add_argument("--skip_sft", action="store_true",
                        help="Skip SFT phase (Phase 2b cho WDPO/CWPO, Phase 1a cho Baseline)")
    parser.add_argument("--skip_weak_model", action="store_true",
                        help="Skip weak model training phase (WDPO Phase 1b+1c)")
    parser.add_argument("--skip_reward_model", action="store_true",
                        help="Skip reward model training (CWPO Phase 1b)")
    parser.add_argument("--skip_labeling", action="store_true",
                        help="Skip weak labeling phase (Phase 2)")
    # ── Pre-computed paths ───────────────────────────────────────────────
    parser.add_argument("--pseudo_labels", type=str, default=None,
                        help="Pre-computed D_weak path (skips labeling + SFT on D_weak)")
    parser.add_argument("--sft_model_path", type=str, default=None,
                        help="Pre-trained SFT model path (skips SFT on D_weak / D)")
    parser.add_argument("--weak_model_path", type=str, default=None,
                        help="Pre-trained π_w^* path (WDPO, skips weak model training)")
    parser.add_argument("--weak_ref_path", type=str, default=None,
                        help="Pre-trained π_w^SFT path used as DPO ref (WDPO)")
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
    sft_model_path = (
        args.sft_model_path
        or cfg.sft.get("output_dir", f"outputs/{method}/sft_strong")
    )
    weak_labels_path = args.pseudo_labels or os.path.join(
        cfg.get("weak_label_output_dir", f"outputs/{method}/weak_labels"),
        "pseudo_labeled.jsonl",
    )
    reward_model_path = args.reward_model_path or os.path.join(
        cfg.reward_model.get("output_dir", f"outputs/{method}/reward_model"),
        "checkpoint-final",   # R2/RT2 fix: pass directory, not model.pt file
    )                         # label_weak.py detects model.pt + metadata.json inside
    weak_model_path = (
        args.weak_model_path
        or cfg.get("weak_model_dpo", {}).get("output_dir", f"outputs/{method}/weak_model_dpo")
    )
    weak_ref_path = (
        args.weak_ref_path
        or cfg.get("weak_model_sft", {}).get("output_dir", f"outputs/{method}/weak_model_sft")
    )

    logger.info(f"Starting pipeline for method: {method}")
    logger.info(f"Dataset: {cfg.dataset_name} | Labeled ratio: {cfg.labeled_ratio}")

    # ════════════════════════════════════════════════════════════════════
    # BASELINE DPO
    # ════════════════════════════════════════════════════════════════════
    if method == "baseline_dpo":
        # ══ Phase 1a: SFT Strong Model on D (toàn bộ dataset) ════════════
        if not args.skip_sft:
            logger.info("═" * 60)
            logger.info("PHASE 1a: Baseline — SFT Strong Model on D (full dataset)")
            logger.info("═" * 60)
            run_script("train_sft.py", "--config", args.config, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping SFT (--skip_sft)")

        # ══ Phase 3: Standard DPO on D ════════════════════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 3: Baseline — Standard DPO on D (full dataset)")
        logger.info("═" * 60)
        extra_args = ["--sft_model_path", sft_model_path]
        run_script("train_strong.py", "--config", args.config, *extra_args, *debug_flag, *args.overrides)

        # ══ Phase 4: Evaluation ═══════════════════════════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 4: Evaluation")
        logger.info("═" * 60)
        eval_args = [
            "--aligned_model_path", output_dir,
            "--sft_model_path", sft_model_path,
        ]
        if args.run_gpt4:
            eval_args.append("--run_gpt4")
        run_script("evaluate.py", "--config", args.config, *eval_args, *args.overrides)

        logger.info("═" * 60)
        logger.info("Pipeline complete for method=baseline_dpo!")
        logger.info("═" * 60)
        return

    # ════════════════════════════════════════════════════════════════════
    # WDPO
    # ════════════════════════════════════════════════════════════════════
    if method == "wdpo":
        # ══ Phase 1b+1c: Train Weak Model (SFT → DPO) on D_l ════════════
        if not args.skip_weak_model and not args.weak_model_path:
            logger.info("═" * 60)
            logger.info("PHASE 1b+1c: WDPO — Train Weak Model on D_l (SFT → DPO → π_w^*)")
            logger.info("═" * 60)
            run_script(
                "train_weak_model.py",
                "--config", args.config,
                *debug_flag,
                *args.overrides,
            )
        else:
            logger.info("Skipping WDPO weak model training (--skip_weak_model or --weak_model_path set)")

        # ══ Phase 2: Weak Labeling D_u → D_weak ═════════════════════════
        if not args.skip_labeling and not args.pseudo_labels:
            logger.info("═" * 60)
            logger.info("PHASE 2: WDPO — Weak Labeling D_u → D_weak (implicit reward scoring)")
            logger.info("═" * 60)
            extra = []
            if os.path.exists(weak_model_path):
                extra += ["--weak_model_path", weak_model_path]
            if os.path.exists(weak_ref_path):
                extra += ["--weak_ref_path", weak_ref_path]
            run_script("label_weak.py", "--config", args.config, *extra, *debug_flag, *args.overrides)
        else:
            if args.pseudo_labels:
                logger.info(f"Using pre-computed D_weak: {args.pseudo_labels}")
            elif args.skip_labeling:
                logger.info("Skipping labeling (--skip_labeling)")

        # ══ Phase 2b: SFT Strong Model on D_weak ═════════════════════════
        if not args.skip_sft:
            logger.info("═" * 60)
            logger.info("PHASE 2b: WDPO — SFT Strong Model on D_weak → π_θ^SFT")
            logger.info("═" * 60)
            run_script(
                "train_sft.py",
                "--config", args.config,
                "--pseudo_labels", weak_labels_path,
                *debug_flag,
                *args.overrides,
            )
        else:
            logger.info("Skipping SFT on D_weak (--skip_sft)")

        # ══ Phase 3: DPO Strong Model on D_weak ══════════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 3: WDPO — DPO Strong Model on D_weak")
        logger.info("═" * 60)
        extra_args = [
            "--sft_model_path", sft_model_path,
            "--pseudo_labels", weak_labels_path,
        ]
        run_script("train_strong.py", "--config", args.config, *extra_args, *debug_flag, *args.overrides)

        # ══ Phase 4: Evaluation ═══════════════════════════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 4: Evaluation")
        logger.info("═" * 60)
        eval_args = [
            "--aligned_model_path", output_dir,
            "--sft_model_path", sft_model_path,
        ]
        if args.run_gpt4:
            eval_args.append("--run_gpt4")
        if os.path.exists(weak_labels_path):
            eval_args += ["--pseudo_labels", weak_labels_path]
        run_script("evaluate.py", "--config", args.config, *eval_args, *args.overrides)

        logger.info("═" * 60)
        logger.info("Pipeline complete for method=wdpo!")
        logger.info("═" * 60)
        return

    # ════════════════════════════════════════════════════════════════════
    # CWPO
    # ════════════════════════════════════════════════════════════════════
    if method == "cwpo":
        # ══ Phase 1b: Train Scalar Reward Model on D_l ════════════════════
        if not args.skip_reward_model and not args.reward_model_path:
            logger.info("═" * 60)
            logger.info("PHASE 1b: CWPO — Train Scalar Reward Model on D_l (Bradley-Terry)")
            logger.info("═" * 60)
            run_script("train_reward_model.py", "--config", args.config, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping CWPO reward model training")

        # ══ Phase 2: Confidence Labeling D_u → D_weak ═══════════════════
        if not args.skip_labeling and not args.pseudo_labels:
            logger.info("═" * 60)
            logger.info("PHASE 2: CWPO — Confidence Labeling D_u → D_weak (C = 2·(σ(s+−s−)−0.5))")
            logger.info("═" * 60)
            extra = []
            if os.path.exists(reward_model_path):
                extra += ["--reward_model_path", reward_model_path]
            run_script("label_weak.py", "--config", args.config, *extra, *debug_flag, *args.overrides)
        else:
            if args.pseudo_labels:
                logger.info(f"Using pre-computed D_weak: {args.pseudo_labels}")
            elif args.skip_labeling:
                logger.info("Skipping labeling (--skip_labeling)")

        # ══ Phase 2b: SFT Strong Model on D_weak ═════════════════════════
        if not args.skip_sft:
            logger.info("═" * 60)
            logger.info("PHASE 2b: CWPO — SFT Strong Model on D_weak → π_θ^SFT")
            logger.info("═" * 60)
            run_script(
                "train_sft.py",
                "--config", args.config,
                "--pseudo_labels", weak_labels_path,
                *debug_flag,
                *args.overrides,
            )
        else:
            logger.info("Skipping SFT on D_weak (--skip_sft)")

        # ══ Phase 3: CW-DPO Strong Model on D_weak ═══════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 3: CWPO — CW-DPO Strong Model on D_weak")
        logger.info("═" * 60)
        extra_args = [
            "--sft_model_path", sft_model_path,
            "--pseudo_labels", weak_labels_path,
        ]
        run_script("train_strong.py", "--config", args.config, *extra_args, *debug_flag, *args.overrides)

        # ══ Phase 4: Evaluation ═══════════════════════════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 4: Evaluation")
        logger.info("═" * 60)
        eval_args = [
            "--aligned_model_path", output_dir,
            "--sft_model_path", sft_model_path,
        ]
        if args.run_gpt4:
            eval_args.append("--run_gpt4")
        if os.path.exists(weak_labels_path):
            eval_args += ["--pseudo_labels", weak_labels_path]
        run_script("evaluate.py", "--config", args.config, *eval_args, *args.overrides)

        logger.info("═" * 60)
        logger.info("Pipeline complete for method=cwpo!")
        logger.info("═" * 60)
        return

    raise ValueError(f"Unknown method: '{method}'. Choose: wdpo, cwpo, baseline_dpo")


if __name__ == "__main__":
    main()
