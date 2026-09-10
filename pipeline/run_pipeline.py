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

  MWDPO pipeline (Phase 1 — Multi-Weak Agreement DPO):
    Phase 1a: Train k scalar reward models on D_l (different seeds)
    Phase 1b: Multi-weak labeling of D_u → D_h (agreement) + D_l (disagreement)
    Phase 2a: SFT strong model on D_h    → π_θ^SFT
    Phase 2b: Standard DPO on D_h       → π_θ^DPO  (Phase 1 of proposal)
    Phase 3:  Evaluation (GRA)

  MWDPO_bootstrap_calibration pipeline (1a + 2a extensions):
    Phase 1a: Train ONE MultiHeadRewardModel on D_l
              — shared backbone + K bootstrap heads (Plan 1a)
              — each head trained on per-batch bootstrap resample of D_l
              → produces K decorrelated reward estimators at ~1× backbone cost
    Phase 1b: Calibrate + Label D_u → D_h / D_l
              — fit per-head temperatures T_k on D_l val split (Plan 2a)
              — aggregate calibrated probabilities p_k=σ(margin_k/T_k)
              — unanimous agreement filter (same as MWDPO) on calibrated votes
              → D_h: high-agreement, calibrated-confidence subset
    Phase 2a: SFT strong model on D_h        → π_θ^SFT   (unchanged from MWDPO)
    Phase 2b: Standard DPO on D_h            → π_θ^DPO   (unchanged from MWDPO)
    Phase 3:  Evaluation (GRA)              (unchanged from MWDPO)

    Ablation options (set in config bootstrap_calibration section):
      use_bootstrap: false  → all heads see full batch (no resampling), multi-init only
      use_calibration: false → T_k = 1.0 (raw margins, same as current MWDPO)
      Both false + num_heads=1 → reproduces MWDPO k=1 (degenerate baseline)

  Baseline DPO pipeline:
    Phase 1a: SFT strong model on D (toàn bộ dataset)
    Phase 3:  Standard DPO strong model on D
    Phase 4:  Evaluation (GRA)

Usage:
    python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml
    python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf.yaml
    python pipeline/run_pipeline.py --config configs/mwdpo_hh_rlhf.yaml
    python pipeline/run_pipeline.py --config configs/mwdpo_bc_hh_rlhf.yaml
    python pipeline/run_pipeline.py --config configs/baseline_dpo_hh_rlhf.yaml

    # Debug mode (small data, fast)
    python pipeline/run_pipeline.py --config configs/mwdpo_hh_rlhf.yaml --debug
    python pipeline/run_pipeline.py --config configs/mwdpo_bc_hh_rlhf.yaml --debug

    # MWDPO_bootstrap_calibration — full run (both 1a and 2a enabled by default)
    python pipeline/run_pipeline.py --config configs/mwdpo_bc_hh_rlhf.yaml

    # Ablation: bootstrap only, no calibration (set use_calibration=false in config)
    python pipeline/run_pipeline.py --config configs/mwdpo_bc_hh_rlhf.yaml \\
        bootstrap_calibration.use_calibration=false

    # Ablation: calibration only, no bootstrap (set use_bootstrap=false in config)
    python pipeline/run_pipeline.py --config configs/mwdpo_bc_hh_rlhf.yaml \\
        bootstrap_calibration.use_bootstrap=false

    # Skip reward model training (already trained), run labeling + SFT + DPO:
    python pipeline/run_pipeline.py --config configs/mwdpo_bc_hh_rlhf.yaml \\
        --skip_reward_model

    # Skip reward model + labeling (use pre-computed D_h), run SFT + DPO only:
    python pipeline/run_pipeline.py --config configs/mwdpo_bc_hh_rlhf.yaml \\
        --skip_reward_model \\
        --pseudo_labels outputs/mwdpo_bc/hh_rlhf/.../weak_labels/d_high/pseudo_labeled.jsonl

    # Skip SFT (pre-trained SFT checkpoint available):
    python pipeline/run_pipeline.py --config configs/mwdpo_bc_hh_rlhf.yaml \\
        --skip_reward_model --skip_sft \\
        --pseudo_labels outputs/mwdpo_bc/hh_rlhf/.../weak_labels/d_high/pseudo_labeled.jsonl \\
        --sft_model_path outputs/mwdpo_bc/hh_rlhf/.../sft_strong

    # Use explicit pre-trained MultiHeadRewardModel checkpoint:
    python pipeline/run_pipeline.py --config configs/mwdpo_bc_hh_rlhf.yaml \\
        --skip_reward_model \\
        --reward_model_path outputs/mwdpo_bc/hh_rlhf/.../reward_model/checkpoint-final

    # MWDPO (original) — skip phases already completed
    python pipeline/run_pipeline.py --config configs/mwdpo_hh_rlhf.yaml \\
        --skip_reward_model \\
        --pseudo_labels outputs/mwdpo/hh_rlhf/.../weak_labels/d_high/pseudo_labeled.jsonl \\
        --sft_model_path outputs/mwdpo/hh_rlhf/.../sft_strong
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
    parser = argparse.ArgumentParser(
        description="Run full WDPO/CWPO/MWDPO/MWDPO_bootstrap_calibration pipeline"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    # ── Phase skips ───────────────────────────────────────────────────
    parser.add_argument("--skip_sft", action="store_true",
                        help="Skip SFT phase (Phase 2a for MWDPO, Phase 2b for WDPO/CWPO)")
    parser.add_argument("--skip_weak_model", action="store_true",
                        help="Skip weak model training phase (WDPO Phase 1b+1c)")
    parser.add_argument("--skip_reward_model", action="store_true",
                        help="Skip reward model training (CWPO Phase 1b; MWDPO Phase 1a)")
    parser.add_argument("--skip_labeling", action="store_true",
                        help="Skip weak labeling phase (Phase 2 / Phase 1b for MWDPO)")
    parser.add_argument(
        "--skip_phase1_dpo", action="store_true",
        help=(
            "[2-phase only] Skip Phase 1 DPO on D_h (Phase 2b). "
            "Use when Phase 1 DPO is already done and you only want to run Phase 2."
        ),
    )
    parser.add_argument(
        "--skip_phase2", action="store_true",
        help=(
            "[2-phase only] Skip Phase 2 (C_strong computation + Debate-Weighted DPO on D_l). "
            "Produces a Phase 1-only model for ablation comparison."
        ),
    )
    # ── Pre-computed paths ────────────────────────────────────────────
    parser.add_argument("--pseudo_labels", type=str, default=None,
                        help="Pre-computed D_h (MWDPO) or D_weak (WDPO/CWPO) path")
    parser.add_argument("--sft_model_path", type=str, default=None,
                        help="Pre-trained SFT model path (skips SFT)")
    parser.add_argument("--weak_model_path", type=str, default=None,
                        help="Pre-trained π_w^* path (WDPO, skips weak model training)")
    parser.add_argument("--weak_ref_path", type=str, default=None,
                        help="Pre-trained π_w^SFT path used as DPO ref (WDPO)")
    parser.add_argument("--reward_model_path", type=str, default=None,
                        help="Pre-trained reward model path (CWPO single model; or "
                             "MultiHeadRewardModel checkpoint-final dir for mwdpo_bootstrap_calibration)")
    # ── Resume checkpoints ───────────────────────────────────────────────
    parser.add_argument("--resume_sft_checkpoint", type=str, default=None,
                        help="Resume SFT training from this checkpoint directory")
    parser.add_argument("--resume_dpo_checkpoint", type=str, default=None,
                        help="Resume DPO strong model training from this checkpoint directory. "
                             "Model weights will be loaded from this checkpoint.")
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


def resolve_phase1_train_labels(
    cfg,
    label_base_dir: str,
    cli_pseudo_labels: str | None = None,
) -> str:
    """
    Resolve Phase-1 SFT + DPO --pseudo_labels path from phase1_data_mode.

    Priority:
      1. CLI --pseudo_labels (explicit override)
      2. phase1_data_mode=d_high         → {label_base}/d_high/pseudo_labeled.jsonl
      3. phase1_data_mode=all_unlabeled  → {label_base}/pseudo_labeled.jsonl
         (full D_u with multi-weak ensemble preferences; already written at labeling time)

    Both SFT and Phase-1 DPO must use the same resolved path.
    """
    if cli_pseudo_labels:
        logger.info(
            f"[phase1_data_mode] Using CLI --pseudo_labels override: {cli_pseudo_labels}"
        )
        return cli_pseudo_labels

    mode = str(cfg.get("phase1_data_mode", "d_high"))
    if mode == "d_high":
        path = os.path.join(label_base_dir, "d_high", "pseudo_labeled.jsonl")
    elif mode == "all_unlabeled":
        path = os.path.join(label_base_dir, "pseudo_labeled.jsonl")
    else:
        raise ValueError(
            f"Unknown phase1_data_mode={mode!r}. "
            "Choose 'd_high' or 'all_unlabeled'."
        )

    logger.info(f"[phase1_data_mode={mode}] Phase-1 SFT/DPO labels: {path}")
    return path


def resolve_phase2_train_labels(
    cfg,
    d_high_path: str,
    d_low_scored_path: str,
    label_base_dir: str,
    debug: bool = False,
    debug_n: int = 64,
) -> str:
    """
    Resolve Phase 2d --pseudo_labels path from phase2_training.phase2_data_mode.

    d_low (default): scored D_l only.
    d_weak_asymmetric: merge D_h (w=1) ∪ scored D_l → d_weak_scored jsonl.

    When debug=True and mode is d_weak_asymmetric, truncate D_h to debug_n rows so
    the ACE union matches the truncated scored D_l from compute_strong_confidence --debug.
    """
    from src.utils.phase2_train_data import build_phase2_train_jsonl

    p2_cfg = cfg.get("phase2_training", {})
    sc_cfg = cfg.get("strong_confidence", {})
    mode = str(p2_cfg.get("phase2_data_mode", "d_low"))
    d_weak_scored_path = sc_cfg.get(
        "d_weak_scored_path",
        os.path.join(label_base_dir, "d_weak_scored", "pseudo_labeled.jsonl"),
    )

    merge_d_high = d_high_path
    if debug and mode == "d_weak_asymmetric":
        # Keep ACE smoke cheap and size-matched to scored D_l (--debug truncates to 64).
        import json as _json
        dbg_dir = os.path.join(label_base_dir, "d_high_debug")
        os.makedirs(dbg_dir, exist_ok=True)
        merge_d_high = os.path.join(dbg_dir, "pseudo_labeled.jsonl")
        n_written = 0
        with open(d_high_path, "r", encoding="utf-8") as src, open(
            merge_d_high, "w", encoding="utf-8"
        ) as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                dst.write(line + "\n")
                n_written += 1
                if n_written >= debug_n:
                    break
        logger.info(
            f"[DEBUG] Truncated D_h to {n_written} samples for ACE merge: {merge_d_high}"
        )

    return build_phase2_train_jsonl(
        phase2_data_mode=mode,
        d_high_path=merge_d_high,
        d_low_scored_path=d_low_scored_path,
        d_weak_scored_path=d_weak_scored_path,
    )


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
        # ══ Phase 1a: SFT Strong Model on D (toàn bộ dataset) ════════════════════
        if not args.skip_sft:
            logger.info("═" * 60)
            logger.info("PHASE 1a: Baseline — SFT Strong Model on D (full dataset)")
            logger.info("═" * 60)
            sft_resume_args = (["--resume_sft_checkpoint", args.resume_sft_checkpoint]
                               if args.resume_sft_checkpoint else [])
            run_script("train_sft.py", "--config", args.config,
                       *sft_resume_args, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping SFT (--skip_sft)")

        # ══ Phase 3: Standard DPO on D ═══════════════════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 3: Baseline — Standard DPO on D (full dataset)")
        logger.info("═" * 60)
        extra_args = ["--sft_model_path", sft_model_path]
        if args.resume_dpo_checkpoint:
            extra_args += ["--resume_dpo_checkpoint", args.resume_dpo_checkpoint]
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

        # ══ Phase 2b: SFT Strong Model on D_weak ═════════════════════════════
        if not args.skip_sft:
            logger.info("═" * 60)
            logger.info("PHASE 2b: WDPO — SFT Strong Model on D_weak → π_θ^SFT")
            logger.info("═" * 60)
            sft_resume_args = (["--resume_sft_checkpoint", args.resume_sft_checkpoint]
                               if args.resume_sft_checkpoint else [])
            run_script(
                "train_sft.py",
                "--config", args.config,
                "--pseudo_labels", weak_labels_path,
                *sft_resume_args,
                *debug_flag,
                *args.overrides,
            )
        else:
            logger.info("Skipping SFT on D_weak (--skip_sft)")

        # ══ Phase 3: DPO Strong Model on D_weak ════════════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 3: WDPO — DPO Strong Model on D_weak")
        logger.info("═" * 60)
        extra_args = [
            "--sft_model_path", sft_model_path,
            "--pseudo_labels", weak_labels_path,
        ]
        if args.resume_dpo_checkpoint:
            extra_args += ["--resume_dpo_checkpoint", args.resume_dpo_checkpoint]
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

        # ══ Phase 2b: SFT Strong Model on D_weak ═════════════════════════════
        if not args.skip_sft:
            logger.info("═" * 60)
            logger.info("PHASE 2b: CWPO — SFT Strong Model on D_weak → π_θ^SFT")
            logger.info("═" * 60)
            sft_resume_args = (["--resume_sft_checkpoint", args.resume_sft_checkpoint]
                               if args.resume_sft_checkpoint else [])
            run_script(
                "train_sft.py",
                "--config", args.config,
                "--pseudo_labels", weak_labels_path,
                *sft_resume_args,
                *debug_flag,
                *args.overrides,
            )
        else:
            logger.info("Skipping SFT on D_weak (--skip_sft)")

        # ══ Phase 3: CW-DPO Strong Model on D_weak ══════════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 3: CWPO — CW-DPO Strong Model on D_weak")
        logger.info("═" * 60)
        extra_args = [
            "--sft_model_path", sft_model_path,
            "--pseudo_labels", weak_labels_path,
        ]
        if args.resume_dpo_checkpoint:
            extra_args += ["--resume_dpo_checkpoint", args.resume_dpo_checkpoint]
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

    # ════════════════════════════════════════════════════════════════════
    # MWDPO — Multi-Weak Agreement DPO (Phase 1)
    # ════════════════════════════════════════════════════════════════════
    if method == "mwdpo":
        mw_cfg = cfg.get("multi_weak", {})
        rm_base_dir = mw_cfg.get(
            "output_dir",
            cfg.get("reward_model", {}).get("output_dir", "outputs/mwdpo/reward_models")
        )
        label_base_dir = cfg.get("multi_weak_label_output_dir",
                                  cfg.get("weak_label_output_dir", "outputs/mwdpo/weak_labels"))

        # Phase-1 SFT + DPO train labels (d_high | all_unlabeled)
        phase1_labels_path = resolve_phase1_train_labels(
            cfg, label_base_dir, args.pseudo_labels
        )

        # ══ Phase 1a: Train k Reward Models on D_l ══════════════════════
        if not args.skip_reward_model:
            logger.info("═" * 60)
            logger.info("PHASE 1a: MWDPO — Train k Reward Models on D_l (Bradley-Terry)")
            logger.info("═" * 60)
            run_script("train_multi_reward_models.py", "--config", args.config,
                       *debug_flag, *args.overrides)
        else:
            logger.info("Skipping MWDPO reward model training (--skip_reward_model)")

        # ══ Phase 1b: Multi-Weak Labeling D_u → D_h ∪ D_l ══════════════
        if not args.skip_labeling and not args.pseudo_labels:
            logger.info("═" * 60)
            logger.info(
                "PHASE 1b: MWDPO — Multi-Weak Labeling D_u → D_h (agreement) + D_l (disagreement)"
            )
            logger.info("═" * 60)
            run_script("label_multi_weak.py", "--config", args.config,
                       *debug_flag, *args.overrides)
        else:
            if args.pseudo_labels:
                logger.info(f"Using pre-computed Phase-1 labels: {args.pseudo_labels}")
            elif args.skip_labeling:
                logger.info("Skipping multi-weak labeling (--skip_labeling)")

        # ══ Phase 2a: SFT Strong Model on phase1_data_mode labels ═══════
        if not args.skip_sft:
            logger.info("═" * 60)
            logger.info(
                f"PHASE 2a: MWDPO — SFT Strong Model "
                f"(phase1_data_mode={cfg.get('phase1_data_mode', 'd_high')}) → π_θ^SFT"
            )
            logger.info("═" * 60)
            sft_resume_args = (["--resume_sft_checkpoint", args.resume_sft_checkpoint]
                               if args.resume_sft_checkpoint else [])
            run_script(
                "train_sft.py",
                "--config", args.config,
                "--pseudo_labels", phase1_labels_path,
                *sft_resume_args,
                *debug_flag,
                *args.overrides,
            )
        else:
            logger.info("Skipping SFT (--skip_sft)")

        # ══ Phase 2b: Standard DPO on phase1_data_mode labels ══════════
        logger.info("═" * 60)
        logger.info(
            f"PHASE 2b: MWDPO Phase 1 — Standard DPO "
            f"(phase1_data_mode={cfg.get('phase1_data_mode', 'd_high')})"
        )
        logger.info("═" * 60)
        extra_args = [
            "--sft_model_path", sft_model_path,
            "--pseudo_labels", phase1_labels_path,
        ]
        if args.resume_dpo_checkpoint:
            extra_args += ["--resume_dpo_checkpoint", args.resume_dpo_checkpoint]
        run_script("train_strong.py", "--config", args.config, *extra_args,
                   *debug_flag, *args.overrides)

        # ══ Phase 3: Evaluation ════════════════════════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 3: MWDPO — Evaluation (GRA)")
        logger.info("═" * 60)
        eval_args = [
            "--aligned_model_path", output_dir,
            "--sft_model_path", sft_model_path,
        ]
        if args.run_gpt4:
            eval_args.append("--run_gpt4")
        if os.path.exists(phase1_labels_path):
            eval_args += ["--pseudo_labels", phase1_labels_path]
        run_script("evaluate.py", "--config", args.config, *eval_args, *args.overrides)

        logger.info("═" * 60)
        logger.info("Pipeline complete for method=mwdpo!")
        logger.info("═" * 60)
        return

    # ════════════════════════════════════════════════════════════════════
    # MWDPO_bootstrap_calibration — Multi-Head Bootstrap + Calibration DPO
    # ════════════════════════════════════════════════════════════════════
    if method == "mwdpo_bootstrap_calibration":
        bc_cfg = cfg.get("bootstrap_calibration", {})

        # Reward model checkpoint dir (one multi-head model, not K separate)
        bc_rm_dir = bc_cfg.get(
            "output_dir",
            cfg.get("reward_model", {}).get("output_dir",
                    "outputs/mwdpo_bootstrap_calibration/reward_model"),
        )
        label_base_dir = cfg.get(
            "multi_weak_label_output_dir",
            cfg.get("weak_label_output_dir",
                    "outputs/mwdpo_bootstrap_calibration/weak_labels"),
        )

        # Phase-1 SFT + DPO train labels (d_high | all_unlabeled)
        phase1_labels_path = resolve_phase1_train_labels(
            cfg, label_base_dir, args.pseudo_labels
        )

        # ══ Phase 1a: Train MultiHeadRewardModel on D_l ══════════════════
        if not args.skip_reward_model:
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
                logger.info(
                    "PHASE 1a: MWDPO_BC — Train MultiHeadRewardModel on D_l "
                    "(shared backbone + K bootstrap heads)"
                )
                logger.info("═" * 60)
                run_script(
                    "train_bootstrap_reward_model.py",
                    "--config", args.config,
                    *debug_flag, *args.overrides,
                )
        else:
            logger.info("Skipping MWDPO_BC reward model training (--skip_reward_model)")

        # ══ Phase 1b: Calibrate + Label D_u → D_h ∪ D_l ═════════════════
        if not args.skip_labeling and not args.pseudo_labels:
            logger.info("═" * 60)
            logger.info(
                "PHASE 1b: MWDPO_BC — Calibrate T_k on D_l val, "
                "then label D_u → D_h (agreement) + D_l (disagreement)"
            )
            logger.info("═" * 60)
            extra = []
            if args.reward_model_path:
                extra += ["--checkpoint_dir", args.reward_model_path]
            run_script(
                "label_bootstrap_calibration.py",
                "--config", args.config,
                *extra, *debug_flag, *args.overrides,
            )
        else:
            if args.pseudo_labels:
                logger.info(f"Using pre-computed Phase-1 labels: {args.pseudo_labels}")
            elif args.skip_labeling:
                logger.info("Skipping labeling (--skip_labeling)")

        # ══ Phase 2a: SFT Strong Model on phase1_data_mode labels ════════
        if not args.skip_sft:
            logger.info("═" * 60)
            logger.info(
                f"PHASE 2a: MWDPO_BC — SFT Strong Model "
                f"(phase1_data_mode={cfg.get('phase1_data_mode', 'd_high')}) → π_θ^SFT"
            )
            logger.info("═" * 60)
            sft_resume_args = (
                ["--resume_sft_checkpoint", args.resume_sft_checkpoint]
                if args.resume_sft_checkpoint else []
            )
            run_script(
                "train_sft.py",
                "--config", args.config,
                "--pseudo_labels", phase1_labels_path,
                *sft_resume_args, *debug_flag, *args.overrides,
            )
        else:
            logger.info("Skipping SFT (--skip_sft)")

        # ══ Phase 2b: Standard DPO on phase1_data_mode labels ═════════════
        logger.info("═" * 60)
        logger.info(
            f"PHASE 2b: MWDPO_BC — Standard DPO "
            f"(phase1_data_mode={cfg.get('phase1_data_mode', 'd_high')})"
        )
        logger.info("═" * 60)
        extra_args = [
            "--sft_model_path", sft_model_path,
            "--pseudo_labels", phase1_labels_path,
        ]
        if args.resume_dpo_checkpoint:
            extra_args += ["--resume_dpo_checkpoint", args.resume_dpo_checkpoint]
        run_script(
            "train_strong.py",
            "--config", args.config,
            *extra_args, *debug_flag, *args.overrides,
        )

        # ══ Phase 3: Evaluation (UNCHANGED from MWDPO) ════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 3: MWDPO_BC — Evaluation (GRA)")
        logger.info("═" * 60)
        eval_args = [
            "--aligned_model_path", output_dir,
            "--sft_model_path", sft_model_path,
        ]
        if args.run_gpt4:
            eval_args.append("--run_gpt4")
        if os.path.exists(phase1_labels_path):
            eval_args += ["--pseudo_labels", phase1_labels_path]
        run_script("evaluate.py", "--config", args.config, *eval_args, *args.overrides)

        logger.info("═" * 60)
        logger.info("Pipeline complete for method=mwdpo_bootstrap_calibration!")
        logger.info("═" * 60)
        return

    # ════════════════════════════════════════════════════════════════════
    # SUPER_MULTI_DPO — 2-Phase LoRA Ensemble DPO
    # ════════════════════════════════════════════════════════════════════
    if method == "super_multi_dpo":
        sm_cfg = cfg.get("super_multi", {})
        rm_base_dir = sm_cfg.get(
            "output_dir",
            cfg.get("reward_model", {}).get("output_dir", "outputs/super_multi_dpo/reward_models")
        )
        label_base_dir = cfg.get(
            "multi_weak_label_output_dir",
            cfg.get("weak_label_output_dir", f"outputs/super_multi_dpo/weak_labels"),
        )
        phase1_labels_path = resolve_phase1_train_labels(
            cfg, label_base_dir, args.pseudo_labels
        )

        # ══ Phase 1a+b: Train Super Multi Reward Model (Phase 1 warmup + Phase 2 LoRA ensemble)
        if not args.skip_reward_model:
            logger.info("═" * 60)
            logger.info("PHASE 1a+b: Super_multi_dpo — 2-Phase Reward Model Training")
            logger.info("  Phase 1: Warmup backbone + K linear heads jointly")
            logger.info("  Phase 2: Freeze backbone, attach K LoRA adapters, train independently")
            logger.info("═" * 60)
            run_script(
                "train_super_multi_reward_model.py",
                "--config", args.config,
                *debug_flag, *args.overrides,
            )
        else:
            logger.info("Skipping reward model training (--skip_reward_model)")

        # ══ Phase 1c: Multi-Weak Labeling D_u → D_h / D_l (reuses label_multi_weak.py) ═
        if not args.skip_labeling and not args.pseudo_labels:
            logger.info("═" * 60)
            logger.info("PHASE 1c: Super_multi_dpo — Multi-Weak Labeling D_u → D_h/D_l")
            logger.info(f"  agreement_mode: {sm_cfg.get('agreement_mode', 'unanimous')}")
            logger.info("═" * 60)
            run_script(
                "label_multi_weak.py",
                "--config", args.config,
                *debug_flag, *args.overrides,
            )
        else:
            if args.pseudo_labels:
                logger.info(f"Using pre-computed Phase-1 labels: {args.pseudo_labels}")
            elif args.skip_labeling:
                logger.info("Skipping labeling (--skip_labeling)")

        # ══ Phase 2a: SFT Strong Model on phase1_data_mode labels ══════════════
        if not args.skip_sft:
            logger.info("═" * 60)
            logger.info(
                f"PHASE 2a: Super_multi_dpo — SFT Strong Model "
                f"(phase1_data_mode={cfg.get('phase1_data_mode', 'd_high')})"
            )
            logger.info("═" * 60)
            sft_resume_args = (
                ["--resume_sft_checkpoint", args.resume_sft_checkpoint]
                if args.resume_sft_checkpoint else []
            )
            run_script(
                "train_sft.py",
                "--config", args.config,
                "--pseudo_labels", phase1_labels_path,
                *sft_resume_args, *debug_flag, *args.overrides,
            )
        else:
            logger.info("Skipping SFT (--skip_sft)")

        # ══ Phase 2b: Standard DPO on phase1_data_mode labels ═══════════════
        logger.info("═" * 60)
        logger.info(
            f"PHASE 2b: Super_multi_dpo — Standard DPO "
            f"(phase1_data_mode={cfg.get('phase1_data_mode', 'd_high')})"
        )
        logger.info("═" * 60)
        extra_args = [
            "--sft_model_path", sft_model_path,
            "--pseudo_labels", phase1_labels_path,
        ]
        if args.resume_dpo_checkpoint:
            extra_args += ["--resume_dpo_checkpoint", args.resume_dpo_checkpoint]
        run_script(
            "train_strong.py",
            "--config", args.config,
            *extra_args, *debug_flag, *args.overrides,
        )

        # ══ Phase 3: Evaluation (GRA) (IDENTICAL to MWDPO) ═════════════════
        logger.info("═" * 60)
        logger.info("PHASE 3: Super_multi_dpo — Evaluation (GRA)")
        logger.info("═" * 60)
        eval_args = [
            "--aligned_model_path", output_dir,
            "--sft_model_path", sft_model_path,
        ]
        if args.run_gpt4:
            eval_args.append("--run_gpt4")
        if os.path.exists(phase1_labels_path):
            eval_args += ["--pseudo_labels", phase1_labels_path]
        run_script("evaluate.py", "--config", args.config, *eval_args, *args.overrides)

        logger.info("═" * 60)
        logger.info("Pipeline complete for method=super_multi_dpo!")
        logger.info("═" * 60)
        return

    # ════════════════════════════════════════════════════════════════════
    # MWDPO_2PHASE — Multi-Weak Agreement DPO with Phase 2 Residual Training
    # ════════════════════════════════════════════════════════════════════
    if method == "mwdpo_2phase":
        label_base_dir = cfg.get("multi_weak_label_output_dir",
                                  cfg.get("weak_label_output_dir", "outputs/mwdpo_2phase/weak_labels"))

        # Canonical D_h for Phase-2 ACE merge (never replaced by all_unlabeled)
        d_high_path = os.path.join(label_base_dir, "d_high", "pseudo_labeled.jsonl")
        # Phase-1 SFT + DPO train labels (d_high | all_unlabeled)
        phase1_labels_path = resolve_phase1_train_labels(
            cfg, label_base_dir, args.pseudo_labels
        )
        d_low_path = os.path.join(label_base_dir, "d_low", "pseudo_labeled.jsonl")

        # D_l with C_strong scores (output of compute_strong_confidence.py)
        sc_cfg = cfg.get("strong_confidence", {})
        d_low_scored_path = sc_cfg.get(
            "d_low_scored_path",
            os.path.join(label_base_dir, "d_low_scored", "pseudo_labeled.jsonl"),
        )

        # Phase 1 DPO output dir (π_Phase1)
        phase1_output_dir = cfg.training.get(
            "output_dir", "outputs/mwdpo_2phase/strong_model_phase1"
        )

        # Phase 2 final model output dir
        p2_cfg = cfg.get("phase2_training", cfg.get("training", {}))
        phase2_output_dir = p2_cfg.get(
            "output_dir", "outputs/mwdpo_2phase/strong_model_phase2"
        )

        # ══ Phase 1a: Train k Reward Models on D_l ════════════════════════════
        if not args.skip_reward_model:
            logger.info("═" * 60)
            logger.info("PHASE 1a: MWDPO_2PHASE — Train k Reward Models on D_l")
            logger.info("═" * 60)
            run_script("train_multi_reward_models.py", "--config", args.config,
                       *debug_flag, *args.overrides)
        else:
            logger.info("Skipping reward model training (--skip_reward_model)")

        # ══ Phase 1b: Multi-Weak Labeling D_u → D_h ∪ D_l ═════════════════
        if not args.skip_labeling and not args.pseudo_labels:
            logger.info("═" * 60)
            logger.info("PHASE 1b: MWDPO_2PHASE — Multi-Weak Labeling D_u → D_h + D_l")
            logger.info("═" * 60)
            run_script("label_multi_weak.py", "--config", args.config,
                       *debug_flag, *args.overrides)
        else:
            if args.pseudo_labels:
                logger.info(f"Using pre-computed Phase-1 labels: {args.pseudo_labels}")
            elif args.skip_labeling:
                logger.info("Skipping labeling (--skip_labeling)")

        # ══ Phase 2a: SFT Strong Model on phase1_data_mode labels ══════════
        if not args.skip_sft:
            logger.info("═" * 60)
            logger.info(
                f"PHASE 2a: MWDPO_2PHASE — SFT Strong Model "
                f"(phase1_data_mode={cfg.get('phase1_data_mode', 'd_high')}) → π_SFT"
            )
            logger.info("═" * 60)
            sft_resume_args = (["--resume_sft_checkpoint", args.resume_sft_checkpoint]
                               if args.resume_sft_checkpoint else [])
            run_script("train_sft.py", "--config", args.config,
                       "--pseudo_labels", phase1_labels_path,
                       *sft_resume_args, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping SFT (--skip_sft)")

        # ══ Phase 2b: Standard DPO on phase1_data_mode labels → π_Phase1 ══
        if not getattr(args, "skip_phase1_dpo", False):
            logger.info("═" * 60)
            logger.info(
                f"PHASE 2b: MWDPO_2PHASE — Standard DPO "
                f"(phase1_data_mode={cfg.get('phase1_data_mode', 'd_high')}) → π_Phase1"
            )
            logger.info("═" * 60)
            extra_args = ["--sft_model_path", sft_model_path,
                          "--pseudo_labels", phase1_labels_path]
            if args.resume_dpo_checkpoint:
                extra_args += ["--resume_dpo_checkpoint", args.resume_dpo_checkpoint]
            run_script("train_strong.py", "--config", args.config,
                       *extra_args, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping Phase 1 DPO (--skip_phase1_dpo)")

        # ══ Phase 2c: Pre-compute C_strong for D_l ═════════════════════════
        if not getattr(args, "skip_phase2", False):
            logger.info("═" * 60)
            logger.info("PHASE 2c: MWDPO_2PHASE — Pre-compute C_strong for D_l")
            logger.info(f"  π_Phase1 : {phase1_output_dir}")
            logger.info(f"  π_SFT    : {sft_model_path}")
            logger.info(f"  D_l      : {d_low_path}")
            logger.info(f"  Output   : {d_low_scored_path}")
            logger.info("═" * 60)

            # weighting_mode: read from config, with backward compat for allow_negative
            _wm = sc_cfg.get("weighting_mode", None)
            if _wm is None and sc_cfg.get("allow_negative", False):
                _wm = "strong_raw"  # backward compat
            weighting_mode_args = (["--weighting_mode", str(_wm)] if _wm else [])
            # alpha: only pass for linear_combined
            _alpha = sc_cfg.get("alpha", None)
            alpha_args = (["--alpha", str(float(_alpha))] if _alpha is not None else [])
            batch_size_arg = ["--batch_size", str(int(sc_cfg.get("batch_size", 8)))]
            run_script(
                "compute_strong_confidence.py",
                "--config", args.config,
                "--phase1_model_path", phase1_output_dir,
                "--sft_model_path", sft_model_path,
                "--d_low_path", d_low_path,
                "--output_path", d_low_scored_path,
                *weighting_mode_args,
                *alpha_args,
                *batch_size_arg,
                *debug_flag, *args.overrides,
            )

            # Resolve Phase 2d data: D_l-only (legacy) or ACE D_weak asymmetric
            phase2_labels = resolve_phase2_train_labels(
                cfg, d_high_path, d_low_scored_path, label_base_dir,
                debug=bool(args.debug),
            )

            # ══ Phase 2d: Debate-Weighted / ACE Coverage DPO → π_Final ════
            _p2_mode = str(cfg.get("phase2_training", {}).get("phase2_data_mode", "d_low"))
            logger.info("═" * 60)
            logger.info(
                f"PHASE 2d: MWDPO_2PHASE — Weighted DPO "
                f"(phase2_data_mode={_p2_mode}) → π_Final"
            )
            logger.info(f"  Phase2 labels: {phase2_labels}")
            logger.info("═" * 60)
            phase2_extra = [
                "--sft_model_path",   sft_model_path,
                "--phase1_model_path", phase1_output_dir,
                "--pseudo_labels",    phase2_labels,
            ]
            if args.resume_dpo_checkpoint:
                phase2_extra += ["--resume_dpo_checkpoint", args.resume_dpo_checkpoint]
            run_script("train_strong.py", "--config", args.config,
                       *phase2_extra, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping Phase 2 (--skip_phase2). Using Phase 1 model as final.")
            phase2_output_dir = phase1_output_dir

        # ══ Phase 3: Evaluation (GRA) ════════════════════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 3: MWDPO_2PHASE — Evaluation (GRA)")
        logger.info("═" * 60)
        eval_args = [
            "--aligned_model_path", phase2_output_dir,
            "--sft_model_path",     sft_model_path,
        ]
        if args.run_gpt4:
            eval_args.append("--run_gpt4")
        if os.path.exists(phase1_labels_path):
            eval_args += ["--pseudo_labels", phase1_labels_path]
        run_script("evaluate.py", "--config", args.config, *eval_args, *args.overrides)

        logger.info("═" * 60)
        logger.info("Pipeline complete for method=mwdpo_2phase!")
        logger.info("═" * 60)
        return

    # ════════════════════════════════════════════════════════════════════
    # MWDPO_BC_2PHASE — Multi-Head Bootstrap Calibration DPO with Phase 2
    # ════════════════════════════════════════════════════════════════════
    if method == "mwdpo_bc_2phase":
        bc_cfg = cfg.get("bootstrap_calibration", {})
        bc_rm_dir = bc_cfg.get(
            "output_dir",
            cfg.get("reward_model", {}).get(
                "output_dir", "outputs/mwdpo_bc_2phase/reward_model"
            ),
        )
        label_base_dir = cfg.get(
            "multi_weak_label_output_dir",
            cfg.get("weak_label_output_dir", "outputs/mwdpo_bc_2phase/weak_labels"),
        )

        # Canonical D_h for Phase-2 ACE merge (never replaced by all_unlabeled)
        d_high_path = os.path.join(label_base_dir, "d_high", "pseudo_labeled.jsonl")
        # Phase-1 SFT + DPO train labels (d_high | all_unlabeled)
        phase1_labels_path = resolve_phase1_train_labels(
            cfg, label_base_dir, args.pseudo_labels
        )
        d_low_path = os.path.join(label_base_dir, "d_low", "pseudo_labeled.jsonl")

        # D_l with C_strong scores (output of compute_strong_confidence.py)
        sc_cfg = cfg.get("strong_confidence", {})
        d_low_scored_path = sc_cfg.get(
            "d_low_scored_path",
            os.path.join(label_base_dir, "d_low_scored", "pseudo_labeled.jsonl"),
        )

        # Phase 1 DPO output dir (π_Phase1)
        phase1_output_dir = cfg.training.get(
            "output_dir", "outputs/mwdpo_bc_2phase/strong_model_phase1"
        )

        # Phase 2 final model output dir
        p2_cfg = cfg.get("phase2_training", cfg.get("training", {}))
        phase2_output_dir = p2_cfg.get(
            "output_dir", "outputs/mwdpo_bc_2phase/strong_model_phase2"
        )

        # ══ Phase 1a: Train MultiHeadRewardModel on D_l ══════════════════════
        if not args.skip_reward_model:
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
                logger.info(
                    "PHASE 1a: MWDPO_BC_2PHASE — Train MultiHeadRewardModel on D_l "
                    "(shared backbone + K bootstrap heads)"
                )
                logger.info("═" * 60)
                run_script("train_bootstrap_reward_model.py", "--config", args.config,
                           *debug_flag, *args.overrides)
        else:
            logger.info("Skipping reward model training (--skip_reward_model)")

        # ══ Phase 1b: Calibrate + Label D_u → D_h ∪ D_l ════════════════════
        if not args.skip_labeling and not args.pseudo_labels:
            logger.info("═" * 60)
            logger.info(
                "PHASE 1b: MWDPO_BC_2PHASE — Calibrate T_k + Label D_u → D_h + D_l"
            )
            logger.info("═" * 60)
            extra = []
            if args.reward_model_path:
                extra += ["--checkpoint_dir", args.reward_model_path]
            run_script("label_bootstrap_calibration.py", "--config", args.config,
                       *extra, *debug_flag, *args.overrides)
        else:
            if args.pseudo_labels:
                logger.info(f"Using pre-computed Phase-1 labels: {args.pseudo_labels}")
            elif args.skip_labeling:
                logger.info("Skipping labeling (--skip_labeling)")

        # ══ Phase 2a: SFT Strong Model on phase1_data_mode labels ══════════
        if not args.skip_sft:
            logger.info("═" * 60)
            logger.info(
                f"PHASE 2a: MWDPO_BC_2PHASE — SFT Strong Model "
                f"(phase1_data_mode={cfg.get('phase1_data_mode', 'd_high')}) → π_SFT"
            )
            logger.info("═" * 60)
            sft_resume_args = (
                ["--resume_sft_checkpoint", args.resume_sft_checkpoint]
                if args.resume_sft_checkpoint else []
            )
            run_script("train_sft.py", "--config", args.config,
                       "--pseudo_labels", phase1_labels_path,
                       *sft_resume_args, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping SFT (--skip_sft)")

        # ══ Phase 2b: Standard DPO on phase1_data_mode labels → π_Phase1 ══
        if not getattr(args, "skip_phase1_dpo", False):
            logger.info("═" * 60)
            logger.info(
                f"PHASE 2b: MWDPO_BC_2PHASE — Standard DPO "
                f"(phase1_data_mode={cfg.get('phase1_data_mode', 'd_high')}) → π_Phase1"
            )
            logger.info("═" * 60)
            extra_args = ["--sft_model_path", sft_model_path,
                          "--pseudo_labels", phase1_labels_path]
            if args.resume_dpo_checkpoint:
                extra_args += ["--resume_dpo_checkpoint", args.resume_dpo_checkpoint]
            run_script("train_strong.py", "--config", args.config,
                       *extra_args, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping Phase 1 DPO (--skip_phase1_dpo)")

        # ══ Phase 2c: Pre-compute C_strong for D_l ═════════════════════════
        if not getattr(args, "skip_phase2", False):
            logger.info("═" * 60)
            logger.info("PHASE 2c: MWDPO_BC_2PHASE — Pre-compute C_strong for D_l")
            logger.info(f"  π_Phase1 : {phase1_output_dir}")
            logger.info(f"  π_SFT    : {sft_model_path}")
            logger.info(f"  D_l      : {d_low_path}")
            logger.info(f"  Output   : {d_low_scored_path}")
            logger.info("═" * 60)

            # weighting_mode: read from config, with backward compat for allow_negative
            _wm = sc_cfg.get("weighting_mode", None)
            if _wm is None and sc_cfg.get("allow_negative", False):
                _wm = "strong_raw"  # backward compat
            weighting_mode_args = (["--weighting_mode", str(_wm)] if _wm else [])
            # alpha: only pass for linear_combined
            _alpha = sc_cfg.get("alpha", None)
            alpha_args = (["--alpha", str(float(_alpha))] if _alpha is not None else [])
            batch_size_arg = ["--batch_size", str(int(sc_cfg.get("batch_size", 4)))]
            run_script(
                "compute_strong_confidence.py",
                "--config", args.config,
                "--phase1_model_path", phase1_output_dir,
                "--sft_model_path", sft_model_path,
                "--d_low_path", d_low_path,
                "--output_path", d_low_scored_path,
                *weighting_mode_args,
                *alpha_args,
                *batch_size_arg,
                *debug_flag, *args.overrides,
            )

            # Resolve Phase 2d data: D_l-only (legacy) or ACE D_weak asymmetric
            phase2_labels = resolve_phase2_train_labels(
                cfg, d_high_path, d_low_scored_path, label_base_dir,
                debug=bool(args.debug),
            )

            # ══ Phase 2d: Debate-Weighted / ACE Coverage DPO → π_Final ════
            _p2_mode = str(cfg.get("phase2_training", {}).get("phase2_data_mode", "d_low"))
            logger.info("═" * 60)
            logger.info(
                f"PHASE 2d: MWDPO_BC_2PHASE — Weighted DPO "
                f"(phase2_data_mode={_p2_mode}) → π_Final"
            )
            logger.info(f"  Phase2 labels: {phase2_labels}")
            logger.info("═" * 60)
            phase2_extra = [
                "--sft_model_path",   sft_model_path,
                "--phase1_model_path", phase1_output_dir,
                "--pseudo_labels",    phase2_labels,
            ]
            if args.resume_dpo_checkpoint:
                phase2_extra += ["--resume_dpo_checkpoint", args.resume_dpo_checkpoint]
            run_script("train_strong.py", "--config", args.config,
                       *phase2_extra, *debug_flag, *args.overrides)
        else:
            logger.info("Skipping Phase 2 (--skip_phase2). Using Phase 1 model as final.")
            phase2_output_dir = phase1_output_dir

        # ══ Phase 3: Evaluation (GRA) ════════════════════════════════════
        logger.info("═" * 60)
        logger.info("PHASE 3: MWDPO_BC_2PHASE — Evaluation (GRA)")
        logger.info("═" * 60)
        eval_args = [
            "--aligned_model_path", phase2_output_dir,
            "--sft_model_path",     sft_model_path,
        ]
        if args.run_gpt4:
            eval_args.append("--run_gpt4")
        if os.path.exists(phase1_labels_path):
            eval_args += ["--pseudo_labels", phase1_labels_path]
        run_script("evaluate.py", "--config", args.config, *eval_args, *args.overrides)

        logger.info("═" * 60)
        logger.info("Pipeline complete for method=mwdpo_bc_2phase!")
        logger.info("═" * 60)
        return

    raise ValueError(
        f"Unknown method: '{method}'. "
        "Choose: wdpo, cwpo, mwdpo, mwdpo_bootstrap_calibration, baseline_dpo, "
        "super_multi_dpo, mwdpo_2phase, mwdpo_bc_2phase"
    )


if __name__ == "__main__":
    main()
