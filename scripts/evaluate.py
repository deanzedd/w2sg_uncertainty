#!/usr/bin/env python3
"""
Evaluation script — compute GRA and optionally GPT-4 win rate.

Usage:
    python scripts/evaluate.py \
        --config configs/wdpo_hh_rlhf.yaml \
        --aligned_model_path outputs/wdpo/hh_rlhf/strong_model \
        --sft_model_path outputs/wdpo/hh_rlhf/sft_strong

    # With GPT-4 win rate:
    python scripts/evaluate.py \
        --config configs/cwpo_hh_rlhf.yaml \
        --aligned_model_path outputs/cwpo/hh_rlhf/strong_model \
        --sft_model_path outputs/cwpo/hh_rlhf/sft_strong \
        --run_gpt4 \
        --pseudo_labels outputs/cwpo/hh_rlhf/weak_labels/pseudo_labeled.jsonl
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataset
from src.evaluation.evaluator import Evaluator
from src.weak_labeler.base_labeler import BaseWeakLabeler
from src.utils import load_config, print_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation (GRA + GPT-4 win rate)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--aligned_model_path", required=True,
                        help="Path to aligned model (WDPO/CWPO) checkpoint")
    parser.add_argument("--sft_model_path", required=True,
                        help="Path to SFT baseline checkpoint")
    parser.add_argument("--run_gpt4", action="store_true",
                        help="Run GPT-4 win rate evaluation (requires OPENAI_API_KEY)")
    parser.add_argument("--pseudo_labels", type=str, default=None,
                        help="Path to pseudo_labeled.jsonl (for preference accuracy)")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Số samples tối đa từ test set đưa vào generation "
                             "(default=500 để tránh chạy vài tiếng, đặt None để dùng toàn bộ)")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg)
    print_config(cfg)
    set_seed(cfg.seed)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Resolve device_map cho aligned + SFT models ────────────────────────────
    # eval_device_map trong config được ưu tiên hơn device_map (training)
    # Vì evaluation load 2 model lớn cùng lúc, cần phân bổ qua nhiều GPU
    eval_device_map = (
        cfg.get("eval", {}).get("device_map", None)  # eval.device_map (specific)
        or cfg.get("device_map", None)               # top-level device_map (training fallback)
    )
    # Nếu vẫn None và có nhiều hơn 2 GPU khả dụng → tự dùng "auto" để tránh OOM
    if eval_device_map is None and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            f"eval_device_map not set but {torch.cuda.device_count()} GPUs available. "
            "Forcing device_map='auto' for evaluation to avoid OOM."
        )
        eval_device_map = "auto"
    logger.info(f"Evaluator device_map: {eval_device_map}")

    # ── Load eval dataset ────────────────────────────────────────────────
    logger.info(f"Loading test dataset: {cfg.dataset_name}")
    eval_ds = get_dataset(
        cfg.dataset_name,
        split="test",
        labeled_ratio=1.0,
        max_samples=args.max_eval_samples,
        cache_dir=cfg.get("cache_dir"),
    )
    logger.info(f"Eval set size: {len(eval_ds)}")

    # ── Load pseudo-labels (for preference accuracy) ─────────────────────
    pseudo_labels = None
    human_labels = None
    if args.pseudo_labels and os.path.exists(args.pseudo_labels):
        pseudo_labels = BaseWeakLabeler.load(args.pseudo_labels)
        # D_l samples serve as "human labels" for comparison
        train_ds = get_dataset(
            cfg.dataset_name,
            split="train",
            labeled_ratio=cfg.labeled_ratio,
            seed=cfg.seed,
            cache_dir=cfg.get("cache_dir"),
        )
        labeled_ds, _ = train_ds.get_labeled_unlabeled_split()
        human_labels = list(labeled_ds)

    # ── Run evaluation ───────────────────────────────────────────────────
    evaluator = Evaluator(
        aligned_model_path=args.aligned_model_path,
        sft_model_path=args.sft_model_path,
        cfg=cfg,
        device=device,
        device_map=eval_device_map,
    )

    metrics = evaluator.run(
        eval_dataset=eval_ds,
        run_gpt4=args.run_gpt4,
        pseudo_labels=pseudo_labels,
        human_labels=human_labels,
    )

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
