#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) — dùng cho nhiều phase khác nhau:

  Phase 2b (WDPO/CWPO): SFT strong model on D_weak
    → Khi có --pseudo_labels: dùng D_weak (chosen responses từ pseudo-labeled data)
    → Output: π_θ^SFT dùng làm reference cho DPO phase tiếp theo

  Phase 1a (Baseline DPO): SFT strong model on D (toàn bộ dataset)
    → Khi method=baseline_dpo và không có --pseudo_labels: dùng toàn bộ D

Usage:
    # WDPO/CWPO Phase 2b: SFT on D_weak (sau khi đã label)
    python scripts/train_sft.py --config configs/wdpo_hh_rlhf.yaml \\
        --pseudo_labels outputs/wdpo/hh_rlhf/weak_labels/pseudo_labeled.jsonl

    # Baseline Phase 1a: SFT on D (full dataset)
    python scripts/train_sft.py --config configs/baseline_dpo_hh_rlhf.yaml

    # Debug mode
    python scripts/train_sft.py --config configs/cwpo_hh_rlhf.yaml --debug --max_samples 100

    # Override config values
    python scripts/train_sft.py --config configs/wdpo_hh_rlhf.yaml \\
        strong_model_name=facebook/opt-2.7b sft.output_dir=outputs/opt2.7b/sft
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataset
from src.models import get_model_wrapper
from src.trainers.sft_trainer import build_sft_args, run_sft
from src.utils import load_config, print_config, set_seed, setup_logging, init_wandb

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for strong model")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--pseudo_labels", type=str, default=None,
        help="Path to D_weak (pseudo_labeled.jsonl). "
             "When provided, SFT is performed on D_weak (Phase 2b). "
             "When absent, SFT is performed on D_l or full D depending on method.",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode with limited data")
    parser.add_argument("--max_samples", type=int, default=None, help="Max training samples")
    parser.add_argument(
        "--resume_sft_checkpoint", type=str, default=None,
        help="Path to a SFT checkpoint directory to resume training from "
             "(e.g. outputs/baseline_dpo/hh_rlhf/sft_strong/checkpoint-500). "
             "HF Trainer restores optimizer/scheduler state and skips completed steps. "
             "Model weights are loaded from the checkpoint directory automatically.",
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides: key=value")
    return parser.parse_args()


def _load_pseudo_labeled_as_sft_dataset(pseudo_labels_path: str, max_samples=None):
    """
    Load D_weak (pseudo_labeled.jsonl) và convert thành SFT dataset.

    Mỗi sample trong D_weak có: prompt, chosen, rejected, [confidence_weight]
    SFT chỉ cần: (prompt + chosen) — train strong model sinh ra chosen response.

    Trả về list of dict với keys: prompt, chosen (để BasePreferenceDataset hiểu được).
    """
    import datasets as hf_datasets

    records = []
    with open(pseudo_labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            records.append({
                "prompt":  sample["prompt"],
                "chosen":  sample["chosen"],
                "rejected": sample.get("rejected", ""),  # SFT không dùng rejected
            })
            if max_samples and len(records) >= max_samples:
                break

    logger.info(f"Loaded {len(records)} samples from D_weak: {pseudo_labels_path}")
    return hf_datasets.Dataset.from_list(records)


def main():
    args = parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.debug:
        cfg.sft.num_train_epochs = 1
        cfg.sft.save_steps = 5
        cfg.use_wandb = False

    setup_logging(cfg)
    print_config(cfg)
    set_seed(cfg.seed)

    method = cfg.get("method", "wdpo")

    # ── Determine training data source ───────────────────────────────────
    if args.pseudo_labels:
        # ── Phase 2b (WDPO/CWPO): SFT on D_weak ─────────────────────────
        logger.info(f"SFT Phase 2b: training on D_weak from {args.pseudo_labels}")
        init_wandb(cfg, tags=["sft", "d_weak", cfg.dataset_name, cfg.strong_model_name])

        max_samples = args.max_samples if args.debug else None
        train_dataset = _load_pseudo_labeled_as_sft_dataset(args.pseudo_labels, max_samples)

        eval_ds = get_dataset(
            cfg.dataset_name,
            split="test",
            labeled_ratio=1.0,
            cache_dir=cfg.get("cache_dir"),
        )

    elif method == "baseline_dpo":
        # ── Phase 1a (Baseline): SFT on D (full dataset) ─────────────────
        logger.info(f"SFT Phase 1a (Baseline): training on full dataset D")
        init_wandb(cfg, tags=["sft", "baseline", "full_D", cfg.dataset_name, cfg.strong_model_name])

        import datasets as hf_datasets
        train_raw = get_dataset(
            cfg.dataset_name,
            split="train",
            labeled_ratio=1.0,          # 1.0 = dùng toàn bộ D
            seed=cfg.seed,
            max_samples=args.max_samples if args.debug else None,
            cache_dir=cfg.get("cache_dir"),
        )
        # With labeled_ratio=1.0, get_labeled_unlabeled_split returns
        # all data as labeled (unlabeled is empty)
        all_ds, _ = train_raw.get_labeled_unlabeled_split()
        logger.info(f"Full dataset D size: {len(all_ds)}")

        train_dataset = hf_datasets.Dataset.from_list([
            {"prompt": all_ds[i]["prompt"], "chosen": all_ds[i]["chosen"],
             "rejected": all_ds[i].get("rejected", "")}
            for i in range(len(all_ds))
        ])
        logger.info(f"Full dataset D size: {len(train_dataset)}")


        eval_ds = get_dataset(
            cfg.dataset_name,
            split="test",
            labeled_ratio=1.0,
            cache_dir=cfg.get("cache_dir"),
        )

    else:
        # ── WDPO/CWPO không có pseudo_labels: SFT on D_l ─────────────────
        # (dùng khi chạy train_sft.py trực tiếp, không qua pipeline)
        logger.info(f"SFT: training on D_l (labeled split)")
        init_wandb(cfg, tags=["sft", "d_l", cfg.dataset_name, cfg.strong_model_name])

        train_raw = get_dataset(
            cfg.dataset_name,
            split="train",
            labeled_ratio=cfg.labeled_ratio,
            seed=cfg.seed,
            max_samples=args.max_samples if args.debug else None,
            cache_dir=cfg.get("cache_dir"),
        )
        labeled_ds, _ = train_raw.get_labeled_unlabeled_split()
        logger.info(f"D_l size: {len(labeled_ds)}")

        import datasets as hf_datasets
        train_dataset = hf_datasets.Dataset.from_list([
            {"prompt": s["prompt"], "chosen": s["chosen"], "rejected": s.get("rejected", "")}
            for s in [labeled_ds[i] for i in range(len(labeled_ds))]
        ])

        eval_ds = get_dataset(
            cfg.dataset_name,
            split="test",
            labeled_ratio=1.0,
            cache_dir=cfg.get("cache_dir"),
        )

    # ── Load strong model ────────────────────────────────────────────────
    logger.info(f"Loading strong model: {cfg.strong_model_name}")
    wrapper = get_model_wrapper(cfg.strong_model_name, cfg)

    # ── SFT training ─────────────────────────────────────────────────────
    sft_args = build_sft_args(cfg, role="strong")
    trainer = run_sft(
        model=wrapper.model,
        tokenizer=wrapper.tokenizer,
        train_dataset=train_dataset,
        args=sft_args,
        eval_dataset=eval_ds,
        resume_from_checkpoint=args.resume_sft_checkpoint,
    )

    logger.info(f"SFT complete! Model saved to: {sft_args.output_dir}")


if __name__ == "__main__":
    main()
