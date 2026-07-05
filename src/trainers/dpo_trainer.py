"""
Baseline DPO Trainer.

Trains the strong model using standard DPO *only on D_l* (human-labeled data).
No weak labeling involved — used as the comparison baseline.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import datasets as hf_datasets
from omegaconf import DictConfig
from trl import DPOConfig

from .sft_trainer import _detect_precision

logger = logging.getLogger(__name__)


def to_hf_dataset(samples) -> hf_datasets.Dataset:
    """
    Convert a list of dicts or a PyTorch Dataset to a HuggingFace Dataset.

    trl >= 1.0's DPOTrainer / SFTTrainer require HuggingFace datasets.Dataset
    (they call .map(), .filter(), .column_names, etc. internally).
    """
    if isinstance(samples, hf_datasets.Dataset):
        return samples
    if hasattr(samples, "__len__") and hasattr(samples, "__getitem__") and not isinstance(samples, list):
        # PyTorch Dataset — iterate and collect
        records = [samples[i] for i in range(len(samples))]
    else:
        records = list(samples)

    # Keep only raw text fields that DPOTrainer needs; drop any pre-tokenized tensors
    # (trl 1.5.x does its own tokenization from raw text).
    clean = []
    for r in records:
        clean.append({
            "prompt":   r["prompt"],
            "chosen":   r["chosen"],
            "rejected": r["rejected"],
        })
    return hf_datasets.Dataset.from_list(clean)


class BaselineDPODataset:
    """
    Thin wrapper around D_l samples for baseline DPO training.

    trl >= 1.0's DPOTrainer performs its own tokenization from raw text.
    This class is now just a lightweight container — call `to_hf_dataset()`
    before passing to DPOTrainer.
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer=None,       # kept for API compatibility; no longer used
        max_length: int = 512,
        max_prompt_length: int = 256,
    ) -> None:
        # Store only raw text fields; trl handles tokenization
        self._data = [
            {"prompt": s["prompt"], "chosen": s["chosen"], "rejected": s["rejected"]}
            for s in samples
        ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def to_hf(self) -> hf_datasets.Dataset:
        """Return as a HuggingFace Dataset ready for DPOTrainer."""
        return hf_datasets.Dataset.from_list(self._data)


def build_baseline_dpo_args(cfg: DictConfig) -> DPOConfig:
    """Build DPOConfig for baseline DPO."""
    train_cfg = cfg.training
    # DT2 fix: use _detect_precision(cfg.training) to validate bf16 GPU support
    # instead of directly reading train_cfg.get("bf16") without checking GPU capability.
    # Consistent with sft_trainer.py and wdpo_trainer.py approach.
    fp16, bf16 = _detect_precision(cfg.training)
    return DPOConfig(
        output_dir=train_cfg.get("output_dir", "outputs/baseline_dpo"),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(train_cfg.get("learning_rate", 5e-6)),  # 5e-6 per spec (Bảng 8)
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        # warmup: prefer warmup_steps (100 per spec, Bảng 8); zero-out ratio to avoid conflict
        warmup_steps=train_cfg.get("warmup_steps", 0),
        warmup_ratio=0.0 if train_cfg.get("warmup_steps", 0) > 0 else train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.05),  # 0.05 per spec
        optim=train_cfg.get("optim", "paged_adamw_32bit"),  # paged adamw 32bit per spec
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        fp16=fp16,
        bf16=bf16,
        beta=float(train_cfg.get("beta", 0.5)),           # 0.5 per spec (Bảng 8)
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),  # True per spec
        remove_unused_columns=False,
        report_to="wandb" if cfg.get("use_wandb", True) else "none",
        run_name=cfg.get("wandb_run_name", None),
    )
