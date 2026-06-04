"""
Baseline DPO Trainer.

Trains the strong model using standard DPO *only on D_l* (human-labeled data).
No weak labeling involved — used as the comparison baseline.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


class BaselineDPODataset(torch.utils.data.Dataset):
    """Wraps D_l samples for baseline DPO training."""

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = [self._tokenize(s) for s in samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _tokenize(self, sample: Dict) -> Dict:
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        def _enc(text):
            return self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_input_ids": _enc(prompt + " " + chosen)["input_ids"],
            "chosen_attention_mask": _enc(prompt + " " + chosen)["attention_mask"],
            "rejected_input_ids": _enc(prompt + " " + rejected)["input_ids"],
            "rejected_attention_mask": _enc(prompt + " " + rejected)["attention_mask"],
            "prompt_input_ids": _enc(prompt)["input_ids"],
            "prompt_attention_mask": _enc(prompt)["attention_mask"],
        }


def build_baseline_dpo_args(cfg: DictConfig) -> DPOConfig:
    """Build DPOConfig for baseline DPO."""
    train_cfg = cfg.training
    return DPOConfig(
        output_dir=train_cfg.get("output_dir", "outputs/baseline_dpo"),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(train_cfg.get("learning_rate", 5e-7)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        beta=float(train_cfg.get("beta", 0.1)),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        remove_unused_columns=False,
        report_to="wandb" if cfg.get("use_wandb", True) else "none",
        run_name=cfg.get("wandb_run_name", None),
    )
