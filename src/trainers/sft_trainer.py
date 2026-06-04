"""
SFT Trainer — Supervised Fine-Tuning.

Used for:
1. Strong model: SFT before DPO/WDPO/CWPO training
2. WDPO weak model: SFT then DPO on D_l to get implicit reward policy

Wraps TRL's SFTTrainer.
"""

from __future__ import annotations

import logging
from typing import Optional

from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def build_sft_args(cfg: DictConfig, role: str = "strong") -> SFTConfig:
    """
    Build SFTConfig from config.

    Args:
        cfg:  full OmegaConf config
        role: "strong" (for strong model SFT) or "weak" (for weak model SFT)
    """
    sft_cfg = cfg.sft
    return SFTConfig(
        output_dir=sft_cfg.get("output_dir", f"outputs/sft_{role}"),
        num_train_epochs=sft_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=sft_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(sft_cfg.get("learning_rate", 2e-5)),
        lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.1),
        weight_decay=sft_cfg.get("weight_decay", 0.01),
        logging_steps=sft_cfg.get("logging_steps", 50),
        save_steps=sft_cfg.get("save_steps", 500),
        eval_steps=sft_cfg.get("eval_steps", 500),
        fp16=sft_cfg.get("fp16", False),
        bf16=sft_cfg.get("bf16", True),
        dataloader_num_workers=sft_cfg.get("dataloader_num_workers", 4),
        max_seq_length=cfg.get("max_length", 512),
        remove_unused_columns=False,
        report_to="wandb" if cfg.get("use_wandb", True) else "none",
        run_name=cfg.get("wandb_run_name", None),
    )


def run_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    args: SFTConfig,
    eval_dataset: Optional[Dataset] = None,
    formatting_func=None,
) -> SFTTrainer:
    """
    Run SFT training and return the trained SFTTrainer.

    Args:
        model:          model to fine-tune
        tokenizer:      tokenizer
        train_dataset:  training data (with 'chosen' field as the target text)
        args:           SFTConfig
        eval_dataset:   optional eval data
        formatting_func: optional function to format samples into strings
    """
    if formatting_func is None:
        # Default: use the 'chosen' response as the training target
        formatting_func = _default_formatting_func

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
    )

    logger.info(f"Starting SFT training... output_dir={args.output_dir}")
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info(f"SFT model saved to {args.output_dir}")
    return trainer


def _default_formatting_func(sample: dict) -> str:
    """
    Format a preference sample for SFT:
    concatenate prompt + chosen response as the training text.
    """
    return sample["prompt"] + " " + sample["chosen"]
