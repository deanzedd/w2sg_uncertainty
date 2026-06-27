"""
SFT Trainer — Supervised Fine-Tuning.

Used for:
1. Strong model: SFT before DPO/WDPO/CWPO training
2. WDPO weak model: SFT then DPO on D_l to get implicit reward policy

Wraps TRL's SFTTrainer.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import datasets as hf_datasets
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def _detect_precision(cfg) -> Tuple[bool, bool]:
    """
    Auto-detect the best training precision for the current GPU setup.

    Priority:
      1. Explicit config override: if fp16=True or bf16=True in config, use that
         (but only if CUDA is actually available — skip if no GPU)
      2. Auto-detect: bf16 if Ampere+ GPU (SM >= 8.0), else fp16 if CUDA available
      3. Fallback: fp32 (bf16=False, fp16=False) if no CUDA

    Returns:
        (fp16, bf16) tuple of booleans

    This handles the common case of a faulty GPU 0 breaking NVML / CUDA init.
    The caller should not set bf16=True unconditionally — always call this.
    """
    cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0

    if not cuda_ok:
        logger.warning(
            "No CUDA GPU detected (or GPU error). Falling back to fp32 training. "
            "If you have GPUs, check 'nvidia-smi' for hardware errors (e.g., GPU 0 Unknown Error). "
            "Try restarting the machine or setting CUDA_VISIBLE_DEVICES to skip faulty GPUs."
        )
        return False, False

    # Check explicit config overrides
    fp16_req = cfg.get("fp16", False)
    bf16_req = cfg.get("bf16", True)   # default True per pipeline spec

    if fp16_req:
        return True, False
    if bf16_req:
        # Verify bf16 is actually supported on this GPU
        bf16_ok = torch.cuda.is_bf16_supported()
        if bf16_ok:
            return False, True
        else:
            logger.warning(
                "bf16 requested but not supported on this GPU "
                f"(capability: {torch.cuda.get_device_capability(0)}). "
                "Falling back to fp16."
            )
            return True, False

    return False, False  # explicit fp32



def build_sft_args(cfg: DictConfig, role: str = "strong") -> SFTConfig:
    """
    Build SFTConfig from config.

    Per pipeline spec:
      - learning_rate:               1e-5
      - per_device_train_batch_size: 4 (with gradient_accumulation_steps=4 → effective 16)
      - num_train_epochs:            1 (WDPO) or 3 (CWPO)

    Args:
        cfg:  full OmegaConf config
        role: "strong" (for strong model SFT) or "weak" (for weak model SFT in WDPO)
    """
    sft_cfg = cfg.sft

    # When device_map="auto" is used (multi-GPU model parallelism),
    # the Trainer must NOT move the model itself — HF handles placement.
    # Also, gradient_checkpointing reduces VRAM usage significantly.
    use_device_map = cfg.get("device_map", None) == "auto" or cfg.get("use_lora", False)

    # Auto-detect precision: handles faulty GPU 0 / no CUDA / bf16 unsupported
    # ST1 fix: pass cfg.sft (section config) not full cfg, so _detect_precision
    # reads sft.bf16 / sft.fp16 correctly instead of top-level defaults.
    fp16, bf16 = _detect_precision(cfg.sft)
    logger.info(f"Training precision: fp16={fp16}, bf16={bf16}")

    return SFTConfig(
        output_dir=sft_cfg.get("output_dir", f"outputs/sft_{role}"),
        num_train_epochs=sft_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=sft_cfg.get("per_device_eval_batch_size", 4),
        # gradient_accumulation: 4*4=16 effective batch size per spec
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(sft_cfg.get("learning_rate", 1e-5)),  # 1e-5 per spec
        lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.1),
        weight_decay=sft_cfg.get("weight_decay", 0.01),
        logging_steps=sft_cfg.get("logging_steps", 50),
        save_steps=sft_cfg.get("save_steps", 500),
        eval_steps=sft_cfg.get("eval_steps", 500),
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=sft_cfg.get("dataloader_num_workers", 4),
        max_length=cfg.get("max_length", 512),
        remove_unused_columns=False,
        report_to="wandb" if cfg.get("use_wandb", True) else "none",
        run_name=cfg.get("wandb_run_name", None),
        # ── Multi-GPU / memory settings ────────────────────────────────
        # gradient_checkpointing: trades compute for VRAM (recomputes activations)
        # Required for 7B+ models. Set sft.gradient_checkpointing=true in config or CLI.
        gradient_checkpointing=sft_cfg.get("gradient_checkpointing", False),
        # With device_map="auto", HF Trainer must not re-assign model devices
        # ddp_find_unused_parameters is only relevant for DDP, not model parallel
        ddp_find_unused_parameters=False if use_device_map else None,
    )


def _to_hf_dataset(dataset: Dataset) -> hf_datasets.Dataset:
    """
    Convert a PyTorch Dataset (or our custom BasePreferenceDataset/_SubsetDataset)
    to a HuggingFace datasets.Dataset.

    trl >= 1.0's SFTTrainer / DPOTrainer heavily rely on HuggingFace Dataset APIs
    (.map, .filter, .select, .column_names, etc.).  Converting once here avoids
    patching each missing method individually.
    """
    if isinstance(dataset, hf_datasets.Dataset):
        return dataset
    samples = [dataset[i] for i in range(len(dataset))]
    return hf_datasets.Dataset.from_list(samples)


def _to_sft_format(hf_ds: hf_datasets.Dataset) -> hf_datasets.Dataset:
    """
    Convert a preference dataset {prompt, chosen, rejected} to the SFT format
    expected by trl >= 1.0: {prompt, completion}.

    trl 1.5.x's SFTTrainer internally looks for example["completion"] during
    tokenization (sft_trainer.py::tokenize_fn). Using a `formatting_func` conflicts
    with this pipeline. The correct approach is to supply the data in
    {prompt, completion} form and let SFTTrainer handle tokenization natively.
    """
    def _rename(example):
        return {"prompt": example["prompt"], "completion": example["chosen"]}

    return hf_ds.map(_rename, remove_columns=hf_ds.column_names)


def run_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    args: SFTConfig,
    eval_dataset: Optional[Dataset] = None,
    formatting_func=None,  # kept for API compatibility but no longer used
) -> SFTTrainer:
    """
    Run SFT training and return the trained SFTTrainer.

    Args:
        model:          model to fine-tune
        tokenizer:      tokenizer
        train_dataset:  training data (with 'chosen' field as the target text)
        args:           SFTConfig
        eval_dataset:   optional eval data
        formatting_func: DEPRECATED — ignored; trl 1.5.x uses {prompt, completion}
                         format natively.
    """
    # trl >= 1.0 requires HuggingFace datasets.Dataset (needs .map, .column_names, etc.)
    # Convert our custom PyTorch Dataset to HF Dataset before passing to SFTTrainer.
    hf_train = _to_hf_dataset(train_dataset)
    hf_eval  = _to_hf_dataset(eval_dataset) if eval_dataset is not None else None

    # trl 1.5.x SFTTrainer natively expects {prompt, completion} columns.
    # Map "chosen" → "completion" so trl's internal tokenize_fn can find the field.
    hf_train = _to_sft_format(hf_train)
    if hf_eval is not None:
        hf_eval = _to_sft_format(hf_eval)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=hf_train,
        eval_dataset=hf_eval,
        # No formatting_func: let SFTTrainer use its native {prompt, completion} pipeline
    )

    logger.info(f"Starting SFT training... output_dir={args.output_dir}")
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info(f"SFT model saved to {args.output_dir}")
    return trainer

