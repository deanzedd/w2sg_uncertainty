"""
Abstract base model wrapper.

All model wrappers must inherit from BaseModelWrapper and implement:
  - load_model_and_tokenizer(): return (model, tokenizer)

The base class handles:
  - LoRA / full fine-tuning mode switching
  - Reference model creation (frozen copy)
  - Common tokenizer fixes (padding, etc.)
  - Multi-GPU device_map resolution (single-card or model parallel)
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BaseModelWrapper(ABC):
    """
    Wraps a HuggingFace causal LM with optional LoRA support.

    Args:
        cfg: OmegaConf DictConfig with at least:
            - model_name or equivalent
            - use_lora, lora_r, lora_alpha, lora_dropout, lora_target_modules
            - cache_dir
            - device_map: null (single GPU), "auto" (multi-GPU, recommended for 7B+),
                          "balanced" (equal sharding), or a dict for manual placement.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self._fix_tokenizer(self.tokenizer)
        if cfg.get("use_lora", False):
            self.model = self._wrap_lora(self.model, cfg)
        if cfg.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_model_and_tokenizer(
        self,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load and return (model, tokenizer) from HuggingFace Hub."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Public helpers                                                      #
    # ------------------------------------------------------------------ #

    def get_ref_model(self) -> PreTrainedModel:
        """
        Return a frozen deep copy of the model for use as DPO reference.
        The copy is put in eval mode with all gradients disabled.
        """
        ref = copy.deepcopy(self.model)
        ref.eval()
        for param in ref.parameters():
            param.requires_grad = False
        return ref

    # ------------------------------------------------------------------ #
    #  Multi-GPU: device_map resolution                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_device_map(cfg) -> Optional[str]:
        """
        Determine the device_map to pass to from_pretrained().

        Priority:
          1. cfg.device_map  (explicit override in YAML or CLI)
          2. "auto" if use_lora=True  (PEFT requires device_map for multi-GPU)
          3. None            (single-GPU default, HF Trainer manages placement)

        device_map values:
          None        — load model entirely on CPU, then Trainer moves to cuda:0
          "auto"      — HF shards layers across all available GPUs by free memory
          "balanced"  — HF shards layers equally across all available GPUs
          dict        — manual placement, e.g. {"model.embed": 0, "lm_head": 1}

        For 4× RTX 3080 (10 GB each) + Qwen2.5-7B (~14 GB bfloat16):
          → use device_map: auto  in your config YAML (see configs/base.yaml)
        """
        explicit = cfg.get("device_map", None)
        if explicit is not None:
            return explicit
        # LoRA with multi-GPU: PEFT requires device_map for gradient offloading
        if cfg.get("use_lora", False):
            return "auto"
        return None

    # ------------------------------------------------------------------ #
    #  LoRA wrapping                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wrap_lora(model: PreTrainedModel, cfg) -> PreTrainedModel:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError(
                "peft is required for LoRA training. Install it: pip install peft"
            )
        target_modules = cfg.get("lora_target_modules", None)
        # If not specified, let PEFT auto-detect
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model

    # ------------------------------------------------------------------ #
    #  Tokenizer fixes                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fix_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Common tokenizer issues:
        - OPT uses eos_token as pad (no explicit pad_token set)
        - Some models need padding_side set to 'left' for generation
        """
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
