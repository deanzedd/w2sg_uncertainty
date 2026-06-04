"""
Abstract base model wrapper.

All model wrappers must inherit from BaseModelWrapper and implement:
  - load_model_and_tokenizer(): return (model, tokenizer)

The base class handles:
  - LoRA / full fine-tuning mode switching
  - Reference model creation (frozen copy)
  - Common tokenizer fixes (padding, etc.)
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
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self._fix_tokenizer(self.tokenizer)
        if cfg.get("use_lora", False):
            self.model = self._wrap_lora(self.model, cfg)

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
