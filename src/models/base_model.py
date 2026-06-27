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
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

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
        # M1 fix: gradient_checkpointing_enable() is NOT called here.
        # GC is managed per-trainer via SFTConfig/DPOConfig.gradient_checkpointing,
        # which TRL applies correctly from the section config (sft.gradient_checkpointing,
        # training.gradient_checkpointing). A top-level cfg.get("gradient_checkpointing")
        # always returns False (base.yaml default) → the old call was dead code.

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
        Return a frozen reference model for DPO training.

        M2 fix: Use safetensors memory-mapped I/O instead of deepcopy.

        Background:
          deepcopy(model) duplicates all weight tensors in VRAM, doubling memory
          usage. For a 7B model (~14 GB bfloat16), this causes OOM on most setups.

        Modern approach (safetensors / mmap):
          HuggingFace transformers >= 4.38 uses memory-mapped .safetensors files
          by default when loading from cache. The OS shares mmap pages between
          multiple loads of the same checkpoint (copy-on-write semantics).
          Reloading the same model a second time consumes near-zero additional VRAM
          for weight tensors, because the physical memory pages are shared.

          This is the current best practice used by TRL, Axolotl, and LLaMA-Factory.

        For LoRA-wrapped models:
          The caller should prefer PEFT's `disable_adapter()` context manager
          to avoid loading a separate ref model entirely.
        """
        model_name = getattr(self, "_model_name", None)
        if model_name is None:
            raise RuntimeError(
                "get_ref_model() requires _model_name to be set by the subclass. "
                "Ensure your ModelWrapper sets self._model_name before super().__init__()."
            )

        dtype = next(self.model.parameters()).dtype
        device_map = self._resolve_device_map(self.cfg)
        cache_dir = self.cfg.get("cache_dir", None)

        load_kwargs: dict = {
            "torch_dtype": dtype,
            # transformers >= 4.38 uses mmap=True for .safetensors by default.
            # Re-loading the same checkpoint reuses OS mmap pages (copy-on-write),
            # so ref model weights share physical memory with the policy model.
        }
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir

        ref = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
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

    @staticmethod
    def _resolve_dtype(cfg) -> "torch.dtype":
        """
        Resolve model loading dtype from config.

        O1/O2 fix: reads from section-level config (sft.bf16, training.bf16,
        reward_model.bf16) with fallback to top-level, instead of always
        reading top-level cfg.get("bf16", True) which defaults to True even when
        a section config sets bf16: false.

        Priority:
          1. sft.bf16 / training.bf16 / reward_model.bf16 (section configs)
          2. Top-level bf16 / fp16
          3. Default: bfloat16
        """
        # Check section configs first
        for section in ("sft", "training", "reward_model"):
            section_cfg = cfg.get(section, {})
            if section_cfg is None:
                continue
            bf16_val = section_cfg.get("bf16", None)
            fp16_val = section_cfg.get("fp16", None)
            if fp16_val:   # explicit fp16
                return torch.float16
            if bf16_val is not None:
                return torch.bfloat16 if bf16_val else torch.float32
        # Top-level fallback
        if cfg.get("fp16", False):
            return torch.float16
        return torch.bfloat16 if cfg.get("bf16", True) else torch.float32

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
