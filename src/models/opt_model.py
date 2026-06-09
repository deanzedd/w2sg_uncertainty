"""
OPT model wrapper.

Supports the full OPT family:
  facebook/opt-125m
  facebook/opt-350m
  facebook/opt-1.3b
  facebook/opt-2.7b
  facebook/opt-6.7b
  facebook/opt-13b
  facebook/opt-30b

OPT-specific quirk: the tokenizer uses EOS as the padding token
(already handled in BaseModelWrapper._fix_tokenizer).

Multi-GPU support:
  Set `device_map: auto` in config to shard the model across all GPUs.
  For small OPT models (≤1.3B) on 10GB GPUs, single-card (device_map: null)
  is fine. For OPT-6.7B+ you need device_map: auto.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from omegaconf import DictConfig

from .base_model import BaseModelWrapper


class OPTModelWrapper(BaseModelWrapper):
    """
    Wrapper for the OPT model family.

    Args:
        cfg:        DictConfig with `model_name`, `cache_dir`, and LoRA settings.
        model_name: if provided, overrides cfg.model_name (for weak/strong distinction).
    """

    def __init__(self, cfg: DictConfig, model_name: str | None = None) -> None:
        self._model_name = model_name or cfg.get("model_name")
        super().__init__(cfg)

    def load_model_and_tokenizer(
        self,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        cache_dir = self.cfg.get("cache_dir", None)
        dtype = torch.bfloat16 if self.cfg.get("bf16", True) else torch.float32

        # ── Multi-GPU device_map (via BaseModelWrapper helper) ────────────
        # null  → single GPU (default for small OPT models)
        # "auto"→ shard across all available GPUs (needed for OPT-6.7B+)
        device_map = self._resolve_device_map(self.cfg)

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir=cache_dir,
            use_fast=True,
        )

        load_kwargs = dict(
            cache_dir=cache_dir,
            torch_dtype=dtype,
        )
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            **load_kwargs,
        )

        return model, tokenizer
