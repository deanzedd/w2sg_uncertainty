"""
Qwen2.5 model wrapper.

Supports the full Qwen2.5 family:
  Qwen/Qwen2.5-0.5B
  Qwen/Qwen2.5-0.5B-Instruct
  Qwen/Qwen2.5-1.5B
  Qwen/Qwen2.5-1.5B-Instruct
  Qwen/Qwen2.5-3B
  Qwen/Qwen2.5-3B-Instruct
  Qwen/Qwen2.5-7B
  Qwen/Qwen2.5-7B-Instruct
  Qwen/Qwen2.5-14B
  ...

Note: Qwen2.5 tokenizers already have a pad_token; no fix needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from omegaconf import DictConfig

from .base_model import BaseModelWrapper


class Qwen25ModelWrapper(BaseModelWrapper):
    """
    Wrapper for the Qwen2.5 model family.

    Args:
        cfg:        DictConfig with `model_name`, `cache_dir`, and LoRA settings.
        model_name: if provided, overrides cfg.model_name.
    """

    def __init__(self, cfg: DictConfig, model_name: str | None = None) -> None:
        self._model_name = model_name or cfg.get("model_name")
        super().__init__(cfg)

    def load_model_and_tokenizer(
        self,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        cache_dir = self.cfg.get("cache_dir", None)
        dtype = torch.bfloat16 if self.cfg.get("bf16", True) else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir=cache_dir,
            use_fast=True,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        return model, tokenizer
