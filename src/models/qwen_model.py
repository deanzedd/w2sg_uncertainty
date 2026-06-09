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

Multi-GPU support:
  Set `device_map: auto` in config to automatically shard the model across
  all available GPUs (model parallelism). This is the recommended approach
  for large models (7B+) on multi-GPU machines with limited VRAM per card.

  For 4× RTX 3080 (10 GB each):
    Qwen2.5-0.5B / 1.5B / 3B  → device_map: null  (single GPU, fits in 10GB)
    Qwen2.5-7B                 → device_map: auto  (shards across 4 GPUs)
    Qwen2.5-14B                → device_map: auto  (shards across 4 GPUs)

  When using LoRA (use_lora: true), device_map defaults to "auto" automatically
  because PEFT requires it for multi-GPU gradient checkpointing.
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

        # ── Multi-GPU device_map (via BaseModelWrapper helper) ────────────
        # null   → single GPU  (default; fine for ≤3B models on 10GB GPU)
        # "auto" → shard model across all available GPUs (needed for 7B+)
        # When use_lora=True, defaults to "auto" automatically.
        device_map = self._resolve_device_map(self.cfg)

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir=cache_dir,
            use_fast=True,
            trust_remote_code=True,
        )

        load_kwargs = dict(
            cache_dir=cache_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            **load_kwargs,
        )

        return model, tokenizer
