"""
Reddit TL;DR Summarization Dataset Loader.

Dataset: CarperAI/openai_summarize_comparisons  (HuggingFace Hub)
  Mirror of openai/summarize_from_feedback "comparisons" subset, stored as
  Parquet — compatible with datasets >= 4.0 (original uses legacy .py scripts).

Paper spec (prompt.txt §3):
  - 123,169 posts with preference labels after filtering
  - ~5% held out for validation
  - Filter out samples with MORE THAN 1024 tokens (Reddit posts = long prompts)
  - Training split is randomly divided: 30% → D_l, 70% → D_u

Task: Summarization preference — given a Reddit post, choose the better summary.

Each sample is converted to:
    {
        "prompt":   "<Reddit post prompt>",
        "chosen":   "<preferred summary>",
        "rejected": "<dispreferred summary>",
    }

Token filtering is done conservatively via whitespace-split word count
(≈ token count) to avoid requiring a tokenizer at data-loading time.
"""

from __future__ import annotations

from typing import Any, Optional

from datasets import load_dataset

from .base_dataset import BasePreferenceDataset, PreferenceSample

# CarperAI mirror: same data as openai/summarize_from_feedback "comparisons" subset
# but stored as Parquet — compatible with datasets >= 4.0.
# Splits: train / valid1 / valid2 / test
_DATASET_ID = "CarperAI/openai_summarize_comparisons"

# Map canonical split names to the names used by this dataset
_SPLIT_MAP = {
    "train":      "train",
    "validation": "valid1",
    "valid":      "valid1",
    "test":       "test",
}

# Paper spec: filter out samples with more than 1024 tokens
_MAX_TOKENS_TLDR = 1024


def _word_count(text: str) -> int:
    """Conservative token proxy: whitespace-split word count."""
    return len(text.split())


class TLDRDataset(BasePreferenceDataset):
    """
    Loader for the OpenAI Reddit TL;DR summarization preference dataset.

    Paper spec:
      - 123,169 samples retained after filtering (max_length=1024)
      - ~5% held out for validation (the "valid1" split in CarperAI mirror)
      - Samples with more than 1024 tokens are filtered out

    Uses the CarperAI/openai_summarize_comparisons mirror (Parquet) because
    the original `openai/summarize_from_feedback` dataset relies on a legacy
    dataset script that is no longer supported by `datasets >= 4.0`.
    """

    def __init__(
        self,
        split: str = "train",
        labeled_ratio: float = 0.3,
        seed: int = 42,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        max_length: int = _MAX_TOKENS_TLDR,
    ) -> None:
        self.max_length = max_length
        super().__init__(
            split=split,
            labeled_ratio=labeled_ratio,
            seed=seed,
            max_samples=max_samples,
            cache_dir=cache_dir,
        )

    def load_raw(self, split: str, cache_dir: Optional[str]) -> Any:
        # Strip any slice expression (e.g. "train[:10%]" → "train")
        base_split = split.split("[")[0].strip()
        hub_split  = _SPLIT_MAP.get(base_split, base_split)

        # Reconstruct the full split string with the mapped split name
        # e.g. "train[:10%]" → "valid1[:10%]" if user asked for "validation[:10%]"
        if "[" in split:
            slice_part = split[split.index("["):]
            hub_split  = hub_split + slice_part

        return load_dataset(
            _DATASET_ID,
            split=hub_split,
            cache_dir=cache_dir,
        )

    def preprocess_sample(self, raw_sample: Any) -> Optional[PreferenceSample]:
        # CarperAI/openai_summarize_comparisons exposes {prompt, chosen, rejected}
        prompt   = raw_sample.get("prompt",   "").strip()
        chosen   = raw_sample.get("chosen",   "").strip()
        rejected = raw_sample.get("rejected", "").strip()

        if not prompt or not chosen or not rejected:
            return None

        # ── Paper spec: filter out samples with more than 1024 tokens ─────
        total_words = _word_count(prompt + " " + chosen + " " + rejected)
        if total_words > self.max_length:
            return None

        return PreferenceSample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
        )
