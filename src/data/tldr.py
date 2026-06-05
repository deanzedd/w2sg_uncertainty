"""
Reddit TL;DR Summarization Dataset Loader.

Original dataset: openai/summarize_from_feedback (HuggingFace Hub)
Mirror used:      CarperAI/openai_summarize_comparisons
  → Same data, already in Parquet format with {prompt, chosen, rejected} schema.
  → Required because `datasets >= 4.0` dropped support for legacy dataset scripts
    (.py), and `openai/summarize_from_feedback` still uses one.

Task:    Summarization preference — given a Reddit post, choose the
         better summary between two candidates.

Each sample is converted to:
    {
        "prompt":   "SUBREDDIT: <sub>\\nPOST: <post>\\nTL;DR:",
        "chosen":   "<preferred summary>",
        "rejected": "<dispreferred summary>",
    }
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
    "train": "train",
    "validation": "valid1",
    "valid": "valid1",
    "test": "test",
}


class TLDRDataset(BasePreferenceDataset):
    """
    Loader for the OpenAI Reddit TL;DR summarization preference dataset.

    Uses the CarperAI/openai_summarize_comparisons mirror (Parquet) because
    the original `openai/summarize_from_feedback` dataset relies on a legacy
    dataset script that is no longer supported by `datasets >= 4.0`.
    """

    def load_raw(self, split: str, cache_dir: Optional[str]) -> Any:
        # Strip any slice expression (e.g. "train[:10%]" → "train")
        base_split = split.split("[")[0].strip()
        hub_split = _SPLIT_MAP.get(base_split, base_split)

        # Reconstruct the full split string with the mapped split name
        # e.g. "train[:10%]" becomes "valid1[:10%]" if user asked for "validation[:10%]"
        if "[" in split:
            slice_part = split[split.index("["):]
            hub_split = hub_split + slice_part

        return load_dataset(
            _DATASET_ID,
            split=hub_split,
            cache_dir=cache_dir,
        )

    def preprocess_sample(self, raw_sample: Any) -> Optional[PreferenceSample]:
        # CarperAI/openai_summarize_comparisons already exposes {prompt, chosen, rejected}
        prompt = raw_sample.get("prompt", "").strip()
        chosen = raw_sample.get("chosen", "").strip()
        rejected = raw_sample.get("rejected", "").strip()

        if not prompt or not chosen or not rejected:
            return None

        return PreferenceSample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
        )
