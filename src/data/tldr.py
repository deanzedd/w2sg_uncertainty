"""
Reddit TL;DR Summarization Dataset Loader.

Dataset: openai/summarize_from_feedback  (HuggingFace Hub)
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

_DATASET_ID = "openai/summarize_from_feedback"
_SUBSET = "comparisons"  # the pairwise comparison subset


class TLDRDataset(BasePreferenceDataset):
    """
    Loader for the OpenAI Reddit TL;DR summarization preference dataset.
    """

    def load_raw(self, split: str, cache_dir: Optional[str]) -> Any:
        return load_dataset(
            _DATASET_ID,
            _SUBSET,
            split=split,
            cache_dir=cache_dir,
        )

    def preprocess_sample(self, raw_sample: Any) -> Optional[PreferenceSample]:
        info = raw_sample.get("info", {})
        summaries = raw_sample.get("summaries", [])
        choice = raw_sample.get("choice", None)

        if len(summaries) < 2 or choice is None:
            return None

        post = info.get("post", "")
        subreddit = info.get("subreddit", "")
        title = info.get("title", "")

        # Build prompt in the standard TL;DR format
        prompt_parts = []
        if subreddit:
            prompt_parts.append(f"SUBREDDIT: r/{subreddit}")
        if title:
            prompt_parts.append(f"TITLE: {title}")
        prompt_parts.append(f"POST: {post}")
        prompt_parts.append("TL;DR:")
        prompt = "\n".join(prompt_parts)

        # choice=0 means summaries[0] is preferred; choice=1 means summaries[1]
        chosen_idx = int(choice)
        rejected_idx = 1 - chosen_idx

        chosen = summaries[chosen_idx].get("text", "").strip()
        rejected = summaries[rejected_idx].get("text", "").strip()

        if not chosen or not rejected:
            return None

        return PreferenceSample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
        )
