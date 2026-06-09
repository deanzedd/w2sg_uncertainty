"""
UltraFeedback Binarized (UFB) Dataset Loader.

Dataset: HuggingFaceH4/ultrafeedback_binarized  (HuggingFace Hub)
Reference: Cui et al. (2024), used to train Zephyr-7B-β.

Paper spec (prompt.txt §2):
  - Original UltraFeedback: 64k prompts, each with 4 model completions
  - GPT-4 scores each completion on helpfulness/honesty
  - Binarization: highest-scored completion → "chosen";
    one of remaining 3 randomly selected → "rejected"
  - Training set: 61.1k samples
  - Test set: 2k samples
  - Filter out samples with MORE THAN 2048 tokens
  - Training split is randomly divided: 30% → D_l, 70% → D_u

HuggingFace Hub schema for "HuggingFaceH4/ultrafeedback_binarized":
  - prompt          : str   — the instruction / question
  - prompt_id       : str
  - chosen          : list  — [{"role": "user", ...}, {"role": "assistant", "content": ...}]
  - rejected        : list  — same structure
  - score_chosen    : float
  - score_rejected  : float

We extract the "content" of the last assistant turn from chosen/rejected lists
and use the "prompt" field directly.

Token filtering is done conservatively via whitespace-split word count
(≈ token count) to avoid requiring a tokenizer at data-loading time.
"""

from __future__ import annotations

from typing import Any, List, Optional

from datasets import load_dataset

from .base_dataset import BasePreferenceDataset, PreferenceSample

_DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"

# HuggingFaceH4/ultrafeedback_binarized split names
# train_prefs  → 61,135 samples  (paper: 61.1k)
# test_prefs   → 2,000  samples  (paper: 2k)
_SPLIT_MAP = {
    "train":      "train_prefs",
    "test":       "test_prefs",
    "validation": "test_prefs",  # no separate val; use test as proxy
}

# Paper spec: filter out samples with more than 2048 tokens
_MAX_TOKENS_UFB = 2048


def _word_count(text: str) -> int:
    """Conservative token proxy: whitespace-split word count."""
    return len(text.split())


def _extract_content(messages: Any) -> str:
    """
    Extract the assistant's response text from a messages list.

    UFB stores chosen/rejected as a list of dicts:
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    We take the last assistant turn's "content".
    If the field is already a plain string (older schema), return it directly.
    """
    if isinstance(messages, str):
        return messages.strip()

    if not isinstance(messages, (list, tuple)):
        return str(messages).strip()

    # Walk backwards to find the last assistant message
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()

    # Fallback: return content of last element
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            return str(last.get("content", "")).strip()
    return ""


class UFBDataset(BasePreferenceDataset):
    """
    Loader for the UltraFeedback Binarized (UFB) preference dataset.

    Paper spec:
      - 61.1k training samples, 2k test samples (after filtering)
      - Samples with more than 2048 tokens are filtered out
      - chosen  = highest GPT-4-scored completion
      - rejected = one of the remaining 3 completions (randomly selected by dataset)

    Each sample is converted to:
        {
            "prompt":   "<instruction / question>",
            "chosen":   "<best-scored assistant response>",
            "rejected": "<randomly-selected inferior response>",
        }
    """

    def __init__(
        self,
        split: str = "train",
        labeled_ratio: float = 0.3,
        seed: int = 42,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        max_length: int = _MAX_TOKENS_UFB,
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
        base_split = split.split("[")[0].strip()
        hub_split  = _SPLIT_MAP.get(base_split, base_split)

        # Reconstruct slice expression if present
        if "[" in split:
            slice_part = split[split.index("["):]
            hub_split  = hub_split + slice_part

        return load_dataset(
            _DATASET_ID,
            split=hub_split,
            cache_dir=cache_dir,
        )

    def preprocess_sample(self, raw_sample: Any) -> Optional[PreferenceSample]:
        # UFB schema: prompt (str), chosen (list|str), rejected (list|str)
        prompt   = raw_sample.get("prompt", "").strip()
        chosen   = _extract_content(raw_sample.get("chosen",   ""))
        rejected = _extract_content(raw_sample.get("rejected", ""))

        if not prompt or not chosen or not rejected:
            return None

        # ── Paper spec: filter out samples with more than 2048 tokens ─────
        total_words = _word_count(prompt + " " + chosen + " " + rejected)
        if total_words > self.max_length:
            return None

        return PreferenceSample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
        )
