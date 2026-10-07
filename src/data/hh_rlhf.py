"""
HH-RLHF Dataset Loader.

Dataset: Anthropic/hh-rlhf  (HuggingFace Hub)
Subsets: "harmless-base" and "helpful-base"

Paper spec (prompt.txt §1):
  - Filter out samples with MORE THAN 512 tokens (total prompt+response length)
  - Training split is randomly divided: 30% → D_l, 70% → D_u
  - Three dataset variants:
      "harmless"  → only Harmless subset  (~35,908 train / 1,927 test)
      "helpful"   → only Helpful subset   (~34,873 train / 1,878 test)
      "hh_rlhf"   → concatenated joint    (~70,781 train / 3,805 test)
  - For the joint dataset, the test set is constructed by randomly sampling
    from the test splits of BOTH subsets and concatenating them.

Dataset format: Each sample has 'chosen' and 'rejected' fields, which are
full dialogue strings of the form:
    "\\n\\nHuman: <question>\\n\\nAssistant: <answer>"

We extract:
  - prompt:   everything up to and including the last "\\n\\nAssistant:"
  - chosen:   the assistant response in the 'chosen' field
  - rejected: the assistant response in the 'rejected' field

Token filtering is done conservatively via whitespace-split word count
(≈ token count) to avoid requiring a tokenizer at data-loading time.
The actual tokenizer truncation is still applied by the trainer via max_length.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from datasets import concatenate_datasets, load_dataset

from .base_dataset import BasePreferenceDataset, PreferenceSample

_DATASET_ID = "Anthropic/hh-rlhf"

# Marker used by the dataset to delimit the final assistant turn
_ASSISTANT_TOKEN = "\n\nAssistant:"

# Paper spec: filter out samples with more than 512 tokens.
# H1 fix: word-count proxy (str.split()) undercounts actual tokens for
# HH-RLHF dialogues by ~10-30% due to special tokens and punctuation.
# Tighten threshold to 400 words (~20% safety margin) to avoid samples
# that exceed 512 actual tokens after tokenization.
_MAX_TOKENS_HH = 400

# Subset name → HuggingFace data_dir argument
_SUBSET_MAP = {
    "harmless": "harmless-base",
    "helpful":  "helpful-base",
}


def _word_count(text: str) -> int:
    """Conservative token proxy: whitespace-split word count."""
    return len(text.split())


def _load_hf_subset(subset_dir: str, split: str, cache_dir: Optional[str]):
    """Load one HF subset split."""
    return load_dataset(
        _DATASET_ID,
        data_dir=subset_dir,
        split=split,
        cache_dir=cache_dir,
    )


class HHRLHFDataset(BasePreferenceDataset):
    """
    Loader for the Anthropic HH-RLHF preference dataset.

    Supports three dataset_variant values (passed as ``subset`` kwarg):
      - ``"harmless"``  : only the Harmless-Base subset
      - ``"helpful"``   : only the Helpful-Base subset
      - ``"hh_rlhf"``   : joint concatenation of both subsets (default)

    Each sample is converted to:
        {
            "prompt":   "<full conversation up to last Assistant turn>",
            "chosen":   "<preferred assistant response>",
            "rejected": "<dispreferred assistant response>",
        }

    Filtering: samples whose total (prompt + chosen + rejected) word count
    exceeds ``max_length`` (default 400 per H1 fix — tightened from paper's
    512 to compensate for the word-count proxy undercounting special tokens)
    are dropped. The actual tokenizer truncation is still applied by the
    trainer via max_length.
    """

    def __init__(
        self,
        split: str = "train",
        labeled_ratio: float = 0.3,
        seed: int = 42,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        subset: str = "hh_rlhf",       # "harmless" | "helpful" | "hh_rlhf"
        max_length: int = _MAX_TOKENS_HH,
    ) -> None:
        self.subset = subset
        self.max_length = max_length
        super().__init__(
            split=split,
            labeled_ratio=labeled_ratio,
            seed=seed,
            max_samples=max_samples,
            cache_dir=cache_dir,
        )

    def load_raw(self, split: str, cache_dir: Optional[str]) -> Any:
        """
        Load raw HF data according to self.subset.

        For "hh_rlhf" (joint), the test set is built by concatenating the
        test splits of both Harmless and Helpful subsets (paper §1, last line).
        """
        if self.subset in ("harmless", "helpful"):
            subset_dir = _SUBSET_MAP[self.subset]
            return _load_hf_subset(subset_dir, split, cache_dir)

        # Joint "hh_rlhf": concatenate both subsets
        harmless = _load_hf_subset("harmless-base", split, cache_dir)
        helpful  = _load_hf_subset("helpful-base",  split, cache_dir)
        return concatenate_datasets([harmless, helpful])

    def preprocess_sample(self, raw_sample: Any) -> Optional[PreferenceSample]:
        chosen_text: str  = raw_sample.get("chosen", "")
        rejected_text: str = raw_sample.get("rejected", "")

        # Extract prompt (everything up to and including the last "\n\nAssistant:")
        prompt = self._extract_prompt(chosen_text)
        if prompt is None:
            return None  # skip malformed samples

        chosen_response   = self._extract_response(chosen_text,   prompt)
        rejected_response = self._extract_response(rejected_text, prompt)

        if not chosen_response or not rejected_response:
            return None

        # ── Paper spec: filter out samples with more than 512 tokens ──────
        total_words = _word_count(prompt + " " + chosen_response + " " + rejected_response)
        if total_words > self.max_length:
            return None

        return PreferenceSample(
            prompt=prompt,
            chosen=chosen_response,
            rejected=rejected_response,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _extract_prompt(self, text: str) -> Optional[str]:
        """Return everything up to and including the LAST '\\n\\nAssistant:'."""
        last_idx = text.rfind(_ASSISTANT_TOKEN)
        if last_idx == -1:
            return None
        # Include the assistant token itself so the model knows it must reply
        return text[: last_idx + len(_ASSISTANT_TOKEN)]

    def _extract_response(self, text: str, prompt: str) -> str:
        """Return the assistant response by stripping the prompt prefix."""
        if not text.startswith(prompt):
            # prompt may differ slightly between chosen/rejected for multi-turn
            last_idx = text.rfind(_ASSISTANT_TOKEN)
            if last_idx == -1:
                return ""
            return text[last_idx + len(_ASSISTANT_TOKEN):].strip()
        return text[len(prompt):].strip()
