"""
HH-RLHF Dataset Loader.

Dataset: Anthropic/hh-rlhf  (HuggingFace Hub)
Format:  Each sample has 'chosen' and 'rejected' fields, which are full
         dialogue strings of the form:
             "\\n\\nHuman: <question>\\n\\nAssistant: <answer>"

We extract:
  - prompt:   everything up to and including the last "\\n\\nAssistant:"
  - chosen:   the assistant response in the 'chosen' field
  - rejected: the assistant response in the 'rejected' field
"""

from __future__ import annotations

from typing import Any, Optional

from datasets import load_dataset

from .base_dataset import BasePreferenceDataset, PreferenceSample

_DATASET_ID = "Anthropic/hh-rlhf"
_SPLIT_MAP = {"train": "train", "test": "test"}

# Marker used by the dataset to delimit the final assistant turn
_ASSISTANT_TOKEN = "\n\nAssistant:"


class HHRLHFDataset(BasePreferenceDataset):
    """
    Loader for the Anthropic HH-RLHF preference dataset.

    Each sample is converted to:
        {
            "prompt":   "<full conversation up to last Assistant turn>",
            "chosen":   "<preferred assistant response>",
            "rejected": "<dispreferred assistant response>",
        }
    """

    def load_raw(self, split: str, cache_dir: Optional[str]) -> Any:
        hf_split = _SPLIT_MAP.get(split, split)
        return load_dataset(
            _DATASET_ID,
            split=hf_split,
            cache_dir=cache_dir,
        )

    def preprocess_sample(self, raw_sample: Any) -> Optional[PreferenceSample]:
        chosen_text: str = raw_sample.get("chosen", "")
        rejected_text: str = raw_sample.get("rejected", "")

        # Extract prompt (everything up to and including the last "\\n\\nAssistant:")
        prompt = self._extract_prompt(chosen_text)
        if prompt is None:
            return None  # skip malformed samples

        chosen_response = self._extract_response(chosen_text, prompt)
        rejected_response = self._extract_response(rejected_text, prompt)

        if not chosen_response or not rejected_response:
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
            # prompt may differ slightly between chosen/rejected for multi-turn; handle gracefully
            last_idx = text.rfind(_ASSISTANT_TOKEN)
            if last_idx == -1:
                return ""
            return text[last_idx + len(_ASSISTANT_TOKEN):].strip()
        return text[len(prompt):].strip()
