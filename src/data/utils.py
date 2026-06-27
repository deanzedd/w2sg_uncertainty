"""
Data utilities: tokenization helpers and DataCollators.

Collators:
  - DPODataCollator:   standard DPO batching (prompt, chosen, rejected)
  - CWPODataCollator:  adds confidence_weights field for CWPO training
  - RewardDataCollator: for reward model training (Bradley-Terry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase


# ─────────────────────────────────────────────────────────────────────────── #
#  Tokenization helper                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def tokenize_preference_sample(
    sample: Dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    max_prompt_length: int = 256,
    separator: str = "\n",
) -> Dict[str, Any]:
    """
    Tokenize a preference sample into the format expected by DPOTrainer.

    Args:
        sample:           dict with 'prompt', 'chosen', 'rejected'
        tokenizer:        HuggingFace tokenizer
        max_length:       max total sequence length
        max_prompt_length: max prompt length
        separator:        DU3 fix — string inserted between prompt and response.
                          Default '\\n' is appropriate for both OPT and Qwen.
                          The old default ' ' (space) caused subtle tokenization
                          boundary issues: the space sometimes merged into the
                          last prompt token or first response token differently
                          than when the model sees prompt + response in context,
                          degrading DPO label quality.

    Returns a dict with:
        prompt_input_ids, prompt_attention_mask,
        chosen_input_ids, chosen_attention_mask,
        rejected_input_ids, rejected_attention_mask
    """
    prompt = sample["prompt"]
    chosen = sample["chosen"]
    rejected = sample["rejected"]

    # Tokenize prompt
    prompt_enc = tokenizer(
        prompt,
        max_length=max_prompt_length,
        truncation=True,
        add_special_tokens=True,
    )

    # Tokenize full sequences (prompt + response) for loss computation
    chosen_full = tokenizer(
        prompt + separator + chosen,
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )
    rejected_full = tokenizer(
        prompt + separator + rejected,
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "prompt_input_ids": prompt_enc["input_ids"],
        "prompt_attention_mask": prompt_enc["attention_mask"],
        "chosen_input_ids": chosen_full["input_ids"],
        "chosen_attention_mask": chosen_full["attention_mask"],
        "rejected_input_ids": rejected_full["input_ids"],
        "rejected_attention_mask": rejected_full["attention_mask"],
    }


def tokenize_for_reward_model(
    sample: Dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    separator: str = "\n",
) -> Dict[str, Any]:
    """
    Tokenize a preference sample for reward model training.
    Returns chosen and rejected encodings separately.

    Args:
        separator: DU3 fix — see tokenize_preference_sample docstring.
    """
    chosen_enc = tokenizer(
        sample["prompt"] + separator + sample["chosen"],
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )
    rejected_enc = tokenizer(
        sample["prompt"] + separator + sample["rejected"],
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )
    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
    }


# ─────────────────────────────────────────────────────────────────────────── #
#  DataCollators                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class DPODataCollator:
    """
    Collator for standard DPO batches.
    Pads chosen and rejected sequences to the same length within a batch.
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return _collate_dpo_features(
            features,
            self.tokenizer,
            self.max_length,
            self.label_pad_token_id,
            include_confidence=False,
        )


@dataclass
class CWPODataCollator:
    """
    Collator for CWPO batches.
    Same as DPODataCollator but also collates the `confidence_weight` field.
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = _collate_dpo_features(
            features,
            self.tokenizer,
            self.max_length,
            self.label_pad_token_id,
            include_confidence=True,
        )
        return batch


@dataclass
class RewardDataCollator:
    """
    Collator for reward model training.
    Pairs (chosen, rejected) into a single batch for Bradley-Terry loss.
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_ids = [f["chosen_input_ids"] for f in features]
        chosen_masks = [f["chosen_attention_mask"] for f in features]
        rejected_ids = [f["rejected_input_ids"] for f in features]
        rejected_masks = [f["rejected_attention_mask"] for f in features]

        chosen_ids_padded, chosen_masks_padded = _pad_sequences(
            chosen_ids, chosen_masks, self.tokenizer.pad_token_id, self.max_length
        )
        rejected_ids_padded, rejected_masks_padded = _pad_sequences(
            rejected_ids, rejected_masks, self.tokenizer.pad_token_id, self.max_length
        )

        return {
            "chosen_input_ids": chosen_ids_padded,
            "chosen_attention_mask": chosen_masks_padded,
            "rejected_input_ids": rejected_ids_padded,
            "rejected_attention_mask": rejected_masks_padded,
        }


# ─────────────────────────────────────────────────────────────────────────── #
#  Internal helpers                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def _pad_sequences(
    ids_list: List[List[int]],
    masks_list: List[List[int]],
    pad_token_id: int,
    max_length: int,
) -> tuple:
    """Right-pad a list of sequences to the same length."""
    # DU1 fix: max() on an empty sequence raises ValueError.
    # Return empty tensors when ids_list is empty (e.g. empty batch).
    if not ids_list:
        return (
            torch.zeros((0, 0), dtype=torch.long),
            torch.zeros((0, 0), dtype=torch.long),
        )
    max_len = min(max(len(s) for s in ids_list), max_length)
    padded_ids = []
    padded_masks = []
    for ids, mask in zip(ids_list, masks_list):
        ids = ids[:max_len]
        mask = mask[:max_len]
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_token_id] * pad_len)
        padded_masks.append(mask + [0] * pad_len)
    return (
        torch.tensor(padded_ids, dtype=torch.long),
        torch.tensor(padded_masks, dtype=torch.long),
    )


def _collate_dpo_features(
    features: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    label_pad_token_id: int,
    include_confidence: bool,
) -> Dict[str, torch.Tensor]:
    """Shared collation logic for DPO and CWPO."""
    chosen_ids = [f["chosen_input_ids"] for f in features]
    chosen_masks = [f["chosen_attention_mask"] for f in features]
    rejected_ids = [f["rejected_input_ids"] for f in features]
    rejected_masks = [f["rejected_attention_mask"] for f in features]

    chosen_ids_t, chosen_masks_t = _pad_sequences(
        chosen_ids, chosen_masks, tokenizer.pad_token_id, max_length
    )
    rejected_ids_t, rejected_masks_t = _pad_sequences(
        rejected_ids, rejected_masks, tokenizer.pad_token_id, max_length
    )

    batch = {
        "chosen_input_ids": chosen_ids_t,
        "chosen_attention_mask": chosen_masks_t,
        "rejected_input_ids": rejected_ids_t,
        "rejected_attention_mask": rejected_masks_t,
    }

    if include_confidence:
        weights = [f.get("confidence_weight", 1.0) for f in features]
        batch["confidence_weights"] = torch.tensor(weights, dtype=torch.float)

    return batch
