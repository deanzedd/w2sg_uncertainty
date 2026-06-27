"""
Abstract base class for weak preference labelers.

All labelers take a dataset of (prompt, response1, response2) triplets
and assign preference labels to create pseudo-labeled dataset D̂.

Subclasses:
  - DPORewardLabeler:    WDPO — uses DPO implicit reward
  - ConfidenceLabeler:   CWPO — uses scalar reward model + confidence margin
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset


class PseudoLabeledSample(Dict):
    """
    Standard format for pseudo-labeled samples:
        {
            "prompt":             str,
            "chosen":             str,   # y+ (weak model's preferred)
            "rejected":           str,   # y- (weak model's dispreferred)
            "confidence_weight":  float, # C in [0,1); 1.0 for WDPO (no weighting)
        }
    """
    pass


class BaseWeakLabeler(ABC):
    """
    Abstract weak labeler interface.

    Args:
        device:     torch device string (e.g. "cuda", "cpu")
        batch_size: inference batch size
    """

    def __init__(self, device: str = "cuda", batch_size: int = 16) -> None:
        self.device = device
        self.batch_size = batch_size

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def label_dataset(
        self,
        dataset: Dataset,
        max_samples: Optional[int] = None,
    ) -> List[PseudoLabeledSample]:
        """
        Label all samples in `dataset` and return pseudo-labeled list D̂.

        Args:
            dataset:     unlabeled dataset (D_u) with 'prompt','chosen','rejected'
            max_samples: if set, only label this many samples (for debugging)

        Returns:
            list of PseudoLabeledSample dicts
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Optional: save / load D̂ to disk                                   #
    # ------------------------------------------------------------------ #

    def save(self, samples: List[PseudoLabeledSample], path: str) -> None:
        """Save pseudo-labeled dataset to a JSON Lines file."""
        import json
        import os
        # BL1 fix: os.path.dirname("bare.jsonl") == "" and os.makedirs("") raises
        # FileNotFoundError. Guard to only call makedirs when there IS a directory.
        if d := os.path.dirname(path):
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    @staticmethod
    def load(path: str) -> List[PseudoLabeledSample]:
        """Load pseudo-labeled dataset from a JSON Lines file."""
        import json
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(PseudoLabeledSample(**json.loads(line)))
        return samples
