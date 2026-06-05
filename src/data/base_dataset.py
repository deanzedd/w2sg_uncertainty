"""
Base dataset class for preference optimization experiments.

All datasets must inherit from BasePreferenceDataset and implement:
  - load_raw(): load raw data from HuggingFace Hub
  - preprocess_sample(): convert a raw sample to standard format

Standard format per sample:
  {
      "prompt":   str,   # the input prompt
      "chosen":   str,   # preferred response
      "rejected": str,   # dispreferred response
  }
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset


class PreferenceSample(Dict):
    """Type alias for a single preference sample dict."""
    pass


class BasePreferenceDataset(ABC, Dataset):
    """
    Abstract base class for preference datasets.

    Subclasses must implement `load_raw` and `preprocess_sample`.

    Args:
        split:          "train" or "test"
        labeled_ratio:  fraction of the training split designated as D_l
                        The rest becomes D_u (unlabeled).
                        Only applies when split="train".
        seed:           random seed for D_l / D_u split
        max_samples:    if set, truncate dataset to this many samples (for debugging)
        cache_dir:      HuggingFace cache directory
    """

    def __init__(
        self,
        split: str = "train",
        labeled_ratio: float = 0.3,
        seed: int = 42,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.split = split
        self.labeled_ratio = labeled_ratio
        self.seed = seed
        self.max_samples = max_samples
        self.cache_dir = cache_dir

        raw = self.load_raw(split, cache_dir)
        self.samples: List[PreferenceSample] = self._process_all(raw)

        if max_samples is not None:
            self.samples = self.samples[: max_samples]

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_raw(self, split: str, cache_dir: Optional[str]) -> Any:
        """Load and return the raw HuggingFace dataset object."""
        raise NotImplementedError

    @abstractmethod
    def preprocess_sample(self, raw_sample: Any) -> Optional[PreferenceSample]:
        """
        Convert one raw sample to standard dict format.

        Return None to skip a sample (will be filtered out).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Dataset interface                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PreferenceSample:
        return self.samples[idx]

    @property
    def column_names(self) -> List[str]:
        """Expose column names so trl >= 1.0's SFTTrainer can introspect the dataset."""
        if not self.samples:
            return ["prompt", "chosen", "rejected"]
        return list(self.samples[0].keys())

    # ------------------------------------------------------------------ #
    #  Labeled / Unlabeled split                                          #
    # ------------------------------------------------------------------ #

    def get_labeled_unlabeled_split(
        self,
    ) -> Tuple["BasePreferenceDataset", "BasePreferenceDataset"]:
        """
        Split `self.samples` into (D_l, D_u) based on `labeled_ratio`.

        Returns two lightweight views wrapping the same base class.
        """
        if self.split != "train":
            raise ValueError("D_l / D_u split is only meaningful for split='train'.")

        indices = list(range(len(self.samples)))
        rng = random.Random(self.seed)
        rng.shuffle(indices)

        n_labeled = max(1, int(len(indices) * self.labeled_ratio))
        labeled_indices = indices[:n_labeled]
        unlabeled_indices = indices[n_labeled:]

        labeled_ds = _SubsetDataset(self, labeled_indices)
        unlabeled_ds = _SubsetDataset(self, unlabeled_indices)
        return labeled_ds, unlabeled_ds

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _process_all(self, raw) -> List[PreferenceSample]:
        processed = []
        for item in raw:
            sample = self.preprocess_sample(item)
            if sample is not None:
                processed.append(sample)
        return processed


class _SubsetDataset(Dataset):
    """Lightweight index-based view over a BasePreferenceDataset."""

    def __init__(self, parent: BasePreferenceDataset, indices: List[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> PreferenceSample:
        return self._parent.samples[self._indices[idx]]

    @property
    def column_names(self) -> List[str]:
        """Expose column names so trl >= 1.0's SFTTrainer can introspect the dataset."""
        return self._parent.column_names
