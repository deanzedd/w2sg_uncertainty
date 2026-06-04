from .base_dataset import BasePreferenceDataset, PreferenceSample
from .hh_rlhf import HHRLHFDataset
from .tldr import TLDRDataset
from .utils import (
    DPODataCollator,
    CWPODataCollator,
    RewardDataCollator,
    tokenize_preference_sample,
    tokenize_for_reward_model,
)

DATASET_REGISTRY = {
    "hh_rlhf": HHRLHFDataset,
    "tldr": TLDRDataset,
}


def get_dataset(name: str, **kwargs) -> BasePreferenceDataset:
    """Factory: load dataset by name from registry."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name](**kwargs)


__all__ = [
    "BasePreferenceDataset",
    "PreferenceSample",
    "HHRLHFDataset",
    "TLDRDataset",
    "DPODataCollator",
    "CWPODataCollator",
    "RewardDataCollator",
    "tokenize_preference_sample",
    "tokenize_for_reward_model",
    "get_dataset",
    "DATASET_REGISTRY",
]
