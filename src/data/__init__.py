from .base_dataset import BasePreferenceDataset, PreferenceSample
from .hh_rlhf import HHRLHFDataset
from .tldr import TLDRDataset
from .ufb import UFBDataset
from .utils import (
    DPODataCollator,
    CWPODataCollator,
    RewardDataCollator,
    tokenize_preference_sample,
    tokenize_for_reward_model,
)

DATASET_REGISTRY = {
    # ── HH-RLHF variants ──────────────────────────────────────────────────
    # "hh_rlhf"   → joint Harmless + Helpful (~70k train / 3.8k test)
    # "harmless"  → only Harmless subset     (~35.9k train / 1.9k test)
    # "helpful"   → only Helpful subset      (~34.9k train / 1.9k test)
    # Token filter: max 512 tokens
    "hh_rlhf":  lambda **kw: HHRLHFDataset(subset="hh_rlhf",  **kw),
    "harmless":  lambda **kw: HHRLHFDataset(subset="harmless", **kw),
    "helpful":   lambda **kw: HHRLHFDataset(subset="helpful",  **kw),

    # ── TL;DR ─────────────────────────────────────────────────────────────
    # 123k Reddit posts with preference labels.  Token filter: max 1024 tokens.
    # ~5% validation held out (valid1 split in CarperAI mirror).
    "tldr":     TLDRDataset,

    # ── UltraFeedback Binarized (UFB) ─────────────────────────────────────
    # 61.1k train / 2k test. GPT-4 scored, binarized. Token filter: max 2048.
    "ufb":      UFBDataset,
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
    "UFBDataset",
    "DPODataCollator",
    "CWPODataCollator",
    "RewardDataCollator",
    "tokenize_preference_sample",
    "tokenize_for_reward_model",
    "get_dataset",
    "DATASET_REGISTRY",
]
