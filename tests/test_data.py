"""Unit tests for data loading and D_l/D_u split."""

import pytest
from src.data.base_dataset import _SubsetDataset, BasePreferenceDataset, PreferenceSample


class MockDataset(BasePreferenceDataset):
    """Minimal mock dataset for testing."""

    def load_raw(self, split, cache_dir):
        return [
            {"chosen": f"chosen_{i}", "rejected": f"rejected_{i}", "prompt": f"prompt_{i}"}
            for i in range(100)
        ]

    def preprocess_sample(self, raw_sample):
        return PreferenceSample(
            prompt=raw_sample["prompt"],
            chosen=raw_sample["chosen"],
            rejected=raw_sample["rejected"],
        )


class TestBaseDataset:
    def test_length(self):
        ds = MockDataset(split="train", labeled_ratio=0.3, seed=42)
        assert len(ds) == 100

    def test_getitem(self):
        ds = MockDataset(split="train", labeled_ratio=0.3, seed=42)
        sample = ds[0]
        assert "prompt" in sample
        assert "chosen" in sample
        assert "rejected" in sample

    def test_labeled_unlabeled_split_ratio(self):
        ds = MockDataset(split="train", labeled_ratio=0.3, seed=42)
        labeled, unlabeled = ds.get_labeled_unlabeled_split()
        assert len(labeled) == 30  # 30% of 100
        assert len(unlabeled) == 70  # 70% of 100
        assert len(labeled) + len(unlabeled) == 100

    def test_split_reproducible(self):
        """Same seed should produce same split."""
        ds1 = MockDataset(split="train", labeled_ratio=0.3, seed=42)
        ds2 = MockDataset(split="train", labeled_ratio=0.3, seed=42)
        l1, _ = ds1.get_labeled_unlabeled_split()
        l2, _ = ds2.get_labeled_unlabeled_split()
        assert [s["prompt"] for s in l1] == [s["prompt"] for s in l2]

    def test_split_different_seeds(self):
        """Different seeds should produce different splits."""
        ds1 = MockDataset(split="train", labeled_ratio=0.3, seed=42)
        ds2 = MockDataset(split="train", labeled_ratio=0.3, seed=99)
        l1, _ = ds1.get_labeled_unlabeled_split()
        l2, _ = ds2.get_labeled_unlabeled_split()
        assert [s["prompt"] for s in l1] != [s["prompt"] for s in l2]

    def test_split_raises_on_test(self):
        """D_l/D_u split should only work on train split."""
        ds = MockDataset(split="test", labeled_ratio=0.3)
        with pytest.raises(ValueError):
            ds.get_labeled_unlabeled_split()

    def test_max_samples(self):
        ds = MockDataset(split="train", labeled_ratio=0.3, max_samples=10)
        assert len(ds) == 10

    def test_no_overlap_between_splits(self):
        """Labeled and unlabeled subsets should be disjoint."""
        ds = MockDataset(split="train", labeled_ratio=0.3, seed=42)
        labeled, unlabeled = ds.get_labeled_unlabeled_split()

        labeled_prompts = set(s["prompt"] for s in labeled)
        unlabeled_prompts = set(s["prompt"] for s in unlabeled)
        assert labeled_prompts.isdisjoint(unlabeled_prompts)
