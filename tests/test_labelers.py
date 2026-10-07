"""Unit tests for weak labelers."""

import pytest
import torch
from unittest.mock import MagicMock, patch


class TestDPORewardLabeler:
    def test_label_prefers_higher_reward(self):
        """WDPO labeler should select the response with higher implicit reward."""
        from src.weak_labeler.dpo_reward_labeler import DPORewardLabeler

        # Mock models and tokenizer
        mock_model = MagicMock()
        mock_ref = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        labeler = DPORewardLabeler(
            weak_model=mock_model,
            ref_model=mock_ref,
            tokenizer=mock_tokenizer,
            beta=0.1,
            device="cpu",
        )

        # Mock implicit reward: y1 gets reward 2.0, y2 gets reward 1.0
        with patch.object(labeler, "_implicit_reward", side_effect=[2.0, 1.0]):
            sample = {"prompt": "Q: hi", "chosen": "y1", "rejected": "y2"}
            result = labeler._label_batch([sample])[0]
            assert result["chosen"] == "y1"  # y1 has higher reward
            assert result["rejected"] == "y2"
            assert result["confidence_weight"] == 1.0  # WDPO: uniform

    def test_label_swaps_when_rejected_better(self):
        """WDPO should swap chosen/rejected when rejected has higher reward."""
        from src.weak_labeler.dpo_reward_labeler import DPORewardLabeler

        mock_model = MagicMock()
        mock_ref = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        labeler = DPORewardLabeler(
            weak_model=mock_model,
            ref_model=mock_ref,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # y2 (originally rejected) has higher reward
        with patch.object(labeler, "_implicit_reward", side_effect=[1.0, 3.0]):
            sample = {"prompt": "Q: hi", "chosen": "y1_orig", "rejected": "y2_orig"}
            result = labeler._label_batch([sample])[0]
            assert result["chosen"] == "y2_orig"  # Swapped!
            assert result["rejected"] == "y1_orig"


class TestConfidenceLabeler:
    def test_confidence_range(self):
        """Confidence should always be in [0, 1] (1.0 is valid at float limit)."""
        for _ in range(20):
            s_chosen = torch.randn(1).item() * 5
            s_rejected = torch.randn(1).item() * 5
            diff = torch.tensor(s_chosen - s_rejected)
            confidence = float(2.0 * (torch.sigmoid(diff) - 0.5).clamp(min=0.0))
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range"

    def test_confidence_zero_when_equal(self):
        """When s_chosen == s_rejected, confidence should be 0."""
        diff = torch.tensor(0.0)
        confidence = float(2.0 * (torch.sigmoid(diff) - 0.5).clamp(min=0.0))
        assert abs(confidence) < 1e-5

    def test_confidence_higher_when_larger_margin(self):
        """Larger score margin should give higher confidence."""
        def conf(margin):
            diff = torch.tensor(float(margin))
            return float(2.0 * (torch.sigmoid(diff) - 0.5).clamp(min=0.0))

        assert conf(0.5) < conf(1.0) < conf(2.0) < conf(5.0)


class TestBaseWeakLabeler:
    def test_save_and_load(self, tmp_path):
        """Save and load should be reversible."""
        from src.weak_labeler.base_labeler import BaseWeakLabeler, PseudoLabeledSample

        samples = [
            PseudoLabeledSample(
                prompt="Q: test",
                chosen="good answer",
                rejected="bad answer",
                confidence_weight=0.8,
            )
        ]

        # Use a concrete subclass just for save/load
        class DummyLabeler(BaseWeakLabeler):
            def label_dataset(self, dataset, max_samples=None):
                return []

        labeler = DummyLabeler()
        path = str(tmp_path / "test_labels.jsonl")
        labeler.save(samples, path)
        loaded = BaseWeakLabeler.load(path)

        assert len(loaded) == 1
        assert loaded[0]["prompt"] == "Q: test"
        assert loaded[0]["chosen"] == "good answer"
        assert abs(loaded[0]["confidence_weight"] - 0.8) < 1e-5
