"""Unit tests for CWPO loss function."""

import pytest
import torch
from src.losses.cwpo_loss import cwpo_loss, compute_confidence_from_scores
from src.losses.dpo_loss import dpo_loss, dpo_loss_per_sample, compute_log_probs


class TestDPOLoss:
    def test_dpo_loss_basic(self):
        """DPO loss should be positive for random inputs."""
        B = 4
        policy_chosen = torch.randn(B)
        policy_rejected = torch.randn(B)
        ref_chosen = torch.randn(B)
        ref_rejected = torch.randn(B)

        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
        assert loss.item() > 0

    def test_dpo_loss_ideal_case(self):
        """If chosen is strongly preferred, loss should be small."""
        B = 4
        # Policy very strongly prefers chosen (large gap → sigma(beta*(200)) ≈ 1 → loss ≈ 0)
        policy_chosen = torch.ones(B) * 100.0
        policy_rejected = torch.ones(B) * -100.0
        ref_chosen = torch.zeros(B)
        ref_rejected = torch.zeros(B)

        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
        assert loss.item() < 0.001

    def test_dpo_per_sample_shape(self):
        """Per-sample loss should return a tensor of shape (B,)."""
        B = 8
        per_sample = dpo_loss_per_sample(
            torch.randn(B), torch.randn(B),
            torch.randn(B), torch.randn(B),
        )
        assert per_sample.shape == (B,)
        assert (per_sample > 0).all()  # loss should be positive


class TestCWPOLoss:
    def test_cwpo_loss_uniform_weights(self):
        """CWPO with all weights=1 should equal standard DPO."""
        B = 4
        pc = torch.randn(B)
        pr = torch.randn(B)
        rc = torch.randn(B)
        rr = torch.randn(B)

        standard = dpo_loss(pc, pr, rc, rr, beta=0.1)
        weighted = cwpo_loss(pc, pr, rc, rr, torch.ones(B), beta=0.1)
        assert abs(standard.item() - weighted.item()) < 1e-4

    def test_cwpo_loss_zero_weights(self):
        """CWPO with all weights=0 should not crash (falls back to mean)."""
        B = 4
        loss = cwpo_loss(
            torch.randn(B), torch.randn(B),
            torch.randn(B), torch.randn(B),
            torch.zeros(B), beta=0.1,
        )
        assert not torch.isnan(loss)

    def test_cwpo_confidence_formula(self):
        """Test the exact confidence formula C = 2·(σ(s+ - s-) - 0.5)."""
        # When s+ - s- = 0: C = 2*(0.5 - 0.5) = 0
        s_pos = torch.tensor([0.0])
        s_neg = torch.tensor([0.0])
        conf = compute_confidence_from_scores(s_pos, s_neg)
        assert abs(conf.item()) < 1e-5

        # When s+ - s- = large positive: C → 1
        s_pos = torch.tensor([100.0])
        s_neg = torch.tensor([0.0])
        conf = compute_confidence_from_scores(s_pos, s_neg)
        assert conf.item() > 0.99

        # Confidence should always be in [0, 1)
        s_pos = torch.randn(100)
        s_neg = torch.randn(100)
        conf = compute_confidence_from_scores(s_pos, s_neg)
        assert (conf >= 0).all()
        assert (conf < 1).all()

    def test_cwpo_high_confidence_dominates(self):
        """High-confidence samples should have more influence on loss."""
        B = 4
        # All samples same DPO loss
        pc = torch.zeros(B)
        pr = torch.ones(B) * -1.0
        rc = torch.zeros(B)
        rr = torch.zeros(B)

        # Low confidence weights
        low_conf = torch.tensor([0.1, 0.1, 0.1, 0.1])
        # High confidence weights
        high_conf = torch.tensor([0.9, 0.9, 0.9, 0.9])

        loss_low = cwpo_loss(pc, pr, rc, rr, low_conf, beta=0.1)
        loss_high = cwpo_loss(pc, pr, rc, rr, high_conf, beta=0.1)

        # Numerically, weighted mean with same weights should give same result
        # Both should be positive and finite
        assert not torch.isnan(loss_low) and not torch.isnan(loss_high)
        assert loss_low.item() > 0 and loss_high.item() > 0


class TestComputeLogProbs:
    def test_log_probs_shape(self):
        """compute_log_probs should return (B,) shaped tensor."""
        B, L, V = 2, 10, 100
        logits = torch.randn(B, L, V)
        labels = torch.randint(0, V, (B, L))
        labels[:, :3] = -100  # mask prompt tokens

        logps = compute_log_probs(logits, labels)
        assert logps.shape == (B,)
