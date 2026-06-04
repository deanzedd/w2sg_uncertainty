"""
Confidence-Weighted Preference Optimization (CWPO) Loss.

From the CWPO paper (eq. 7 & 8):

    L_CW-PO = E_{(x,y+,y-) ∈ D̂} [ C(x,y+,y-) · ℓ(πs; x, y+, y-) ]

where:
    ℓ(πs; x, y+, y-)  = standard DPO loss per sample
    C(x, y+, y-)       = 2 · (σ(πw(x,y+) − πw(x,y-)) − 0.5)

The confidence C is computed by the weak labeler (ScalarRewardModel)
and stored alongside each pseudo-labeled sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .dpo_loss import dpo_loss_per_sample


def cwpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    confidence_weights: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    CWPO loss: confidence-weighted DPO.

    Args:
        policy_chosen_logps:   (batch,) log π_s(y+|x)
        policy_rejected_logps: (batch,) log π_s(y-|x)
        ref_chosen_logps:      (batch,) log π_ref(y+|x)
        ref_rejected_logps:    (batch,) log π_ref(y-|x)
        confidence_weights:    (batch,) C(x,y+,y-) in [0, 1)
        beta:                  KL coefficient

    Returns:
        scalar weighted loss
    """
    # Per-sample DPO loss  (batch,)
    per_sample_loss = dpo_loss_per_sample(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
    )

    # Confidence-weighted average
    # If all weights are zero (degenerate), fall back to unweighted
    weight_sum = confidence_weights.sum()
    if weight_sum < 1e-8:
        return per_sample_loss.mean()

    weighted_loss = (confidence_weights * per_sample_loss).sum() / weight_sum
    return weighted_loss


def compute_confidence_from_scores(
    score_chosen: torch.Tensor,
    score_rejected: torch.Tensor,
) -> torch.Tensor:
    """
    Compute CWPO confidence weight from weak model scalar scores.

    Formula (eq. 8):
        C(x, y+, y-) = 2 · (σ(πw(x,y+) − πw(x,y-)) − 0.5)

    Args:
        score_chosen:   (batch,) πw(x, y+)
        score_rejected: (batch,) πw(x, y-)

    Returns:
        confidence: (batch,) in [0, 1)
    """
    diff = score_chosen - score_rejected
    return (2.0 * (torch.sigmoid(diff) - 0.5)).clamp(min=0.0)
