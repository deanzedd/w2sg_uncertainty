"""
Standard DPO Loss (per-sample and batched).

DPO loss (Rafailov et al., 2023):
    L_DPO = -log σ(β · (log π(y+|x)/π_ref(y+|x) - log π(y-|x)/π_ref(y-|x)))

where:
    π      = current policy
    π_ref  = frozen reference policy
    β      = KL divergence coefficient
    y+     = chosen (preferred) response
    y-     = rejected response
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """
    Compute per-token log probabilities and sum/average over the sequence.

    Args:
        logits:           (batch, seq_len, vocab_size)
        labels:           (batch, seq_len)  — use -100 for tokens to ignore
        average_log_prob: if True, return mean instead of sum

    Returns:
        (batch,) — log probability (summed or averaged over non-ignored tokens)
    """
    # Shift: predict token t+1 from token t
    logits = logits[:, :-1, :]        # (B, L-1, V)
    labels = labels[:, 1:].clone()    # (B, L-1)

    # Mask padding / prompt tokens
    loss_mask = labels != -100

    # L1 fix: clamp(min=1) silently hides the case where a sample has no
    # response tokens at all (loss_mask all-False). Log a warning so the caller
    # is aware — this typically indicates a tokenization/data issue.
    zero_response_rows = (loss_mask.sum(dim=-1) == 0)
    if zero_response_rows.any():
        logger.warning(
            f"L1: {zero_response_rows.sum().item()} sample(s) in the batch have "
            "no response tokens (loss_mask all-False). Their log-prob contribution "
            "will be 0.0 due to clamp(min=1). Check tokenization / label masking."
        )

    # Replace -100 with 0 for gather (won't affect masked positions)
    labels[labels == -100] = 0

    # Gather log-softmax at label positions
    log_probs = F.log_softmax(logits, dim=-1)  # (B, L-1, V)
    per_token_logps = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)  # (B, L-1)

    per_token_logps = per_token_logps * loss_mask.float()  # explicit cast: bool * bf16 can be wrong under autocast

    if average_log_prob:
        return per_token_logps.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)
    return per_token_logps.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Standard DPO loss (batched, reduction='mean').

    Args:
        policy_chosen_logps:   (batch,) log π(y+|x)
        policy_rejected_logps: (batch,) log π(y-|x)
        ref_chosen_logps:      (batch,) log π_ref(y+|x)
        ref_rejected_logps:    (batch,) log π_ref(y-|x)
        beta:                  KL coefficient

    Returns:
        scalar loss
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()
    return loss


def dpo_loss_per_sample(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DPO loss without reduction — returns per-sample loss for CWPO weighting.

    Returns:
        (batch,) per-sample loss values
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_rewards - rejected_rewards
    return -F.logsigmoid(logits)  # (batch,)
