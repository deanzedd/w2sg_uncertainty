"""
Evaluation Metrics.

Implements two primary metrics from the papers:

1. Gold Reward Accuracy (GRA):
   How often the aligned model's response gets a higher reward score
   from a pretrained reward model than the SFT model's response.
   GRA = P(r(aligned) > r(sft))

2. GPT-4 Win Rate:
   GPT-4 as a proxy for human evaluation. Given two responses,
   GPT-4 rates each on a scale of 1-10. Win rate = fraction of
   prompts where aligned model scores higher.

3. Preference Accuracy (sanity check):
   How often weak model labels agree with human labels.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
#  Preference Accuracy (weak label quality check)                            #
# ─────────────────────────────────────────────────────────────────────────── #

def preference_accuracy(
    pseudo_labels: List[Dict],
    human_labels: List[Dict],
) -> float:
    """
    Measure how often weak model labels agree with human labels.

    Args:
        pseudo_labels: list of {prompt, chosen, rejected} from weak labeler
        human_labels:  list of {prompt, chosen, rejected} from human

    Returns:
        accuracy in [0, 1]
    """
    if not pseudo_labels or not human_labels:
        return 0.0

    # Build lookup by prompt
    human_lookup: Dict[str, str] = {
        h["prompt"]: h["chosen"] for h in human_labels
    }

    correct = 0
    total = 0
    for pseudo in pseudo_labels:
        prompt = pseudo["prompt"]
        if prompt in human_lookup:
            if pseudo["chosen"] == human_lookup[prompt]:
                correct += 1
            total += 1

    return correct / max(1, total)


# ─────────────────────────────────────────────────────────────────────────── #
#  Gold Reward Accuracy (GRA)                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def gold_reward_accuracy(
    aligned_scores: List[float],
    sft_scores: List[float],
) -> Dict[str, float]:
    """
    Compute GRA: how often aligned model scores higher than SFT model.

    GRA = P(r(y_aligned) > r(y_sft))

    Args:
        aligned_scores: reward scores for aligned model responses
        sft_scores:     reward scores for SFT model responses

    Returns:
        dict with 'gra', 'avg_aligned_reward', 'avg_sft_reward', 'reward_gap'
    """
    assert len(aligned_scores) == len(sft_scores), \
        "aligned_scores and sft_scores must have equal length"

    wins = sum(a > s for a, s in zip(aligned_scores, sft_scores))
    ties = sum(a == s for a, s in zip(aligned_scores, sft_scores))
    n = len(aligned_scores)

    return {
        "gra": wins / max(1, n),
        "gra_with_ties": (wins + 0.5 * ties) / max(1, n),
        "avg_aligned_reward": sum(aligned_scores) / max(1, n),
        "avg_sft_reward": sum(sft_scores) / max(1, n),
        "reward_gap": (sum(aligned_scores) - sum(sft_scores)) / max(1, n),
    }


# ─────────────────────────────────────────────────────────────────────────── #
#  GPT-4 Win Rate Aggregation                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_gpt4_win_rate(
    gpt4_results: List[Dict],
) -> Dict[str, float]:
    """
    Aggregate GPT-4 win rate from evaluation results.

    Args:
        gpt4_results: list of {
            "prompt": str,
            "score_a": float,   # score for response A (aligned)
            "score_b": float,   # score for response B (SFT/baseline)
        }

    Returns:
        dict with 'win_rate', 'tie_rate', 'loss_rate', 'avg_score_a', 'avg_score_b'
    """
    wins = ties = losses = 0
    total_a = total_b = 0.0
    n = len(gpt4_results)

    for r in gpt4_results:
        sa = r["score_a"]
        sb = r["score_b"]
        total_a += sa
        total_b += sb
        if sa > sb:
            wins += 1
        elif sa == sb:
            ties += 1
        else:
            losses += 1

    return {
        "win_rate": wins / max(1, n),
        "tie_rate": ties / max(1, n),
        "loss_rate": losses / max(1, n),
        "avg_score_a": total_a / max(1, n),
        "avg_score_b": total_b / max(1, n),
    }
