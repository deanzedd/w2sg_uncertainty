"""
CWPO Confidence Labeler — Scalar Reward Model + Confidence Margin.

Algorithm (from CWPO paper):
    For each (x, y1, y2) in D_u:
        s1 = πw(x, y1)    # scalar score from weak reward model
        s2 = πw(x, y2)
        y+ = y1 if s1 >= s2 else y2
        y- = the other
        C(x,y+,y-) = 2 · (σ(s_chosen − s_rejected) − 0.5)   # eq. 8

    D̂ = {(x, y+, y-, C)}   ← used for confidence-weighted DPO training
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from ..models.reward_model import ScalarRewardModel
from .base_labeler import BaseWeakLabeler, PseudoLabeledSample

logger = logging.getLogger(__name__)


class ConfidenceLabeler(BaseWeakLabeler):
    """
    CWPO weak preference labeler using a scalar reward model.

    The reward model is a pretrained LM backbone with a scalar output head,
    fine-tuned on D_l via Bradley-Terry loss.

    Args:
        reward_model:   trained ScalarRewardModel
        tokenizer:      tokenizer for the reward model
        max_length:     max token length
        device:         compute device
        batch_size:     inference batch size
    """

    def __init__(
        self,
        reward_model: ScalarRewardModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        device: str = "cuda",
        batch_size: int = 16,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        self.reward_model = reward_model.to(device).eval()
        self.tokenizer = tokenizer
        self.max_length = max_length

        for param in self.reward_model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def label_dataset(
        self,
        dataset: Dataset,
        max_samples: Optional[int] = None,
    ) -> List[PseudoLabeledSample]:
        """
        Label D_u with CWPO confidence scores and return pseudo-labeled D̂.
        """
        samples = list(dataset)
        if max_samples is not None:
            samples = samples[:max_samples]

        logger.info(f"[CWPO] Labeling {len(samples)} samples...")
        pseudo_labeled = []
        confidence_scores = []

        for i in tqdm(range(0, len(samples), self.batch_size), desc="CWPO labeling"):
            batch = samples[i : i + self.batch_size]
            batch_results = self._label_batch(batch)
            pseudo_labeled.extend(batch_results)
            confidence_scores.extend([r["confidence_weight"] for r in batch_results])

        # Statistics
        n_swapped = sum(
            1 for orig, pseudo in zip(samples, pseudo_labeled)
            if orig["chosen"] != pseudo["chosen"]
        )
        avg_conf = sum(confidence_scores) / max(1, len(confidence_scores))
        logger.info(
            f"[CWPO] Done. {n_swapped}/{len(samples)} labels flipped "
            f"({100*n_swapped/max(1,len(samples)):.1f}%). "
            f"Avg confidence: {avg_conf:.4f}"
        )
        return pseudo_labeled

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _label_batch(self, batch: list) -> List[PseudoLabeledSample]:
        """
        CL2 fix: score the entire batch with two vectorized forward passes
        (one for chosen, one for rejected) instead of 2 * len(batch) sequential
        single-sample forward passes.
        """
        prompts   = [s["prompt"]   for s in batch]
        chosens   = [s["chosen"]   for s in batch]
        rejecteds = [s["rejected"] for s in batch]

        scores_chosen   = self._score_batch(prompts, chosens)    # (N,)
        scores_rejected = self._score_batch(prompts, rejecteds)  # (N,)

        results = []
        for i, sample in enumerate(batch):
            s1, s2 = scores_chosen[i], scores_rejected[i]

            if s1 >= s2:
                chosen, rejected = sample["chosen"], sample["rejected"]
                s_chosen, s_rejected = s1, s2
            else:
                chosen, rejected = sample["rejected"], sample["chosen"]
                s_chosen, s_rejected = s2, s1

            # Confidence: C = 2·(σ(s+ - s-) - 0.5)  [eq. 8]
            diff = torch.tensor(s_chosen - s_rejected)
            confidence = float(2.0 * (torch.sigmoid(diff) - 0.5).clamp(min=0.0))

            results.append(PseudoLabeledSample(
                prompt=sample["prompt"],
                chosen=chosen,
                rejected=rejected,
                confidence_weight=confidence,
            ))
        return results

    def _score_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        """
        CL2 fix: batch-score (prompt, response) pairs with a single forward pass.

        Args:
            prompts:   list of N prompt strings
            responses: list of N response strings (parallel to prompts)

        Returns:
            list of N scalar reward scores (float)
        """
        texts = [p + "\n" + r for p, r in zip(prompts, responses)]
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        scores = self.reward_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )  # (N,)
        return scores.tolist()

    def _score(self, prompt: str, response: str) -> float:
        """
        Single-sample scoring (kept for API compatibility).
        For batch inference, use _score_batch().
        """
        return self._score_batch([prompt], [response])[0]

