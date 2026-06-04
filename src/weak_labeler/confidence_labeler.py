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
        results = []
        for sample in batch:
            prompt = sample["prompt"]
            y1 = sample["chosen"]
            y2 = sample["rejected"]

            s1 = self._score(prompt, y1)
            s2 = self._score(prompt, y2)

            if s1 >= s2:
                chosen, rejected = y1, y2
                s_chosen, s_rejected = s1, s2
            else:
                chosen, rejected = y2, y1
                s_chosen, s_rejected = s2, s1

            # Confidence: C = 2·(σ(s+ - s-) - 0.5)  [eq. 8]
            diff = torch.tensor(s_chosen - s_rejected)
            confidence = float(2.0 * (torch.sigmoid(diff) - 0.5).clamp(min=0.0))

            results.append(PseudoLabeledSample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                confidence_weight=confidence,
            ))
        return results

    def _score(self, prompt: str, response: str) -> float:
        """Compute πw(x, y) — scalar reward for (prompt, response) pair."""
        full_text = prompt + " " + response
        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding=False,
        ).to(self.device)

        score = self.reward_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        return score.item()
