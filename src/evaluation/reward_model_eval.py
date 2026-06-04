"""
Gold Reward Accuracy (GRA) Evaluator.

Uses a pretrained reward model from HuggingFace to score model responses.
GRA = P(r(aligned_response) > r(sft_response))

Recommended reward model: OpenAssistant/reward-model-deberta-v3-large-v2
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .metrics import gold_reward_accuracy

logger = logging.getLogger(__name__)


class RewardModelEvaluator:
    """
    Evaluates models using a pretrained scalar reward model.

    Args:
        reward_model_name: HF model ID for the reward model
        device:            compute device
        batch_size:        inference batch size
        max_length:        max token length
        cache_dir:         HF cache directory
    """

    def __init__(
        self,
        reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(f"Loading reward model: {reward_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            reward_model_name, cache_dir=cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def score_responses(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[float]:
        """
        Score a list of (prompt, response) pairs.

        Args:
            prompts:   list of prompts
            responses: list of responses

        Returns:
            list of scalar reward scores
        """
        all_scores = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_responses = responses[i : i + self.batch_size]

            # Format: prompt + response
            texts = [p + " " + r for p, r in zip(batch_prompts, batch_responses)]

            enc = self.tokenizer(
                texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**enc)
            scores = outputs.logits.squeeze(-1).cpu().tolist()

            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        return all_scores

    def compute_gra(
        self,
        prompts: List[str],
        aligned_responses: List[str],
        sft_responses: List[str],
    ) -> Dict[str, float]:
        """
        Compute Gold Reward Accuracy.

        Args:
            prompts:            list of eval prompts
            aligned_responses:  responses from aligned model (WDPO/CWPO)
            sft_responses:      responses from SFT baseline

        Returns:
            dict with GRA metrics
        """
        logger.info(f"Scoring {len(prompts)} aligned responses...")
        aligned_scores = self.score_responses(prompts, aligned_responses)

        logger.info(f"Scoring {len(prompts)} SFT responses...")
        sft_scores = self.score_responses(prompts, sft_responses)

        metrics = gold_reward_accuracy(aligned_scores, sft_scores)
        logger.info(
            f"GRA: {metrics['gra']:.4f} | "
            f"Aligned avg: {metrics['avg_aligned_reward']:.4f} | "
            f"SFT avg: {metrics['avg_sft_reward']:.4f}"
        )
        return metrics
