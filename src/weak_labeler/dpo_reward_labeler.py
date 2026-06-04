"""
WDPO Weak Labeler — DPO Implicit Reward.

Algorithm (from WDPO paper):
    For each (x, y1, y2) in D_u:
        r_w(x, y1) = β · (log π_w(y1|x) − log π_ref(y1|x))
        r_w(x, y2) = β · (log π_w(y2|x) − log π_ref(y2|x))
        y+ = y1 if r_w(x,y1) >= r_w(x,y2) else y2
        y- = the other

    D̂ = {(x, y+, y-)}   ← only pseudo-labeled data used for strong model training

confidence_weight is set to 1.0 for all samples (no weighting in WDPO).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base_labeler import BaseWeakLabeler, PseudoLabeledSample

logger = logging.getLogger(__name__)


class DPORewardLabeler(BaseWeakLabeler):
    """
    WDPO weak preference labeler using DPO implicit reward.

    Args:
        weak_model:     fine-tuned weak LM (π_w)
        ref_model:      frozen reference LM (π_ref), typically SFT checkpoint
        tokenizer:      tokenizer for weak model
        beta:           DPO KL coefficient
        max_length:     max token length
        device:         compute device
        batch_size:     inference batch size
    """

    def __init__(
        self,
        weak_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        beta: float = 0.1,
        max_length: int = 512,
        device: str = "cuda",
        batch_size: int = 8,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        self.weak_model = weak_model.to(device).eval()
        self.ref_model = ref_model.to(device).eval()
        self.tokenizer = tokenizer
        self.beta = beta
        self.max_length = max_length

        for param in self.ref_model.parameters():
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
        Label D_u with WDPO implicit reward and return pseudo-labeled D̂.
        """
        samples = list(dataset)
        if max_samples is not None:
            samples = samples[:max_samples]

        logger.info(f"[WDPO] Labeling {len(samples)} samples...")
        pseudo_labeled = []

        for i in tqdm(range(0, len(samples), self.batch_size), desc="WDPO labeling"):
            batch = samples[i : i + self.batch_size]
            batch_results = self._label_batch(batch)
            pseudo_labeled.extend(batch_results)

        # Statistics
        n_swapped = sum(
            1 for orig, pseudo in zip(samples, pseudo_labeled)
            if orig["chosen"] != pseudo["chosen"]
        )
        logger.info(
            f"[WDPO] Done. {n_swapped}/{len(samples)} labels flipped by weak model "
            f"({100*n_swapped/max(1,len(samples)):.1f}%)"
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

            r1 = self._implicit_reward(prompt, y1)
            r2 = self._implicit_reward(prompt, y2)

            if r1 >= r2:
                chosen, rejected = y1, y2
            else:
                chosen, rejected = y2, y1

            results.append(PseudoLabeledSample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                confidence_weight=1.0,  # WDPO: uniform weights
            ))
        return results

    def _implicit_reward(self, prompt: str, response: str) -> float:
        """
        Compute DPO implicit reward:
            r_w(x, y) = β · (log π_w(y|x) − log π_ref(y|x))
        """
        full_text = prompt + " " + response
        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding=False,
        ).to(self.device)

        # Log prob under weak model
        weak_logp = self._sequence_log_prob(self.weak_model, enc)
        ref_logp = self._sequence_log_prob(self.ref_model, enc)

        return self.beta * (weak_logp - ref_logp).item()

    @staticmethod
    def _sequence_log_prob(
        model: PreTrainedModel,
        enc: dict,
    ) -> torch.Tensor:
        """
        Compute sum of token log-probabilities for a sequence.
        Returns a scalar tensor.
        """
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs.logits  # (1, L, V)
        # Shift: predict token t+1
        logits = logits[:, :-1, :]       # (1, L-1, V)
        labels = input_ids[:, 1:]        # (1, L-1)

        log_probs = F.log_softmax(logits, dim=-1)
        token_logps = torch.gather(
            log_probs, dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)  # (1, L-1)

        # Mask prompt tokens — only sum over response tokens
        # (simplified: sum all tokens; can be refined to mask prompt)
        return token_logps.sum()
