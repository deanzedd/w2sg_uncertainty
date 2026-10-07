"""
Multi-Weak Ensemble Labeler — Phase 1 of the MWDPO proposal.

Algorithm:
  Given k trained scalar reward models {π_w1, ..., π_wk} and unlabeled data D_u,
  for each triplet (x, y_1, y_2) ∈ D_u:

  1. Compute per-model scores:
       s_i(x, y) = π_wi(x, y)   for i in 1..k

  2. Compute ensemble average score:
       S(x, y) = 1/k Σ_i s_i(x, y)

  3. Determine preference ranking via ensemble:
       y_w > y_l  if  S(x, y_1) > S(x, y_2)   [eq. 1 from proposal]

  4. Compute ensemble confidence (F=1, Phase 1):
       C_weak(x, y_w, y_l) = σ(S(x, y_w) − S(x, y_l))   ∈ (0.5, 1]

  5. Check agreement:
       - "unanimous": all k models individually rank the same way
       - "unanimous_with_threshold": unanimous AND mean confidence > threshold

  6. Split D_u:
       D_h = {samples where agreement == True}
       D_l = {samples where agreement == False}

Note:
  - F = 1 (Phase 1 simplification). Future versions can extend F.
  - Confidence is stored in PseudoLabeledSample["confidence_weight"] for use
    in Phase 1 DPO (standard DPO ignores it; Phase 2 can use it).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from ..models.reward_model import ScalarRewardModel
from .base_labeler import BaseWeakLabeler, PseudoLabeledSample

logger = logging.getLogger(__name__)


class MultiWeakLabeler(BaseWeakLabeler):
    """
    Multi-weak ensemble labeler that scores D_u with k reward models,
    determines preference by majority ensemble score, and filters into
    D_h (high-agreement) and D_l (disagreement) subsets.

    Args:
        reward_models:       list of k trained ScalarRewardModel instances
        tokenizer:           tokenizer (shared by all models — same backbone)
        max_length:          max token length for scoring
        device:              compute device
        batch_size:          inference batch size
        agreement_mode:      "unanimous" or "unanimous_with_threshold"
        confidence_threshold: mean confidence threshold (only for "unanimous_with_threshold")
    """

    def __init__(
        self,
        reward_models: List[ScalarRewardModel],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        device: str = "cuda",
        batch_size: int = 16,
        agreement_mode: str = "unanimous",
        confidence_threshold: float = 0.8,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        if len(reward_models) == 0:
            raise ValueError("MultiWeakLabeler requires at least 1 reward model.")

        self.reward_models = [m.to(device).eval() for m in reward_models]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.agreement_mode = agreement_mode
        self.confidence_threshold = confidence_threshold
        self.k = len(reward_models)

        # Freeze all models
        for rm in self.reward_models:
            for param in rm.parameters():
                param.requires_grad = False

        logger.info(
            f"[MultiWeakLabeler] Initialized with k={self.k} models, "
            f"agreement_mode='{self.agreement_mode}', "
            f"confidence_threshold={self.confidence_threshold}"
        )

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
        Label D_u and return ALL pseudo-labeled samples (D_h ∪ D_l combined).
        Each sample includes:
          - "in_d_high": bool — True if unanimous agreement → D_h
          - "confidence_weight": float — C_weak = σ(S(y_w) - S(y_l))
          - "ensemble_agreement": bool — alias for in_d_high
          - "individual_agreements": list[bool] — per-model ranking

        To get D_h / D_l splits, use label_and_filter_dataset() instead.
        """
        d_high, d_low = self.label_and_filter_dataset(dataset, max_samples=max_samples)
        return d_high + d_low

    @torch.no_grad()
    def label_and_filter_dataset(
        self,
        dataset: Dataset,
        max_samples: Optional[int] = None,
    ) -> Tuple[List[PseudoLabeledSample], List[PseudoLabeledSample]]:
        """
        Label D_u and split into D_h (high-agreement) and D_l (low-agreement).

        Returns:
            (D_h, D_l) — two lists of PseudoLabeledSample
        """
        samples = list(dataset)
        if max_samples is not None:
            samples = samples[:max_samples]

        logger.info(f"[MultiWeakLabeler] Scoring {len(samples)} samples with k={self.k} models...")

        d_high: List[PseudoLabeledSample] = []
        d_low: List[PseudoLabeledSample] = []

        for i in tqdm(range(0, len(samples), self.batch_size), desc="Multi-weak labeling"):
            batch = samples[i : i + self.batch_size]
            batch_high, batch_low = self._process_batch(batch)
            d_high.extend(batch_high)
            d_low.extend(batch_low)

        # Statistics
        total = len(d_high) + len(d_low)
        ratio = len(d_high) / max(1, total)
        avg_conf_high = (
            sum(s["confidence_weight"] for s in d_high) / max(1, len(d_high))
            if d_high else 0.0
        )
        avg_conf_low = (
            sum(s["confidence_weight"] for s in d_low) / max(1, len(d_low))
            if d_low else 0.0
        )

        logger.info(
            f"[MultiWeakLabeler] Done.\n"
            f"  D_h (high-agreement): {len(d_high)}/{total} ({100*ratio:.1f}%)\n"
            f"  D_l (disagreement):   {len(d_low)}/{total} ({100*(1-ratio):.1f}%)\n"
            f"  Avg confidence D_h:   {avg_conf_high:.4f}\n"
            f"  Avg confidence D_l:   {avg_conf_low:.4f}"
        )

        if len(d_high) == 0:
            logger.warning(
                "[MultiWeakLabeler] D_h is EMPTY! All samples went to D_l. "
                "Consider relaxing the agreement criterion or confidence threshold."
            )

        return d_high, d_low

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _process_batch(
        self,
        batch: list,
    ) -> Tuple[List[PseudoLabeledSample], List[PseudoLabeledSample]]:
        """
        Process a batch of samples:
          1. Score y_1 and y_2 with all k models → per_model_scores shape: (k, N)
          2. Compute ensemble score S = mean over k → shape: (N,) for each response
          3. Compute confidence C = σ(S(y_w) - S(y_l))
          4. Check agreement and split into D_h / D_l
        """
        prompts   = [s["prompt"]   for s in batch]
        chosens   = [s["chosen"]   for s in batch]   # y_1 (original labels)
        rejecteds = [s["rejected"] for s in batch]   # y_2

        N = len(batch)

        # Score y_1 and y_2 with each of the k models: shape (k, N)
        per_model_scores_y1 = torch.zeros(self.k, N)
        per_model_scores_y2 = torch.zeros(self.k, N)

        for idx, rm in enumerate(self.reward_models):
            scores_y1 = self._score_batch_with_model(rm, prompts, chosens)    # (N,)
            scores_y2 = self._score_batch_with_model(rm, prompts, rejecteds)  # (N,)
            per_model_scores_y1[idx] = torch.tensor(scores_y1)
            per_model_scores_y2[idx] = torch.tensor(scores_y2)

        # Ensemble average: S(x, y) = 1/k Σ π_wi(x, y)
        S_y1 = per_model_scores_y1.mean(dim=0)  # (N,)
        S_y2 = per_model_scores_y2.mean(dim=0)  # (N,)

        # Per-model individual ranking: True if model_i prefers y_1 over y_2
        # individual_prefers_y1[i, n] = True means model i prefers y_1 for sample n
        individual_prefers_y1 = per_model_scores_y1 > per_model_scores_y2  # (k, N) bool

        d_high = []
        d_low = []

        for n in range(N):
            sample = batch[n]
            s1 = S_y1[n].item()
            s2 = S_y2[n].item()

            # Ensemble preference: y_w is the one with higher ensemble score
            ensemble_prefers_y1 = (s1 >= s2)
            if ensemble_prefers_y1:
                chosen, rejected = sample["chosen"], sample["rejected"]
                S_chosen, S_rejected = s1, s2
            else:
                chosen, rejected = sample["rejected"], sample["chosen"]
                S_chosen, S_rejected = s2, s1

            # Confidence: C_weak = σ(S(y_w) - S(y_l))  [F=1]
            diff = S_chosen - S_rejected  # always ≥ 0 since y_w = argmax S
            confidence = float(torch.sigmoid(torch.tensor(diff)))

            # Individual agreement check: does each model agree with ensemble?
            # Each model's ranking wrt same ensemble preference direction
            if ensemble_prefers_y1:
                individual_agrees = individual_prefers_y1[:, n]  # (k,) bool
            else:
                individual_agrees = ~individual_prefers_y1[:, n]  # flipped

            # Agreement criterion
            agreed = self._check_agreement(
                individual_agrees=individual_agrees.tolist(),
                confidence=confidence,
            )

            pseudo = PseudoLabeledSample(
                prompt=sample["prompt"],
                chosen=chosen,
                rejected=rejected,
                confidence_weight=confidence,
                # Extra metadata for analysis
                ensemble_agreement=agreed,
                individual_agreements=individual_agrees.tolist(),
                in_d_high=agreed,
            )

            if agreed:
                d_high.append(pseudo)
            else:
                d_low.append(pseudo)

        return d_high, d_low

    def _check_agreement(
        self,
        individual_agrees: List[bool],
        confidence: float,
    ) -> bool:
        """
        Determine if a sample belongs to D_h based on the configured agreement criterion.

        Args:
            individual_agrees: list of k bools — does each model agree with ensemble ranking?
            confidence:        ensemble confidence C_weak ∈ (0.5, 1]

        Returns:
            True if sample should go into D_h
        """
        unanimous = all(individual_agrees)

        if self.agreement_mode == "unanimous":
            return unanimous

        elif self.agreement_mode == "unanimous_with_threshold":
            return unanimous and (confidence >= self.confidence_threshold)

        else:
            raise ValueError(
                f"Unknown agreement_mode: '{self.agreement_mode}'. "
                "Choose 'unanimous' or 'unanimous_with_threshold'."
            )

    def _score_batch_with_model(
        self,
        model: ScalarRewardModel,
        prompts: List[str],
        responses: List[str],
    ) -> List[float]:
        """
        Score (prompt, response) pairs with a single reward model.

        Args:
            model:     ScalarRewardModel (already on device)
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

        scores = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )  # (N,)
        return scores.tolist()
