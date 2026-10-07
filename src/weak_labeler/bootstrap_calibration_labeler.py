"""
Bootstrap Calibration Labeler — MWDPO_bootstrap_calibration (1a + 2a).

Algorithm:
    Given a trained MultiHeadRewardModel and unlabeled data D_u:

    [Step 1 — Temperature calibration (2a), run ONCE on D_l val split]
        For each head k:
            T_k* = argmin_{T>0}  -Σ log σ(margin_k(x,y_w,y_l) / T)
        where margin_k = score_k(y_w) - score_k(y_l) on D_l validation samples.
        Optimization: LBFGS (few iterations, scalar T per head).
        If use_calibration=False: T_k = 1.0 for all k (ablation).

    [Step 2 — Labeling D_u]
        For each (x, y_1, y_2) ∈ D_u:
            1. scores_y1, scores_y2: (K,) via one backbone + K-head forward pass
            2. margins_k = scores_y1_k - scores_y2_k  for k in 1..K
            3. S(y) = mean_k[score_k(y)]              ensemble score (score space, same as MultiWeak)
            4. y_w = y_1 if S(y_1) >= S(y_2)         ≡ mean_k[margin_k] >= 0
            5. C = σ(S(y_w) - S(y_l))               ∈ (0.5, 1]  (same formula as MultiWeak)
            6. p_k = σ(margin_k / T_k)               per-head calibrated prob (for agreement only)
            7. unanimous = all heads agree with y_w direction  (margin_k ≥ 0 ⇔ p_k ≥ 0.5, T_k-invariant)
            8. in_d_high = unanimous [AND C >= threshold if mode='unanimous_with_threshold']

    Why score-space ensemble:
        mean_k[σ(m_k)] can be dominated by heads with large |margin| (sigmoid saturation).
        mean_k[m_k] > 0  is equivalent to "total evidence across K heads favors y_1",
        matching the MultiWeakLabeler criterion exactly (eq. 1 from proposal).
        Temperature calibration (T_k) does NOT affect ranking direction (only magnitude),
        so individual agreement is unchanged by this switch.

    Output: D_h (high-agreement) + D_l (disagreement) in PseudoLabeledSample format,
    compatible with the existing SFT/DPO training pipeline.

Degenerate case:
    K=1, use_bootstrap=False, use_calibration=False → reproduces MWDPO with k=1
    (binary vote with raw confidence).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from ..models.multi_head_reward_model import MultiHeadRewardModel
from .base_labeler import BaseWeakLabeler, PseudoLabeledSample

logger = logging.getLogger(__name__)


class BootstrapCalibrationLabeler(BaseWeakLabeler):
    """
    Labeler that uses a shared-backbone multi-head reward model with
    per-head temperature calibration to produce calibrated confidence weights.

    Args:
        model:                 trained MultiHeadRewardModel (K heads)
        tokenizer:             tokenizer (shared backbone)
        max_length:            max token length for scoring
        device:                compute device
        batch_size:            inference batch size
        use_calibration:       if True (default), fit T_k on D_l val split (2a).
                               if False, T_k = 1.0 for all heads (ablation).
        agreement_mode:        "unanimous" or "unanimous_with_threshold"
        confidence_threshold:  threshold for unanimous_with_threshold mode
    """

    def __init__(
        self,
        model: MultiHeadRewardModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        device: str = "cuda",
        batch_size: int = 16,
        use_calibration: bool = True,
        agreement_mode: str = "unanimous",
        confidence_threshold: float = 0.8,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_calibration = use_calibration
        self.agreement_mode = agreement_mode
        self.confidence_threshold = confidence_threshold
        self.num_heads = model.num_heads

        # Temperatures (one per head): initialized to 1.0, updated by calibrate()
        self.temperatures: List[float] = [1.0] * self.num_heads

        for param in self.model.parameters():
            param.requires_grad = False

        cal_str = "WITH calibration" if use_calibration else "NO calibration (ablation)"
        logger.info(
            f"[BootstrapCalibrationLabeler] K={self.num_heads} heads, {cal_str}, "
            f"agreement_mode='{agreement_mode}', threshold={confidence_threshold}"
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def calibrate(self, cal_dataset: Dataset, max_cal_samples: Optional[int] = None) -> None:
        """
        Fit per-head temperatures on a labeled calibration set (D_l val split).

        For each head k, minimizes negative log-likelihood:
            L(T_k) = -Σ log σ(margin_k(x, y_w, y_l) / T_k)
        where y_w, y_l are the ground-truth preferred/rejected responses in cal_dataset.

        This is a 1-D optimization per head; we use LBFGS (fast convergence).

        Args:
            cal_dataset:      labeled dataset with 'prompt', 'chosen', 'rejected'.
                              Uses ground-truth labels as y_w/y_l (no flipping).
            max_cal_samples:  limit samples for speed (default: use all)
        """
        if not self.use_calibration:
            logger.info("[Calibration] Skipped (use_calibration=False). T_k = 1.0 for all heads.")
            return

        logger.info(f"[Calibration] Fitting temperatures on {len(cal_dataset)} cal samples...")

        # Collect all margins (batch, K) on cal set
        all_margins = self._collect_margins_on_labeled(cal_dataset, max_cal_samples)
        # all_margins: (N_cal, K)   margin_k = score_k(y_w) - score_k(y_l) > 0 ideally

        if all_margins.size(0) == 0:
            logger.warning("[Calibration] Empty calibration set — using T_k = 1.0")
            return

        fitted_temps = []
        for k in range(self.num_heads):
            margins_k = all_margins[:, k].float()  # (N_cal,) on CPU

            # T_k as a learnable scalar > 0 (log-space to enforce positivity)
            log_T = torch.zeros(1, requires_grad=True)
            opt = torch.optim.LBFGS([log_T], max_iter=100, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                T = log_T.exp()
                nll = -F.logsigmoid(margins_k / T).mean()
                nll.backward()
                return nll

            opt.step(closure)

            T_fitted = float(log_T.exp().item())
            # Sanity clamp: T ∈ [0.1, 10.0] to prevent degenerate solutions
            T_fitted = max(0.1, min(10.0, T_fitted))
            fitted_temps.append(T_fitted)

            before_nll = float(-F.logsigmoid(margins_k).mean())
            after_nll  = float(-F.logsigmoid(margins_k / T_fitted).mean())
            logger.info(
                f"  Head {k}: T_k = {T_fitted:.4f} | "
                f"NLL before={before_nll:.4f}, after={after_nll:.4f}"
            )

        self.temperatures = fitted_temps
        logger.info(f"[Calibration] Done. Temperatures: {self.temperatures}")

    @torch.no_grad()
    def label_dataset(
        self,
        dataset: Dataset,
        max_samples: Optional[int] = None,
    ) -> List[PseudoLabeledSample]:
        """
        Label D_u and return ALL samples (D_h ∪ D_l combined).
        Use label_and_filter_dataset() to get the split.
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

        logger.info(
            f"[BootstrapCalibrationLabeler] Scoring {len(samples)} samples, "
            f"K={self.num_heads} heads, T={[f'{t:.3f}' for t in self.temperatures]}..."
        )

        d_high: List[PseudoLabeledSample] = []
        d_low:  List[PseudoLabeledSample] = []

        for i in tqdm(range(0, len(samples), self.batch_size), desc="BC labeling"):
            batch = samples[i : i + self.batch_size]
            batch_high, batch_low = self._process_batch(batch)
            d_high.extend(batch_high)
            d_low.extend(batch_low)

        # Summary statistics
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
            f"[BootstrapCalibrationLabeler] Done.\n"
            f"  D_h (high-agreement): {len(d_high)}/{total} ({100*ratio:.1f}%)\n"
            f"  D_l (disagreement):   {len(d_low)}/{total} ({100*(1-ratio):.1f}%)\n"
            f"  Avg confidence D_h:   {avg_conf_high:.4f}\n"
            f"  Avg confidence D_l:   {avg_conf_low:.4f}"
        )

        if len(d_high) == 0:
            logger.warning(
                "[BootstrapCalibrationLabeler] D_h is EMPTY! "
                "Consider relaxing agreement_mode or confidence_threshold."
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
        Score a batch with all K heads, compute ensemble in score space (like MultiWeakLabeler),
        check per-head agreement (with calibration), and partition into D_h / D_l.

        Ensemble ranking: y_w = y_1 if mean_k[score_k(y1)] >= mean_k[score_k(y2)]
                               ≡ mean_k[margin_k] ≥ 0   (score-space, same as MultiWeak eq. 1)
        Confidence:       C = σ(S(y_w) - S(y_l)) ∈ (0.5, 1]   (same as MultiWeak)
        Agreement:        each head k agrees if margin_k >= 0  ⇔ p_k = σ(m_k/T_k) >= 0.5
                          (T_k only scales magnitude, does NOT change head direction)
        """
        prompts   = [s["prompt"]   for s in batch]
        chosens   = [s["chosen"]   for s in batch]   # y_1 (original label direction)
        rejecteds = [s["rejected"] for s in batch]   # y_2

        # Score with all K heads: (N, K)
        scores_y1 = self._score_batch(prompts, chosens)    # (N, K)
        scores_y2 = self._score_batch(prompts, rejecteds)  # (N, K)

        # Per-head margins: (N, K) — positive if head prefers y_1
        margins = scores_y1 - scores_y2

        # ── Ensemble ranking in score space (eq. 1 from MultiWeak proposal) ───
        # S(y) = mean_k[score_k(y)] — average across K heads
        S_y1 = scores_y1.mean(dim=-1)   # (N,)
        S_y2 = scores_y2.mean(dim=-1)   # (N,)
        # S_y1 - S_y2 = mean_k[margin_k] — total evidence
        ensemble_prefers_y1 = (S_y1 >= S_y2)  # (N,) bool

        # ── Per-head individual agreement (still uses calibrated probs) ──────
        # Note: σ(m_k / T_k) > 0.5  ⇔  m_k > 0  (T_k > 0 always)
        # Temperature does NOT affect direction — only magnitude for confidence calibration.
        temps = torch.tensor(self.temperatures, dtype=torch.float32)  # (K,)
        calibrated_probs = torch.sigmoid(margins / temps.unsqueeze(0))  # (N, K)
        individual_prefers_y1 = calibrated_probs > 0.5  # ≡ margins > 0   (K-invariant)

        d_high = []
        d_low  = []

        for n in range(len(batch)):
            sample = batch[n]
            s1 = float(S_y1[n])
            s2 = float(S_y2[n])

            # Ensemble preference
            pref_y1 = bool(ensemble_prefers_y1[n])
            if pref_y1:
                chosen, rejected = sample["chosen"], sample["rejected"]
                S_w, S_l = s1, s2
            else:
                chosen, rejected = sample["rejected"], sample["chosen"]
                S_w, S_l = s2, s1

            # Confidence: σ(S(y_w) - S(y_l)) ∈ (0.5, 1]  — same formula as MultiWeakLabeler
            # diff = S_w - S_l ≥ 0 always (y_w = argmax S)
            confidence = float(torch.sigmoid(torch.tensor(S_w - S_l, dtype=torch.float32)))

            # Per-head agreement with ensemble direction
            if pref_y1:
                individual_agrees = individual_prefers_y1[n]   # (K,) bool
            else:
                individual_agrees = ~individual_prefers_y1[n]  # flip: agree if margin_k < 0

            agreed = self._check_agreement(
                individual_agrees=individual_agrees.tolist(),
                confidence=confidence,
            )

            # Ensemble score margin (raw, for analysis)
            ensemble_margin = S_w - S_l

            pseudo = PseudoLabeledSample(
                prompt=sample["prompt"],
                chosen=chosen,
                rejected=rejected,
                confidence_weight=confidence,
                ensemble_agreement=agreed,
                individual_agreements=individual_agrees.tolist(),
                in_d_high=agreed,
                # Store both new and old-style fields for compatibility
                p_ensemble=float(confidence),          # σ(S_w - S_l), analogous to p_ens
                temperatures=list(self.temperatures),  # kept for calibration bookkeeping
            )

            if agreed:
                d_high.append(pseudo)
            else:
                d_low.append(pseudo)

        return d_high, d_low

    @torch.no_grad()
    def _score_batch(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """
        Score (prompt, response) pairs with all K heads in one forward pass.

        Returns:
            (N, K) float tensor of raw scores (on CPU)
        """
        texts = [p + "\n" + r for p, r in zip(prompts, responses)]
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        scores = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )  # (N, K)

        return scores.float().cpu()

    @torch.no_grad()
    def _collect_margins_on_labeled(
        self,
        dataset: Dataset,
        max_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute per-head margins on the calibration set (labeled D_l val split).
        Uses the ground-truth chosen/rejected direction — no preference flipping.

        Returns:
            (N_cal, K) margins tensor (CPU float32)
        """
        samples = list(dataset)
        if max_samples is not None:
            samples = samples[:max_samples]

        all_margins = []
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i : i + self.batch_size]
            prompts   = [s["prompt"]   for s in batch]
            chosens   = [s["chosen"]   for s in batch]
            rejecteds = [s["rejected"] for s in batch]

            scores_chosen   = self._score_batch(prompts, chosens)    # (N, K)
            scores_rejected = self._score_batch(prompts, rejecteds)  # (N, K)

            margins = scores_chosen - scores_rejected  # (N, K)
            all_margins.append(margins)

        return torch.cat(all_margins, dim=0) if all_margins else torch.zeros(0, self.num_heads)

    def _check_agreement(
        self,
        individual_agrees: List[bool],
        confidence: float,
    ) -> bool:
        """
        Determine if a sample belongs to D_h based on configured agreement criterion.
        Matches the logic in MultiWeakLabeler for consistency.
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
