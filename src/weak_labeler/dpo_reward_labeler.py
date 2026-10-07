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

        Prompt is fed to the model for context (causal LM requires full sequence),
        but only response token log-probs are summed — matching log π(y|x).
        """
        # Tokenize full sequence (prompt + response) — model needs full context
        full_text = prompt + " " + response
        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding=False,
        ).to(self.device)

        # WL1 fix: compute prompt_len via token ID matching instead of
        # tokenizing prompt separately (which can be off-by-one due to
        # BOS token and separator whitespace handling differences).
        prompt_len = self._get_prompt_len(prompt, enc)

        # Compute log π(y|x) for both models (only response tokens)
        weak_logp = self._sequence_log_prob(self.weak_model, enc, prompt_len)
        ref_logp  = self._sequence_log_prob(self.ref_model,  enc, prompt_len)

        return self.beta * (weak_logp - ref_logp).item()

    def _get_prompt_len(self, prompt: str, full_enc: dict) -> int:
        """
        WL1 fix: Compute the number of prompt tokens in the full tokenized sequence
        by matching token IDs, not by tokenizing the prompt separately.

        Problem with the old approach:
          tokenize(prompt, add_special_tokens=True)  → [BOS, t1, ..., tN]
          tokenize(prompt + " " + response, ...)      → [BOS, t1, ..., tN, ..., tR]
          The " " separator can cause the tokenizer to merge the space into
          the last prompt token or the first response token differently,
          making the lengths differ by ±1.

        Fix:
          Tokenize the prompt alone (no separator), then match its IDs as a
          prefix of the full sequence. This finds the exact boundary where
          prompt tokens end in the full tokenized sequence.

        Args:
            prompt:   raw prompt string
            full_enc: tokenized {input_ids, attention_mask} of (prompt + " " + response)

        Returns:
            Number of prompt tokens in full_enc["input_ids"]
        """
        prompt_enc = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_tensors="pt",
            padding=False,
        )
        prompt_ids = prompt_enc["input_ids"][0].tolist()  # list of ints
        full_ids   = full_enc["input_ids"][0].tolist()    # list of ints

        # Find the longest matching prefix between prompt_ids and full_ids
        match_len = 0
        for i, (p, f) in enumerate(zip(prompt_ids, full_ids)):
            if p == f:
                match_len = i + 1
            else:
                break  # first mismatch → stop

        # Fallback: if no match at all (edge case), use prompt token count
        return match_len if match_len > 0 else len(prompt_ids)

    @staticmethod
    def _sequence_log_prob(
        model: PreTrainedModel,
        enc: dict,
        prompt_len: int = 0,
    ) -> torch.Tensor:
        """
        Compute sum of response token log-probabilities: log π(y|x).

        The full sequence (prompt + response) is fed to the model so that
        the causal LM has proper context. However, only response token
        positions are included in the sum — prompt positions are zeroed out.

        Args:
            model:      causal LM (π_w or π_ref)
            enc:        tokenized full sequence {input_ids, attention_mask}
            prompt_len: number of prompt tokens; positions 0..prompt_len-2
                        (after the shift) are masked out.

        Returns:
            scalar tensor — sum of log-probs over response tokens only
        """
        input_ids      = enc["input_ids"]       # (1, L)
        attention_mask = enc["attention_mask"]  # (1, L)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Shift: position i predicts token i+1
        # After shift: index j in token_logps corresponds to predicting token j+1
        logits = outputs.logits[:, :-1, :]   # (1, L-1, V)
        labels = input_ids[:, 1:]            # (1, L-1)

        log_probs   = F.log_softmax(logits, dim=-1)
        token_logps = torch.gather(
            log_probs, dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)  # (1, L-1)

        # Zero out prompt token positions.
        # After the shift, predicting token at position j uses index j-1.
        # Prompt tokens occupy indices 0..prompt_len-1 in the original sequence.
        # Their predictions land at shifted indices 0..prompt_len-2.
        # Response tokens start at shifted index prompt_len-1.
        #
        # WL2 fix: old condition was `if prompt_len > 1`, which skipped masking
        # for prompt_len=0 (no BOS at all) and prompt_len=1 (only BOS).
        # For prompt_len=0: nothing to mask ([:max(0,-1)] = [:0] → no-op). OK.
        # For prompt_len=1: old code didn't mask either — but there is a BOS token
        # at shifted index 0 that predicts token 1 (first response token). The BOS
        # prediction itself (index 0 after shift) should be excluded → mask [:0] → no-op.
        # This means prompt_len=1 (BOS only) is actually correct with either condition.
        # The real fix: make the masking unconditional to handle edge cases uniformly.
        if prompt_len > 0:
            token_logps[:, : max(0, prompt_len - 1)] = 0.0

        return token_logps.sum()
