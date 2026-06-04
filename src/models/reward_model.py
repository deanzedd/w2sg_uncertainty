"""
Scalar Reward Model for CWPO weak annotator.

Architecture (per CWPO paper):
  backbone (pretrained LM, e.g. OPT-125M or Qwen2.5-0.5B)
      ↓  [last hidden state of the final token]
  nn.Linear(hidden_size, 1)   ← scalar reward head
      ↓
  scalar score  πw(x, y)

The backbone LM head is bypassed — we only use the transformer body.
Training uses Bradley-Terry loss on labeled preference pairs (D_l).

Confidence formula (from CWPO paper, eq. 8):
  C(x, y+, y-) = 2 · (σ(πw(x,y+) − πw(x,y-)) − 0.5)
               = sigmoid(πw(x,y+) − πw(x,y-)) * 2 − 1
  which maps the score difference to [0, 1).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


class ScalarRewardModel(nn.Module):
    """
    Lightweight reward model built from a pretrained LM backbone.

    The backbone's language-model head is ignored; instead we add a
    single linear layer on top of the last-token hidden state.

    Args:
        backbone_name: HuggingFace model ID (e.g. "facebook/opt-125m")
        cache_dir:     HF cache directory
        dtype:         torch dtype for the backbone
    """

    def __init__(
        self,
        backbone_name: str,
        cache_dir: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        # Load transformer body only (no LM head)
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        )

        hidden_size = self.backbone.config.hidden_size
        self.scalar_head = nn.Linear(hidden_size, 1, bias=False)

        # Initialize scalar head with small weights
        nn.init.normal_(self.scalar_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scalar reward score πw(x, y).

        Args:
            input_ids:       (batch_size, seq_len)
            attention_mask:  (batch_size, seq_len)

        Returns:
            scores: (batch_size,)  — scalar reward per sequence
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use the hidden state of the last non-padded token
        last_token_hidden = self._get_last_token_hidden(
            outputs.last_hidden_state, attention_mask
        )
        scores = self.scalar_head(last_token_hidden).squeeze(-1)  # (batch_size,)
        return scores

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_last_token_hidden(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the hidden state of the last non-padding token for each sample.

        Args:
            hidden_states:   (batch, seq_len, hidden_size)
            attention_mask:  (batch, seq_len)   1=real token, 0=pad

        Returns:
            (batch, hidden_size)
        """
        # Last real token index per sample
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_size = hidden_states.size(0)
        batch_idx = torch.arange(batch_size, device=hidden_states.device)
        return hidden_states[batch_idx, seq_lengths]

    @staticmethod
    def compute_confidence(
        score_chosen: torch.Tensor,
        score_rejected: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CWPO confidence weight (eq. 8 from paper):

            C(x, y+, y-) = 2 · (σ(πw(x,y+) − πw(x,y-)) − 0.5)

        Range: [0, 1)  where 0.5 → 0 (coin flip) and large diff → ~1.

        Args:
            score_chosen:   (batch_size,)  πw(x, y+)
            score_rejected: (batch_size,)  πw(x, y-)

        Returns:
            confidence: (batch_size,)
        """
        diff = score_chosen - score_rejected
        confidence = 2.0 * (torch.sigmoid(diff) - 0.5)
        return confidence.clamp(min=0.0)  # ensure non-negative

    def bradley_terry_loss(
        self,
        score_chosen: torch.Tensor,
        score_rejected: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bradley-Terry loss for reward model training:
            L = -log σ(s_chosen - s_rejected)

        Args:
            score_chosen:   (batch_size,)
            score_rejected: (batch_size,)

        Returns:
            scalar loss
        """
        loss = -F.logsigmoid(score_chosen - score_rejected).mean()
        return loss


def load_reward_model_and_tokenizer(
    backbone_name: str,
    cache_dir: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[ScalarRewardModel, PreTrainedTokenizerBase]:
    """
    Convenience function: load reward model + matching tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        backbone_name,
        cache_dir=cache_dir,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = ScalarRewardModel(backbone_name, cache_dir=cache_dir, dtype=dtype)
    return model, tokenizer
