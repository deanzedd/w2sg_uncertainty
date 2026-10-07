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

import json
import os
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

        # RM-dtype fix: cast scalar_head to the same dtype as backbone.
        # nn.Linear defaults to float32 regardless of what dtype the backbone uses.
        # When backbone is loaded with torch_dtype=bfloat16, the last_token_hidden
        # state is bfloat16 but scalar_head.weight is float32 → RuntimeError:
        # "expected mat1 and mat2 to have the same dtype, but got: BFloat16 != float"
        self.scalar_head = self.scalar_head.to(dtype)

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
        # R1 fix: if attention_mask is all-zeros (padding-only sequence),
        # seq_lengths becomes -1, and hidden_states[idx, -1] silently returns the
        # LAST hidden state via Python negative indexing instead of raising an error.
        if (attention_mask.sum(dim=1) == 0).any():
            raise ValueError(
                "R1: attention_mask has at least one all-zero row (padding-only sequence). "
                "This causes seq_lengths=-1 and incorrect hidden state extraction. "
                "Ensure all sequences have at least one real token."
            )
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
    checkpoint_path: Optional[str] = None,
) -> Tuple[ScalarRewardModel, PreTrainedTokenizerBase]:
    """
    R2 fix: Load reward model + tokenizer, supporting two modes:

    Mode A — HuggingFace pretrained backbone (original behaviour):
        load_reward_model_and_tokenizer("facebook/opt-125m")
        Loads a fresh ScalarRewardModel with random scalar_head weights.

    Mode B — checkpoint directory (new):
        load_reward_model_and_tokenizer(
            "facebook/opt-125m",              # still used as fallback
            checkpoint_path="outputs/.../checkpoint-final"
        )
        Or auto-detect: if backbone_name is a directory path containing
        metadata.json, treats it as a checkpoint directory directly.

        In Mode B:
          1. Reads backbone_name from metadata.json (if available)
          2. Loads backbone architecture from HF
          3. Loads tokenizer from checkpoint dir (captures special token mods)
          4. Loads fine-tuned weights from model.pt

    Args:
        backbone_name:   HuggingFace model ID  OR  checkpoint directory path.
                         If it's a directory with metadata.json, Mode B is used.
        cache_dir:       HF cache directory (used for backbone download)
        dtype:           torch dtype for backbone
        checkpoint_path: explicit checkpoint directory (overrides auto-detect)

    Returns:
        (ScalarRewardModel with fine-tuned weights, tokenizer)
    """
    # ── Detect checkpoint directory ──────────────────────────────────────
    # Allow backbone_name itself to be a checkpoint dir for ergonomic API.
    effective_checkpoint = checkpoint_path
    if effective_checkpoint is None and os.path.isdir(backbone_name):
        metadata_file = os.path.join(backbone_name, "metadata.json")
        if os.path.exists(metadata_file):
            effective_checkpoint = backbone_name

    if effective_checkpoint is not None:
        # Mode B: load from checkpoint directory
        metadata_file = os.path.join(effective_checkpoint, "metadata.json")
        weights_file = os.path.join(effective_checkpoint, "model.pt")

        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"Checkpoint directory '{effective_checkpoint}' is missing model.pt. "
                f"Expected: {weights_file}"
            )

        # Read backbone_name from metadata if available
        resolved_backbone = backbone_name
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            if metadata.get("backbone_name"):
                resolved_backbone = metadata["backbone_name"]

        # Load tokenizer from checkpoint dir (captures pad_token, special tokens)
        tok_files = [
            "tokenizer_config.json", "vocab.json", "tokenizer.json",
            "special_tokens_map.json", "merges.txt",
        ]
        has_saved_tokenizer = any(
            os.path.exists(os.path.join(effective_checkpoint, f)) for f in tok_files
        )
        if has_saved_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(
                effective_checkpoint,
                use_fast=True,
                trust_remote_code=True,
            )
        else:
            # Fallback: load from backbone (tokenizer not saved in checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_backbone,
                cache_dir=cache_dir,
                use_fast=True,
                trust_remote_code=True,
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Build model architecture from backbone
        model = ScalarRewardModel(resolved_backbone, cache_dir=cache_dir, dtype=dtype)

        # Load fine-tuned weights
        state_dict = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(state_dict)

        return model, tokenizer

    # ── Mode A: fresh pretrained backbone (original behaviour) ───────────
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
