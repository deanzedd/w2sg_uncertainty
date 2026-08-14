"""
Multi-Head Reward Model for MWDPO bootstrap calibration (1a), 2-phase training,
and Super_multi_dpo (LoRA ensemble).

Architecture (original / Phase 2):
    backbone (pretrained LM, shared)
        ↓  [last hidden state of the final token]
    K × head_k   ← K independent heads (linear or 2-layer MLP)
        ↓
    (batch, K) score tensor

2-Phase training (MWDPO_BC):
    Phase 1: Train ScalarRewardModel (backbone + 1 linear head) on D_l.
             Backbone learns a high-quality reward feature representation.
    Phase 2: Load Phase 1 backbone, freeze it, train K heads independently
             (MLPRewardHead with dropout for forced diversity + bootstrap masks).

Super_multi_dpo:
    Phase 1 (Warmup): Train MultiHeadRewardModel (backbone + K linear heads) jointly.
    Phase 2 (Ensemble): Freeze backbone, attach K LoRA adapters, train each
                        LoRA_k + head_k independently → K diverse reward models.
    Each (backbone + LoRA_k + head_k) is wrapped as LoRARewardModel — interface-
    compatible with ScalarRewardModel for use with MultiWeakLabeler.

Head types (configurable via head_type):
    "linear" — nn.Linear(hidden_size, 1)          [backward compatible default]
    "mlp"    — Linear → LayerNorm → GELU → Dropout → Linear  [diverse projections]
"""

import copy
import json
import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  MLP Reward Head                                                              #
# --------------------------------------------------------------------------- #

class MLPRewardHead(nn.Module):
    """
    2-layer MLP reward head for Phase 2 frozen-backbone training.

    Architecture:
        Linear(hidden_size, mlp_hidden) → LayerNorm → GELU → Dropout → Linear(mlp_hidden, 1)

    Design rationale:
        - LayerNorm: stabilises gradients across heads, keeps activations at same scale
        - GELU: smooth nonlinearity better suited to LM-derived features than ReLU
        - Dropout: stochastically masks features each forward pass → forces each head
          to find diverse, non-overlapping subspace of the backbone representation.
          Combined with bootstrap batch masks, this is the primary source of
          head decorrelation when backbone is frozen.

    Args:
        hidden_size: input dim (backbone hidden_size)
        mlp_hidden:  intermediate projection dim (e.g. hidden_size // 4)
        dropout:     Dropout probability (0.3–0.5 recommended for diversity)
        dtype:       torch dtype matching backbone
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden: int,
        dropout: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=False),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, 1, bias=False),
        )
        # Init all Linear layers with small normal weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # Cast to backbone dtype (same fix as ScalarRewardModel scalar_head)
        self.net = self.net.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, hidden_size) → output: (batch, 1)"""
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Backbone loading from Phase 1 checkpoint                                    #
# --------------------------------------------------------------------------- #

def _load_backbone_from_scalar_checkpoint(
    checkpoint_path: str,
    backbone_name: str,
    cache_dir: Optional[str],
    dtype: torch.dtype,
) -> nn.Module:
    """
    Extract backbone weights from a ScalarRewardModel Phase 1 checkpoint.

    ScalarRewardModel state_dict keys:
        "backbone.<layer_name>"   ← transformer body (what we want)
        "scalar_head.<...>"       ← discarded (single head, not used in Phase 2)

    Args:
        checkpoint_path: directory containing model.pt (ScalarRewardModel checkpoint-final)
        backbone_name:   HuggingFace model ID used to build the architecture
        cache_dir:       HF cache directory
        dtype:           torch dtype for the backbone

    Returns:
        nn.Module: backbone with Phase 1 fine-tuned weights loaded
    """
    weights_path = os.path.join(checkpoint_path, "model.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Phase 1 backbone checkpoint not found: {weights_path}\n"
            f"Run Phase 1 first (ScalarRewardModel training) and pass the "
            f"checkpoint-final directory as backbone_checkpoint."
        )

    state_dict = torch.load(weights_path, map_location="cpu")

    # Strip "backbone." prefix, discard "scalar_head.*"
    backbone_state = {
        k[len("backbone."):]: v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }

    if not backbone_state:
        raise ValueError(
            f"No 'backbone.*' keys found in {weights_path}. "
            "Is this a ScalarRewardModel checkpoint?"
        )

    backbone = AutoModel.from_pretrained(
        backbone_name, cache_dir=cache_dir, torch_dtype=dtype
    )
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=True)
    if missing:
        import logging
        logging.getLogger(__name__).warning(
            f"[Phase 1 load] Missing keys in backbone state_dict: {missing}"
        )

    return backbone


# --------------------------------------------------------------------------- #
#  MultiHeadRewardModel                                                        #
# --------------------------------------------------------------------------- #

class MultiHeadRewardModel(nn.Module):
    """
    Shared-backbone reward model with K independent heads.

    Supports two training modes:
        Mode A (original): backbone + K heads train jointly (freeze_backbone=False)
        Mode B (2-phase):  backbone frozen from Phase 1, only K heads train (freeze_backbone=True)

    Args:
        backbone_name:       HuggingFace model ID (e.g. "Qwen/Qwen2.5-0.5B")
        num_heads:           K — number of independent reward heads
        cache_dir:           HF cache directory
        dtype:               torch dtype for backbone and heads
        head_type:           "linear" | "mlp"  (default "linear" for backward compat)
        mlp_hidden:          Intermediate dim for MLPRewardHead. None = hidden_size // 4
        head_dropout:        Dropout in MLPRewardHead (0.0 = disabled, 0.3–0.5 recommended)
        freeze_backbone:     If True, backbone.requires_grad=False and backbone.eval()
                             is enforced. Used in Phase 2 (frozen backbone + K head training).
        backbone_checkpoint: Path to ScalarRewardModel Phase 1 checkpoint-final dir.
                             If provided, backbone weights are loaded from there instead
                             of fresh HF pretrained weights.
    """

    def __init__(
        self,
        backbone_name: str,
        num_heads: int = 3,
        cache_dir: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        # New params (all have backward-compatible defaults):
        head_type: str = "linear",
        mlp_hidden: Optional[int] = None,
        head_dropout: float = 0.0,
        freeze_backbone: bool = False,
        backbone_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()

        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if head_type not in ("linear", "mlp"):
            raise ValueError(f"head_type must be 'linear' or 'mlp', got '{head_type}'")

        # ── Load backbone ──────────────────────────────────────────────────
        if backbone_checkpoint is not None:
            # Phase 2: load Phase 1 fine-tuned backbone weights
            self.backbone = _load_backbone_from_scalar_checkpoint(
                backbone_checkpoint, backbone_name, cache_dir, dtype
            )
        else:
            # Original mode: fresh pretrained backbone
            self.backbone = AutoModel.from_pretrained(
                backbone_name,
                cache_dir=cache_dir,
                torch_dtype=dtype,
            )

        # ── Freeze backbone (Phase 2 mode) ─────────────────────────────────
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()   # disable backbone dropout during head training

        # ── Build K heads ──────────────────────────────────────────────────
        self.num_heads = num_heads
        hidden_size = self.backbone.config.hidden_size
        _mlp_hidden = mlp_hidden if mlp_hidden is not None else (hidden_size // 4)

        if head_type == "mlp":
            self.heads = nn.ModuleList([
                MLPRewardHead(hidden_size, _mlp_hidden, head_dropout, dtype)
                for _ in range(num_heads)
            ])
        else:  # "linear" — backward compatible
            self.heads = nn.ModuleList([
                nn.Linear(hidden_size, 1, bias=False)
                for _ in range(num_heads)
            ])
            # Init + dtype cast (same as original)
            for head in self.heads:
                nn.init.normal_(head.weight, mean=0.0, std=0.02)
                head.to(dtype)

        # ── Store config for save/load ─────────────────────────────────────
        self.head_type = head_type
        self.mlp_hidden = _mlp_hidden if head_type == "mlp" else None
        self.head_dropout = head_dropout
        self.backbone_frozen = freeze_backbone

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute K reward scores for each sequence.

        Args:
            input_ids:       (batch, seq_len)
            attention_mask:  (batch, seq_len)

        Returns:
            scores: (batch, K)  — one score per head per sequence
        """
        last_hidden = self._get_last_token_hidden(
            self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state,
            attention_mask,
        )  # (batch, hidden)

        # Stack K head outputs: each head → (batch, 1) → cat → (batch, K)
        scores = torch.cat(
            [head(last_hidden) for head in self.heads], dim=-1
        )  # (batch, K)
        return scores

    # ------------------------------------------------------------------ #
    #  Per-head loss (used by trainer with bootstrap masks)                #
    # ------------------------------------------------------------------ #

    def bradley_terry_loss_per_head(
        self,
        scores_chosen: torch.Tensor,   # (batch, K)
        scores_rejected: torch.Tensor, # (batch, K)
        bootstrap_masks: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Bradley-Terry loss averaged over K heads.

        Each head k uses its own bootstrap_mask to select which batch samples
        contribute to its gradient — this is the mechanism that decorrelates heads.

        Args:
            scores_chosen:    (batch, K) — chosen scores from forward()
            scores_rejected:  (batch, K) — rejected scores from forward()
            bootstrap_masks:  list of K boolean masks, each (batch,).
                              If None, all samples used for all heads (no bootstrap).

        Returns:
            scalar loss (mean over K heads)
        """
        head_losses = []

        for k in range(self.num_heads):
            s_chosen_k   = scores_chosen[:, k]    # (batch,)
            s_rejected_k = scores_rejected[:, k]  # (batch,)

            if bootstrap_masks is not None:
                mask = bootstrap_masks[k]  # (batch,) bool
                if mask.sum() == 0:
                    # Skip head if mask is all-False (shouldn't happen in practice)
                    continue
                s_chosen_k   = s_chosen_k[mask]
                s_rejected_k = s_rejected_k[mask]

            loss_k = -F.logsigmoid(s_chosen_k - s_rejected_k).mean()
            head_losses.append(loss_k)

        if not head_losses:
            return torch.tensor(0.0, requires_grad=True)

        return torch.stack(head_losses).mean()

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
        if (attention_mask.sum(dim=1) == 0).any():
            raise ValueError(
                "attention_mask has at least one all-zero row (padding-only sequence). "
                "Ensure all sequences have at least one real token."
            )
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_size = hidden_states.size(0)
        batch_idx = torch.arange(batch_size, device=hidden_states.device)
        return hidden_states[batch_idx, seq_lengths]  # (batch, hidden)

    @staticmethod
    def make_bootstrap_masks(
        batch_size: int,
        num_heads: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Generate K bootstrap masks by sampling batch_size indices WITH replacement.

        Each mask is a boolean tensor of shape (batch_size,) where True means
        the sample is included in that head's bootstrap resample.
        Samples not drawn (~37%) are excluded for that head.

        Args:
            batch_size: number of samples in the current batch
            num_heads:  K
            device:     torch device

        Returns:
            list of K boolean tensors, each (batch_size,)
        """
        masks = []
        for _ in range(num_heads):
            # Sample batch_size indices with replacement
            sampled_idx = torch.randint(0, batch_size, (batch_size,), device=device)
            # Convert to boolean mask: True if index was drawn at least once
            mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            mask[sampled_idx] = True
            masks.append(mask)
        return masks

    def attach_lora_adapters(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ) -> List[Tuple[nn.Module, nn.Module]]:
        """
        Phase 2 — Super_multi_dpo: Freeze backbone, create K LoRA-adapted copies.

        For each head k:
            1. Deep-copy the (now frozen) backbone.
            2. Wrap the copy with a fresh LoraConfig (random LoRA init → diversity).
            3. Pair with self.heads[k].

        The returned list contains K (lora_backbone_k, head_k) tuples ready for
        independent training by SuperMultiRewardTrainer.  The original self.backbone
        is left frozen in-place; self.heads are NOT copied (each tuple holds a
        reference to the original head nn.Module).

        Args:
            lora_r:               LoRA rank
            lora_alpha:           LoRA alpha scaling
            lora_dropout:         LoRA dropout (adds diversity on top of random init)
            lora_target_modules:  target module names (None = PEFT auto-detect)

        Returns:
            list of K (lora_backbone, head) tuples, length == self.num_heads
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("peft is required for LoRA. Install: pip install peft")

        # Freeze backbone (in-place, permanent for this instance)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.backbone_frozen = True
        logger.info(
            f"[attach_lora_adapters] Backbone frozen. "
            f"Creating {self.num_heads} LoRA-adapted copies "
            f"(r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})."
        )

        lora_pairs: List[Tuple[nn.Module, nn.Module]] = []
        for k in range(self.num_heads):
            # Independent deep-copy → each adapter gets its own random init
            backbone_k = copy.deepcopy(self.backbone)
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            lora_backbone_k = get_peft_model(backbone_k, lora_cfg)
            n_lora = sum(p.numel() for p in lora_backbone_k.parameters() if p.requires_grad)
            n_head = sum(p.numel() for p in self.heads[k].parameters())
            logger.info(
                f"  LoRA model {k}: lora_params={n_lora:,}, head_params={n_head:,}"
            )
            lora_pairs.append((lora_backbone_k, self.heads[k]))

        return lora_pairs


# --------------------------------------------------------------------------- #
#  LoRA Reward Model — Super_multi_dpo                                         #
# --------------------------------------------------------------------------- #

class LoRARewardModel(nn.Module):
    """
    Single LoRA-adapted reward model for Super_multi_dpo Phase 2.

    Architecture: frozen_backbone + LoRA_k adapter (trainable) + linear head_k (trainable).

    Interface-compatible with ScalarRewardModel:
        forward(input_ids, attention_mask) → (batch,) scores

    This allows LoRARewardModel instances to be passed directly to MultiWeakLabeler
    as drop-in replacements for ScalarRewardModel instances.

    Created by MultiHeadRewardModel.attach_lora_adapters().
    """

    def __init__(self, lora_backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = lora_backbone  # PeftModel wrapping frozen AutoModel
        self.head = head               # nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scalar reward score.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            scores: (batch,)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state   # (batch, seq_len, hidden)
        # Last non-padding token (same as ScalarRewardModel._get_last_token_hidden)
        seq_lengths = attention_mask.sum(dim=-1) - 1   # (batch,)
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        last_token_hidden = last_hidden[batch_idx, seq_lengths]  # (batch, hidden)
        scores = self.head(last_token_hidden).squeeze(-1)        # (batch,)
        return scores

    def bradley_terry_loss(
        self,
        score_chosen: torch.Tensor,
        score_rejected: torch.Tensor,
    ) -> torch.Tensor:
        """BT loss: -log σ(s_chosen - s_rejected)"""
        return -F.logsigmoid(score_chosen - score_rejected).mean()


# --------------------------------------------------------------------------- #
#  Save / Load                                                                 #
# --------------------------------------------------------------------------- #

def save_multi_head_reward_model(
    model: MultiHeadRewardModel,
    tokenizer,
    output_dir: str,
    step,
    backbone_name: str,
    optimizer=None,
) -> None:
    """
    Save MultiHeadRewardModel checkpoint.

    Saves to output_dir/checkpoint-{step}/:
        model.pt           — state dict (backbone + all K heads)
        metadata.json      — backbone_name, num_heads, hidden_size, head config
        tokenizer_*        — tokenizer files
        optimizer_state.pt — optimizer state (if provided)
        trainer_state.json — global_step
    """
    import json as _json

    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_path, exist_ok=True)

    # 1. Model weights
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

    # 2. Metadata (extended with head architecture info)
    metadata = {
        "backbone_name": backbone_name,
        "num_heads": model.num_heads,
        "hidden_size": model.backbone.config.hidden_size,
        "step": str(step),
        "model_type": "multi_head_reward_model",
        # New: head architecture config for correct reconstruction on load
        "head_type": getattr(model, "head_type", "linear"),
        "mlp_hidden": getattr(model, "mlp_hidden", None),
        "head_dropout": getattr(model, "head_dropout", 0.0),
        "backbone_frozen": getattr(model, "backbone_frozen", False),
    }
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        _json.dump(metadata, f, indent=2)

    # 3. Tokenizer
    try:
        tokenizer.save_pretrained(save_path)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not save tokenizer: {e}")

    # 4. Optimizer state
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer_state.pt"))

    # 5. Trainer state
    trainer_state = {"global_step": step if isinstance(step, int) else 0}
    with open(os.path.join(save_path, "trainer_state.json"), "w") as f:
        _json.dump(trainer_state, f, indent=2)


def load_multi_head_reward_model(
    checkpoint_path: str,
    backbone_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple["MultiHeadRewardModel", "PreTrainedTokenizerBase"]:
    """
    Load MultiHeadRewardModel from a checkpoint directory.

    Reads head_type, mlp_hidden, head_dropout from metadata.json to correctly
    reconstruct the architecture before loading state_dict weights.

    Args:
        checkpoint_path: directory containing model.pt + metadata.json
        backbone_name:   override backbone (reads from metadata.json by default)
        cache_dir:       HF cache dir for backbone download
        dtype:           torch dtype

    Returns:
        (MultiHeadRewardModel, tokenizer)
    """
    import json as _json

    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    weights_path  = os.path.join(checkpoint_path, "model.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"model.pt not found in checkpoint: {checkpoint_path}"
        )

    # Read metadata
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = _json.load(f)
        resolved_backbone = backbone_name or metadata.get("backbone_name")
        num_heads    = int(metadata.get("num_heads", 3))
        head_type    = str(metadata.get("head_type", "linear"))
        mlp_hidden   = metadata.get("mlp_hidden", None)
        head_dropout = float(metadata.get("head_dropout", 0.0))
    else:
        if backbone_name is None:
            raise ValueError(
                f"metadata.json not found at {metadata_path} and backbone_name not provided."
            )
        resolved_backbone = backbone_name
        num_heads    = 3
        head_type    = "linear"
        mlp_hidden   = None
        head_dropout = 0.0

    # Tokenizer: prefer saved, fallback to backbone
    tok_files = [
        "tokenizer_config.json", "vocab.json", "tokenizer.json",
        "special_tokens_map.json", "merges.txt",
    ]
    has_saved_tokenizer = any(
        os.path.exists(os.path.join(checkpoint_path, f)) for f in tok_files
    )
    if has_saved_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, use_fast=True, trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_backbone, cache_dir=cache_dir, use_fast=True, trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build architecture (freeze_backbone=False at load time: inference uses all params)
    model = MultiHeadRewardModel(
        resolved_backbone,
        num_heads=num_heads,
        cache_dir=cache_dir,
        dtype=dtype,
        head_type=head_type,
        mlp_hidden=mlp_hidden,
        head_dropout=head_dropout,
        freeze_backbone=False,       # inference: no freeze needed
        backbone_checkpoint=None,    # weights loaded directly from state_dict below
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)

    return model, tokenizer


# --------------------------------------------------------------------------- #
#  LoRA reward model save / load — Super_multi_dpo                             #
# --------------------------------------------------------------------------- #

def save_lora_reward_models(
    lora_model_pairs: List[Tuple[nn.Module, nn.Module]],
    tokenizer,
    output_dir: str,
    backbone_name: str,
) -> None:
    """
    Save K LoRA reward models to output_dir/model_{k}/checkpoint-final/.

    Each model k saves:
        adapter_model.bin / adapter_config.json  — PEFT LoRA adapter weights
        head.pt                                  — linear head state dict
        tokenizer_*                              — tokenizer files
        metadata.json                            — model_type, backbone_name, index

    Args:
        lora_model_pairs: list of (lora_backbone_k, head_k) tuples, length K
        tokenizer:        shared tokenizer
        output_dir:       base directory; model k → output_dir/model_k/checkpoint-final/
        backbone_name:    HF model ID of the base backbone
    """
    for k, (lora_backbone, head) in enumerate(lora_model_pairs):
        model_dir = os.path.join(output_dir, f"model_{k}", "checkpoint-final")
        os.makedirs(model_dir, exist_ok=True)

        # 1. LoRA adapter weights (adapter_model.bin + adapter_config.json)
        lora_backbone.save_pretrained(model_dir)

        # 2. Head weights
        torch.save(head.state_dict(), os.path.join(model_dir, "head.pt"))

        # 3. Tokenizer
        try:
            tokenizer.save_pretrained(model_dir)
        except Exception as e:
            logger.warning(f"Could not save tokenizer for LoRA model {k}: {e}")

        # 4. Metadata
        metadata = {
            "backbone_name": backbone_name,
            "model_type": "lora_reward_model",
            "lora_model_index": k,
        }
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[LoRA] Saved model {k} → {model_dir}")


def load_lora_reward_model(
    checkpoint_dir: str,
    cache_dir: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple["LoRARewardModel", "PreTrainedTokenizerBase"]:
    """
    Load one LoRA reward model from a checkpoint directory.

    Expects:
        checkpoint_dir/adapter_model.bin   — LoRA adapter weights
        checkpoint_dir/adapter_config.json — LoRA config
        checkpoint_dir/head.pt             — linear head state dict
        checkpoint_dir/metadata.json       — backbone_name

    Returns:
        (LoRARewardModel, tokenizer)
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("peft is required. Install: pip install peft")

    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {checkpoint_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    backbone_name = metadata["backbone_name"]

    # Load base backbone
    backbone = AutoModel.from_pretrained(
        backbone_name, cache_dir=cache_dir, torch_dtype=dtype
    )
    # Wrap with LoRA adapter
    lora_backbone = PeftModel.from_pretrained(backbone, checkpoint_dir)

    # Load head
    hidden_size = backbone.config.hidden_size
    head = nn.Linear(hidden_size, 1, bias=False).to(dtype)
    head_path = os.path.join(checkpoint_dir, "head.pt")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"head.pt not found in {checkpoint_dir}")
    head.load_state_dict(torch.load(head_path, map_location="cpu"))

    # Tokenizer
    tok_files = [
        "tokenizer_config.json", "vocab.json", "tokenizer.json",
        "special_tokens_map.json", "merges.txt",
    ]
    has_saved_tok = any(
        os.path.exists(os.path.join(checkpoint_dir, f)) for f in tok_files
    )
    if has_saved_tok:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, use_fast=True, trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            backbone_name, cache_dir=cache_dir, use_fast=True, trust_remote_code=True
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return LoRARewardModel(lora_backbone, head), tokenizer

