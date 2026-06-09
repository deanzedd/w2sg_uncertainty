"""
CWPO Strong Model Trainer — TRL 1.5.x compatible.

Implements Confidence-Weighted DPO (CW-DPO) loss:

    L_CW-DPO = E_{(x,y+,y-) ∈ D̂} [
        C(x,y+,y-) · (-log σ(β·(log π_θ(y+|x)/π_ref(y+|x) - log π_θ(y-|x)/π_ref(y-|x))))
    ]

where C = 2·(σ(πw(x,y+) − πw(x,y-)) − 0.5) is computed by the scalar weak model
and stored in the pseudo-labeled dataset as `confidence_weight`.

TRL 1.5.1 design notes:
  - TRL 1.5.x uses `_compute_loss()` internally (not `concatenated_forward()`).
  - `_compute_loss` builds `model_kwargs` from `inputs` by filtering:
      _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
    Any extra key NOT in this set gets passed to `model()` — which would crash.
  - Solution: Override `compute_loss` to pop `confidence_weight` before delegating
    to `_compute_loss`, then rescale the returned scalar loss by a batch confidence weight.
  - Limitation: TRL 1.5.x `_compute_loss` returns a scalar (mean reduced) loss.
    We approximate CW-DPO by using the AVERAGE confidence weight over the batch
    since we can't inject per-sample weights inside the private TRL loop.
    This is mathematically equivalent only when the batch is homogeneous, but
    is practically close to the exact formulation at typical batch sizes.

For exact per-sample weighting, see `_compute_cwpo_loss_exact()` which fully
reimplements the DPO forward pass outside TRL.

Hyperparameters (per CWPO spec):
    learning_rate:               5e-6
    beta:                        0.5
    num_train_epochs:            3–5
    per_device_train_batch_size: 4 (gradient_accumulation=4 → effective 16)
    lora_r:                      8, lora_alpha: 16
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets as hf_datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from trl import DPOConfig, DPOTrainer

from ..losses.dpo_loss import compute_log_probs
from .sft_trainer import _detect_precision

logger = logging.getLogger(__name__)


class CWPOTrainer(DPOTrainer):
    """
    Extends TRL 1.5.x DPOTrainer to implement CW-DPO (Confidence-Weighted DPO).

    Strategy:
      1. Override `compute_loss` (public hook) to pop `confidence_weight` from the batch.
      2. Stash the confidence weights in `self._current_confidence_weights`.
      3. Override `_compute_loss` (private TRL method) to apply confidence weighting
         to the per-sequence loss before mean reduction.

    The `remove_unused_columns=False` in DPOConfig ensures TRL's tokenization
    pipeline does NOT drop the `confidence_weight` column from the dataset.

    This trainer requires the dataset to have columns:
        - prompt, chosen, rejected   (tokenized by TRL internally)
        - confidence_weight          (float scalar per sample)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Thread-local storage for confidence weights during the loss computation
        self._current_confidence_weights: Optional[torch.Tensor] = None

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Public override: pop confidence_weight from inputs before TRL processes them,
        stash for use in _compute_loss.
        """
        # Pop confidence_weight so TRL doesn't try to pass it to model()
        confidence_weights = inputs.pop("confidence_weight", None)

        # Stash on instance for use by _compute_loss
        self._current_confidence_weights = confidence_weights

        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Clean up
        self._current_confidence_weights = None
        return result

    def _compute_loss(self, model, inputs, return_outputs):
        """
        Override TRL's private _compute_loss to inject per-batch confidence weighting.

        TRL 1.5.x computes `per_sequence_loss` (one value per batch element) and
        then does `loss = per_sequence_loss.mean()`.

        We intercept after per_sequence_loss computation and apply confidence weighting:
            loss = weighted_mean(confidence_weights * per_sequence_loss)
        """
        # Pop confidence_weight that may still be in inputs if compute_loss wasn't called
        confidence_weights = inputs.pop("confidence_weight", None)
        if confidence_weights is None:
            confidence_weights = self._current_confidence_weights

        # Standard TRL processing — but we need to hook into per_sequence_loss
        # We accomplish this by overriding the internals via a monkey-patch context
        # However, _compute_loss is complex. The cleanest approach for TRL 1.5.x:
        # implement CW-DPO loss fully here, bypassing TRL's internal loop.

        return self._compute_cwpo_loss_exact(model, inputs, confidence_weights, return_outputs)

    def _compute_cwpo_loss_exact(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        confidence_weights: Optional[torch.Tensor],
        return_outputs: bool,
    ):
        """
        Exact CW-DPO loss implementation for TRL 1.5.x.

        Computes the confidence-weighted DPO loss:
            L = -weighted_mean[C · log σ(β · (log π(y+|x)/π_ref(y+|x) - log π(y-|x)/π_ref(y-|x)))]

        The batch from TRL 1.5.x is structured as [chosen_1, ..., chosen_n, rejected_1, ..., rejected_n]
        in a concatenated format via `input_ids`, `attention_mask`, and `completion_mask`.

        Args:
            model:              the policy model (πθ)
            inputs:             TRL-formatted batch dict (tokenized, concatenated chosen+rejected)
            confidence_weights: (batch_size_half,) confidence C per sample, or None
            return_outputs:     whether to return model outputs

        Returns:
            loss tensor (scalar), or (loss, outputs) if return_outputs=True
        """
        from trl.trainer.dpo_trainer import selective_log_softmax

        device = model.device if hasattr(model, 'device') else next(model.parameters()).device

        # ── Build model_kwargs (exclude TRL-internal keys) ────────────────
        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False

        # ── Policy forward pass ──────────────────────────────────────────
        outputs = model(**model_kwargs)

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()

        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0
        logps = per_token_logps.sum(dim=1)  # (2*batch_size,)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        # ── Reference forward pass ────────────────────────────────────────
        with torch.no_grad():
            if self.ref_model is not None:
                ref_outputs = self.ref_model(**model_kwargs)
            else:
                # PEFT: use adapter disabled for reference
                from peft.utils import disable_adapter
                unwrapped = self.accelerator.unwrap_model(model)
                with disable_adapter(unwrapped):
                    ref_outputs = unwrapped(**model_kwargs)

        ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
        ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
        ref_per_token_logps[shift_completion_mask == 0] = 0.0
        ref_logps = ref_per_token_logps.sum(dim=1)
        ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        # ── CW-DPO loss ──────────────────────────────────────────────────
        # Standard DPO per-sample loss: -log σ(β · (log ratio_chosen - log ratio_rejected))
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        delta_score = chosen_logratios - rejected_logratios
        per_sequence_loss = -F.logsigmoid(self.beta * delta_score)   # (batch_size,)

        # Apply confidence weighting
        if confidence_weights is not None and torch.is_tensor(confidence_weights):
            confidence_weights = confidence_weights.to(per_sequence_loss.device).float()
            # Weighted mean: E[C · loss] / E[C] (normalized to avoid scale issues)
            weight_sum = confidence_weights.sum().clamp(min=1e-8)
            loss = (confidence_weights * per_sequence_loss).sum() / weight_sum
        else:
            # Fallback: standard DPO loss (uniform weights)
            loss = per_sequence_loss.mean()

        # ── Metrics logging (mirrors TRL's internal tracking) ─────────────
        mode = "train" if model.training else "eval"
        try:
            chosen_rewards = (self.beta * chosen_logratios).detach()
            rejected_rewards = (self.beta * rejected_logratios).detach()
            agg_chosen = self.accelerator.gather(chosen_rewards)
            agg_rejected = self.accelerator.gather(rejected_rewards)
            self._metrics[mode]["rewards/chosen"].append(agg_chosen.mean().item())
            self._metrics[mode]["rewards/rejected"].append(agg_rejected.mean().item())
            reward_acc = (chosen_rewards > rejected_rewards).float()
            self._metrics[mode]["rewards/accuracies"].append(
                self.accelerator.gather(reward_acc).mean().item()
            )
            margins = chosen_rewards - rejected_rewards
            self._metrics[mode]["rewards/margins"].append(
                self.accelerator.gather(margins).mean().item()
            )
            self._metrics[mode]["logps/chosen"].append(
                self.accelerator.gather(chosen_logps).mean().item()
            )
            self._metrics[mode]["logps/rejected"].append(
                self.accelerator.gather(rejected_logps).mean().item()
            )
            if confidence_weights is not None and torch.is_tensor(confidence_weights):
                self._metrics[mode].setdefault("confidence_weight/mean", []).append(
                    confidence_weights.mean().item()
                )
        except Exception:
            pass  # Metrics logging is optional; don't crash training

        return (loss, outputs) if return_outputs else loss


def build_cwpo_dataset(samples: List[Dict]) -> hf_datasets.Dataset:
    """
    Convert pseudo-labeled CWPO samples to HuggingFace Dataset.

    Includes the `confidence_weight` field alongside raw text fields.
    TRL DPOTrainer tokenizes prompt/chosen/rejected internally via `_prepare_dataset`.
    The `confidence_weight` column passes through because `remove_unused_columns=False`
    is set in `build_cwpo_training_args`.

    Args:
        samples: list of dicts with keys: prompt, chosen, rejected, confidence_weight

    Returns:
        HuggingFace Dataset with columns: prompt, chosen, rejected, confidence_weight
    """
    records = []
    for s in samples:
        records.append({
            "prompt":            s["prompt"],
            "chosen":            s["chosen"],
            "rejected":          s["rejected"],
            "confidence_weight": float(s.get("confidence_weight", 1.0)),
        })
    return hf_datasets.Dataset.from_list(records)


def build_cwpo_training_args(cfg: DictConfig) -> DPOConfig:
    """
    Build DPOConfig (TRL) for CW-DPO strong model training.

    CWPO spec:
      - learning_rate: 5e-6
      - beta:          0.5
      - epochs:        3–5
      - batch_size:    4 per device, gradient_accumulation=4 (→ effective 16)
      - lora_r:        8, lora_alpha: 16
    """
    train_cfg = cfg.training
    use_device_map = cfg.get("device_map", None) == "auto" or cfg.get("use_lora", False)
    fp16, bf16 = _detect_precision(cfg)

    return DPOConfig(
        output_dir=train_cfg.get("output_dir", "outputs/cwpo"),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        eval_steps=train_cfg.get("eval_steps", 200),
        fp16=fp16,
        bf16=bf16,
        beta=float(train_cfg.get("beta", 0.5)),           # 0.5 for CWPO per spec
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        # CRITICAL: must be False so that `confidence_weight` column is NOT stripped
        # before the batch reaches compute_loss / _compute_loss
        remove_unused_columns=False,
        # gradient_checkpointing: required for 7B+ models to avoid OOM
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        ddp_find_unused_parameters=False if use_device_map else None,
        report_to="wandb" if cfg.get("use_wandb", True) else "none",
        run_name=cfg.get("wandb_run_name", None),
    )
