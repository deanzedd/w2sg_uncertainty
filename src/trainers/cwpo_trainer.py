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
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets as hf_datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers.trainer import TRAINER_STATE_NAME
from trl import DPOConfig, DPOTrainer

from ..losses.dpo_loss import compute_log_probs
from .sft_trainer import _detect_precision

logger = logging.getLogger(__name__)


class SafeCheckpointMixin:
    """
    Mixin that guarantees trainer_state.json is always written when a checkpoint
    is saved, even when PEFT's LoRA save hook bypasses the normal HF Trainer
    checkpoint path.

    Root cause: PEFT overrides _save_checkpoint and calls model.save_pretrained()
    directly, which only writes adapter_model.safetensors / adapter_config.json.
    The call to self.state.save_to_json() inside the original _save_checkpoint
    may be skipped or happen before PEFT finishes, leaving trainer_state.json
    absent from the checkpoint directory.

    Fix: call super()._save_checkpoint() first (handles everything HF/PEFT does),
    then unconditionally write trainer_state.json afterwards.
    """

    def _save_checkpoint(self, model, trial):
        # Let HF Trainer (and any PEFT override) do its normal save first.
        super()._save_checkpoint(model, trial)

        # Then guarantee trainer_state.json exists in every checkpoint dir.
        # self.state.global_step is the step that was just checkpointed.
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        state_path = os.path.join(output_dir, TRAINER_STATE_NAME)

        if not os.path.exists(state_path):
            self.state.save_to_json(state_path)
            logger.info(
                f"[SafeCheckpointMixin] Wrote missing {TRAINER_STATE_NAME} "
                f"to {output_dir} (global_step={self.state.global_step})"
            )


class CWPOTrainer(SafeCheckpointMixin, DPOTrainer):
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
        # CT2 fix: do NOT pop "confidence_weight" here again.
        # compute_loss (the public hook called first) already pops it and stashes
        # it in self._current_confidence_weights. A second pop always returns None
        # because the key was already removed, making the stash mechanism the only
        # source of truth. Use the stashed value directly.
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
        # ── Device resolution ─────────────────────────────────────────────
        # CT3 fix: use accelerator.device instead of model.device
        # (model.device is "meta" or wrong with device_map="auto")
        device = self.accelerator.device

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

        # CT1 fix: inline implementation instead of private TRL API
        # selective_log_softmax from TRL is internal and breaks across versions.
        # Equivalent: log_softmax then gather the token log-prob.
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # CT4 fix: use multiplication instead of in-place zeroing.
        # In-place assignment on a tensor that participates in the autograd graph
        # raises: "RuntimeError: one of the variables needed for gradient
        # computation has been modified by an inplace operation".
        # Multiplying by a float mask is equivalent and autograd-safe.
        per_token_logps = per_token_logps * shift_completion_mask.float()
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
        # Use inline log_softmax (same CT1 fix as above; no gradient needed here)
        ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
        ref_per_token_logps = ref_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        # Safe to use multiplication here too (inside no_grad, but consistent style)
        ref_per_token_logps = ref_per_token_logps * shift_completion_mask.float()
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
            # Paper spec (detail.txt §3): "nhân trực tiếp trọng số C vào loss của
            # từng sample, trước khi tính mean cho toàn batch"
            # → loss = mean(C · per_sample_loss)
            # NOT sum(C·loss)/sum(C) which inflates loss when C is small.
            loss = (confidence_weights * per_sequence_loss).mean()
        else:
            # Fallback: standard DPO loss (uniform weights, equivalent to C=1)
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

    CWPO spec (Bảng 8):
      - learning_rate: 5e-6
      - beta:          0.5
      - epochs:        3–5
      - batch_size:    4 per device, gradient_accumulation=4 (→ effective 16)
      - weight_decay:  0.05
      - optimizer:     paged_adamw_32bit
      - warmup_steps:  100
      - gradient_checkpointing: True
      - lora_r:        8, lora_alpha: 16
    """
    train_cfg = cfg.training
    # Mirror _resolve_device_map logic: device_map is used whenever any non-null
    # value is explicitly set, or LoRA is on, or CUDA is available (auto-detect fires).
    _explicit_dm = cfg.get("device_map", None)
    use_device_map = (_explicit_dm is not None) or cfg.get("use_lora", False) or torch.cuda.is_available()
    # Linked fix (DT2 pattern): use _detect_precision(cfg.training) to validate
    # bf16 GPU support instead of _detect_precision(cfg) which reads top-level config.
    fp16, bf16 = _detect_precision(cfg.training)

    return DPOConfig(
        output_dir=train_cfg.get("output_dir", "outputs/cwpo"),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        # warmup: prefer warmup_steps (100 per spec, Bảng 8); zero-out ratio to avoid conflict
        warmup_steps=train_cfg.get("warmup_steps", 0),
        warmup_ratio=0.0 if train_cfg.get("warmup_steps", 0) > 0 else train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.05),  # 0.05 per spec
        optim=train_cfg.get("optim", "paged_adamw_32bit"),  # paged adamw 32bit per spec
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
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),  # True per spec
        ddp_find_unused_parameters=False if use_device_map else None,
        report_to="wandb" if cfg.get("use_wandb", True) else "none",
        run_name=cfg.get("wandb_run_name", None),
    )
