"""
CWPO Strong Model Trainer.

Overrides DPOTrainer.compute_loss to inject confidence weights:

    L_CWPO = E_{(x,y+,y-) ∈ D̂} [ C(x,y+,y-) · ℓ_DPO(πs; x, y+, y-) ]

where C = 2·(σ(πw(x,y+) − πw(x,y-)) − 0.5) is stored in the dataset
as `confidence_weight`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

from ..losses.cwpo_loss import cwpo_loss
from ..losses.dpo_loss import compute_log_probs

logger = logging.getLogger(__name__)


class CWPODataset(torch.utils.data.Dataset):
    """
    Wraps pseudo-labeled samples for CWPO training.
    Each item: {prompt, chosen, rejected, confidence_weight, *tokenized fields*}
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.data = [self._tokenize(s) for s in samples]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

    def _tokenize(self, sample: Dict) -> Dict:
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]
        confidence = float(sample.get("confidence_weight", 1.0))

        def _enc(text):
            return self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "confidence_weight": confidence,
            # TRL DPOTrainer expects these keys
            "chosen_input_ids": _enc(prompt + " " + chosen)["input_ids"],
            "chosen_attention_mask": _enc(prompt + " " + chosen)["attention_mask"],
            "rejected_input_ids": _enc(prompt + " " + rejected)["input_ids"],
            "rejected_attention_mask": _enc(prompt + " " + rejected)["attention_mask"],
            "prompt_input_ids": _enc(prompt)["input_ids"],
            "prompt_attention_mask": _enc(prompt)["attention_mask"],
        }


class CWPOTrainer(DPOTrainer):
    """
    Extends TRL's DPOTrainer to implement confidence-weighted loss.

    Key override: `compute_loss` multiplies per-sample DPO loss by C.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        args: DPOConfig,
        train_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        eval_dataset: Optional[Dataset] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            eval_dataset=eval_dataset,
            **kwargs,
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Override DPOTrainer loss with confidence weighting.
        """
        # Extract confidence weights before passing to parent
        confidence_weights = inputs.pop("confidence_weights", None)

        # Compute log probs via parent class internals
        (
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            _,
        ) = self.concatenated_forward(model, inputs)

        beta = self.args.beta

        if confidence_weights is not None:
            confidence_weights = confidence_weights.to(policy_chosen_logps.device)
            loss = cwpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                confidence_weights=confidence_weights,
                beta=beta,
            )
        else:
            # Fallback to standard DPO if no confidence weights
            from ..losses.dpo_loss import dpo_loss
            loss = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=beta,
            )

        if return_outputs:
            return loss, {}
        return loss


def build_cwpo_training_args(cfg: DictConfig) -> DPOConfig:
    """Build DPOConfig (TRL) from OmegaConf training config."""
    train_cfg = cfg.training
    return DPOConfig(
        output_dir=train_cfg.get("output_dir", "outputs/cwpo"),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(train_cfg.get("learning_rate", 5e-7)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        eval_steps=train_cfg.get("eval_steps", 200),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        beta=float(train_cfg.get("beta", 0.1)),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        remove_unused_columns=False,
        report_to="wandb" if cfg.get("use_wandb", True) else "none",
        run_name=cfg.get("wandb_run_name", None),
    )
