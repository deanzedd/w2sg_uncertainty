"""
Multi-Head Reward Model Trainer — MWDPO bootstrap calibration (1a) and 2-phase training.

Trains a MultiHeadRewardModel on D_l with per-head bootstrap masks so that
each head sees a different random resample of the batch (with replacement),
producing K decorrelated reward estimators from one shared backbone.

When use_bootstrap=False, all heads see the full batch (no resampling) —
useful as an ablation to isolate the effect of bootstrapping vs. multi-head alone.

2-Phase mode (freeze_backbone=True):
    Backbone is frozen (requires_grad=False, eval mode enforced every step).
    Only the K heads are in the optimizer — backbone receives no gradient updates.
    This preserves the Phase 1 feature representation and forces heads to
    find diverse projections via dropout + bootstrap masks.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from ..data.utils import RewardDataCollator, tokenize_for_reward_model
from ..models.multi_head_reward_model import (
    MultiHeadRewardModel,
    save_multi_head_reward_model,
)

logger = logging.getLogger(__name__)


class MultiHeadRewardTrainer:
    """
    Trainer for MultiHeadRewardModel with optional bootstrap-per-head masking.

    Args:
        model:          MultiHeadRewardModel instance
        tokenizer:      tokenizer compatible with the model
        cfg:            DictConfig — reads from cfg.reward_model (same schema as
                        RewardModelTrainer for drop-in compatibility)
        device:         compute device
        backbone_name:  HF model ID (for metadata.json)
        use_bootstrap:  if True (default), each head trains on bootstrap resample
                        of the batch — provides true decorrelation.
                        if False, all heads see the full batch (ablation mode).
    """

    def __init__(
        self,
        model: MultiHeadRewardModel,
        tokenizer,
        cfg: DictConfig,
        device: str = "cuda",
        backbone_name: Optional[str] = None,
        use_bootstrap: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.cfg = cfg.reward_model
        self.device = device
        self.backbone_name = backbone_name
        self.use_bootstrap = use_bootstrap
        self.freeze_backbone = freeze_backbone

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """
        Train MultiHeadRewardModel on D_l.

        Bootstrap logic (when use_bootstrap=True):
            For each batch B of size N:
              - Generate K binary masks via make_bootstrap_masks(N, K)
              - Each head k computes BT loss only on samples where mask_k == True
              - Backward pass updates backbone + head_k gradients jointly
              - Result: K heads with different gradient trajectories → decorrelated

        Args:
            train_dataset:          labeled preference dataset (D_l)
            eval_dataset:           optional eval dataset for accuracy logging
            resume_from_checkpoint: checkpoint dir to resume from
        """
        max_length = self.cfg.get("max_length", 512)
        collator   = RewardDataCollator(self.tokenizer, max_length=max_length)

        # Tokenize
        train_tok = [
            tokenize_for_reward_model(s, self.tokenizer, max_length)
            for s in train_dataset
        ]
        train_loader = DataLoader(
            train_tok,
            batch_size=self.cfg.get("per_device_train_batch_size", 4),
            shuffle=True,
            collate_fn=collator,
            num_workers=self.cfg.get("dataloader_num_workers", 2),
        )

        eval_loader = None
        if eval_dataset is not None:
            eval_tok = [
                tokenize_for_reward_model(s, self.tokenizer, max_length)
                for s in eval_dataset
            ]
            eval_loader = DataLoader(
                eval_tok,
                batch_size=self.cfg.get("per_device_eval_batch_size", 4),
                shuffle=False,
                collate_fn=collator,
            )

        # Optimizer: Phase 2 → only K heads; Phase 1-style → backbone + all heads
        if self.freeze_backbone:
            trainable_params = list(self.model.heads.parameters())
            n_trainable = sum(p.numel() for p in trainable_params)
            logger.info(
                f"[MultiHeadRewardTrainer] Backbone FROZEN (Phase 2). "
                f"Training only K={self.model.num_heads} heads "
                f"({n_trainable:,} params)"
            )
        else:
            trainable_params = list(self.model.parameters())
            logger.info(
                "[MultiHeadRewardTrainer] Training backbone + all heads jointly."
            )

        optimizer = AdamW(
            trainable_params,
            lr=float(self.cfg.get("learning_rate", 1e-5)),
            weight_decay=float(self.cfg.get("weight_decay", 0.01)),
        )

        num_epochs  = self.cfg.get("num_train_epochs", 5)
        num_steps   = len(train_loader) * num_epochs
        warmup_steps = int(num_steps * self.cfg.get("warmup_ratio", 0.1))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_steps,
        )

        # ── Resume from checkpoint ─────────────────────────────────────────
        start_global_step = 0
        if resume_from_checkpoint is not None:
            weights_path = os.path.join(resume_from_checkpoint, "model.pt")
            opt_path     = os.path.join(resume_from_checkpoint, "optimizer_state.pt")
            state_path   = os.path.join(resume_from_checkpoint, "trainer_state.json")

            if os.path.isfile(weights_path):
                logger.info(f"Resuming model weights from: {weights_path}")
                self.model.load_state_dict(
                    torch.load(weights_path, map_location=self.device)
                )
            if os.path.isfile(opt_path):
                logger.info(f"Resuming optimizer from: {opt_path}")
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location=self.device)
                )
            if os.path.isfile(state_path):
                with open(state_path) as f:
                    state = json.load(f)
                start_global_step = int(state.get("global_step", 0))
                logger.info(f"Resuming from global_step={start_global_step}")

        logging_steps = self.cfg.get("logging_steps", 50)
        save_steps    = self.cfg.get("save_steps", 500)
        output_dir    = self.cfg.get("output_dir", "outputs/reward_model")
        os.makedirs(output_dir, exist_ok=True)

        bootstrap_str = "with bootstrap" if self.use_bootstrap else "NO bootstrap (ablation)"
        freeze_str    = "FROZEN backbone (Phase 2)" if self.freeze_backbone else "joint backbone+heads"
        logger.info(
            f"[MultiHeadRewardTrainer] K={self.model.num_heads} heads, "
            f"{bootstrap_str}, {freeze_str}"
        )

        global_step = 0
        self.model.train()
        # Phase 2: ensure backbone stays in eval mode even after model.train()
        if self.freeze_backbone:
            self.model.backbone.eval()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            steps_this_epoch = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                global_step += 1
                if global_step <= start_global_step:
                    continue

                # Phase 2: re-enforce backbone.eval() each step
                # (optimizer.step() or other calls could inadvertently enable train mode)
                if self.freeze_backbone:
                    self.model.backbone.eval()

                chosen_ids    = batch["chosen_input_ids"].to(self.device)
                chosen_mask   = batch["chosen_attention_mask"].to(self.device)
                rejected_ids  = batch["rejected_input_ids"].to(self.device)
                rejected_mask = batch["rejected_attention_mask"].to(self.device)

                # Forward: (batch, K) for chosen and rejected
                scores_chosen   = self.model(chosen_ids, chosen_mask)
                scores_rejected = self.model(rejected_ids, rejected_mask)

                # Bootstrap masks: one boolean (batch,) tensor per head
                bootstrap_masks = None
                if self.use_bootstrap:
                    batch_size = chosen_ids.size(0)
                    bootstrap_masks = MultiHeadRewardModel.make_bootstrap_masks(
                        batch_size, self.model.num_heads, device=self.device
                    )

                # BT loss averaged over K heads (each head uses its own mask)
                loss = self.model.bradley_terry_loss_per_head(
                    scores_chosen, scores_rejected, bootstrap_masks
                )

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.get("max_grad_norm", 1.0),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                steps_this_epoch += 1

                if global_step % logging_steps == 0:
                    logger.info(
                        f"Step {global_step} | Loss: {loss.item():.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

                if global_step % save_steps == 0:
                    save_multi_head_reward_model(
                        self.model, self.tokenizer, output_dir,
                        global_step, self.backbone_name, optimizer=optimizer,
                    )

            avg_loss = epoch_loss / max(1, steps_this_epoch)
            logger.info(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

            if eval_loader is not None:
                acc = self._evaluate(eval_loader)
                logger.info(f"Epoch {epoch+1} eval pref accuracy: {acc:.4f}")

        # Final save
        save_multi_head_reward_model(
            self.model, self.tokenizer, output_dir,
            "final", self.backbone_name, optimizer=optimizer,
        )
        logger.info(f"MultiHeadRewardModel saved to {output_dir}/checkpoint-final")

    # ------------------------------------------------------------------ #
    #  Eval                                                                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        """
        Preference accuracy: fraction of samples where ensemble mean
        score for chosen > score for rejected.
        """
        self.model.eval()
        correct = total = 0
        for batch in loader:
            chosen_ids    = batch["chosen_input_ids"].to(self.device)
            chosen_mask   = batch["chosen_attention_mask"].to(self.device)
            rejected_ids  = batch["rejected_input_ids"].to(self.device)
            rejected_mask = batch["rejected_attention_mask"].to(self.device)

            s_chosen   = self.model(chosen_ids, chosen_mask).mean(dim=-1)   # (batch,)
            s_rejected = self.model(rejected_ids, rejected_mask).mean(dim=-1)

            correct += (s_chosen > s_rejected).sum().item()
            total   += s_chosen.size(0)

        self.model.train()
        return correct / max(1, total)
