"""
Reward Model Trainer (CWPO Phase 1).

Trains the weak annotator (ScalarRewardModel) on D_l using Bradley-Terry loss.

The weak model uses:
    backbone (pretrained LM) → Linear(hidden_size, 1) scalar head

Training objective:
    L = -log σ(score_chosen - score_rejected)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from ..data.utils import RewardDataCollator, tokenize_for_reward_model
from ..models.reward_model import ScalarRewardModel

logger = logging.getLogger(__name__)


class RewardModelTrainer:
    """
    Trainer for the CWPO scalar reward model (weak annotator).

    Args:
        model:      ScalarRewardModel instance
        tokenizer:  tokenizer compatible with the model
        cfg:        DictConfig with reward_model training parameters
        device:     compute device
    """

    def __init__(
        self,
        model: ScalarRewardModel,
        tokenizer,
        cfg: DictConfig,
        device: str = "cuda",
        backbone_name: Optional[str] = None,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.cfg = cfg.reward_model
        self.device = device
        # R2/RT2 fix: store backbone_name so _save can write metadata.json,
        # enabling load_reward_model_and_tokenizer to reconstruct the model
        # standalone from a checkpoint directory.
        self.backbone_name = backbone_name

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """Train the reward model on D_l.

        Args:
            train_dataset:          training dataset
            eval_dataset:           optional eval dataset
            resume_from_checkpoint: path to a checkpoint directory produced by _save()
                                    (e.g. ``outputs/cwpo/reward_model/checkpoint-500``).
                                    Resumes model weights, optimizer state, and
                                    skips steps already completed in that run.
        """
        max_length = self.cfg.get("max_length", 512)
        collator = RewardDataCollator(self.tokenizer, max_length=max_length)

        # Tokenize datasets
        train_tok = self._tokenize_dataset(train_dataset, max_length)
        train_loader = DataLoader(
            train_tok,
            batch_size=self.cfg.get("per_device_train_batch_size", 4),
            shuffle=True,
            collate_fn=collator,
            num_workers=self.cfg.get("dataloader_num_workers", 2),
        )

        eval_loader = None
        if eval_dataset is not None:
            eval_tok = self._tokenize_dataset(eval_dataset, max_length)
            eval_loader = DataLoader(
                eval_tok,
                batch_size=self.cfg.get("per_device_eval_batch_size", 4),
                shuffle=False,
                collate_fn=collator,
            )

        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.cfg.get("learning_rate", 1e-5)),
            weight_decay=float(self.cfg.get("weight_decay", 0.01)),
        )

        num_epochs = self.cfg.get("num_train_epochs", 5)
        num_steps = len(train_loader) * num_epochs
        warmup_steps = int(num_steps * self.cfg.get("warmup_ratio", 0.1))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps
        )

        # ── Resume from checkpoint ─────────────────────────────────────────
        start_global_step = 0
        if resume_from_checkpoint is not None:
            weights_path = os.path.join(resume_from_checkpoint, "model.pt")
            opt_path     = os.path.join(resume_from_checkpoint, "optimizer_state.pt")
            state_path   = os.path.join(resume_from_checkpoint, "trainer_state.json")

            if os.path.isfile(weights_path):
                logger.info(f"Resuming reward model weights from: {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                logger.warning(
                    f"resume_from_checkpoint set but model.pt not found at {weights_path}. "
                    "Starting from current model weights."
                )

            if os.path.isfile(opt_path):
                logger.info(f"Resuming optimizer state from: {opt_path}")
                opt_state = torch.load(opt_path, map_location=self.device)
                optimizer.load_state_dict(opt_state)
            else:
                logger.warning(
                    f"optimizer_state.pt not found at {opt_path}. "
                    "Optimizer will start fresh (LR schedule may be inconsistent)."
                )

            if os.path.isfile(state_path):
                import json as _json
                with open(state_path) as _f:
                    _state = _json.load(_f)
                start_global_step = int(_state.get("global_step", 0))
                logger.info(
                    f"Resuming from global_step={start_global_step} "
                    f"(epoch ≈ {start_global_step // max(1, len(train_loader))})"
                )

        logging_steps = self.cfg.get("logging_steps", 50)
        save_steps = self.cfg.get("save_steps", 500)
        output_dir = self.cfg.get("output_dir", "outputs/reward_model")
        os.makedirs(output_dir, exist_ok=True)

        global_step = 0
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            steps_this_epoch = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                global_step += 1
                # Skip steps already completed before the resume point
                if global_step <= start_global_step:
                    continue
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                chosen_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_mask = batch["rejected_attention_mask"].to(self.device)

                score_chosen = self.model(chosen_ids, chosen_mask)
                score_rejected = self.model(rejected_ids, rejected_mask)

                loss = self.model.bradley_terry_loss(score_chosen, score_rejected)

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
                    self._save(output_dir, global_step, optimizer=optimizer)

            logger.info(
                f"Epoch {epoch+1} avg loss: "
                f"{epoch_loss / max(1, steps_this_epoch):.4f}"
            )

            if eval_loader is not None:
                acc = self._evaluate(eval_loader)
                logger.info(f"Epoch {epoch+1} eval preference accuracy: {acc:.4f}")

        # Final save
        self._save(output_dir, "final", optimizer=optimizer)
        logger.info(f"Reward model saved to {output_dir}/final")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _tokenize_dataset(self, dataset: Dataset, max_length: int) -> List[Dict]:
        tokenized = []
        for sample in dataset:
            tokenized.append(
                tokenize_for_reward_model(sample, self.tokenizer, max_length)
            )
        return tokenized

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        """Compute preference accuracy on eval set."""
        self.model.eval()
        correct = total = 0
        for batch in loader:
            chosen_ids = batch["chosen_input_ids"].to(self.device)
            chosen_mask = batch["chosen_attention_mask"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_mask = batch["rejected_attention_mask"].to(self.device)

            s_chosen = self.model(chosen_ids, chosen_mask)
            s_rejected = self.model(rejected_ids, rejected_mask)

            correct += (s_chosen > s_rejected).sum().item()
            total += len(s_chosen)

        self.model.train()
        return correct / max(1, total)

    def _save(self, output_dir: str, step, optimizer=None) -> None:
        """
        R2/RT2 fix: Save full checkpoint — state dict + tokenizer + metadata.

        Saves to: output_dir/checkpoint-{step}/
            - model.pt           : ScalarRewardModel state_dict (weights)
            - optimizer_state.pt : AdamW state dict for exact resume
            - trainer_state.json : global_step for resume bookkeeping
            - tokenizer_*        : tokenizer files via save_pretrained()
            - metadata.json      : backbone_name, scalar_head config for
                                   standalone reconstruction

        This allows load_reward_model_and_tokenizer() to load the checkpoint
        from the directory path without needing external backbone_name.
        """
        import json
        save_path = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)

        # 1. Save model weights (state dict)
        weights_path = os.path.join(save_path, "model.pt")
        torch.save(self.model.state_dict(), weights_path)

        # 2. Save optimizer state (enables exact resume of LR schedule)
        if optimizer is not None:
            opt_path = os.path.join(save_path, "optimizer_state.pt")
            torch.save(optimizer.state_dict(), opt_path)

        # 3. Save tokenizer (captures any pad_token / vocab modifications)
        try:
            self.tokenizer.save_pretrained(save_path)
        except Exception as e:
            logger.warning(f"Could not save tokenizer: {e}")

        # 4. Save trainer state for resume bookkeeping
        trainer_state = {"global_step": step if isinstance(step, int) else 0}
        trainer_state_path = os.path.join(save_path, "trainer_state.json")
        with open(trainer_state_path, "w") as f:
            json.dump(trainer_state, f, indent=2)

        # 5. Save metadata for standalone checkpoint loading
        hidden_size = self.model.backbone.config.hidden_size
        metadata = {
            "backbone_name": self.backbone_name,
            "hidden_size": hidden_size,
            "step": str(step),
        }
        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Saved reward model checkpoint to {save_path} "
            f"(model.pt + optimizer_state.pt + trainer_state.json + tokenizer + metadata.json)"
        )
