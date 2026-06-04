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
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.cfg = cfg.reward_model
        self.device = device

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        """Train the reward model on D_l."""
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

        num_epochs = self.cfg.get("num_train_epochs", 2)
        num_steps = len(train_loader) * num_epochs
        warmup_steps = int(num_steps * self.cfg.get("warmup_ratio", 0.1))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps
        )

        logging_steps = self.cfg.get("logging_steps", 50)
        save_steps = self.cfg.get("save_steps", 500)
        output_dir = self.cfg.get("output_dir", "outputs/reward_model")
        os.makedirs(output_dir, exist_ok=True)

        global_step = 0
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
                global_step += 1

                if global_step % logging_steps == 0:
                    logger.info(
                        f"Step {global_step} | Loss: {loss.item():.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

                if global_step % save_steps == 0:
                    self._save(output_dir, global_step)

            logger.info(
                f"Epoch {epoch+1} avg loss: {epoch_loss/len(train_loader):.4f}"
            )

            if eval_loader is not None:
                acc = self._evaluate(eval_loader)
                logger.info(f"Epoch {epoch+1} eval preference accuracy: {acc:.4f}")

        # Final save
        self._save(output_dir, "final")
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

    def _save(self, output_dir: str, step) -> None:
        save_path = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, "model.pt"))
        logger.info(f"Saved reward model checkpoint to {save_path}")
