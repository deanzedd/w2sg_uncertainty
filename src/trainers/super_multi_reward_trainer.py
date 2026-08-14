"""
Super_multi_dpo Phase 2 Trainer — LoRA Ensemble Reward Model Training.

Trains K LoRA-adapted reward models either sequentially (default) or in parallel,
starting from a MultiHeadRewardModel warmed up in Phase 1.

Training modes (configurable via super_multi.train_mode):
    "sequential" (default): train model_0 → save → free VRAM → train model_1 → ...
                            VRAM: 1 × (LoRA backbone + head), identical results to parallel
    "parallel":             train all K models simultaneously in same training loop
                            VRAM: K × (LoRA backbone + head), faster wall-clock

The BT loss and convergence are IDENTICAL between modes since K models are
trained independently with no gradient coupling.

Architecture for each LoRA model k:
    frozen_backbone + LoRA_k (trainable, ~r*d params) + head_k (trainable, ~d params)
    
Optional bootstrap masking in Phase 2:
    use_bootstrap=True → each model k sees a bootstrap resample of the batch
                         (additional decorrelation on top of LoRA random init)
    use_bootstrap=False → all models see full batch (LoRA diversity only)
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from ..data.utils import RewardDataCollator, tokenize_for_reward_model
from ..models.multi_head_reward_model import (
    MultiHeadRewardModel,
    LoRARewardModel,
    save_lora_reward_models,
)

logger = logging.getLogger(__name__)


class SuperMultiRewardTrainer:
    """
    Phase 2 Trainer for Super_multi_dpo.

    Starting from a warmed-up MultiHeadRewardModel, this trainer:
        1. Calls model.attach_lora_adapters() → K (lora_backbone_k, head_k) pairs
        2. Trains each pair independently with Bradley-Terry loss
        3. Saves K LoRA reward models to output_dir/model_{k}/checkpoint-final/

    The saved models are compatible with MultiWeakLabeler via load_lora_reward_model().

    Args:
        phase1_model:           MultiHeadRewardModel (warmed up, Phase 1)
        tokenizer:              shared tokenizer
        cfg:                    full OmegaConf DictConfig (reads reward_model + super_multi)
        device:                 compute device
        backbone_name:          HF model ID
        use_bootstrap:          if True, each model trains on bootstrap resample of batch
        train_mode:             "sequential" (default) or "parallel"
        lora_r:                 LoRA rank
        lora_alpha:             LoRA alpha
        lora_dropout:           LoRA dropout (adds diversity on top of different random init)
        lora_target_modules:    LoRA target modules (None = PEFT auto-detect)
        phase2_epochs:          number of training epochs for Phase 2
        phase2_lr:              learning rate for Phase 2 (LoRA adapters + heads)
    """

    def __init__(
        self,
        phase1_model: MultiHeadRewardModel,
        tokenizer,
        cfg,
        device: str = "cuda",
        backbone_name: Optional[str] = None,
        use_bootstrap: bool = False,
        train_mode: str = "sequential",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        phase2_epochs: int = 2,
        phase2_lr: float = 2e-4,
    ) -> None:
        if train_mode not in ("sequential", "parallel"):
            raise ValueError(f"train_mode must be 'sequential' or 'parallel', got '{train_mode}'")

        self.phase1_model = phase1_model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.rm_cfg = cfg.reward_model
        self.device = device
        self.backbone_name = backbone_name
        self.use_bootstrap = use_bootstrap
        self.train_mode = train_mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.phase2_epochs = phase2_epochs
        self.phase2_lr = phase2_lr

    # ------------------------------------------------------------------ #
    #  Public: train                                                        #
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        Run Phase 2 training.

        1. Attach K LoRA adapters to the Phase 1 backbone.
        2. Train according to train_mode (sequential or parallel).
        3. Save K LoRA reward models.
        """
        output_dir = self.rm_cfg.get("output_dir", "outputs/reward_model")

        # ── Build tokenised data loaders ──────────────────────────────────
        max_length = self.rm_cfg.get("max_length", 512)
        collator = RewardDataCollator(self.tokenizer, max_length=max_length)
        batch_size = int(self.rm_cfg.get("per_device_train_batch_size", 16))

        train_tok = [
            tokenize_for_reward_model(s, self.tokenizer, max_length)
            for s in train_dataset
        ]
        train_loader = DataLoader(
            train_tok,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.rm_cfg.get("dataloader_num_workers", 2),
        )

        eval_loader = None
        if eval_dataset is not None:
            eval_tok = [
                tokenize_for_reward_model(s, self.tokenizer, max_length)
                for s in eval_dataset
            ]
            eval_loader = DataLoader(
                eval_tok,
                batch_size=int(self.rm_cfg.get("per_device_eval_batch_size", 16)),
                shuffle=False,
                collate_fn=collator,
            )

        # ── Attach LoRA adapters ──────────────────────────────────────────
        logger.info("=" * 60)
        logger.info(
            f"Phase 2 (Ensemble): attaching {self.phase1_model.num_heads} LoRA adapters "
            f"[r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}]"
        )
        logger.info(f"  train_mode   = {self.train_mode}")
        logger.info(f"  use_bootstrap= {self.use_bootstrap}")
        logger.info(f"  phase2_epochs= {self.phase2_epochs}, phase2_lr= {self.phase2_lr}")
        logger.info("=" * 60)

        lora_pairs = self.phase1_model.attach_lora_adapters(
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_target_modules=self.lora_target_modules,
        )

        # ── Train ─────────────────────────────────────────────────────────
        if self.train_mode == "sequential":
            self._train_sequential(lora_pairs, train_loader, eval_loader, output_dir)
        else:
            self._train_parallel(lora_pairs, train_loader, eval_loader, output_dir)

        # ── Save all K models ─────────────────────────────────────────────
        # Note: sequential mode saves each model inline; parallel saves all here.
        if self.train_mode == "parallel":
            logger.info("Saving all K LoRA reward models...")
            save_lora_reward_models(
                lora_pairs, self.tokenizer, output_dir, self.backbone_name
            )

        logger.info("=" * 60)
        logger.info(f"Phase 2 complete. {len(lora_pairs)} LoRA models saved to {output_dir}/")
        logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    #  Sequential training                                                 #
    # ------------------------------------------------------------------ #

    def _train_sequential(
        self,
        lora_pairs: List[Tuple[nn.Module, nn.Module]],
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        output_dir: str,
    ) -> None:
        """
        Train K LoRA models one after another.
        VRAM: 1 × (LoRA backbone + head) at a time.
        Saves each model immediately and frees VRAM before next.
        """
        K = len(lora_pairs)
        saved_pairs: List[Tuple[nn.Module, nn.Module]] = []

        for k, (lora_backbone_k, head_k) in enumerate(lora_pairs):
            logger.info(f"═══ Sequential: training LoRA model {k+1}/{K} ═══")
            model_k = LoRARewardModel(lora_backbone_k, head_k).to(self.device)

            self._train_single_model(model_k, k, train_loader, eval_loader)

            # Save immediately (no need to keep in VRAM)
            model_k.cpu()
            save_lora_reward_models(
                [(lora_backbone_k, head_k)],
                self.tokenizer,
                output_dir.rstrip("/") + f"_tmp_{k}",  # temp, will rename
                self.backbone_name,
            )
            # Move saved files to correct model_{k} path
            import shutil
            src = os.path.join(output_dir.rstrip("/") + f"_tmp_{k}", "model_0", "checkpoint-final")
            dst = os.path.join(output_dir, f"model_{k}", "checkpoint-final")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
            shutil.rmtree(output_dir.rstrip("/") + f"_tmp_{k}", ignore_errors=True)

            del model_k
            torch.cuda.empty_cache()
            logger.info(f"  LoRA model {k} saved → {dst}. VRAM freed.")

    # ------------------------------------------------------------------ #
    #  Parallel training                                                   #
    # ------------------------------------------------------------------ #

    def _train_parallel(
        self,
        lora_pairs: List[Tuple[nn.Module, nn.Module]],
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        output_dir: str,
    ) -> None:
        """
        Train all K LoRA models simultaneously.
        VRAM: K × (LoRA backbone + head) — higher but faster wall-clock.
        Results are IDENTICAL to sequential training.
        """
        K = len(lora_pairs)
        logger.info(f"Parallel mode: moving {K} LoRA models to device {self.device}...")
        lora_models = [
            LoRARewardModel(bb, h).to(self.device)
            for bb, h in lora_pairs
        ]

        # Build K optimizers and schedulers
        num_steps = len(train_loader) * self.phase2_epochs
        warmup_steps = int(num_steps * self.rm_cfg.get("warmup_ratio", 0.05))

        optimizers = []
        schedulers = []
        for k, lm in enumerate(lora_models):
            trainable = [p for p in lm.parameters() if p.requires_grad]
            opt = AdamW(trainable, lr=self.phase2_lr,
                        weight_decay=float(self.rm_cfg.get("weight_decay", 0.01)))
            sch = get_cosine_schedule_with_warmup(opt, warmup_steps, num_steps)
            optimizers.append(opt)
            schedulers.append(sch)

        logging_steps = int(self.rm_cfg.get("logging_steps", 50))

        for epoch in range(self.phase2_epochs):
            logger.info(f"Epoch {epoch+1}/{self.phase2_epochs} [parallel, K={K}]")
            for lm in lora_models:
                lm.train()

            global_step = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                global_step += 1
                chosen_ids   = batch["chosen_input_ids"].to(self.device)
                chosen_mask  = batch["chosen_attention_mask"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_mask= batch["rejected_attention_mask"].to(self.device)
                batch_size_n = chosen_ids.size(0)

                total_loss = 0.0
                for k, (lm, opt, sch) in enumerate(zip(lora_models, optimizers, schedulers)):
                    if self.use_bootstrap:
                        mask = MultiHeadRewardModel.make_bootstrap_masks(
                            batch_size_n, 1, device=self.device
                        )[0]
                        s_c = lm(chosen_ids[mask], chosen_mask[mask])
                        s_r = lm(rejected_ids[mask], rejected_mask[mask])
                    else:
                        s_c = lm(chosen_ids, chosen_mask)
                        s_r = lm(rejected_ids, rejected_mask)

                    loss_k = lm.bradley_terry_loss(s_c, s_r)
                    loss_k.backward()
                    nn.utils.clip_grad_norm_(
                        [p for p in lm.parameters() if p.requires_grad],
                        float(self.rm_cfg.get("max_grad_norm", 1.0))
                    )
                    opt.step()
                    sch.step()
                    opt.zero_grad()
                    total_loss += loss_k.item()

                if global_step % logging_steps == 0:
                    logger.info(
                        f"Step {global_step} | avg_loss/K={total_loss/K:.4f} "
                        f"| LR={schedulers[0].get_last_lr()[0]:.2e}"
                    )

            if eval_loader is not None:
                for k, lm in enumerate(lora_models):
                    acc = self._evaluate_single(lm, eval_loader)
                    logger.info(f"  Epoch {epoch+1} | Model {k} eval acc: {acc:.4f}")

        # Move lora_pairs back to CPU before saving
        for lm in lora_models:
            lm.cpu()

    # ------------------------------------------------------------------ #
    #  Single model training (used by sequential mode)                     #
    # ------------------------------------------------------------------ #

    def _train_single_model(
        self,
        model: LoRARewardModel,
        model_idx: int,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
    ) -> None:
        """Train a single LoRARewardModel with BT loss."""
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        logger.info(f"  Model {model_idx}: {n_trainable:,} trainable params")

        num_steps = len(train_loader) * self.phase2_epochs
        warmup_steps = int(num_steps * self.rm_cfg.get("warmup_ratio", 0.05))

        optimizer = AdamW(
            trainable,
            lr=self.phase2_lr,
            weight_decay=float(self.rm_cfg.get("weight_decay", 0.01)),
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps)

        logging_steps = int(self.rm_cfg.get("logging_steps", 50))
        global_step = 0

        for epoch in range(self.phase2_epochs):
            model.train()
            epoch_loss = 0.0
            n_steps = 0

            for batch in tqdm(train_loader, desc=f"Model {model_idx} Epoch {epoch+1}/{self.phase2_epochs}"):
                global_step += 1
                chosen_ids   = batch["chosen_input_ids"].to(self.device)
                chosen_mask  = batch["chosen_attention_mask"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_mask= batch["rejected_attention_mask"].to(self.device)

                if self.use_bootstrap:
                    # Bootstrap: random resample of batch (with replacement)
                    bs = chosen_ids.size(0)
                    mask = MultiHeadRewardModel.make_bootstrap_masks(bs, 1, device=self.device)[0]
                    if mask.sum() == 0:
                        mask[0] = True  # safety: at least 1 sample
                    s_c = model(chosen_ids[mask], chosen_mask[mask])
                    s_r = model(rejected_ids[mask], rejected_mask[mask])
                else:
                    s_c = model(chosen_ids, chosen_mask)
                    s_r = model(rejected_ids, rejected_mask)

                loss = model.bradley_terry_loss(s_c, s_r)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    trainable, float(self.rm_cfg.get("max_grad_norm", 1.0))
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                n_steps += 1

                if global_step % logging_steps == 0:
                    logger.info(
                        f"  Model {model_idx} | Step {global_step} | "
                        f"loss={loss.item():.4f} | LR={scheduler.get_last_lr()[0]:.2e}"
                    )

            avg_loss = epoch_loss / max(1, n_steps)
            logger.info(f"  Model {model_idx} | Epoch {epoch+1} avg loss: {avg_loss:.4f}")

            if eval_loader is not None:
                acc = self._evaluate_single(model, eval_loader)
                logger.info(f"  Model {model_idx} | Epoch {epoch+1} eval acc: {acc:.4f}")

    # ------------------------------------------------------------------ #
    #  Eval                                                                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _evaluate_single(self, model: LoRARewardModel, loader: DataLoader) -> float:
        """Preference accuracy for one LoRARewardModel."""
        model.eval()
        correct = total = 0
        for batch in loader:
            chosen_ids   = batch["chosen_input_ids"].to(self.device)
            chosen_mask  = batch["chosen_attention_mask"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_mask= batch["rejected_attention_mask"].to(self.device)

            s_c = model(chosen_ids, chosen_mask)
            s_r = model(rejected_ids, rejected_mask)
            correct += (s_c > s_r).sum().item()
            total   += s_c.size(0)

        model.train()
        return correct / max(1, total)
