"""
Unified Evaluator — runs all evaluation metrics in sequence.

Pipeline:
1. Generate responses from aligned model and SFT baseline
2. Compute GRA (Gold Reward Accuracy)
3. Compute GPT-4 Win Rate (optional, requires API key)
4. Save all results to output_dir
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import preference_accuracy
from .reward_model_eval import RewardModelEvaluator
from .gpt4_eval import GPT4Evaluator

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Unified evaluator: generates responses and computes all metrics.

    Args:
        aligned_model_path: path to aligned model checkpoint (WDPO/CWPO)
        sft_model_path:     path to SFT baseline checkpoint
        cfg:                OmegaConf DictConfig with eval settings
        device:             compute device
    """

    def __init__(
        self,
        aligned_model_path: str,
        sft_model_path: str,
        cfg: DictConfig,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.eval_cfg = cfg.eval

        logger.info(f"Loading aligned model from {aligned_model_path}")
        self.aligned_model = AutoModelForCausalLM.from_pretrained(
            aligned_model_path, torch_dtype=torch.bfloat16
        ).to(device).eval()

        logger.info(f"Loading SFT model from {sft_model_path}")
        self.sft_model = AutoModelForCausalLM.from_pretrained(
            sft_model_path, torch_dtype=torch.bfloat16
        ).to(device).eval()

        # Use the aligned model's tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(aligned_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(
        self,
        eval_dataset,
        run_gpt4: bool = False,
        pseudo_labels: Optional[List[Dict]] = None,
        human_labels: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Run full evaluation pipeline.

        Args:
            eval_dataset:  test dataset with {prompt, chosen, rejected}
            run_gpt4:      whether to run GPT-4 win rate evaluation
            pseudo_labels: weak model pseudo-labels (for preference accuracy)
            human_labels:  human labels for D_l samples (for preference accuracy)

        Returns:
            dict with all evaluation metrics
        """
        output_dir = self.eval_cfg.get("output_dir", "outputs/eval")
        os.makedirs(output_dir, exist_ok=True)
        all_metrics = {}

        # ── Step 1: Extract prompts ──────────────────────────────────────
        samples = list(eval_dataset)
        prompts = [s["prompt"] for s in samples]

        # ── Step 2: Generate responses ───────────────────────────────────
        logger.info("Generating responses from aligned model...")
        aligned_responses = self._generate_responses(self.aligned_model, prompts)

        logger.info("Generating responses from SFT model...")
        sft_responses = self._generate_responses(self.sft_model, prompts)

        # Save generated responses
        responses_path = os.path.join(output_dir, "generated_responses.json")
        with open(responses_path, "w") as f:
            json.dump(
                [{"prompt": p, "aligned": a, "sft": s}
                 for p, a, s in zip(prompts, aligned_responses, sft_responses)],
                f, indent=2, ensure_ascii=False,
            )

        # ── Step 3: GRA ──────────────────────────────────────────────────
        gra_evaluator = RewardModelEvaluator(
            reward_model_name=self.eval_cfg.get(
                "reward_model_name",
                "OpenAssistant/reward-model-deberta-v3-large-v2",
            ),
            device=self.device,
            cache_dir=self.cfg.get("cache_dir", None),
        )
        gra_metrics = gra_evaluator.compute_gra(
            prompts, aligned_responses, sft_responses
        )
        all_metrics["gra"] = gra_metrics
        logger.info(f"GRA results: {gra_metrics}")

        # ── Step 4: GPT-4 Win Rate (optional) ───────────────────────────
        if run_gpt4:
            gpt4_evaluator = GPT4Evaluator(
                model=self.eval_cfg.get("gpt4_model", "gpt-4o"),
                max_samples=self.eval_cfg.get("gpt4_max_samples", 500),
            )
            gpt4_output = os.path.join(output_dir, "gpt4_results.json")
            gpt4_metrics = gpt4_evaluator.evaluate(
                prompts, aligned_responses, sft_responses,
                output_path=gpt4_output,
            )
            all_metrics["gpt4_winrate"] = gpt4_metrics
            logger.info(f"GPT-4 Win Rate results: {gpt4_metrics}")

        # ── Step 5: Preference Accuracy (optional) ───────────────────────
        if pseudo_labels is not None and human_labels is not None:
            pref_acc = preference_accuracy(pseudo_labels, human_labels)
            all_metrics["preference_accuracy"] = pref_acc
            logger.info(f"Preference accuracy: {pref_acc:.4f}")

        # ── Save all metrics ─────────────────────────────────────────────
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"All metrics saved to {metrics_path}")

        return all_metrics

    @torch.no_grad()
    def _generate_responses(
        self,
        model: AutoModelForCausalLM,
        prompts: List[str],
        max_new_tokens: int = 256,
        batch_size: int = 4,
    ) -> List[str]:
        """Generate responses for a list of prompts."""
        responses = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.get("max_prompt_length", 256),
            ).to(self.device)

            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode only the new tokens (exclude prompt)
            prompt_len = enc["input_ids"].shape[1]
            for out in outputs:
                new_tokens = out[prompt_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                responses.append(text.strip())

        return responses
