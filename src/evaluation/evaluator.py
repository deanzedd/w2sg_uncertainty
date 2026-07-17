"""
Unified Evaluator — runs all evaluation metrics in sequence.

Pipeline:
1. Generate responses from aligned model and SFT baseline
2. Compute GRA (Gold Reward Accuracy)
   - HH-RLHF / UFB  → Skywork/Skywork-Reward-V2-Llama-3.1-8B
                       (SkyworkChatRewardAdapter, device_map="auto")
   - TL;DR          → OpenAssistant/reward-model-deberta-v3-large-v2
                       (ClassificationRewardAdapter, single GPU)
3. Compute GPT-4 Win Rate (optional, requires API key)
4. Save all results to output_dir
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Sequence

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import preference_accuracy
from .reward_model_eval import RewardModelEvaluator
from .gpt4_eval import GPT4Evaluator


# ── Dataset → GRA reward model mapping (paper setting) ───────────────────────
# HH-RLHF và UFB: dùng Skywork-Reward-V2 (LLM-based, chat template)
# TL;DR           : dùng OpenAssistant DeBERTa (classification, concat format)
DATASET_REWARD_MODEL_MAP: dict = {
    "hh_rlhf": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "ufb":     "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "tldr":    "OpenAssistant/reward-model-deberta-v3-large-v2",
}

_DEFAULT_REWARD_MODEL = "OpenAssistant/reward-model-deberta-v3-large-v2"


def _select_reward_model(cfg) -> str:
    """
    Chọn reward model cho GRA theo thứ tự ưu tiên:
      1. eval.reward_model_name trong config (nếu user set thủ công)
      2. DATASET_REWARD_MODEL_MAP theo dataset_name
      3. Fallback về OpenAssistant DeBERTa
    """
    # Ưu tiên override thủ công từ config
    manual = cfg.get("eval", {}).get("reward_model_name", None)
    if manual:  # None hoặc chuỗi rỗng → bỏ qua
        logger.info(f"[GRA] Using manually configured reward model: {manual}")
        return manual

    # Auto-select theo dataset
    dataset_name = cfg.get("dataset_name", "")
    model_name = DATASET_REWARD_MODEL_MAP.get(dataset_name, _DEFAULT_REWARD_MODEL)
    logger.info(
        f"[GRA] Auto-selected reward model for dataset='{dataset_name}': {model_name}"
    )
    return model_name

logger = logging.getLogger(__name__)


# ── Max prompt length per dataset (evaluation generation) ───────────────────
# HH-RLHF / UFB: dialogue ~400 words → 512 tokens đủ
# TL;DR         : Reddit posts ~200-850 words → cần 1024 tokens
_DATASET_MAX_PROMPT_LEN: dict = {
    "hh_rlhf": 512,
    "ufb":     512,
    "tldr":    1024,
}

# ── Stop strings per dataset ──────────────────────────────────────────────────
# HH-RLHF / UFB sử dụng format raw text: "\n\nHuman: ...\n\nAssistant:"
# Model phải dừng trước khi bắt đầu fake Human turn tiếp theo.
# TL;DR dùng format "SUBREDDIT: ... TL;DR:" → không cần stop string.
_DATASET_STOP_STRINGS: dict = {
    "hh_rlhf": ["\n\nHuman:"],
    "ufb":     ["\n\nHuman:"],
    "tldr":    [],
}


def _get_stop_strings(cfg) -> List[str]:
    """
    Trả về danh sách stop strings phù hợp với dataset.
    Ưu tiên: eval.stop_strings (manual) > _DATASET_STOP_STRINGS > [].
    """
    manual = cfg.get("eval", {}).get("stop_strings", None)
    if manual is not None:  # cho phép override thủ công từ config
        logger.info(f"[generate] Using manually configured stop_strings: {manual}")
        return list(manual)
    dataset_name = cfg.get("dataset_name", "")
    stops = _DATASET_STOP_STRINGS.get(dataset_name, [])
    if stops:
        logger.info(
            f"[generate] Auto stop_strings for dataset='{dataset_name}': {stops}"
        )
    return stops


def _get_eval_max_prompt_length(cfg) -> int:
    """
    Chọn max_prompt_length cho evaluation generation theo thứ tự ưu tiên:
      1. eval.max_prompt_length (manual override trong config)
      2. _DATASET_MAX_PROMPT_LEN theo dataset_name
      3. cfg.max_prompt_length (training value)
      4. Fallback = 512
    """
    # 1. Eval-specific override
    eval_override = cfg.get("eval", {}).get("max_prompt_length", None)
    if eval_override is not None:
        return int(eval_override)
    # 2. Auto-select theo dataset
    dataset_name = cfg.get("dataset_name", "")
    auto = _DATASET_MAX_PROMPT_LEN.get(dataset_name, None)
    if auto is not None:
        logger.info(
            f"[generate] Auto max_prompt_length for dataset='{dataset_name}': {auto}"
        )
        return auto
    # 3. Fallback tới training config hoặc 512
    return int(cfg.get("max_prompt_length", None) or 512)


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
        device_map: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.device_map = device_map  # lưu lại để _generate_responses biết cách move tensors
        self.eval_cfg = cfg.eval

        # EV1 fix: support device_map for large models (7B+) to avoid OOM.
        # With device_map="auto", HF shards model layers across all available GPUs.
        # With device_map=None (default), model is loaded entirely on `device`.
        # .to(device) is only called when NOT using device_map (can conflict with sharding).
        _load_kwargs: dict = {"torch_dtype": torch.bfloat16}
        if device_map is not None:
            _load_kwargs["device_map"] = device_map

        logger.info(f"Loading aligned model from {aligned_model_path}")
        self.aligned_model = AutoModelForCausalLM.from_pretrained(
            aligned_model_path, **_load_kwargs
        )
        if device_map is None:
            self.aligned_model = self.aligned_model.to(device)
        self.aligned_model.eval()

        logger.info(f"Loading SFT model from {sft_model_path}")
        self.sft_model = AutoModelForCausalLM.from_pretrained(
            sft_model_path, **_load_kwargs
        )
        if device_map is None:
            self.sft_model = self.sft_model.to(device)
        self.sft_model.eval()

        # Use the aligned model's tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(aligned_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # EV3 fix: left-padding is the standard for batched generation.
        # With right-padding, the model sees padding AFTER the prompt, which
        # can confuse EOS detection and produce truncated / blank responses.
        self.tokenizer.padding_side = "left"


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
        # Số samples được giới hạn bởi --max_eval_samples trong evaluate.py (default=500)
        # gen_batch_size: dùng int() + or-fallback để handle OmegaConf None values
        gen_batch_size = int(self.eval_cfg.get("gen_batch_size", None) or 4)
        logger.info(
            f"Generating responses from aligned model "
            f"(n={len(prompts)}, batch_size={gen_batch_size})..."
        )
        aligned_responses = self._generate_responses(
            self.aligned_model, prompts, batch_size=gen_batch_size
        )

        logger.info(f"Generating responses from SFT model (batch_size={gen_batch_size})...")
        sft_responses = self._generate_responses(
            self.sft_model, prompts, batch_size=gen_batch_size
        )

        # Save generated responses
        responses_path = os.path.join(output_dir, "generated_responses.json")
        with open(responses_path, "w") as f:
            json.dump(
                [{"prompt": p, "aligned": a, "sft": s}
                 for p, a, s in zip(prompts, aligned_responses, sft_responses)],
                f, indent=2, ensure_ascii=False,
            )

        # ── Step 3: GRA ──────────────────────────────────────────────────
        # Auto-select reward model theo dataset (có thể override qua config)
        gra_model_name = _select_reward_model(self.cfg)
        gra_device_map = self.eval_cfg.get("gra_device_map", "auto")
        gra_batch_size = self.eval_cfg.get("gra_batch_size", 8)  # adapter tự điều chỉnh cho Skywork

        gra_evaluator = RewardModelEvaluator(
            reward_model_name=gra_model_name,
            device=self.device,
            device_map=gra_device_map,
            batch_size=gra_batch_size,
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
        max_new_tokens: int = 512,
        batch_size: int = 4,
    ) -> List[str]:
        """Generate responses for a list of prompts."""
        # Paper spec: temperature=0.95 (sampling), max_new_tokens=512
        # Read from eval config; fall back to paper defaults.
        gen_max_new_tokens = int(
            self.eval_cfg.get("gen_max_new_tokens", None) or max_new_tokens
        )
        gen_temperature = float(
            self.eval_cfg.get("gen_temperature", None) or 0.95
        )
        gen_do_sample = bool(
            self.eval_cfg.get("gen_do_sample", True)
        )
        gen_top_p = float(
            self.eval_cfg.get("gen_top_p", None) or 1.0
        )

        # EV4 fix: stop_strings để ngăn multi-turn hallucination.
        #
        # Vấn đề: HH-RLHF dùng raw-text format "\n\nHuman: ...\n\nAssistant:"
        # Model học pattern này và tiếp tục sinh ra fake Human turns sau khi
        # kết thúc assistant response → hallucinate toàn bộ cuộc hội thoại.
        # Kết quả: 48.7% responses bị hallucinate → reward model chấm điểm thấp
        # → GRA giảm xuống 0.368 (dưới cả random baseline 0.5).
        #
        # Fix: truyền stop_strings=["\n\nHuman:"] vào generate().
        # HF sẽ truncate output ngay khi gặp chuỗi này.
        # stop_strings yêu cầu tokenizer được pass vào generate().
        stop_strings: List[str] = _get_stop_strings(self.cfg)

        logger.info(
            f"Generation settings: do_sample={gen_do_sample}, "
            f"temperature={gen_temperature}, top_p={gen_top_p}, "
            f"max_new_tokens={gen_max_new_tokens}, "
            f"stop_strings={stop_strings}"
        )

        # Khi device_map="auto", model layers nằm trên nhiều GPU khác nhau.
        # Input tensor phải được move tới device của embedding layer (GPU đầu tiên).
        # next(model.parameters()).device trả về đúng điều này trong mọi trường hợp.
        input_device = next(model.parameters()).device
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        responses = []
        for i in tqdm(
            range(0, len(prompts), batch_size),
            desc="Generating",
            total=n_batches,
        ):
            batch = prompts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=_get_eval_max_prompt_length(self.cfg),
            ).to(input_device)  # move tới GPU chứa embedding layer của model

            # Xây dựng generate kwargs
            gen_kwargs = dict(
                max_new_tokens=gen_max_new_tokens,  # 512 per paper spec
                do_sample=gen_do_sample,             # True per paper spec
                temperature=gen_temperature,          # 0.95 per paper spec
                top_p=gen_top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            # stop_strings yêu cầu tokenizer được truyền vào generate()
            # (HF >= 4.40). Chỉ add nếu có stop strings để tránh overhead.
            if stop_strings:
                gen_kwargs["stop_strings"] = stop_strings
                gen_kwargs["tokenizer"] = self.tokenizer

            outputs = model.generate(**enc, **gen_kwargs)

            # EV3 fix: decode per-sample using the padded input length.
            #
            # With left-padded inputs (padding_side="left"), the generate() output
            # has shape (batch_size, padded_input_len + max_new_tokens).
            # New tokens are appended starting at exactly padded_input_len (= shape[1])
            # for ALL samples in the batch, regardless of individual prompt lengths.
            #
            # Using enc["input_ids"].shape[1] (padded batch length) is correct.
            # Per-sample attention_mask.sum() would be WRONG here because it gives
            # the number of non-pad tokens, not the position where new tokens start.
            padded_input_len = enc["input_ids"].shape[1]
            for out in outputs:
                new_tokens = out[padded_input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                # EV4 fix: post-hoc truncation dự phòng nếu stop_strings không
                # catch được (transformers < 4.40 hoặc tokenizer mismatch).
                # Cắt tại \n\nHuman: đầu tiên trong decoded text.
                for stop in stop_strings:
                    if stop in text:
                        text = text[: text.index(stop)]
                responses.append(text.strip())

        return responses
