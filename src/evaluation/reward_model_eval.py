"""
Gold Reward Accuracy (GRA) Evaluator.

Hỗ trợ hai loại reward model với input format khác nhau:

1. ClassificationRewardAdapter — DeBERTa-style (AutoModelForSequenceClassification):
   - Input: "prompt response" (concat đơn giản)
   - Ví dụ: OpenAssistant/reward-model-deberta-v3-large-v2
   - Dùng cho: TL;DR dataset

2. SkyworkChatRewardAdapter — LLM chat-based (AutoModelForSequenceClassification với chat template):
   - Input: apply_chat_template([user: prompt, assistant: response])
   - device_map="auto" để load 8B model qua nhiều GPU
   - Ví dụ: Skywork/Skywork-Reward-V2-Llama-3.1-8B
   - Dùng cho: HH-RLHF, UFB datasets

GRA = P(r(aligned_response) > r(sft_response))
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .metrics import gold_reward_accuracy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base Adapter
# ─────────────────────────────────────────────────────────────────────────────

class BaseRewardAdapter(ABC):
    """
    Abstract base class cho reward model adapters.
    Mỗi subclass định nghĩa cách format input và score (prompt, response) pairs.
    """

    @abstractmethod
    def score(
        self,
        prompts: List[str],
        responses: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        """
        Score a list of (prompt, response) pairs.

        Args:
            prompts:   list of prompt strings
            responses: list of response strings (same length as prompts)
            batch_size: inference batch size

        Returns:
            list of scalar reward scores (float)
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Adapter 1: Classification-style (DeBERTa / OpenAssistant)
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationRewardAdapter(BaseRewardAdapter):
    """
    Adapter cho DeBERTa-style reward models (AutoModelForSequenceClassification).

    Input format: prompt + " " + response  (simple concatenation)

    Recommended for:
        - OpenAssistant/reward-model-deberta-v3-large-v2  (TL;DR dataset)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.device = device
        self.max_length = max_length

        logger.info(f"[ClassificationAdapter] Loading reward model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
        ).to(device)
        self.model.eval()
        logger.info(f"[ClassificationAdapter] Model loaded on device: {device}")

    @torch.no_grad()
    def score(
        self,
        prompts: List[str],
        responses: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        """
        Score bằng cách concat 'prompt + response' và forward qua classifier.
        """
        all_scores: List[float] = []

        for i in tqdm(
            range(0, len(prompts), batch_size),
            desc="[ClassificationAdapter] Scoring",
            leave=False,
        ):
            batch_prompts = prompts[i : i + batch_size]
            batch_responses = responses[i : i + batch_size]

            # Format: "<prompt> <response>"
            texts = [f"{p} {r}" for p, r in zip(batch_prompts, batch_responses)]

            enc = self.tokenizer(
                texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**enc)
            # Shape: (batch, num_labels) — squeeze sang scalar
            scores = outputs.logits.squeeze(-1).cpu().tolist()

            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        return all_scores


# ─────────────────────────────────────────────────────────────────────────────
# Adapter 2: Skywork Chat-style (LLM-based reward, chat template)
# ─────────────────────────────────────────────────────────────────────────────

class SkyworkChatRewardAdapter(BaseRewardAdapter):
    """
    Adapter cho Skywork-Reward-V2 (và các LLM-based reward models khác).

    Input format: apply_chat_template([user: prompt, assistant: response])
    KHÔNG dùng system prompt theo khuyến nghị của Skywork team.

    Device handling:
        - Dùng device_map="auto" để phân bổ 8B model qua nhiều GPU tự động.
        - Không gọi .to(device) sau khi load vì device_map đã handle.

    Recommended for:
        - Skywork/Skywork-Reward-V2-Llama-3.1-8B  (HH-RLHF, UFB datasets)
        - Skywork/Skywork-Reward-V2-Qwen3-8B
        - Skywork/Skywork-Reward-V2-Qwen3-1.7B
    """

    def __init__(
        self,
        model_name: str,
        device_map: Union[str, None] = "auto",
        max_length: int = 4096,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.device_map = device_map
        self.max_length = max_length

        logger.info(f"[SkyworkAdapter] Loading reward model: {model_name}")
        logger.info(f"[SkyworkAdapter] device_map={device_map}, max_length={max_length}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        # Skywork-V2 yêu cầu num_labels=1 để output scalar reward score
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            num_labels=1,
        )
        self.model.eval()

        # Xác định device để move tensors về đúng chỗ
        # Khi device_map="auto", model.device trả về device của embedding layer
        self._input_device = next(self.model.parameters()).device
        logger.info(
            f"[SkyworkAdapter] Model loaded. Input device: {self._input_device}"
        )

    def _format_single(self, prompt: str, response: str) -> str:
        """
        Format một cặp (prompt, response) thành chat template string.

        Theo Skywork-V2 paper:
          - Không có system prompt
          - user = prompt, assistant = response
        """
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        # add_generation_prompt=False vì đây là complete conversation
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    @torch.no_grad()
    def score(
        self,
        prompts: List[str],
        responses: List[str],
        batch_size: int = 1,
    ) -> List[float]:
        """
        Score bằng chat template format và forward qua LLM-based classifier.

        Note: batch_size mặc định = 1 vì Skywork-8B tiêu tốn nhiều VRAM.
              Với max_length=4096 và batch=1 cần ~16-18 GB VRAM.
        """
        all_scores: List[float] = []

        for i in tqdm(
            range(0, len(prompts), batch_size),
            desc="[SkyworkAdapter] Scoring",
            leave=False,
        ):
            batch_prompts = prompts[i : i + batch_size]
            batch_responses = responses[i : i + batch_size]

            # Format thành chat template strings
            formatted_texts = [
                self._format_single(p, r)
                for p, r in zip(batch_prompts, batch_responses)
            ]

            enc = self.tokenizer(
                formatted_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self._input_device)

            outputs = self.model(**enc)
            # logits shape: (batch, 1) → squeeze → scalar per sample
            scores = outputs.logits.squeeze(-1).cpu().tolist()

            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        return all_scores


# ─────────────────────────────────────────────────────────────────────────────
# RewardModelEvaluator (public interface — giữ nguyên API)
# ─────────────────────────────────────────────────────────────────────────────

# Mapping model name prefix → adapter class
_SKYWORK_V2_PREFIXES = (
    "Skywork/Skywork-Reward-V2",
    "skywork/skywork-reward-v2",
)


def _is_skywork_v2(model_name: str) -> bool:
    """Phát hiện model thuộc Skywork-Reward-V2 series dựa trên tên."""
    lower = model_name.lower()
    return any(lower.startswith(p.lower()) for p in _SKYWORK_V2_PREFIXES)


class RewardModelEvaluator:
    """
    Unified GRA evaluator — tự động chọn adapter phù hợp theo model name.

    - Nếu model_name là Skywork-Reward-V2-*  → dùng SkyworkChatRewardAdapter
    - Ngược lại                               → dùng ClassificationRewardAdapter

    Args:
        reward_model_name: HF model ID
        device:            compute device (dùng cho ClassificationAdapter)
        device_map:        device map (dùng cho SkyworkAdapter, mặc định "auto")
        batch_size:        inference batch size
        max_length:        max token length
        cache_dir:         HF cache directory
    """

    def __init__(
        self,
        reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
        device: str = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.batch_size = batch_size
        self.reward_model_name = reward_model_name

        if _is_skywork_v2(reward_model_name):
            logger.info(
                f"[RewardModelEvaluator] Detected Skywork-Reward-V2 series → "
                f"using SkyworkChatRewardAdapter (chat template, device_map={device_map})"
            )
            self._adapter = SkyworkChatRewardAdapter(
                model_name=reward_model_name,
                device_map=device_map,
                max_length=max(max_length, 4096),  # Skywork cần context dài hơn
                cache_dir=cache_dir,
            )
            # Skywork-8B tiêu tốn VRAM → ép batch_size=1 nếu user không ghi đè
            if batch_size == 8:  # giá trị default → dùng 1 cho Skywork
                self.batch_size = 1
        else:
            logger.info(
                f"[RewardModelEvaluator] Using ClassificationRewardAdapter "
                f"(concat format, device={device})"
            )
            self._adapter = ClassificationRewardAdapter(
                model_name=reward_model_name,
                device=device,
                max_length=max_length,
                cache_dir=cache_dir,
            )

    def score_responses(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[float]:
        """
        Score a list of (prompt, response) pairs.
        Delegate sang adapter tương ứng.
        """
        return self._adapter.score(prompts, responses, batch_size=self.batch_size)

    def compute_gra(
        self,
        prompts: List[str],
        aligned_responses: List[str],
        sft_responses: List[str],
    ) -> Dict[str, float]:
        """
        Compute Gold Reward Accuracy.

        GRA = P(r(aligned) > r(sft))

        Args:
            prompts:            list of eval prompts
            aligned_responses:  responses from aligned model (WDPO/CWPO)
            sft_responses:      responses from SFT baseline

        Returns:
            dict with GRA metrics (gra, avg_aligned_reward, avg_sft_reward)
        """
        logger.info(
            f"[GRA] Scoring {len(prompts)} aligned responses "
            f"with {self.reward_model_name}..."
        )
        aligned_scores = self.score_responses(prompts, aligned_responses)

        logger.info(f"[GRA] Scoring {len(prompts)} SFT responses...")
        sft_scores = self.score_responses(prompts, sft_responses)

        metrics = gold_reward_accuracy(aligned_scores, sft_scores)
        logger.info(
            f"[GRA] Result: GRA={metrics['gra']:.4f} | "
            f"Aligned avg={metrics['avg_aligned_reward']:.4f} | "
            f"SFT avg={metrics['avg_sft_reward']:.4f}"
        )
        return metrics
