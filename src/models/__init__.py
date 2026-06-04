from .base_model import BaseModelWrapper
from .opt_model import OPTModelWrapper
from .qwen_model import Qwen25ModelWrapper
from .reward_model import ScalarRewardModel, load_reward_model_and_tokenizer

MODEL_REGISTRY = {
    "facebook/opt-125m": OPTModelWrapper,
    "facebook/opt-350m": OPTModelWrapper,
    "facebook/opt-1.3b": OPTModelWrapper,
    "facebook/opt-2.7b": OPTModelWrapper,
    "facebook/opt-6.7b": OPTModelWrapper,
    "facebook/opt-13b":  OPTModelWrapper,
    "Qwen/Qwen2.5-0.5B": Qwen25ModelWrapper,
    "Qwen/Qwen2.5-0.5B-Instruct": Qwen25ModelWrapper,
    "Qwen/Qwen2.5-1.5B": Qwen25ModelWrapper,
    "Qwen/Qwen2.5-1.5B-Instruct": Qwen25ModelWrapper,
    "Qwen/Qwen2.5-3B": Qwen25ModelWrapper,
    "Qwen/Qwen2.5-3B-Instruct": Qwen25ModelWrapper,
    "Qwen/Qwen2.5-7B": Qwen25ModelWrapper,
    "Qwen/Qwen2.5-7B-Instruct": Qwen25ModelWrapper,
}


def get_model_wrapper(model_name: str, cfg, **kwargs) -> BaseModelWrapper:
    """
    Factory: return the appropriate model wrapper for `model_name`.
    Falls back to OPTModelWrapper for unknown names (safe default for HF models).
    """
    # Match by prefix to handle variants not explicitly listed
    wrapper_cls = None
    for key, cls in MODEL_REGISTRY.items():
        if model_name.startswith(key) or key.startswith(model_name.split("/")[0]):
            wrapper_cls = cls
            break
    if wrapper_cls is None:
        # Fallback: try to auto-detect from model name
        name_lower = model_name.lower()
        if "qwen" in name_lower:
            wrapper_cls = Qwen25ModelWrapper
        else:
            wrapper_cls = OPTModelWrapper

    return wrapper_cls(cfg, model_name=model_name, **kwargs)


__all__ = [
    "BaseModelWrapper",
    "OPTModelWrapper",
    "Qwen25ModelWrapper",
    "ScalarRewardModel",
    "load_reward_model_and_tokenizer",
    "get_model_wrapper",
    "MODEL_REGISTRY",
]
