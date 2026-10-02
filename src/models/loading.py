"""
Load a causal LM from a Hub id, a full checkpoint, or a LoRA adapter directory.

An adapter directory is merged into its base weights, so the caller always gets a plain
PreTrainedModel. Chains (e.g. a DPO adapter trained on top of a merged SFT adapter) are
resolved recursively through `base_model_name_or_path` in adapter_config.json.
"""

from __future__ import annotations

import json
import os

from transformers import AutoModelForCausalLM, PreTrainedModel


def is_adapter_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def load_causal_lm(path: str, **kwargs) -> PreTrainedModel:
    if not is_adapter_dir(path):
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)

    from peft import PeftModel

    with open(os.path.join(path, "adapter_config.json")) as f:
        base_path = json.load(f)["base_model_name_or_path"]
    model = PeftModel.from_pretrained(load_causal_lm(base_path, **kwargs), path).merge_and_unload()

    # PEFT records `model.name_or_path` as the base of any adapter trained on top of this model.
    # Point it at this adapter dir so the chain can be rebuilt at evaluation time.
    abs_path = os.path.abspath(path)
    model.name_or_path = abs_path
    model.config._name_or_path = abs_path
    return model
