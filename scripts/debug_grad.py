#!/usr/bin/env python3
"""Quick debug: check model device, requires_grad, and a mini forward pass."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models import get_model_wrapper
from src.utils import load_config

cfg = load_config("configs/wdpo_tldr.yaml", [])
model_name = "outputs/wdpo/tldr/sft_strong"
wrapper = get_model_wrapper(model_name, cfg)
model = wrapper.model
ref_model = wrapper.get_ref_model()

print(f"=== Policy model ===")
print(f"  type: {type(model).__name__}")
print(f"  device: {model.device}")
print(f"  has hf_device_map: {hasattr(model, 'hf_device_map')}")
if hasattr(model, 'hf_device_map'):
    print(f"  hf_device_map: {model.hf_device_map}")

n_params = sum(1 for p in model.parameters())
n_grad = sum(1 for p in model.parameters() if p.requires_grad)
print(f"  params: {n_params}, requires_grad: {n_grad}")

first_param = next(model.parameters())
print(f"  first param device: {first_param.device}, requires_grad: {first_param.requires_grad}")

print(f"\n=== Ref model ===")
print(f"  type: {type(ref_model).__name__}")
print(f"  device: {ref_model.device}")
print(f"  has hf_device_map: {hasattr(ref_model, 'hf_device_map')}")
if hasattr(ref_model, 'hf_device_map'):
    print(f"  hf_device_map: {ref_model.hf_device_map}")

n_ref_params = sum(1 for p in ref_model.parameters())
n_ref_grad = sum(1 for p in ref_model.parameters() if p.requires_grad)
print(f"  params: {n_ref_params}, requires_grad: {n_ref_grad}")

# Quick forward pass test
print(f"\n=== Forward pass test ===")
tok = wrapper.tokenizer
inputs = tok("Hello world", return_tensors="pt").to(first_param.device)
model.train()
out = model(**inputs)
logits = out.logits
print(f"  logits.requires_grad: {logits.requires_grad}")
print(f"  logits.grad_fn: {logits.grad_fn}")

# Try computing a simple loss
loss = logits.sum()
print(f"  loss.requires_grad: {loss.requires_grad}")
print(f"  loss.grad_fn: {loss.grad_fn}")
try:
    loss.backward()
    print("  backward() SUCCESS!")
except Exception as e:
    print(f"  backward() FAILED: {e}")
