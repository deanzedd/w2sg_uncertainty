# Weak-to-Strong Preference Optimization Baselines

> Codebase baseline cho **WDPO** và **CWPO** — hai phương pháp sử dụng weak model để tạo nhãn preference và train strong model.

## Tổng quan

| Method | Weak Labeler | Strong Training | Key Feature |
|---|---|---|---|
| **Baseline DPO** | — | DPO on D_l | Only human labels |
| **WDPO** | DPO implicit reward | DPO on D̂ | Weak model labels D_u |
| **CWPO** | Scalar reward model | Confidence-weighted DPO on D̂ | C = 2·(σ(s+−s−)−0.5) |

## Cấu trúc

```
w2sg_uncertainty/
├── configs/                  # YAML experiment configs
│   ├── base.yaml             # Shared defaults
│   ├── wdpo_hh_rlhf.yaml    # WDPO + HH-RLHF
│   ├── wdpo_tldr.yaml       # WDPO + TL;DR
│   ├── cwpo_hh_rlhf.yaml   # CWPO + HH-RLHF
│   ├── cwpo_tldr.yaml       # CWPO + TL;DR
│   └── baseline_dpo_*.yaml  # Baseline configs
├── src/
│   ├── data/                 # Dataset loaders (HH-RLHF, TL;DR)
│   ├── models/               # Model wrappers (OPT, Qwen2.5, RewardModel)
│   ├── weak_labeler/         # WDPO + CWPO labelers
│   ├── losses/               # DPO + CWPO loss functions
│   ├── trainers/             # SFT, RewardModel, WDPO, CWPO trainers
│   ├── evaluation/           # GRA + GPT-4 win rate evaluators
│   └── utils/                # Config, logging, seed utils
├── scripts/
│   ├── train_sft.py          # Phase 1: SFT
│   ├── train_reward_model.py # Phase 1b: CWPO weak annotator
│   ├── label_weak.py         # Phase 2: Weak labeling
│   ├── train_strong.py       # Phase 3: Strong model training
│   └── evaluate.py           # Phase 4: Evaluation
├── pipeline/
│   └── run_pipeline.py       # End-to-end pipeline
└── tests/                    # Unit tests
```

## Quick Start

### 1. Cài đặt

```bash
pip install -e .
# hoặc
pip install -r requirements.txt
```

### 2. Chạy end-to-end pipeline

```bash
# WDPO trên HH-RLHF với OPT-125M (weak) → OPT-1.3B (strong)
python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf.yaml

# CWPO trên HH-RLHF
python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml

# Baseline DPO
python pipeline/run_pipeline.py --config configs/baseline_dpo_hh_rlhf.yaml

# Debug mode (fast, dữ liệu nhỏ)
python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml --debug
```

### 3. Chạy từng bước

```bash
# Phase 1: SFT
python scripts/train_sft.py --config configs/wdpo_hh_rlhf.yaml

# Phase 1b (CWPO only): Train reward model
python scripts/train_reward_model.py --config configs/cwpo_hh_rlhf.yaml

# Phase 2: Weak labeling
python scripts/label_weak.py --config configs/wdpo_hh_rlhf.yaml
python scripts/label_weak.py --config configs/cwpo_hh_rlhf.yaml \
    --reward_model_path outputs/cwpo/hh_rlhf/reward_model/checkpoint-final/model.pt

# Phase 3: Train strong model
python scripts/train_strong.py --config configs/wdpo_hh_rlhf.yaml \
    --pseudo_labels outputs/wdpo/hh_rlhf/weak_labels/pseudo_labeled.jsonl \
    --sft_model_path outputs/wdpo/hh_rlhf/sft_strong

# Phase 4: Evaluate (GRA)
python scripts/evaluate.py \
    --config configs/wdpo_hh_rlhf.yaml \
    --aligned_model_path outputs/wdpo/hh_rlhf/strong_model \
    --sft_model_path outputs/wdpo/hh_rlhf/sft_strong
```

### 4. Thay đổi model size

Chỉ cần override trong YAML hoặc CLI:

```bash
# Strong model OPT-2.7B thay vì OPT-1.3B
python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf.yaml \
    strong_model_name=facebook/opt-2.7b \
    training.output_dir=outputs/wdpo/hh_rlhf/opt2.7b

# Qwen2.5-1.5B
python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml \
    strong_model_name=Qwen/Qwen2.5-1.5B

# Weak model Qwen2.5-0.5B
python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml \
    weak_model_name=Qwen/Qwen2.5-0.5B
```

### 5. Bật LoRA

```bash
python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml \
    use_lora=true lora_r=16 lora_alpha=32
```

## Chạy Unit Tests

```bash
pytest tests/ -v
```

## Evaluation Metrics

| Metric | Mô tả |
|---|---|
| **GRA** (Gold Reward Accuracy) | P(r(aligned) > r(SFT)) theo pretrained reward model |
| **GPT-4 Win Rate** | GPT-4 đánh điểm 1-10, tính win rate aligned vs SFT |
| **Preference Accuracy** | Weak label accuracy so với human labels (sanity check) |

## Extending

### Thêm dataset mới

```python
# src/data/my_dataset.py
from src.data.base_dataset import BasePreferenceDataset, PreferenceSample

class MyDataset(BasePreferenceDataset):
    def load_raw(self, split, cache_dir):
        return load_dataset("my/dataset", split=split)

    def preprocess_sample(self, raw):
        return PreferenceSample(
            prompt=raw["input"],
            chosen=raw["output_a"],
            rejected=raw["output_b"],
        )
```

Sau đó đăng ký trong `src/data/__init__.py`:
```python
DATASET_REGISTRY["my_dataset"] = MyDataset
```

### Thêm model family mới

```python
# src/models/llama_model.py
from src.models.base_model import BaseModelWrapper

class LlamaModelWrapper(BaseModelWrapper):
    def load_model_and_tokenizer(self):
        ...
```

### Thêm phương pháp labeling mới

```python
# src/weak_labeler/my_labeler.py
from src.weak_labeler.base_labeler import BaseWeakLabeler, PseudoLabeledSample

class MyLabeler(BaseWeakLabeler):
    def label_dataset(self, dataset, max_samples=None):
        ...
```

## Configuration

Tất cả config có thể override từ CLI:

```bash
python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf.yaml \
    labeled_ratio=0.2 \
    training.beta=0.05 \
    training.learning_rate=1e-6 \
    seed=123
```

## Models Supported

| Weak Models | Strong Models |
|---|---|
| `facebook/opt-125m` | `facebook/opt-1.3b` |
| `Qwen/Qwen2.5-0.5B` | `facebook/opt-2.7b` |
| | `facebook/opt-6.7b` |
| | `Qwen/Qwen2.5-1.5B` |
| | `Qwen/Qwen2.5-3B` |
| | `Qwen/Qwen2.5-7B` |

## Papers

- **WDPO**: Weak Supervision DPO — weak model labels D_u via implicit reward
- **CWPO**: Confidence-Weighted PO — `L = E[C(x,y+,y−)·ℓDPO]` where `C = 2·(σ(s+−s−)−0.5)`
