# Weak-to-Strong Preference Optimization

Code for training a strong language model from preference labels produced by a weak model, with four methods:

| Method (`method:` in config) | Weak annotator | Strong-model training |
|---|---|---|
| Baseline DPO (`baseline_dpo`) | none (human labels) | SFT, then DPO on the full dataset |
| WDPO (`wdpo`) | DPO-trained weak model, implicit reward | SFT, then DPO on the weakly labeled data |
| CWPO (`cwpo`) | scalar reward model, confidence per pair | SFT, then confidence-weighted DPO |
| MWDPO-BC (`mwdpo_bootstrap_calibration`) | multi-head bootstrap reward model | SFT, then DPO on the high-agreement subset |

The strong model is trained with LoRA (r=8, alpha=16). Datasets: TL;DR and HH-RLHF.
The labeled split D_l (30%) trains the weak annotator; the remaining 70% (D_u) is relabeled by it.

## Setup

```bash
conda create -n w2sg python=3.10 && conda activate w2sg
pip install -r requirements.txt
```

Tested with torch 2.11 (CUDA 12.8), transformers 5.17, trl 1.14, peft 0.21, accelerate 1.15, bitsandbytes 0.50.
Models and datasets are downloaded from the Hugging Face Hub into `.cache/` on first use.
An NVIDIA GPU with bf16 support is required; OPT-13B and Qwen2.5-14B need roughly 27-45 GB per GPU.

Optional: `export WANDB_API_KEY=...` for logging (disable with `use_wandb=false`).

## Run one experiment

Each experiment is one config file; the pipeline runs every stage for its method
(weak model or reward model, labeling, strong SFT, strong DPO, evaluation):

```bash
python pipeline/run_pipeline.py --config configs/mwdpo_bc_tldr_opt1.3b.yaml
```

Config names follow `configs/<method>_<dataset>[_<strong model>].yaml`, for example:

```bash
python pipeline/run_pipeline.py --config configs/baseline_dpo_tldr_qwen3b.yaml
python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf_opt6.7b.yaml
python pipeline/run_pipeline.py --config configs/cwpo_tldr_qwen7b.yaml
```

Any config value can be overridden on the command line:

```bash
python pipeline/run_pipeline.py --config configs/cwpo_tldr_qwen3b.yaml use_wandb=false eval.max_gen_samples=100
```

## Useful options

| Option | Effect |
|---|---|
| `--debug` | tiny smoke test (small data, 1 epoch) |
| `--skip_weak_model` / `--skip_reward_model` / `--skip_labeling` | reuse an existing weak model, reward model or labels (e.g. one weak stage shared by several strong-model sizes) |
| `--skip_sft` | reuse an existing strong SFT model |
| `--resume_sft_checkpoint PATH` / `--resume_dpo_checkpoint PATH` | continue SFT or DPO from a saved checkpoint |
| `--run_gpt4` | also compute the GPT-4 win rate (needs `OPENAI_API_KEY`) |
| `W2SG_NUM_GPUS=4` (environment variable) | run strong SFT/DPO data-parallel over the visible GPUs |

## Outputs

All paths are set in the config and live under `outputs/` (one folder per method, dataset, model and seed):
the weak model or reward model, `weak_labels/pseudo_labeled.jsonl`, `sft_strong/`, `strong_model/` (LoRA checkpoints)
and `eval/metrics.json`.
`metrics.json` holds the GRA score: the fraction of 500 test prompts where a reward model prefers the aligned model's
response over the SFT model's (TL;DR: `OpenAssistant/reward-model-deberta-v3-large-v2`; HH-RLHF: `Skywork/Skywork-Reward-V2-Llama-3.1-8B`).

## Tests

```bash
pip install pytest && python -m pytest tests -q
```
