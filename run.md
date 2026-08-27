
## Quick Start

### 1. Cài đặt

```bash
conda create -n w2sg python=3.10.0
pip install -r requirements.txt
```

### 2. Run pipeline mwdpo phase 1 and mwdpo_bc
```bash
python pipeline/run_pipeline.py --config configs/mwdpo_bc_ace_hh_rlhf.yaml
python pipeline/run_pipeline.py --config configs/mwdpo_hh_rlhf.yaml
python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf.yaml
python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml
```

