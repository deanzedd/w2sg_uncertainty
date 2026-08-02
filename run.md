
## Quick Start

### 1. Cài đặt

```bash
conda create -n w2sg python=3.10.0
pip install -r requirements.txt
```

### 2. Run pipeline mwdpo phase 1 and mwdpo_bc
```bash
### mwdpo_bc
python pipeline/run_pipeline.py --config configs/mwdpo_bc_hh_rlhf.yaml

### mwdpo
python pipeline/run_pipeline.py --config configs/mwdpo_hh_rlhf.yaml

### wdpo
python pipeline/run_pipeline.py --config configs/wdpo_hh_rlhf.yaml
```

