# python scripts/evaluate.py --config configs/mwdpo_bc_hh_rlhf.yaml --aligned_model_path outputs/mwdpo_1phase_threshold0.8_linear_boots/hh_rlhf/opt-1.3b/seed42/strong_model --sft_model_path outputs/mwdpo_1phase_threshold0.8_linear_boots/hh_rlhf/opt-1.3b/seed42/sft_strong
# python pipeline/run_pipeline.py --config configs/baseline_dpo_hh_rlhf.yaml
# /home/adminn/theanh28/w2sg_uncertainty/outputs/mwdpo_1phase_threshold0.8_linear_boots/hh_rlhf/opt-1.3b/seed42/eval
python pipeline/run_pipeline.py --config configs/baseline_dpo_tldr.yaml
python pipeline/run_pipeline.py --config configs/cwpo_hh_rlhf.yaml

# python scripts/evaluate.py --config configs/baseline_dpo_hh_rlhf.yaml --aligned_model_path /home/adminn/theanh28/w2sg_uncertainty/outputs/baseline_dpo/hh_rlhf/opt-1.3b/seed42/strong_model --sft_model_path /home/adminn/theanh28/w2sg_uncertainty/outputs/baseline_dpo/hh_rlhf/opt-1.3b/seed42/sft_strong

# python scripts/evaluate.py --config configs/wdpo_hh_rlhf.yaml --aligned_model_path /home/adminn/theanh28/w2sg_uncertainty/outputs/wdpo/hh_rlhf/opt-1.3b/seed42/strong_model --sft_model_path /home/adminn/theanh28/w2sg_uncertainty/outputs/wdpo/hh_rlhf/opt-1.3b/seed42/sft_strong
