# AAAI Extension Plan: Ensemble-Uncertainty Weak-to-Strong Preference Optimization

## Context

The repo implements two published weak-to-strong (W2SG) alignment baselines — **WDPO**
(Tao & Li, ICLR'25: weak model labels via DPO implicit reward) and **CWPO**
(Afzali et al., ICLR'26: single scalar reward model produces a per-sample confidence
`C = 2·(σ(s⁺−s⁻)−0.5)` that reweights the DPO loss). Both let a tiny weak LLM annotate
preferences to align a stronger model, sometimes beating 100% human labels.

**Gap / thesis for the paper.** CWPO's confidence comes from a *single* weak model, so it
conflates two things: how *confident* that model is versus how *reliable* the label is.
A single model can be confidently wrong, and its softmax confidence is known to be
miscalibrated. Our contribution: use a **diverse ensemble of K weak models** to obtain an
*epistemic* uncertainty signal (inter-model disagreement) that is separable from raw
confidence, and show that down-weighting high-disagreement samples produces a
better-calibrated weak-supervision signal and stronger W2SG alignment than single-model
CWPO. A WPO-style importance-weighting term (correcting the off-policy distribution gap of
DPO on weak-labeled data) is included as a secondary component / ablation.

Because the proposed method is the K>1 generalization of CWPO, it reuses the existing
reward-model, labeler, and CW-DPO trainer infrastructure almost entirely. Baselines
(Baseline-DPO, WDPO, CWPO) are **already implemented**, which is what makes the ~2–3 week
timeline feasible.

**Target:** AAAI 2026 main track. **Compute:** up to Qwen2.5-7B / OPT-6.7B strong (LoRA,
`device_map=auto`). **New method key:** `ecwpo`.

---

## Method (what the paper proposes)

Train **K diverse weak scalar reward models** `{r_1..r_K}` (different families to avoid
correlated errors: e.g. OPT-125M, Qwen2.5-0.5B, Pythia-160M) on the labeled split `D_l`
(Bradley-Terry, exactly as CWPO does today). For each unlabeled pair `(x, y_a, y_b)`:

- per-model preference margin `m_k = r_k(x,y_a) − r_k(x,y_b)`
- **label** by majority vote: `y⁺ = y_a` iff `sign(mean_k m_k)` (ties → mean margin)
- **ensemble confidence** `C_mean = 2·(σ(mean_k m_k) − 0.5)` (CWPO formula on the mean)
- **epistemic disagreement** `U = Var_k(m_k)` (normalized by `τ²`)
- **combined weight** `W = C_mean · (1 − U/τ²)⁺`, clamped to `[0,1]`

`W` replaces CWPO's `confidence_weight` and flows through the *unchanged* CW-DPO trainer.

**Secondary / ablation — WPO term.** Add a detached importance weight
`w_IS = clip(exp((logπ_θ−logπ_ref)_{y⁺} + (logπ_θ−logπ_ref)_{y⁻}), ε, M)` into the
per-sample loss. The exact log-ratios already exist in the trainer
(`src/trainers/cwpo_trainer.py:242-243`), so this is a one-block insertion at the loss
multiply site (`src/trainers/cwpo_trainer.py:254`).

Paper claims to support with experiments: (1) ensemble-mean > single CWPO; (2)
disagreement-down-weighting > ensemble-mean; (3) disagreement is *calibrated* — correlates
with gold-reward label error, unlike single-model confidence; (4) diverse ensemble >
same-family ensemble; (5) +WPO adds a further bump.

---

## Implementation (minimal surface; reuses existing code)

New method `ecwpo` must be registered at the four existing dispatch points that currently
`raise ValueError` on unknown methods:
`pipeline/run_pipeline.py:351`, `scripts/train_strong.py:118-125`,
`scripts/label_weak.py:87-92`, and the trainer imports in `scripts/train_strong.py:44-46`.

1. **Config** — add to `configs/base.yaml`: `weak_model_names: [...]` (list; CLI list
   overrides need bracket syntax, so keep the ensemble in the YAML file), and an
   `ensemble:` block (`tau`, `aggregation: mean|vote`, `use_wpo: false`, `wpo_clip_min: 0.1`,
   `wpo_clip_max: 5.0`). New keys are read via `cfg.get(...)` (safe when absent; loader
   merges base→exp→CLI dotlist per `src/utils/config_utils.py:63-68`). New experiment
   configs: `configs/ecwpo_hh_rlhf.yaml`, `configs/ecwpo_ufb.yaml` (mirror
   `configs/cwpo_hh_rlhf.yaml`).

2. **Train K reward models** — add `scripts/train_reward_ensemble.py`: loop over
   `cfg.weak_model_names`, for each call the *existing* `RewardModelTrainer`
   (`src/trainers/reward_model_trainer.py`) with `backbone_name=<model>` and a per-model
   `output_dir` (e.g. `reward_model/member_{i}/checkpoint-final`). No trainer change —
   reuses the current Bradley-Terry loop and `load_reward_model_and_tokenizer`
   (`src/models/reward_model.py:174`).

3. **Ensemble labeler** — new `src/weak_labeler/ensemble_labeler.py`,
   `EnsembleConfidenceLabeler(BaseWeakLabeler)`. Load K `ScalarRewardModel`s; reuse the
   scoring pattern from `ConfidenceLabeler._score_batch`
   (`src/weak_labeler/confidence_labeler.py:142-166`). Compute `W` as above; emit
   `PseudoLabeledSample(prompt, chosen, rejected, confidence_weight=W,
   agreement=..., variance=U, per_model_margins=[...])`. Extra keys serialize for free —
   `PseudoLabeledSample`/`save`/`load` are a plain `dict` subclass + generic JSON
   (`src/weak_labeler/base_labeler.py:20-94`), so the analysis fields round-trip with zero
   schema work. Export it from `src/weak_labeler/__init__.py`.

4. **label_weak.py** — add `_build_ecwpo_labeler` (mirror `_build_cwpo_labeler:183`) that
   loads K reward models from their member dirs; dispatch `method=="ecwpo"` at `:87-92`.

5. **Strong training** — dispatch `ecwpo` to the **existing** `_train_cwpo`
   (`scripts/train_strong.py:170`): it already loads `confidence_weight` from the pseudo
   labels and runs `CWPOTrainer`. Core method needs **no new trainer**.
   - WPO ablation: subclass `CWPOTrainer` → `WPOCWPOTrainer` in
     `src/trainers/cwpo_trainer.py`, overriding only the weighting block at `:249-254`
     to multiply in `w_IS` (log-ratios from `:242-243`, detached). Store `use_wpo`/clip
     bounds as trainer instance attributes — **not** `DPOConfig` kwargs (TRL rejects
     unknown kwargs). Gate on `cfg.ensemble.use_wpo`. Follow the float-multiply masking
     convention (`src/trainers/cwpo_trainer.py:216`).

6. **Pipeline** — add an `ecwpo` branch in `pipeline/run_pipeline.py` mirroring the `cwpo`
   branch (`:277-349`): Phase 1b `train_reward_ensemble.py` → Phase 2 `label_weak.py` →
   Phase 2b `train_sft.py` → Phase 3 `train_strong.py` → Phase 4 `evaluate.py`. Evaluation
   is method-agnostic (`Evaluator.run` branches only on paths), so GRA works unchanged.

7. **Uncertainty analysis** — new `scripts/analyze_uncertainty.py`: score each `D_u` pair's
   `y⁺/y⁻` with the gold reward model via `RewardModelEvaluator`
   (`src/evaluation/reward_model_eval.py:349`), define label-correct = ensemble label agrees
   with gold, and report correlation / calibration curves of `variance` (and single-model
   confidence) vs. label error. This produces the paper's key calibration figure (claim 3).

8. **Tests** — extend `tests/test_labelers.py` with ensemble aggregation math
   (vote, `C_mean`, variance→weight, ties) and `tests/test_losses.py` with the WPO
   clip/detach behavior. Keep the existing CWPO tests green (regression guard for reuse).

---

## Experiments (scoped for the deadline)

Reuse existing baselines — do **not** reimplement them.

| Axis | Setting |
|---|---|
| Datasets | HH-RLHF (primary) + UltraFeedback (UFB) — both use the Skywork GRA model, so eval is uniform. TL;DR only if time permits (different GRA model). |
| Weak ensemble | K=3 diverse: OPT-125M, Qwen2.5-0.5B, Pythia-160M (diversity ablation vs. 3× OPT-125M seeds). |
| Strong models | OPT-1.3B, OPT-2.7B (fast) + **Qwen2.5-7B headline** (LoRA r=8, `device_map=auto`, grad-checkpointing). |
| Baselines (done) | Baseline-DPO (100% labels), WDPO, CWPO (K=1). |
| Metric | GRA (`gold_reward_accuracy`, exists); GPT-4 win-rate on a subset if time. |
| Ablations | K∈{1,3,5}; mean-only vs. mean+disagreement; diverse vs. same-family; +WPO; `labeled_ratio`∈{0.2,0.3}; calibration figure. |

Rough order (parallelize across GPUs): small-model full sweep first (validates method +
ablations), Qwen-7B headline runs second, calibration/analysis + GPT-4 last. If 7B runs
slip, the small/mid-model results + ablations + calibration still stand as a complete paper;
Qwen-7B is the headline upgrade, not a dependency.

---

## Verification

- **Unit:** `pytest tests/ -v` — new ensemble/WPO tests pass, existing CWPO/DPO/data tests stay green.
- **Smoke (debug mode, tiny data, no wandb):**
  `python pipeline/run_pipeline.py --config configs/ecwpo_hh_rlhf.yaml --debug` runs all
  phases end-to-end and writes `outputs/ecwpo/hh_rlhf/{reward_model/member_*,weak_labels/pseudo_labeled.jsonl,sft_strong,strong_model,eval/metrics.json}`.
- **Label sanity:** inspect `pseudo_labeled.jsonl` — every row has `confidence_weight`,
  `variance`, `per_model_margins`; `preference_accuracy` (vs. held-out human labels) prints
  in `label_weak.py`.
- **Method correctness:** with K=1 and `use_wpo=false`, `ecwpo` must reproduce CWPO GRA
  (numerical equivalence check — guards the reuse path).
- **Headline:** `ecwpo` GRA > CWPO GRA > WDPO ≈ Baseline-DPO on HH-RLHF, and disagreement
  vs. label-error correlation is significant in `analyze_uncertainty.py` output.
