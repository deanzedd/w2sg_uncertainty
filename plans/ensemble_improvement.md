# Making the Weak-Model Ensemble More Sophisticated

Three complementary plans that upgrade the baseline ensemble (K independently-trained
reward models → average margins → variance as disagreement). Each plan lists its method,
the concrete code touchpoints (the reward model is `AutoModel` backbone + a single
`nn.Linear(hidden, 1)` head — see `src/models/reward_model.py:43-69`), and the ablation row
it produces. Priority/effort tags assume the AAAI ~2–3 week window.

Shared integration point: all of this feeds a single per-sample weight `W` (and a hard
label) into `EnsembleConfidenceLabeler`, which writes `confidence_weight` (+ analysis
fields) to `pseudo_labeled.jsonl` and flows through the **unchanged** CW-DPO trainer. See
the parent plan `plans/research.md`.

---

## Plan 1 — Richer / cheaper members (the diversity source)

**Goal:** obtain K decorrelated members without paying K× training cost, and mix members
that fail differently (not just different seeds).

### 1a. Multi-head bootstrap ensemble on a shared backbone  ⭐ (HIGH priority, LOW cost)
One shared backbone, **K scalar heads** with independent init, each head trained on a
bootstrap resample of `D_l`. Gives full ensemble disagreement at ~1× backbone cost — the
key enabler that makes a sophisticated ensemble affordable at 7B.
- **Code:** generalize `ScalarRewardModel` head to `nn.Linear(hidden, K, bias=False)`
  (`src/models/reward_model.py:59`); `forward` returns `(batch, K)`; keep
  `_get_last_token_hidden` (`:101`) unchanged. Add per-head Bradley-Terry loss (mean over
  heads) in `bradley_terry_loss` (`:154`). In `RewardModelTrainer.train`
  (`src/trainers/reward_model_trainer.py:61`) draw a per-head bootstrap mask over the batch.
- **Members = heads.** `EnsembleConfidenceLabeler` reads the `(batch, K)` score tensor
  directly — no K model loads.

### 1b. LoRA-adapter / snapshot ensembles (MEDIUM priority, LOW cost)
Alternative cheap members: K LoRA adapters over one frozen backbone (BatchEnsemble-style),
or **checkpoint snapshots** taken along a cyclic-LR run (free — reuse the existing
`save_steps` checkpoints from `RewardModelTrainer._save`, `:244`).

### 1c. Family-diverse full models (HIGH priority, MEDIUM cost)
Keep the cross-family option (OPT-125M + Qwen2.5-0.5B + Pythia-160M) as the **diversity
ablation** vs. same-family/multi-head — same-family members have correlated errors that
naive variance under-counts. Uses `scripts/train_reward_ensemble.py` (parent plan step 2).

### 1d. Heterogeneous members (MEDIUM priority, MEDIUM cost)
Mix *error modes*, not just inits: BT scalar RMs + one WDPO-style DPO-implicit-reward scorer
(`src/weak_labeler/dpo_reward_labeler.py:129`) + optionally the **strong SFT model as a
member** (folds in the self-preference idea). Requires normalizing heterogeneous scores —
see Plan 2a.

**Ablation rows produced:** multi-head vs. K-full-models (cost/quality); diverse-family vs.
same-family (correlated-error effect); homogeneous vs. heterogeneous members.

---

## Plan 2 — Smarter aggregation (the intellectual core)

**Goal:** replace uniform-mean + raw-variance with a calibrated, competence-weighted
estimate that separates *epistemic* from *aleatoric* uncertainty.

### 2a. Per-member temperature calibration  ⭐ (HIGH priority, LOW cost — also a correctness fix)
Raw logits from OPT-125M and Qwen-0.5B are on different scales, so averaging margins is
ill-defined. Fit a per-member temperature `T_k` on `D_l` (minimize NLL of the held-out
human labels), then aggregate calibrated probabilities `σ(m_k / T_k)`.
- **Code:** fit step in `scripts/train_reward_ensemble.py` after training; store `T_k` in
  each member's `metadata.json` (`reward_model.py` save path). Apply in
  `EnsembleConfidenceLabeler._score_batch`.

### 2b. Competence-weighted / Dawid–Skene latent-truth aggregation  ⭐ (HIGH priority, MEDIUM cost)
Weight members by held-out reliability rather than uniformly. Taken to its principled limit:
a **Dawid–Skene** model jointly estimates each member's confusion/reliability and the latent
true preference; the posterior `P(y⁺ correct | votes)` is *both* the label and the weight,
eliminating the ad-hoc `C_mean·(1−Var/τ²)⁺`.
- **Method:** EM over member votes on `D_u` (initialized from `D_l` reliabilities). Weight
  `W = P(y⁺ correct)`; label = argmax posterior.
- **Code:** offline in `EnsembleConfidenceLabeler.label_dataset`; pure numpy/torch, no model
  changes. This is also significance-lever #1 in the parent research plan.

### 2c. Mutual-information (BALD) disagreement — epistemic vs. aleatoric  ⭐ (HIGH priority, LOW cost)
Decompose total predictive entropy of the ensemble preference into:
- **aleatoric** = mean per-member entropy → the pair is genuinely close (humans would
  disagree too; matches WDPO's "near-indistinguishable" finding),
- **epistemic** = total entropy − aleatoric (mutual information) → the *labelers* are unsure.

Down-weight **epistemic** uncertainty (untrustworthy labels) while keeping aleatoric-
uncertain pairs as legitimately soft. Raw variance conflates the two; this decomposition is
the memorable contribution.
- **Weight:** `W = C_mean · (1 − Î_epistemic / log2)`, or fold into the Dawid–Skene
  posterior. `Î_epistemic` = MI over the K calibrated Bernoulli preference probs.
- **Code:** aggregation math in `EnsembleConfidenceLabeler`; store `epistemic`, `aleatoric`,
  `C_mean` per row for the calibration figure.

### 2d. Correlation-corrected disagreement (LOW priority, LOW cost)
Estimate the member error-correlation matrix on `D_l` and use a decorrelated
(Mahalanobis-style) disagreement, or greedy diversity selection of members. Honestly
addresses the correlated-errors weakness of same-family ensembles.

**Ablation rows produced:** uncalibrated-mean vs. calibrated; uniform vs. competence-weighted
vs. Dawid–Skene; variance vs. MI-epistemic weight; ± correlation correction. Plus the
**calibration figure**: epistemic uncertainty vs. gold-reward label error (the paper's key
plot; reuse `scripts/analyze_uncertainty.py` from the parent plan).

---

## Plan 3 — Adaptive / dynamic ensembling (stretch)

**Goal:** move beyond a static flat combine toward input-dependent and iterative use.

### 3a. Per-sample gating / mixture-of-experts (MEDIUM priority, MEDIUM cost)
A small gate network assigns per-sample member weights conditioned on `x` (e.g. lean on a
code-savvy member for code prompts). Turns the flat average into learned routing.
- **Code:** tiny MLP over the pooled prompt embedding; trained on `D_l` to predict which
  member matches the human label. Applied in `EnsembleConfidenceLabeler`.

### 3b. Iterative ensemble growth / self-member (MEDIUM priority, MEDIUM cost)
After round 1, add the trained strong model as a new heterogeneous member and re-estimate
uncertainty → a 1–2 round self-training loop where the ensemble sharpens. Bridges to the
self-preference anchor in `paper/pro.md`.
- **Code:** a `--round` loop in `pipeline/run_pipeline.py` reusing the `ecwpo` branch; round
  N's strong model becomes a member for round N+1's labeling.

### 3c. Cascade for efficiency (LOW priority, LOW cost)
Score with cheap members first; only invoke expensive/7B members on still-uncertain samples.
Efficiency story; reuses the calibrated per-member scores.

**Ablation rows produced:** flat vs. gated aggregation; round-1 vs. round-2 (self-member);
cascade cost/quality trade-off.

---

## Recommended stack for the deadline

Members: **1a multi-head bootstrap** (+ **1c** as the diversity ablation) →
Aggregation: **2a calibration → 2b Dawid–Skene → 2c MI/BALD epistemic weight** (the core) →
Stretch: **3b self-member iteration**.

This reads as a *designed* uncertainty estimator, not "average of K models," and the
epistemic/aleatoric decomposition (2c) is the part reviewers remember.

## Verification (additions to the parent plan's checks)

- **Unit:** multi-head `ScalarRewardModel` returns `(batch, K)`; bootstrap masks differ per
  head; calibration reduces held-out NLL on `D_l`; MI = 0 when all members agree and is
  maximal at even split; Dawid–Skene EM converges on a synthetic toy.
- **Sanity:** `pseudo_labeled.jsonl` rows carry `confidence_weight`, `epistemic`,
  `aleatoric`, `per_model_margins`.
- **Calibration figure:** epistemic uncertainty correlates with gold-reward label error
  (via `scripts/analyze_uncertainty.py`) more strongly than single-model confidence does.
- **Degenerate check:** K=1, no calibration, variance weight ⇒ reproduces CWPO GRA.
