# Attention Notes — Core Methods Comparison (HH-RLHF)

Shared setup unless noted: **HH-RLHF**, `labeled_ratio=0.3` → \(D_l\) (30% human labels) / \(D_u\) (70% unlabeled), weak ≈ Qwen2.5-0.5B, strong = Qwen2.5-7B + LoRA.

Configs:
- `configs/baseline_dpo_hh_rlhf.yaml`
- `configs/wdpo_hh_rlhf.yaml`
- `configs/cwpo_hh_rlhf.yaml`
- `configs/mwdpo_hh_rlhf.yaml`
- `configs/mwdpo_bc_hh_rlhf.yaml`

---

## 1. One-line idea

| Method | Config `method` | Core idea |
|--------|-----------------|-----------|
| **Baseline DPO** | `baseline_dpo` | No weak model. Train strong on **full human data \(D\)**. |
| **WDPO** | `wdpo` | Weak LM (SFT→DPO) labels \(D_u\) via **implicit reward**; strong trains on all pseudo-labels (weight=1). |
| **CWPO** | `cwpo` | Scalar reward model labels \(D_u\) + **confidence weight \(C\)**; strong uses **CW-DPO** \(L=\mathbb{E}[C\cdot\ell_{\mathrm{DPO}}]\). |
| **MWDPO** | `mwdpo` | **\(k\) independent** reward models; keep only **high-agreement** subset \(D_h\); strong SFT+DPO on \(D_h\) only (standard DPO). |
| **MWDPO-BC** | `mwdpo_bootstrap_calibration` | Same \(D_h\) idea, but **1 shared backbone + \(K\) bootstrap heads** (cheaper ensemble); optional temp calibration + confidence gate. |

---

## 2. Pipeline sketch

```
Baseline:   D ──SFT──► π_SFT ──DPO──► π*
            (full D, no weak labeling)

WDPO:       D_l ──SFT→DPO──► π_w* ──label D_u──► D̂ ──SFT+DPO──► π*
            (implicit reward; C=1 for all)

CWPO:       D_l ──BT reward──► r_w ──label D_u──► D̂(C) ──SFT+CW-DPO──► π*
            (soft weights on all labeled pairs)

MWDPO:      D_l ──k×BT RM (diff seeds)──► label D_u ──► D_h ∪ D_ℓ
            D_h ──SFT+standard DPO──► π*   (discard / ignore D_ℓ in Phase 1)

MWDPO-BC:   D_l ──1 MultiHead RM (K bootstrap heads)──► label D_u ──► D_h ∪ D_ℓ
            D_h ──SFT+standard DPO──► π*   (same strong phase as MWDPO)
```

---

## 3. Weak annotator design

| | Baseline | WDPO | CWPO | MWDPO | MWDPO-BC |
|---|----------|------|------|-------|----------|
| **Weak model** | none | LM policy \(\pi_w\) | 1× scalar RM | \(k\) full RMs | 1 backbone + \(K\) heads |
| **Train on** | — | \(D_l\) (SFT then DPO) | \(D_l\) (Bradley–Terry) | \(D_l\) (BT, seeds differ) | \(D_l\) (BT + bootstrap masks) |
| **Diversity** | — | n/a (single) | n/a (single) | different seeds / models | per-batch bootstrap resample |
| **Cost** | lowest | 1 weak LM | 1 RM | \(k\times\) RM cost | ~1× backbone + \(K\) heads |
| **This config** | — | 0.5B SFT+DPO | 0.5B scalar | `num_models: 3`, seeds `[42,123,456]` | `num_heads: 3`, `use_bootstrap: true` |

**WDPO scoring:** \(r_w(x,y)=\beta(\log\pi_w-\log\pi_{\mathrm{ref}})\); pick argmax as chosen.

**CWPO / MWDPO / BC scoring:** scalar scores \(s(x,y)\); pick higher score as chosen.

---

## 4. How \(D_u\) becomes training data

| | WDPO | CWPO | MWDPO | MWDPO-BC |
|---|------|------|-------|----------|
| **Output** | all of \(D_u\) → \(\hat{D}\) | all of \(D_u\) → \(\hat{D}\) | split \(D_h\) / \(D_\ell\) | split \(D_h\) / \(D_\ell\) |
| **Filter** | none | none | agreement | agreement (+ threshold) |
| **Weight** | \(C=1\) always | \(C=2(\sigma(s_+-s_-)-0.5)\in[0,1]\) | store \(C=\sigma(S_w-S_l)\); Phase-1 DPO **unweighted** | same \(C\); gate with threshold |
| **Agreement** | — | — | `unanimous` (default) | `unanimous_with_threshold`, thr=**0.8** |
| **Ensemble score** | — | — | \(S=\frac1k\sum_i r_i\) | \(S=\frac1K\sum_k r_k\) (score space) |

**MWDPO \(D_h\) rule (`unanimous`):** all \(k\) models pick the same winner as the ensemble.

**MWDPO-BC \(D_h\) rule (`unanimous_with_threshold`):**
\[
\text{in }D_h \iff \text{all heads agree with ensemble}\;\wedge\; C\ge 0.8,\quad C=\sigma(S(y_w)-S(y_l)).
\]
(Config has `use_calibration: false` → \(T_k=1\); calibration is optional ablation.)

---

## 5. Strong-model training

| | Baseline | WDPO | CWPO | MWDPO / MWDPO-BC |
|---|----------|------|------|------------------|
| **SFT data** | full \(D\) | \(\hat{D}\) (chosen) | \(\hat{D}\) (chosen) | \(D_h\) only |
| **Preference loss** | standard DPO on full \(D\) | standard DPO on \(\hat{D}\) | **CW-DPO** on \(\hat{D}\) | standard DPO on \(D_h\) |
| **Uses weak confidence?** | no | no | **yes** (multiplies loss) | filter only (not in Phase-1 loss) |
| **Uses unlabeled \(D_u\)?** | no (oracle full labels) | yes (all) | yes (all, soft) | yes (hard subset) |

Typical HPs (shared across methods in these configs): SFT 3 epochs lr \(1\mathrm{e}{-5}\); DPO 5 epochs lr \(5\mathrm{e}{-6}\), \(\beta=0.5\).

---

## 6. What each method is testing

1. **Baseline** — upper-ish / oracle-style reference with **full human preferences** (no weak supervision). Not a fair “same label budget” baseline vs weak methods; it uses all of \(D\).
2. **WDPO** — can a **small DPO policy**’s implicit reward transfer preferences to a strong model?
3. **CWPO** — does a **scalar RM + soft confidence weighting** beat hard pseudo-labels?
4. **MWDPO** — does **ensemble agreement filtering** beat labeling everything (noise reduction via \(D_h\))?
5. **MWDPO-BC** — can **bootstrap multi-head** match MWDPO’s agreement idea at **~1 backbone cost**, and does a **confidence threshold** further clean \(D_h\)?

---

## 7. Quick decision tree

```
Need human-only oracle on full D?     → baseline_dpo
Single weak LM + hard labels?         → wdpo
Single RM + soft weights on all D_u?  → cwpo
k separate RMs + keep agreement only? → mwdpo
Cheap ensemble + optional C≥τ gate?   → mwdpo_bc  (this repo’s bootstrap-calibration variant)
```

---

## 8. Config knobs that matter most

| Knob | Where | Effect |
|------|--------|--------|
| `labeled_ratio` | all | size of \(D_l\) vs \(D_u\) |
| `weak_model_name` | WDPO/CWPO/MWDPO/BC | capacity of annotator |
| `multi_weak.num_models` / `seeds` | MWDPO | ensemble size / diversity |
| `multi_weak.agreement_mode` | MWDPO | unanimous vs +threshold |
| `bootstrap_calibration.num_heads` | MWDPO-BC | \(K\) heads |
| `use_bootstrap` | MWDPO-BC | decorrelate heads via resampling |
| `use_calibration` | MWDPO-BC | fit \(T_k\) (off in current BC HH config) |
| `agreement_mode` + `confidence_threshold` | MWDPO-BC | how strict \(D_h\) is (here: unanimous + 0.8) |

---

## 9. Mental model (shortest)

- **Baseline** = strong on true full data.
- **WDPO** = weak policy votes on all \(D_u\), equal weight.
- **CWPO** = weak RM votes on all \(D_u\), down-weight uncertain pairs in the loss.
- **MWDPO** = many RMs; **throw away** disagreements; train strong only on clean \(D_h\).
- **MWDPO-BC** = MWDPO’s filter idea with a **bootstrap multi-head** weak model (+ optional confidence gate).
