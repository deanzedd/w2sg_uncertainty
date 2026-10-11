#!/usr/bin/env python3
"""
Compare final policies (and their SFT models) from several runs on the SAME prompts with ONE reward model.

GRA compares a policy with its own SFT model, so it cannot rank runs whose SFT models differ (e.g. the
cells of a 2x2 SFT-data x DPO-data ablation). This script instead re-scores the generations that
`scripts/evaluate.py` saved (`generated_responses.json` in each run's eval dir) with one reward model and reports,
with 95% bootstrap confidence intervals over prompts:

  * absolute reward of every run's final policy,
  * absolute reward of every run's SFT model,
  * mean per-prompt improvement over the run's own SFT model,
  * head-to-head win rate of every pair of final policies on identical prompts (ties count 1/2),
    and the same for every pair of SFT models.

Per-prompt scores are cached next to each `generated_responses.json` (`scores_<reward model>.json`).
Intervals reflect the finite prompt set and the sampling of the single generation per prompt; they do NOT
include variation across training seeds.

Usage (repo root), one `--cell NAME=EVAL_DIR` per run:
  python scripts/compare_policies.py --dataset tldr --out outputs/ablation/tldr_qwen1.5b \\
      --cell "SFT-Dh/DPO-Dh=outputs/.../qwen2.5-1.5b/seed42/eval" \\
      --cell "SFT-Du/DPO-Dh=outputs/.../qwen2.5-1.5b_abl_sftall_dpoh/seed42/eval"
"""

import argparse
import itertools
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
#  Statistics (pure functions)                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def bootstrap_ci(values, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Mean and (1-alpha) percentile bootstrap CI of `values`, resampling prompts."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def win_indicator(a, b) -> np.ndarray:
    """Per-prompt 1 if a>b, 0.5 if tie, 0 if a<b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (a > b).astype(np.float64) + 0.5 * (a == b)


def head_to_head(a, b, n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Win rate of a over b on identical prompts and mean reward difference, both with 95% CI."""
    w = bootstrap_ci(win_indicator(a, b), n_boot, seed)
    d = bootstrap_ci(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), n_boot, seed)
    return {"win_rate": w[0], "win_lo": w[1], "win_hi": w[2], "diff": d[0], "diff_lo": d[1], "diff_hi": d[2]}


# ─────────────────────────────────────────────────────────────────────────── #
#  Scoring                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def load_generations(eval_dir: str) -> Tuple[List[str], List[str], List[str]]:
    path = os.path.join(eval_dir, "generated_responses.json")
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [r["prompt"] for r in rows], [r["aligned"] for r in rows], [r["sft"] for r in rows]


def score_cell(eval_dir: str, evaluator, rm_name: str) -> Dict[str, List[float]]:
    prompts, aligned, sft = load_generations(eval_dir)
    cache = os.path.join(eval_dir, "scores_%s.json" % rm_name.replace("/", "__"))
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            c = json.load(f)
        if c.get("n") == len(prompts):
            logger.info(f"reusing cached scores: {cache}")
            return {"aligned": c["aligned"], "sft": c["sft"]}
    logger.info(f"scoring {len(prompts)} aligned + {len(prompts)} SFT responses from {eval_dir}")
    out = {
        "aligned": [float(s) for s in evaluator.score_responses(prompts, aligned)],
        "sft": [float(s) for s in evaluator.score_responses(prompts, sft)],
    }
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"reward_model": rm_name, "n": len(prompts), **out}, f)
    return out


# ─────────────────────────────────────────────────────────────────────────── #
#  Report                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def fmt(m: float, lo: float, hi: float, nd: int = 3) -> str:
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def build_report(cells: Dict[str, Dict[str, List[float]]], rm_name: str, n_boot: int, seed: int) -> Tuple[Dict, str]:
    names = list(cells)
    n = len(next(iter(cells.values()))["aligned"])
    result = {"reward_model": rm_name, "n_prompts": n, "n_boot": n_boot, "cells": {}, "policy_pairs": [], "sft_pairs": []}
    md = [f"Reward model: `{rm_name}` | {n} identical prompts | 95% bootstrap CI over prompts | single training seed", ""]

    md += ["| Run | Final policy reward | SFT model reward | Policy - own SFT |", "|---|---|---|---|"]
    for name in names:
        a, s = np.asarray(cells[name]["aligned"]), np.asarray(cells[name]["sft"])
        ra, rs, rd = bootstrap_ci(a, n_boot, seed), bootstrap_ci(s, n_boot, seed), bootstrap_ci(a - s, n_boot, seed)
        result["cells"][name] = {"policy": ra, "sft": rs, "improvement": rd}
        md.append(f"| {name} | {fmt(*ra)} | {fmt(*rs)} | {fmt(*rd)} |")

    for key, label in (("aligned", "Final policies"), ("sft", "SFT models")):
        md += ["", f"**{label}: head-to-head on identical prompts** (win rate of row over column, ties = 1/2; mean reward difference in brackets)", ""]
        pairs = []
        for i, j in itertools.combinations(range(len(names)), 2):
            h = head_to_head(cells[names[i]][key], cells[names[j]][key], n_boot, seed)
            pairs.append({"a": names[i], "b": names[j], **h})
        result["policy_pairs" if key == "aligned" else "sft_pairs"] = pairs
        md += ["| A | B | A wins | reward A - B |", "|---|---|---|---|"]
        for p in pairs:
            md.append(f"| {p['a']} | {p['b']} | {fmt(p['win_rate'], p['win_lo'], p['win_hi'])} | {fmt(p['diff'], p['diff_lo'], p['diff_hi'])} |")
    return result, "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True, metavar="NAME=EVAL_DIR",
                    help="a run to include; EVAL_DIR holds generated_responses.json (repeatable)")
    ap.add_argument("--dataset", default="tldr", help="selects the default reward model (tldr | hh_rlhf | ufb)")
    ap.add_argument("--reward_model", default=None, help="override the reward model id")
    ap.add_argument("--out", required=True, help="output directory for comparison.json / comparison.md")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device_map", default=None, help="'auto' for Skywork-8B (HH-RLHF); leave unset for DeBERTa")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.evaluation.evaluator import DATASET_REWARD_MODEL_MAP, _DEFAULT_REWARD_MODEL
    from src.evaluation.reward_model_eval import RewardModelEvaluator

    rm_name = args.reward_model or DATASET_REWARD_MODEL_MAP.get(args.dataset, _DEFAULT_REWARD_MODEL)
    specs: List[Tuple[str, str]] = []
    for c in args.cell:
        name, _, path = c.partition("=")
        if not path:
            ap.error(f"--cell expects NAME=EVAL_DIR, got {c!r}")
        specs.append((name, path))

    # all runs must use the same prompts in the same order
    ref_prompts: Optional[List[str]] = None
    for name, path in specs:
        prompts = load_generations(path)[0]
        if ref_prompts is None:
            ref_prompts = prompts
        elif prompts != ref_prompts:
            raise SystemExit(f"Run {name!r} was evaluated on different prompts than {specs[0][0]!r}; cannot compare.")

    evaluator = RewardModelEvaluator(
        reward_model_name=rm_name, device="cuda", device_map=args.device_map, batch_size=args.batch_size,
    )
    cells = {name: score_cell(path, evaluator, rm_name) for name, path in specs}

    result, md = build_report(cells, rm_name, args.n_boot, args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
