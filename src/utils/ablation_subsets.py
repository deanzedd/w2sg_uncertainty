"""Selection-control subsets of the weakly labeled set D_u for the MWDPO-BC filtering ablation.

The labeler writes `<label_base_dir>/pseudo_labeled.jsonl` (every pair of D_u, oriented by the ensemble
preference, with `confidence_weight` = sigma(S_w - S_l) and per-head `individual_agreements`) and
`<label_base_dir>/d_high/pseudo_labeled.jsonl` (the agreement-filtered set D_h).  Two controls are
built from the full file, both with exactly |D_h| pairs so that only the selection rule differs:

  random_subset : |D_h| pairs drawn uniformly at random   (controls for "less data")
  conf_only     : the |D_h| pairs with the highest ensemble confidence, ignoring whether the K heads
                  agree                                    (isolates the value of head agreement)

Files go to `<label_base_dir>/ablation_<mode>/` together with a `meta.json` of selection statistics.
Building is deterministic (seed) and idempotent.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List

logger = logging.getLogger(__name__)

MODES = ("random_subset", "conf_only")


def _read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_subset(label_base_dir: str, mode: str, seed: int = 42) -> str:
    """Create (if missing) the subset for `mode` and return the path of its pseudo_labeled.jsonl."""
    if mode not in MODES:
        raise ValueError(f"Unknown ablation subset mode {mode!r}; choose from {MODES}")

    full_path = os.path.join(label_base_dir, "pseudo_labeled.jsonl")
    d_high_path = os.path.join(label_base_dir, "d_high", "pseudo_labeled.jsonl")
    out_dir = os.path.join(label_base_dir, f"ablation_{mode}")
    out_path = os.path.join(out_dir, "pseudo_labeled.jsonl")

    k = _count_lines(d_high_path)
    if k == 0:
        raise ValueError(f"D_h is empty ({d_high_path}); cannot size the {mode} control")

    if os.path.exists(out_path) and _count_lines(out_path) == k:
        logger.info(f"[{mode}] reusing existing subset ({k} pairs): {out_path}")
        return out_path

    rows = _read_jsonl(full_path)
    n = len(rows)
    if k > n:
        raise ValueError(f"|D_h|={k} exceeds |D_u|={n}")

    if mode == "random_subset":
        chosen_idx = sorted(random.Random(seed).sample(range(n), k))
    else:  # conf_only: highest ensemble confidence; ties broken by file order
        order = sorted(range(n), key=lambda i: (-float(rows[i]["confidence_weight"]), i))
        chosen_idx = sorted(order[:k])

    selected = [rows[i] for i in chosen_idx]
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def unanimous(r: Dict) -> bool:
        return all(r.get("individual_agreements", []))

    meta = {
        "mode": mode,
        "seed": seed,
        "n_total": n,
        "n_selected": k,
        "n_in_d_high": sum(1 for r in selected if r.get("in_d_high")),
        "frac_unanimous": sum(unanimous(r) for r in selected) / k,
        "mean_confidence": sum(float(r["confidence_weight"]) for r in selected) / k,
        "min_confidence": min(float(r["confidence_weight"]) for r in selected),
        "d_high_frac_unanimous": sum(unanimous(r) for r in rows if r.get("in_d_high")) / k,
        "d_high_mean_confidence": sum(float(r["confidence_weight"]) for r in rows if r.get("in_d_high")) / k,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(
        f"[{mode}] built {k}/{n} pairs: overlap with D_h {meta['n_in_d_high']}/{k}, "
        f"unanimous {meta['frac_unanimous']:.1%}, mean confidence {meta['mean_confidence']:.3f}"
    )
    return out_path
