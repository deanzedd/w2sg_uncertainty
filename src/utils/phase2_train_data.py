"""
Phase 2 training jsonl resolution for mwdpo_*_2phase / ACE-DPO.

Modes:
  d_low              — use scored D_l only (legacy residual Phase 2)
  d_weak_asymmetric  — merge D_h (w=1) ∪ scored D_l (C_combined) for ACE coverage
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

PHASE2_DATA_MODES = ("d_low", "d_weak_asymmetric")


def _load_jsonl_dicts(path: str) -> List[Dict]:
    """Load a JSONL file as a list of plain dicts. Fail fast if missing/empty."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSONL not found: {path}")
    samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"JSONL is empty: {path}")
    return samples


def _write_jsonl_dicts(path: str, samples: List[Dict]) -> None:
    """Write samples as JSONL, creating parent dirs as needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def build_phase2_train_jsonl(
    phase2_data_mode: str,
    d_high_path: str,
    d_low_scored_path: str,
    d_weak_scored_path: str,
) -> str:
    """
    Resolve the jsonl path for Phase 2d (Debate-Weighted / ACE coverage) training.

    Modes:
      d_low (default / legacy):
          return d_low_scored_path unchanged (must already exist).
      d_weak_asymmetric (ACE-DPO):
          merge D_h (confidence_weight=1.0, in_d_high=True) with scored D_l
          (keep C_combined weights, in_d_high=False) → write d_weak_scored_path.

    Returns:
        Path to pass as --pseudo_labels for Phase 2d.
    """
    mode = (phase2_data_mode or "d_low").strip()
    if mode not in PHASE2_DATA_MODES:
        raise ValueError(
            f"Invalid phase2_data_mode='{mode}'. Choose from: {PHASE2_DATA_MODES}"
        )

    if mode == "d_low":
        if not os.path.isfile(d_low_scored_path):
            raise FileNotFoundError(
                f"phase2_data_mode=d_low requires scored D_l at: {d_low_scored_path}"
            )
        logger.info(
            f"[Phase2 data] mode=d_low → using scored D_l only: {d_low_scored_path}"
        )
        return d_low_scored_path

    # ── d_weak_asymmetric: D_h (w=1) ∪ D_l (C_combined) ─────────────────────
    d_high = _load_jsonl_dicts(d_high_path)
    d_low = _load_jsonl_dicts(d_low_scored_path)

    merged: List[Dict] = []
    for s in d_high:
        if "prompt" not in s or "chosen" not in s or "rejected" not in s:
            raise ValueError(
                f"D_h sample missing prompt/chosen/rejected in {d_high_path}"
            )
        row = dict(s)
        row["confidence_weight"] = 1.0
        row["in_d_high"] = True
        merged.append(row)

    for s in d_low:
        if "prompt" not in s or "chosen" not in s or "rejected" not in s:
            raise ValueError(
                f"D_l scored sample missing prompt/chosen/rejected in {d_low_scored_path}"
            )
        if "confidence_weight" not in s:
            raise ValueError(
                f"D_l scored sample missing confidence_weight in {d_low_scored_path}. "
                "Run compute_strong_confidence.py first."
            )
        try:
            w = float(s["confidence_weight"])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Non-numeric confidence_weight in scored D_l ({d_low_scored_path}): {e}"
            ) from e
        if not math.isfinite(w):
            raise ValueError(
                f"Non-finite confidence_weight in scored D_l ({d_low_scored_path}): {w}"
            )
        row = dict(s)
        row["confidence_weight"] = w
        row["in_d_high"] = False
        merged.append(row)

    n_dh, n_dl = len(d_high), len(d_low)
    w_dh = [float(s["confidence_weight"]) for s in merged if s.get("in_d_high")]
    w_dl = [float(s["confidence_weight"]) for s in merged if not s.get("in_d_high")]
    mean_dh = sum(w_dh) / len(w_dh) if w_dh else float("nan")
    mean_dl = sum(w_dl) / len(w_dl) if w_dl else float("nan")

    if abs(mean_dh - 1.0) > 1e-6:
        raise RuntimeError(
            f"ACE merge invariant violated: mean(w|D_h)={mean_dh} (expected 1.0)"
        )

    _write_jsonl_dicts(d_weak_scored_path, merged)
    logger.info(
        f"[Phase2 data] mode=d_weak_asymmetric → wrote {len(merged):,} samples "
        f"(D_h={n_dh:,}, D_l={n_dl:,}) to {d_weak_scored_path}"
    )
    logger.info(
        f"[Phase2 data] mean(w|D_h)={mean_dh:.6f}, mean(w|D_l)={mean_dl:.6f}"
    )
    return d_weak_scored_path
