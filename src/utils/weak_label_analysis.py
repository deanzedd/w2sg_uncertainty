"""
Weak-label analysis utility.

Computes and writes an ``analysis.txt`` report into the weak_labels output
directory after any labeling script finishes.

Supports three labeling modes:
  - mwdpo / super_multi_dpo / mwdpo_bootstrap_calibration :
        D_h (high-agreement) + D_l (disagreement) split
  - cwpo / wdpo :
        D_weak only (all D_u samples receive a label)

Proxy accuracy: a sample is "correct" if the weak label preserved the
original dataset order of (chosen, rejected).  This is valid because datasets
like HH-RLHF already carry human-preference ground-truth labels.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def generate_analysis_txt(
    output_dir: str,
    cfg: Any,
    method: str,
    *,
    # For mwdpo / super_multi_dpo (D_h / D_l split)
    d_high: Optional[List[Dict]] = None,
    d_low: Optional[List[Dict]] = None,
    # For cwpo / wdpo (single D_weak list)
    d_all: Optional[List[Dict]] = None,
    # Original D_u samples (same order) used to compute proxy accuracy
    original_samples: Optional[List[Dict]] = None,
    # Extra metadata
    config_path: str = "",
    num_models: int = 1,
) -> str:
    """
    Compute weak-label statistics and write ``analysis.txt`` to *output_dir*.

    Args:
        output_dir:       Folder where ``analysis.txt`` will be written
                          (the same folder that holds ``pseudo_labeled.jsonl``).
        cfg:              OmegaConf / dict config object.
        method:           One of 'mwdpo', 'super_multi_dpo',
                          'mwdpo_bootstrap_calibration', 'cwpo', 'wdpo'.
        d_high:           D_h samples (mwdpo/super_multi_dpo only).
        d_low:            D_l samples (mwdpo/super_multi_dpo only).
        d_all:            All pseudo-labeled samples (cwpo/wdpo only).
        original_samples: Raw D_u samples **in the same order** as the labeled
                          output; used to compute proxy accuracy.
        config_path:      Path to the YAML config (for display in the report).
        num_models:       Number of weak reward models used.

    Returns:
        Absolute path to the written ``analysis.txt``.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "analysis.txt")

    lines = _build_report(
        cfg=cfg,
        method=method,
        d_high=d_high,
        d_low=d_low,
        d_all=d_all,
        original_samples=original_samples,
        config_path=config_path,
        num_models=num_models,
        output_dir=output_dir,
    )

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"[WeakLabelAnalysis] Report written to: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_report(
    cfg: Any,
    method: str,
    output_dir: str,
    d_high: Optional[List[Dict]],
    d_low: Optional[List[Dict]],
    d_all: Optional[List[Dict]],
    original_samples: Optional[List[Dict]],
    config_path: str,
    num_models: int,
) -> List[str]:
    lines: List[str] = []
    _sep = "=" * 68
    _dash = "-" * 68

    # ── Header ──────────────────────────────────────────────────────────────
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines += [
        _sep,
        "  WEAK LABEL ANALYSIS REPORT",
        _sep,
        f"  Generated at : {now}",
        f"  Config file  : {config_path or '(not specified)'}",
        f"  Method       : {method}",
        f"  Dataset      : {_get(cfg, 'dataset_name', '?')}",
        f"  Seed         : {_get(cfg, 'seed', '?')}",
        f"  Labeled ratio: {_get(cfg, 'labeled_ratio', '?')}",
        f"  Weak model   : {_get(cfg, 'weak_model_name', '?')}",
        f"  Num RM       : {num_models}",
        _dash,
        "",
    ]

    has_split = (d_high is not None and d_low is not None)

    if has_split:
        lines += _section_split(
            cfg=cfg,
            method=method,
            d_high=d_high,
            d_low=d_low,
            original_samples=original_samples,
            num_models=num_models,
            output_dir=output_dir,
        )
    else:
        # cwpo / wdpo — single D_weak
        samples = d_all or []
        lines += _section_flat(
            cfg=cfg,
            method=method,
            samples=samples,
            original_samples=original_samples,
            output_dir=output_dir,
        )

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy helpers (lookup-based — order-independent)
# ─────────────────────────────────────────────────────────────────────────────

def _build_original_lookup(
    original_samples: List[Dict],
) -> Dict[tuple, str]:
    """
    Build a lookup dict: (prompt, frozenset({resp_a, resp_b})) → original chosen.

    This allows us to find the ground-truth chosen response for any
    pseudo-labeled sample regardless of ordering, because the labeler may
    reorder samples into D_h / D_l buckets.
    """
    lookup: Dict[tuple, str] = {}
    for o in original_samples:
        prompt = o.get("prompt", "")
        chosen = o.get("chosen", "")
        rejected = o.get("rejected", "")
        key = (prompt, frozenset([chosen, rejected]))
        lookup[key] = chosen
    return lookup


def _check_correct(pseudo_sample: Dict, lookup: Dict[tuple, str]) -> Optional[bool]:
    """
    Check if a pseudo-labeled sample preserved the original chosen/rejected.

    Returns:
        True  — weak label kept original chosen (correct)
        False — weak label flipped chosen/rejected (incorrect)
        None  — could not find matching original sample
    """
    prompt = pseudo_sample.get("prompt", "")
    chosen = pseudo_sample.get("chosen", "")
    rejected = pseudo_sample.get("rejected", "")
    key = (prompt, frozenset([chosen, rejected]))
    orig_chosen = lookup.get(key)
    if orig_chosen is None:
        return None
    return chosen == orig_chosen


def _count_correct_lookup(
    pseudo_list: List[Dict],
    lookup: Dict[tuple, str],
) -> tuple:
    """
    Count correct / incorrect / unmatched samples using lookup.

    Returns:
        (n_correct, n_incorrect, n_unmatched)
    """
    correct = 0
    incorrect = 0
    unmatched = 0
    for p in pseudo_list:
        result = _check_correct(p, lookup)
        if result is True:
            correct += 1
        elif result is False:
            incorrect += 1
        else:
            unmatched += 1
    return correct, incorrect, unmatched


def _split_correct_incorrect(
    pseudo_list: List[Dict],
    lookup: Dict[tuple, str],
) -> tuple:
    """
    Split pseudo_list into (correct_samples, incorrect_samples) using lookup.

    Returns:
        (correct_samples: List[Dict], incorrect_samples: List[Dict])
    """
    correct = []
    incorrect = []
    for p in pseudo_list:
        result = _check_correct(p, lookup)
        if result is True:
            correct.append(p)
        elif result is False:
            incorrect.append(p)
        # Skip unmatched
    return correct, incorrect


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────

def _section_split(
    cfg: Any,
    method: str,
    d_high: List[Dict],
    d_low: List[Dict],
    original_samples: Optional[List[Dict]],
    num_models: int,
    output_dir: str,
) -> List[str]:
    """Sections for methods with D_h / D_l split (mwdpo, super_multi_dpo)."""
    lines: List[str] = []
    _sep = "=" * 68
    _dash = "-" * 68

    total = len(d_high) + len(d_low)

    # ── Dataset split ────────────────────────────────────────────────────────
    lines += [
        "  [1] DATASET SPLIT",
        _dash,
        f"  D_u total              : {total:>7,}",
        f"  D_h (agree / filtered) : {len(d_high):>7,}  ({_pct(len(d_high), total)})",
        f"  D_l (disagree / rest)  : {len(d_low):>7,}  ({_pct(len(d_low), total)})",
        "",
    ]

    # ── Agreement mode ───────────────────────────────────────────────────────
    # For mwdpo_bootstrap_calibration, agreement_mode lives in bootstrap_calibration block.
    mw_cfg = (
        _get(cfg, "multi_weak", {})
        or _get(cfg, "super_multi", {})
        or _get(cfg, "bootstrap_calibration", {})
        or {}
    )
    agreement_mode = _dictget(mw_cfg, "agreement_mode", "unanimous")
    conf_threshold = _dictget(mw_cfg, "confidence_threshold", 0.8)
    lines += [
        "  [2] AGREEMENT SETTINGS",
        _dash,
        f"  Agreement mode     : {agreement_mode}",
        f"  Conf. threshold    : {conf_threshold}  (only for unanimous_with_threshold)",
        f"  Num reward models  : {num_models}",
        "",
    ]

    # ── Proxy accuracy (lookup-based, order-independent) ─────────────────────
    lines += [
        "  [3] PROXY ACCURACY  (vs. original dataset labels = human ground truth)",
        _dash,
    ]
    if original_samples:
        lookup = _build_original_lookup(original_samples)

        correct_dh, incorrect_dh, unmatched_dh = _count_correct_lookup(d_high, lookup)
        correct_dl, incorrect_dl, unmatched_dl = _count_correct_lookup(d_low, lookup)
        correct_all = correct_dh + correct_dl
        matched_dh = correct_dh + incorrect_dh
        matched_dl = correct_dl + incorrect_dl
        matched_all = matched_dh + matched_dl

        acc_dh  = correct_dh  / max(1, matched_dh)
        acc_dl  = correct_dl  / max(1, matched_dl)
        acc_all = correct_all / max(1, matched_all)

        flip_dh  = incorrect_dh
        flip_dl  = incorrect_dl
        flip_all = flip_dh + flip_dl

        lines += [
            f"  D_h accuracy  : {acc_dh*100:6.2f}%   ({correct_dh:,} / {matched_dh:,} correct)",
            f"  D_l accuracy  : {acc_dl*100:6.2f}%   ({correct_dl:,} / {matched_dl:,} correct)",
            f"  Overall acc.  : {acc_all*100:6.2f}%   ({correct_all:,} / {matched_all:,} correct)",
            f"  D_h flip rate : {(flip_dh/max(1,matched_dh))*100:6.2f}%   ({flip_dh:,} labels flipped)",
            f"  D_l flip rate : {(flip_dl/max(1,matched_dl))*100:6.2f}%   ({flip_dl:,} labels flipped)",
            f"  Total flips   : {(flip_all/max(1,matched_all))*100:6.2f}%   ({flip_all:,} labels flipped)",
        ]

        # Report unmatched if any
        total_unmatched = unmatched_dh + unmatched_dl
        if total_unmatched > 0:
            lines.append(
                f"  [WARN] Unmatched samples (no original found): {total_unmatched:,} "
                f"({_pct(total_unmatched, total)})"
            )
    else:
        lines.append("  [SKIP] original_samples not provided — proxy accuracy not computed.")
    lines.append("")

    # ── Confidence statistics ─────────────────────────────────────────────────
    lines += ["  [4] CONFIDENCE STATISTICS", _dash]
    lines += _conf_stats_block("D_h", d_high)
    lines += _conf_stats_block("D_l", d_low)
    lines.append("")

    # ── Correct vs Incorrect (confidence breakdown, lookup-based) ────────────
    lines += ["  [5] CONFIDENCE BY CORRECTNESS", _dash]
    if original_samples:
        lookup = _build_original_lookup(original_samples)
        for label, group in [("D_h", d_high), ("D_l", d_low)]:
            correct_samp, wrong_samp = _split_correct_incorrect(group, lookup)
            lines += _conf_correct_block(label, correct_samp, wrong_samp)
    else:
        lines.append("  [SKIP] original_samples not available.")
    lines.append("")

    # ── Per-model agreement breakdown (MWDPO only) ──────────────────────────
    all_pseudo = list(d_high) + list(d_low)
    has_individual = any("individual_agreements" in s for s in all_pseudo)
    if has_individual and num_models > 1:
        lines += ["  [6] PER-MODEL AGREEMENT BREAKDOWN", _dash]
        lines += _per_model_stats(all_pseudo, num_models)
        lines.append("")

    # ── Output paths ─────────────────────────────────────────────────────────
    lines += [
        "  [7] OUTPUT PATHS",
        _dash,
        f"  D_h pseudo_labeled.jsonl : {os.path.join(output_dir, 'd_high', 'pseudo_labeled.jsonl')}",
        f"  D_l pseudo_labeled.jsonl : {os.path.join(output_dir, 'd_low', 'pseudo_labeled.jsonl')}",
        f"  Combined (D_h + D_l)     : {os.path.join(output_dir, 'pseudo_labeled.jsonl')}",
        f"  This analysis file       : {os.path.join(output_dir, 'analysis.txt')}",
        "",
        "=" * 68,
    ]
    return lines


def _section_flat(
    cfg: Any,
    method: str,
    samples: List[Dict],
    original_samples: Optional[List[Dict]],
    output_dir: str,
) -> List[str]:
    """Sections for methods without D_h/D_l split (cwpo, wdpo)."""
    lines: List[str] = []
    _sep = "=" * 68
    _dash = "-" * 68

    total = len(samples)

    # ── Dataset ───────────────────────────────────────────────────────────────
    lines += [
        "  [1] DATASET",
        _dash,
        f"  D_u total (labeled)  : {total:>7,}",
        f"  Coverage             : 100%  (all D_u samples receive a label)",
        "",
    ]

    # ── Proxy accuracy (lookup-based for safety) ──────────────────────────────
    lines += [
        "  [2] PROXY ACCURACY  (vs. original dataset labels = human ground truth)",
        _dash,
    ]
    if original_samples:
        lookup = _build_original_lookup(original_samples)
        correct, incorrect, unmatched = _count_correct_lookup(samples, lookup)
        matched = correct + incorrect
        acc = correct / max(1, matched)
        flip_rate = incorrect / max(1, matched)
        lines += [
            f"  Accuracy  (same as original) : {acc*100:6.2f}%   ({correct:,} / {matched:,})",
            f"  Flip rate (label reversed)   : {flip_rate*100:6.2f}%   ({incorrect:,} / {matched:,})",
        ]
        if unmatched > 0:
            lines.append(
                f"  [WARN] Unmatched samples: {unmatched:,} ({_pct(unmatched, total)})"
            )
    else:
        lines.append("  [SKIP] original_samples not provided.")
    lines.append("")

    # ── Confidence statistics ─────────────────────────────────────────────────
    lines += ["  [3] CONFIDENCE STATISTICS", _dash]
    lines += _conf_stats_block("D_weak", samples)
    lines.append("")

    # ── Correct vs Incorrect (lookup-based) ──────────────────────────────────
    lines += ["  [4] CONFIDENCE BY CORRECTNESS", _dash]
    if original_samples:
        lookup = _build_original_lookup(original_samples)
        correct_samp, wrong_samp = _split_correct_incorrect(samples, lookup)
        lines += _conf_correct_block("D_weak", correct_samp, wrong_samp)
    else:
        lines.append("  [SKIP] original_samples not available.")
    lines.append("")

    # ── Output paths ──────────────────────────────────────────────────────────
    lines += [
        "  [5] OUTPUT PATHS",
        _dash,
        f"  D_weak pseudo_labeled.jsonl : {os.path.join(output_dir, 'pseudo_labeled.jsonl')}",
        f"  This analysis file          : {os.path.join(output_dir, 'analysis.txt')}",
        "",
        "=" * 68,
    ]
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _conf_stats_block(label: str, samples: List[Dict]) -> List[str]:
    """Return lines with percentile/mean/std of confidence_weight for *samples*."""
    if not samples:
        return [f"  {label} : (empty — no samples)"]

    confs = sorted(s.get("confidence_weight", 0.0) for s in samples)
    n = len(confs)
    mean = sum(confs) / n
    variance = sum((c - mean) ** 2 for c in confs) / n
    std = math.sqrt(variance)

    def pct(p):
        idx = max(0, min(n - 1, int(p / 100 * n)))
        return confs[idx]

    return [
        f"  {label} (N={n:,}):",
        f"    mean={mean:.4f}  std={std:.4f}  min={confs[0]:.4f}  max={confs[-1]:.4f}",
        f"    p10={pct(10):.4f}  p25={pct(25):.4f}  p50={pct(50):.4f}  "
        f"p75={pct(75):.4f}  p90={pct(90):.4f}",
    ]


def _conf_correct_block(
    label: str,
    correct: List[Dict],
    wrong: List[Dict],
) -> List[str]:
    lines = []
    for sublabel, group in [("Correct", correct), ("Incorrect", wrong)]:
        n = len(group)
        if n == 0:
            lines.append(f"  {label} — {sublabel} (N=0): —")
            continue
        confs = [s.get("confidence_weight", 0.0) for s in group]
        mean = sum(confs) / n
        variance = sum((c - mean) ** 2 for c in confs) / n
        std = math.sqrt(variance)
        lines.append(
            f"  {label} — {sublabel} (N={n:,}):  "
            f"mean_conf={mean:.4f}  std={std:.4f}"
        )
    return lines


def _per_model_stats(samples: List[Dict], num_models: int) -> List[str]:
    """Per-model agreement rates and pairwise inter-model agreement."""
    lines = []
    n = len(samples)
    if n == 0:
        return ["  (no samples)"]

    # Per-model agreement rate with ensemble decision
    for i in range(num_models):
        agreed = sum(
            1 for s in samples
            if i < len(s.get("individual_agreements", []))
            and s["individual_agreements"][i]
        )
        lines.append(f"  Model {i} agreement rate with ensemble : {agreed/n*100:.1f}%  ({agreed:,}/{n:,})")

    # Pairwise inter-model agreement
    for i in range(num_models):
        for j in range(i + 1, num_models):
            match = 0
            for s in samples:
                ia = s.get("individual_agreements", [])
                if i < len(ia) and j < len(ia) and (ia[i] == ia[j]):
                    match += 1
            lines.append(
                f"  Pairwise agreement (model {i} \u2194 model {j}) : {match/n*100:.1f}%  ({match:,}/{n:,})"
            )

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safe attribute/key access for both OmegaConf and plain dict."""
    try:
        # OmegaConf
        if hasattr(cfg, "__getattr__"):
            val = getattr(cfg, key, None)
            if val is not None:
                return val
        # dict-like
        return cfg.get(key, default)
    except Exception:
        return default


def _dictget(d: Any, key: str, default: Any = None) -> Any:
    """Access dict or OmegaConf safely."""
    if d is None:
        return default
    try:
        if hasattr(d, "get"):
            return d.get(key, default)
        return getattr(d, key, default)
    except Exception:
        return default


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "0.0%"
    return f"{100 * num / denom:.1f}%"
