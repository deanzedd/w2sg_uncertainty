"""Tests for the statistics and report of scripts/compare_policies.py (no model or GPU needed)."""

import importlib.util
import os

import numpy as np

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "compare_policies.py")
_spec = importlib.util.spec_from_file_location("compare_policies", _PATH)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def test_bootstrap_ci_brackets_the_mean_and_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = rng.normal(1.0, 1.0, 50)
    large = rng.normal(1.0, 1.0, 5000)
    m, lo, hi = cp.bootstrap_ci(small, n_boot=2000)
    assert lo <= m <= hi
    _, lo2, hi2 = cp.bootstrap_ci(large, n_boot=2000)
    assert (hi2 - lo2) < (hi - lo)


def test_win_indicator_counts_ties_as_half():
    w = cp.win_indicator([2.0, 1.0, 0.0], [1.0, 1.0, 3.0])
    assert w.tolist() == [1.0, 0.5, 0.0]


def test_head_to_head_is_antisymmetric_and_detects_a_dominant_policy():
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 1.0, 300)
    better = base + 0.5
    h = cp.head_to_head(better, base, n_boot=2000)
    assert h["win_rate"] == 1.0 and h["win_lo"] == 1.0
    assert h["diff"] > 0.49 and h["diff_lo"] > 0
    g = cp.head_to_head(base, better, n_boot=2000)
    assert abs(h["win_rate"] + g["win_rate"] - 1.0) < 1e-9
    assert abs(h["diff"] + g["diff"]) < 1e-9


def test_build_report_lists_every_run_and_pair():
    rng = np.random.default_rng(2)
    cells = {name: {"aligned": rng.normal(m, 1, 100).tolist(), "sft": rng.normal(0, 1, 100).tolist()}
             for name, m in (("A", 1.0), ("B", 0.5), ("C", 0.0))}
    result, md = cp.build_report(cells, "rm/test", n_boot=500, seed=0)
    assert set(result["cells"]) == {"A", "B", "C"}
    assert len(result["policy_pairs"]) == 3 and len(result["sft_pairs"]) == 3
    assert "| A |" in md and "rm/test" in md and "100 identical prompts" in md
