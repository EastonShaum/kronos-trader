"""Tests for the project-facing bootstrap wrapper over vendored arch."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats_bootstrap import block_bootstrap_ci, resolve_block_size, spa_test  # noqa: E402


@pytest.fixture
def sample():
    return np.random.default_rng(12345).normal(0.4, 1.0, 150)


def test_same_seed_gives_identical_ci(sample):
    a = block_bootstrap_ci(sample, seed=42, reps=400)
    b = block_bootstrap_ci(sample, seed=42, reps=400)
    assert a == b


def test_different_seed_changes_ci(sample):
    a = block_bootstrap_ci(sample, seed=1, reps=400)
    b = block_bootstrap_ci(sample, seed=2, reps=400)
    assert (a["lower"], a["upper"]) != (b["lower"], b["upper"])


def test_ci_contains_the_sample_mean(sample):
    result = block_bootstrap_ci(sample, seed=7, reps=800)
    assert result["lower"] <= result["mean"] <= result["upper"]
    assert result["mean"] == pytest.approx(float(np.mean(sample)))


def test_constant_series_is_degenerate_and_does_not_crash():
    result = block_bootstrap_ci([2.5] * 40, seed=3, reps=300)
    assert result["mean"] == pytest.approx(2.5)
    assert result["lower"] == pytest.approx(2.5)
    assert result["upper"] == pytest.approx(2.5)
    assert result["n"] == 40


def test_small_n_does_not_crash_and_reports_method_used():
    result = block_bootstrap_ci(
        np.random.default_rng(0).normal(size=10), seed=5, reps=300
    )
    assert result["n"] == 10
    assert result["method_used"] in {"bca", "percentile"}
    assert result["lower"] <= result["mean"] <= result["upper"]


def test_bca_fallback_is_recorded_not_hidden():
    """A constant series degenerates bca's acceleration term."""
    result = block_bootstrap_ci([1.0] * 25, seed=0, reps=200, method="bca")
    assert result["method_used"] == "percentile"


def test_auto_block_size_resolves_to_positive_int(sample):
    result = block_bootstrap_ci(sample, block_size=None, seed=11, reps=300)
    assert isinstance(result["block_size"], int)
    assert result["block_size"] >= 1


def test_resolve_block_size_handles_degenerate_input():
    assert resolve_block_size([1.0] * 30) >= 1
    assert resolve_block_size([1.0]) == 1
    assert resolve_block_size([]) == 1


def test_explicit_block_size_is_respected(sample):
    result = block_bootstrap_ci(sample, block_size=24, seed=0, reps=300)
    assert result["block_size"] == 24


def test_block_size_is_clamped_to_sample_length():
    result = block_bootstrap_ci([1.0, 2.0, 3.0], block_size=99, seed=0, reps=200)
    assert result["block_size"] == 3


def test_empty_input_returns_nan_without_crashing():
    result = block_bootstrap_ci([], seed=0)
    assert result["n"] == 0
    assert np.isnan(result["mean"])


def test_non_finite_values_are_dropped():
    values = [1.0, 2.0, float("nan"), 3.0, float("inf")]
    result = block_bootstrap_ci(values, seed=0, reps=200)
    assert result["n"] == 3


def test_kind_must_be_known(sample):
    with pytest.raises(ValueError, match="kind must be one of"):
        block_bootstrap_ci(sample, kind="bogus", seed=0)


@pytest.mark.parametrize("kind", ["moving", "circular", "stationary"])
def test_all_bootstrap_kinds_run(sample, kind):
    result = block_bootstrap_ci(sample, kind=kind, seed=0, reps=200)
    assert result["kind"] == kind
    assert result["lower"] <= result["upper"]


def test_spa_test_runs_and_is_reproducible():
    rng = np.random.default_rng(3)
    benchmark = rng.normal(0.0, 1.0, 120)
    models = rng.normal(-0.2, 1.0, (120, 3))

    a = spa_test(benchmark, models, block_size=24, reps=250, seed=9)
    b = spa_test(benchmark, models, block_size=24, reps=250, seed=9)

    assert a == b
    assert a["n"] == 120
    assert a["n_models"] == 3
    assert 0.0 <= a["pvalue_consistent"] <= 1.0


def test_spa_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="obs"):
        spa_test(np.zeros(50), np.zeros((40, 2)), block_size=24, reps=100)


def test_spa_accepts_one_dimensional_models():
    rng = np.random.default_rng(4)
    result = spa_test(rng.normal(size=80), rng.normal(size=80),
                      block_size=24, reps=200, seed=1)
    assert result["n_models"] == 1
