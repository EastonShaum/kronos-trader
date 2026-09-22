"""Tests for the fill models, overshoot fit, and directionality statistics."""
from __future__ import annotations

from datetime import datetime

import pytest

from backtest.fill_data import SL_PCT, FillObservation, tp_pct_at
from backtest.fill_models import (
    Bar,
    EmpiricalFill,
    ExactFill,
    WorstCaseFill,
    build_empirical,
    make_rng,
)
from backtest.overshoot_fit import fit
from backtest.stats import moving_block_bootstrap_median, sign_test

BAR = Bar(open=100.0, high=104.0, low=96.0, close=101.0)


def _obs(exit_type: str, overshoot: float, threshold_pct: float = 0.03) -> FillObservation:
    ts = datetime(2026, 9, 10, 15, 0, 0)
    return FillObservation(
        symbol="TEST", market="stocks", exit_type=exit_type,
        exit_route="price_monitor", entry_ts=ts, entry_price=100.0, exit_ts=ts,
        threshold_pct=threshold_pct, threshold_price=100.0 * (1 - threshold_pct),
        fill_price=100.0, overshoot=overshoot, hour_utc=15, crossed_session=False,
        bar_open=100.0, bar_high=104.0, bar_low=96.0, bar_close=101.0,
        outside_bar=False, atr_pct=0.01,
    )


# --- ExactFill: the null model --------------------------------------------

def test_exact_fill_returns_threshold():
    """Reproduces Patch 1 behaviour exactly - this is the bug, kept as null."""
    model = ExactFill()
    assert model.price("SL", 97.0, BAR, None) == 97.0
    assert model.price("TP", 103.0, BAR, None) == 103.0


def test_worst_case_fill_uses_bar_extremes():
    model = WorstCaseFill()
    assert model.price("SL", 97.0, BAR, None) == 96.0
    assert model.price("TP", 103.0, BAR, None) == 104.0


# --- EmpiricalFill: direction, determinism, and required seeding -----------

def test_empirical_fill_requires_explicit_rng():
    model = build_empirical([_obs("SL", 0.005)])
    with pytest.raises(ValueError, match="requires an explicit rng"):
        model.price("SL", 97.0, BAR, None)


def test_empirical_fill_overshoot_direction():
    """
    Overshoot is positive past the threshold in the direction of travel:
    a stop fills BELOW its threshold, a take-profit ABOVE.
    """
    sl_model = build_empirical([_obs("SL", 0.01)])
    tp_model = build_empirical([_obs("TP", 0.01)])

    sl_fill = sl_model.price("SL", 97.0, BAR, make_rng(0, 0))
    tp_fill = tp_model.price("TP", 103.0, BAR, make_rng(0, 0))

    assert sl_fill < 97.0
    assert tp_fill > 103.0
    assert sl_fill == pytest.approx(97.0 * 0.99)
    assert tp_fill == pytest.approx(103.0 * 1.01)


def test_empirical_fill_is_reproducible_and_seed_sensitive():
    obs = [_obs("SL", x) for x in (0.001, 0.004, 0.009, 0.02, 0.05)]
    model = build_empirical(obs)

    run_a = [model.price("SL", 97.0, BAR, make_rng(7, d)) for d in range(20)]
    run_b = [model.price("SL", 97.0, BAR, make_rng(7, d)) for d in range(20)]
    run_c = [model.price("SL", 97.0, BAR, make_rng(99, d)) for d in range(20)]

    assert run_a == run_b, "same seed must reproduce exactly"
    assert run_a != run_c, "different seed must produce a different stream"


def test_make_rng_does_not_depend_on_process_hash_seed():
    """Seeding must be stable across processes, so no reliance on hash()."""
    assert make_rng(3, 4).random() == make_rng(3, 4).random()
    assert make_rng(3, 4).random() != make_rng(4, 3).random()


def test_clamping_pulls_fill_into_bar():
    model = build_empirical([_obs("SL", 0.50)], clamp_to_bar=True)
    fill = model.price("SL", 97.0, BAR, make_rng(0, 0))
    assert fill == pytest.approx(BAR.low)
    assert model.out_of_bar_draws == 1


def test_unclamped_fill_may_leave_bar_and_is_counted():
    model = build_empirical([_obs("SL", 0.50)], clamp_to_bar=False)
    fill = model.price("SL", 97.0, BAR, make_rng(0, 0))
    assert fill < BAR.low
    assert model.out_of_bar_draws == 1


# --- Overshoot fit: body/tail separation ----------------------------------

def test_body_and_tail_are_separated():
    """
    A -37% stop is gap risk; a -0.3% stop is sampling latency. Fitting one
    distribution across both would erase one or exaggerate the other.
    """
    obs = [_obs("SL", x) for x in (0.002, 0.003, 0.004, 0.005, 0.35)]
    result = fit(obs, "SL")

    assert len(result.body) == 4
    assert result.tail == [0.35]
    assert result.tail_rate == pytest.approx(0.2)
    # The outlier must not drag the body summary.
    assert result.median == pytest.approx(0.0035)
    assert result.maximum == pytest.approx(0.005)


def test_tp_threshold_era_switch():
    assert tp_pct_at(datetime(2026, 8, 1)) == 0.06
    assert tp_pct_at(datetime(2026, 9, 10)) == 0.03
    assert SL_PCT == 0.03


# --- Statistics ------------------------------------------------------------

def test_sign_test_detects_systematic_bias():
    biased = [0.5] * 40 + [-0.1] * 5
    result = sign_test(biased)
    assert result["skew"] == "favours simulator"
    assert result["p_value"] < 0.001


def test_sign_test_balanced_sample_is_not_significant():
    balanced = [0.3, -0.3] * 30
    result = sign_test(balanced)
    assert result["p_value"] > 0.5


def test_sign_test_counts_ties_separately():
    result = sign_test([0.0, 0.0, 1.0, -1.0])
    assert result["ties"] == 2
    assert result["n"] == 2


def test_moving_block_bootstrap_straddles_zero_for_centred_data():
    centred = [0.1, -0.1] * 50
    result = moving_block_bootstrap_median(centred, block=5, n_resamples=500, seed=0)
    assert result["straddles_zero"]


def test_moving_block_bootstrap_excludes_zero_for_shifted_data():
    shifted = [1.0 + 0.01 * i for i in range(100)]
    result = moving_block_bootstrap_median(shifted, block=5, n_resamples=500, seed=0)
    assert not result["straddles_zero"]
    assert result["ci_low"] > 0


def test_bootstrap_is_reproducible():
    values = [0.1 * ((-1) ** i) + 0.01 * i for i in range(60)]
    a = moving_block_bootstrap_median(values, block=5, n_resamples=300, seed=42)
    b = moving_block_bootstrap_median(values, block=5, n_resamples=300, seed=42)
    assert a == b
