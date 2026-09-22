"""
Regression tests for the audit evidence guards.

Each case below is a recommendation the 2026-09-22 monthly audit actually
produced and that was actually applied to live settings before being reverted.
The prompt now tells the model not to make these, but a prompt is a request;
_validate_evidence is the enforcement, and these tests are what keep it honest.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit_engine import MIN_TRADES_FOR_CI, _validate_evidence  # noqa: E402


def _data(**over):
    base = {
        "symbol_performance": {
            "PLENTY": {"trade_count": 40, "insufficient_sample": False,
                       "ci_excludes_zero": True, "mean_pnl_95ci": [-2.0, -0.5]},
            "NOISY": {"trade_count": 40, "insufficient_sample": False,
                      "ci_excludes_zero": False, "mean_pnl_95ci": [-1.2, 0.8]},
            "THIN": {"trade_count": 5, "insufficient_sample": True,
                     "ci_excludes_zero": False, "mean_pnl_95ci": None},
        },
        "model_comparison": {
            "comparison_validity": {"overlap_hours": 0.0, "confounded_by_period": True},
        },
        "current_settings": {"risk": {"min_predicted_move_pct": 0.02}},
        "prediction_accuracy": {
            "X": {"calibration_by_predicted_move": {
                "predicted_move_0-2%": {"sample_count": 100, "hit_rate_pct": 53.0},
                "predicted_move_2-4%": {"sample_count": 100, "hit_rate_pct": 48.0},
                "predicted_move_4-6%": {"sample_count": 600, "hit_rate_pct": 47.0},
                "predicted_move_>=10%": {"sample_count": 2, "hit_rate_pct": 33.0},
            }},
        },
    }
    base.update(over)
    return base


def _rec(param, value, category="symbol_list", confidence="medium"):
    return {"category": category, "param": param, "proposed_value": value,
            "confidence": confidence, "reasoning": "original reasoning"}


# --- Symbol removal on thin samples (the GDX and HOOD case) ---------------

def test_removal_blocked_when_sample_is_insufficient():
    out = _validate_evidence(_rec("remove_symbol", "THIN"), _data())
    assert out["confidence"] == "insufficient_evidence"
    assert str(MIN_TRADES_FOR_CI) in out["reasoning"]
    assert "original reasoning" in out["reasoning"], "must not discard the original text"


def test_removal_blocked_when_ci_includes_zero():
    out = _validate_evidence(_rec("remove_symbol", "NOISY"), _data())
    assert out["confidence"] == "ci_includes_zero"


def test_removal_allowed_when_ci_excludes_zero():
    out = _validate_evidence(_rec("remove_symbol", "PLENTY"), _data())
    assert out["confidence"] == "medium", "a well-evidenced removal must pass through"
    assert out["reasoning"] == "original reasoning"


def test_removal_of_unknown_symbol_is_flagged():
    out = _validate_evidence(_rec("remove_symbol", "GHOST"), _data())
    assert out["confidence"] == "no_evidence"


# --- Model switch across non-overlapping windows (the active_model case) ---

def test_model_switch_blocked_when_windows_do_not_overlap():
    out = _validate_evidence(
        _rec("active_model", "small", category="model_strategy", confidence="high"),
        _data())
    assert out["confidence"] == "confounded"
    assert "overlap_hours=0.0" in out["reasoning"]


def test_model_switch_allowed_when_windows_overlap():
    data = _data(model_comparison={
        "comparison_validity": {"overlap_hours": 240.0, "confounded_by_period": False}})
    out = _validate_evidence(
        _rec("active_model", "small", category="model_strategy", confidence="high"),
        data)
    assert out["confidence"] == "high"


# --- Entry-threshold change that guts trade volume (the 2% -> 10% case) ---

def test_threshold_raise_that_collapses_volume_is_flagged():
    """The live 2% -> 10% change: only the >=10% bucket survives, 2 of 802."""
    out = _validate_evidence(
        _rec("min_predicted_move_pct", "10.0", category="risk_param"), _data())
    assert out["confidence"] == "volume_collapse"
    assert "2/802" in out["reasoning"]


def test_threshold_raise_reports_volume_impact_even_when_survivable():
    """A raise that keeps most of the population is allowed, but must be quantified."""
    out = _validate_evidence(
        _rec("min_predicted_move_pct", "4.0", category="risk_param"), _data())
    assert out["confidence"] != "volume_collapse"
    assert "VOLUME IMPACT" in out["reasoning"]
    assert "602/802" in out["reasoning"]


def test_threshold_held_at_current_value_is_not_a_raise():
    out = _validate_evidence(
        _rec("min_predicted_move_pct", "2.0", category="risk_param"), _data())
    assert out["reasoning"] == "original reasoning"


def test_threshold_lowering_is_not_volume_checked():
    out = _validate_evidence(
        _rec("min_predicted_move_pct", "1.0", category="risk_param"), _data())
    assert out["confidence"] == "medium"
    assert out["reasoning"] == "original reasoning"


# --- Unrelated recommendations pass through untouched ---------------------

def test_unrelated_param_is_untouched():
    out = _validate_evidence(
        _rec("stop_loss_pct", "-2.5", category="risk_param"), _data())
    assert out["confidence"] == "medium"
    assert out["reasoning"] == "original reasoning"
