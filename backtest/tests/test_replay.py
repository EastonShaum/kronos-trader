"""
Tests for the replay harness.

Covers the five required properties: determinism, entry-side linkage,
model-boundary refusal, read-only enforcement, and a hand-worked fixture whose
expected output is asserted literally.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

import pytest

from backtest.replay import (
    EntrySideLinkageError,
    SimulatedTrade,
    assert_entry_side_linkage,
    replay_symbol,
)
from backtest.settings_resolver import (
    ModelBoundaryError,
    SettingsUnavailableError,
    check_model_boundary,
    resolve,
)
from backtest.snapshot import open_readonly
from backtest.tests.conftest import BASE

RANGE = (date(2026, 8, 20), date(2026, 8, 21))


def _replay(db_path, settings):
    conn = open_readonly(db_path)
    try:
        return replay_symbol(conn, "stocks", "TEST", *RANGE, settings)
    finally:
        conn.close()


# --- 1. Determinism --------------------------------------------------------

def test_replay_is_deterministic(fixture_db, stable_settings):
    first = _replay(fixture_db, stable_settings)
    second = _replay(fixture_db, stable_settings)

    assert first == second
    assert [t.as_row() for t in first] == [t.as_row() for t in second]


# --- 2. Entry-side linkage -------------------------------------------------

def test_entry_linkage_accepts_entry_side_prediction(fixture_db, stable_settings):
    trades = _replay(fixture_db, stable_settings)
    assert len(trades) == 1
    # Prediction 1 is the BUY-side entry prediction, not the SELL's id 109.
    assert trades[0].entry_prediction_id == 1


def test_entry_linkage_rejects_exit_side_prediction():
    """
    The known prior bug: replay reading the SELL trade's exit prediction
    instead of the BUY trade's entry prediction. A prediction that post-dates
    the entry must be rejected.
    """
    entry_ts = BASE + timedelta(minutes=17)
    exit_ts = BASE + timedelta(hours=8)

    mislinked = SimulatedTrade(
        symbol="TEST", market="stocks",
        entry_ts=entry_ts, entry_price=100.0,
        exit_ts=exit_ts, exit_price=103.0,
        exit_reason="take_profit", return_pct=0.03,
        entry_prediction_id=109,          # the SELL's exit-side prediction
        entry_confidence=0.0, entry_model="small",
    )
    # Prediction 109 was created at exit time, not entry time.
    prediction_times = {109: exit_ts, 1: entry_ts}

    with pytest.raises(EntrySideLinkageError, match="post-dates entry"):
        assert_entry_side_linkage(mislinked, prediction_times)


def test_entry_linkage_rejects_unknown_prediction():
    trade = SimulatedTrade(
        symbol="TEST", market="stocks",
        entry_ts=BASE, entry_price=100.0, exit_ts=None, exit_price=100.0,
        exit_reason="horizon", return_pct=0.0, entry_prediction_id=999,
        entry_confidence=0.9, entry_model="small",
    )
    with pytest.raises(EntrySideLinkageError, match="not found"):
        assert_entry_side_linkage(trade, {1: BASE})


# --- 3. Model-boundary refusal --------------------------------------------

def test_model_boundary_refused_without_flag(fixture_db):
    conn = open_readonly(fixture_db)
    try:
        with pytest.raises(ModelBoundaryError, match="2026-07-23"):
            check_model_boundary(conn, date(2026, 7, 20), date(2026, 7, 30),
                                 allow_model_boundary=False)
    finally:
        conn.close()


def test_model_boundary_allowed_with_flag(fixture_db):
    conn = open_readonly(fixture_db)
    try:
        crossed = check_model_boundary(conn, date(2026, 7, 20), date(2026, 7, 30),
                                       allow_model_boundary=True)
        assert date(2026, 7, 23) in crossed
    finally:
        conn.close()


def test_resolve_refuses_without_operator_settings(fixture_db):
    """No silent fallback to the current settings.json."""
    conn = open_readonly(fixture_db)
    try:
        with pytest.raises(SettingsUnavailableError, match="No historical record"):
            resolve(conn, *RANGE, operator_settings=None)
    finally:
        conn.close()


# --- 4. Read-only enforcement ---------------------------------------------

def test_snapshot_is_read_only(fixture_db):
    conn = open_readonly(fixture_db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM trades")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute(
                "INSERT INTO candles (market, symbol, timestamp, open, high, low, close) "
                "VALUES ('stocks','TEST','2026-08-20 13:00:00',1,1,1,1)"
            )
    finally:
        conn.close()


# --- 5. Hand-worked fixture ------------------------------------------------

def test_hand_worked_fixture(fixture_db, stable_settings):
    """
    Worked by hand from conftest.CANDLES:

    Entry  : prediction 1 at 2026-08-20 13:17, spot 100.0, confidence 0.90.
             Predicted move |106.0 - 100.0| / 100.0 = 6% >= 2% minimum. Entry taken.
    Levels : take-profit 100.0 * 1.03 = 103.0, stop-loss 100.0 * 0.97 = 97.0.
    Walk   : bars 0..7 have highs 100.8, 101.0, 101.4, 101.9, 102.2, 102.5,
             102.8, 102.9 - all below 103.0; lows never reach 97.0.
             Bar 8 (2026-08-20 21:00) has high 103.5 >= 103.0 -> take-profit.
    Fill   : _walk_candles fills at the threshold, so exit_price = 103.0 exactly.
    Result : return_pct = (103.0 - 100.0) / 100.0 = 0.03
    """
    trades = _replay(fixture_db, stable_settings)

    assert len(trades) == 1
    trade = trades[0]

    assert trade.symbol == "TEST"
    assert trade.entry_ts == BASE + timedelta(minutes=17)
    assert trade.entry_price == 100.0
    assert trade.exit_ts == BASE + timedelta(hours=8)
    assert trade.exit_price == pytest.approx(103.0)
    assert trade.exit_reason == "take_profit"
    assert trade.return_pct == pytest.approx(0.03)
    assert trade.entry_prediction_id == 1
    assert trade.entry_confidence == pytest.approx(0.90)


def test_fixture_reconciles_against_its_live_trade(fixture_db, stable_settings):
    """The fixture's live trade was built to match the simulated one exactly."""
    from backtest.reconcile import compare, load_live_trades

    conn = open_readonly(fixture_db)
    try:
        live = load_live_trades(conn, "TEST", *RANGE)
        sim = replay_symbol(conn, "stocks", "TEST", *RANGE, stable_settings)
    finally:
        conn.close()

    assert len(live) == 1
    # Entry side must come from the BUY row's prediction, not the SELL's.
    assert live[0].entry_prediction_id == 1

    result = compare("TEST", sim, live, tolerance=0.005)
    assert result.matched == 1
    assert result.divergent == 0
    assert result.missing == 0
    assert result.spurious == 0
    assert result.match_rate == 100.0
