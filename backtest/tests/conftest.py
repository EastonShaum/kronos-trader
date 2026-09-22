"""Hand-built fixture database for the replay tests."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SCHEMA = """
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY, symbol VARCHAR(20) NOT NULL, market VARCHAR(6) NOT NULL,
    current_price FLOAT NOT NULL, predicted_high FLOAT, predicted_low FLOAT,
    predicted_close FLOAT NOT NULL, confidence FLOAT, horizon_hours INTEGER,
    triggered_by VARCHAR(20), raw_output TEXT, created_at DATETIME,
    model_name VARCHAR(20) DEFAULT 'small'
);
CREATE TABLE trades (
    id INTEGER PRIMARY KEY, portfolio_id INTEGER, symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(4) NOT NULL, quantity FLOAT NOT NULL, price FLOAT NOT NULL,
    total_value FLOAT NOT NULL, status VARCHAR(6), realized_pnl FLOAT,
    prediction_id INTEGER, opened_at DATETIME, closed_at DATETIME, notes TEXT,
    gross_pnl REAL, net_pnl REAL, entry_prediction_id INTEGER
);
CREATE TABLE candles (
    id INTEGER PRIMARY KEY, market VARCHAR(20) NOT NULL, symbol VARCHAR(20) NOT NULL,
    timestamp DATETIME NOT NULL, open FLOAT, high FLOAT, low FLOAT, close FLOAT,
    volume FLOAT
);
CREATE TABLE audit_recommendations (
    id INTEGER PRIMARY KEY, created_at DATETIME, audit_type VARCHAR, category VARCHAR,
    param VARCHAR, current_value VARCHAR, proposed_value VARCHAR, confidence VARCHAR,
    reasoning TEXT, status VARCHAR, applied_at DATETIME, reviewer_notes TEXT, source TEXT
);
"""

BASE = datetime(2026, 8, 20, 13, 0, 0)

# 14 hourly candles. Entry is driven by the prediction at 13:17 at spot 100.0.
# Highs stay below the +3% take-profit (103.0) until bar 8, where high=103.5
# crosses it. Lows stay above the -3% stop (97.0) throughout.
CANDLES = [
    # (hour_offset, open,  high,  low,   close)
    (0,  100.0, 100.8,  99.5, 100.2),
    (1,  100.2, 101.0,  99.8, 100.7),
    (2,  100.7, 101.4, 100.1, 101.0),
    (3,  101.0, 101.9, 100.4, 101.5),
    (4,  101.5, 102.2, 100.9, 101.8),
    (5,  101.8, 102.5, 101.2, 102.1),
    (6,  102.1, 102.8, 101.5, 102.4),
    (7,  102.4, 102.9, 101.9, 102.6),
    (8,  102.6, 103.5, 102.3, 103.2),   # take-profit bar: high 103.5 >= 103.0
    (9,  103.2, 104.0, 102.8, 103.8),
    (10, 103.8, 104.4, 103.1, 104.0),
    (11, 104.0, 104.6, 103.4, 104.2),
    (12, 104.2, 104.8, 103.7, 104.5),
    (13, 104.5, 105.0, 104.0, 104.8),
]


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    """
    Minimal snapshot with one symbol, 14 candles and one known trade.

    Entry  : prediction id 1, 2026-08-20 13:17, spot 100.0, confidence 0.90,
             predicted_close 106.0 (+6% move, clears the 2% minimum)
    Exit   : take_profit at candle index 8 (2026-08-20 21:00), fill 103.0
    Return : +3.0% exactly, because _walk_candles fills at the threshold
    """
    db_path = tmp_path / "fixture.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    for offset, o, h, lo, c in CANDLES:
        conn.execute(
            "INSERT INTO candles (market, symbol, timestamp, open, high, low, close, volume) "
            "VALUES ('stocks', 'TEST', ?, ?, ?, ?, ?, 1000)",
            ((BASE + timedelta(hours=offset)).strftime("%Y-%m-%d %H:%M:%S"), o, h, lo, c),
        )

    # Entry prediction, 17 minutes past the hour like the live engine.
    conn.execute(
        "INSERT INTO predictions (id, symbol, market, current_price, predicted_high, "
        "predicted_low, predicted_close, confidence, horizon_hours, model_name, created_at) "
        "VALUES (1, 'TEST', 'stocks', 100.0, 107.0, 99.0, 106.0, 0.90, 24, 'small', ?)",
        ((BASE + timedelta(minutes=17)).strftime("%Y-%m-%d %H:%M:%S.%f"),),
    )

    # Neutral follow-up predictions: confidence 0.5 is above the bearish
    # threshold (0.25) and below entry threshold, so they neither exit nor re-enter.
    for i in range(1, 14):
        conn.execute(
            "INSERT INTO predictions (id, symbol, market, current_price, predicted_high, "
            "predicted_low, predicted_close, confidence, horizon_hours, model_name, created_at) "
            "VALUES (?, 'TEST', 'stocks', 101.0, 102.0, 100.0, 101.2, 0.5, 24, 'small', ?)",
            (100 + i, (BASE + timedelta(hours=i, minutes=17)).strftime("%Y-%m-%d %H:%M:%S.%f")),
        )

    # Matching live trade pair: BUY at entry, SELL at take-profit.
    conn.execute(
        "INSERT INTO trades (id, portfolio_id, symbol, direction, quantity, price, "
        "total_value, status, prediction_id, opened_at, closed_at, notes) "
        "VALUES (1, 1, 'TEST', 'BUY', 10.0, 100.0, 1000.0, 'CLOSED', 1, ?, ?, 'Opened by Kronos')",
        ((BASE + timedelta(minutes=17)).strftime("%Y-%m-%d %H:%M:%S.%f"),
         (BASE + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S.%f")),
    )
    conn.execute(
        "INSERT INTO trades (id, portfolio_id, symbol, direction, quantity, price, "
        "total_value, status, realized_pnl, gross_pnl, net_pnl, prediction_id, "
        "opened_at, closed_at, notes) "
        "VALUES (2, 1, 'TEST', 'SELL', 10.0, 103.0, 1030.0, 'CLOSED', 30.0, 30.0, 29.0, 109, ?, ?, "
        "'Closed: take_profit (3.0%)')",
        ((BASE + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S.%f"),
         (BASE + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S.%f")),
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def stable_settings() -> dict:
    return {
        "position_size_pct": 0.15,
        "min_confidence": 0.75,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.03,
        "min_predicted_move_pct": 0.02,
    }
