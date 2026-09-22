"""
Replay engine.

Re-simulates the system's own trading history from a read-only snapshot, one
symbol at a time, using the live shared candle-walk from backend/diagnostics.py.

No portfolio-level simulation in this patch: no shared capital pool, no
correlation, no exposure cap. Position sizing therefore cannot be reproduced,
so simulated trades carry a quantity-independent return_pct and reconciliation
compares gross P&L at the matched live trade's quantity.

Determinism: this path contains no RNG. SEED is set and logged anyway so that
any transitive use stays reproducible.
"""
from __future__ import annotations

import logging
import random
import sqlite3
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 0
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"

# Live re-entry cooldown after a close (trading_engine.py: cooldown_seconds).
COOLDOWN_SECONDS = 3600


def _import_walk_candles():
    """Import the live shared candle-walk without touching the live DB."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from diagnostics import _walk_candles, BEARISH_THRESHOLD  # noqa: E402
    return _walk_candles, BEARISH_THRESHOLD


class EntrySideLinkageError(AssertionError):
    """
    Raised when a simulated trade is attributed to a prediction that is not the
    BUY-side entry prediction.

    Guards the known prior bug in benchmarks.py, where replay read the SELL
    trade's exit prediction instead of the BUY trade's entry prediction.
    """


@dataclass(frozen=True)
class SimulatedTrade:
    symbol: str
    market: str
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime | None
    exit_price: float
    exit_reason: str          # stop_loss | take_profit | bearish_signal | horizon
    return_pct: float         # quantity-independent
    entry_prediction_id: int
    entry_confidence: float
    entry_model: str

    def sort_key(self) -> tuple:
        return (self.symbol, self.entry_ts, self.entry_prediction_id)

    def as_row(self) -> dict:
        row = asdict(self)
        row["entry_ts"] = self.entry_ts.isoformat()
        row["exit_ts"] = self.exit_ts.isoformat() if self.exit_ts else None
        return row


def assert_entry_side_linkage(trade: SimulatedTrade,
                              prediction_times: dict[int, datetime]) -> None:
    """
    The attached prediction must be the entry-side one.

    Fails if the prediction post-dates the entry - which is exactly what
    happens when exit-side data is read for entry attribution.
    """
    pred_ts = prediction_times.get(trade.entry_prediction_id)
    if pred_ts is None:
        raise EntrySideLinkageError(
            f"{trade.symbol}: entry_prediction_id={trade.entry_prediction_id} "
            f"not found in predictions"
        )
    if pred_ts > trade.entry_ts:
        raise EntrySideLinkageError(
            f"{trade.symbol}: prediction {trade.entry_prediction_id} at {pred_ts} "
            f"post-dates entry at {trade.entry_ts} - exit-side prediction used "
            f"for entry attribution"
        )
    if trade.exit_ts is not None and pred_ts > trade.exit_ts:
        raise EntrySideLinkageError(
            f"{trade.symbol}: prediction {trade.entry_prediction_id} at {pred_ts} "
            f"post-dates exit at {trade.exit_ts}"
        )


def _parse_ts(raw) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def load_candles(conn: sqlite3.Connection, market: str, symbol: str,
                 start: date, end: date) -> pd.DataFrame:
    """
    Candles for one symbol, ascending. _walk_candles assumes caller-sorted input.

    Loaded per symbol rather than all at once - the full candle table is ~21k
    rows across 21 symbols and this harness runs on a Pi.
    """
    rows = conn.execute(
        "SELECT timestamp, open, high, low, close FROM candles "
        "WHERE market = ? AND symbol = ? AND DATE(timestamp) BETWEEN ? AND ? "
        "ORDER BY timestamp",
        (market, symbol, start.isoformat(), end.isoformat()),
    ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [{"timestamp": _parse_ts(r["timestamp"]), "open": r["open"], "high": r["high"],
          "low": r["low"], "close": r["close"]} for r in rows]
    ).set_index("timestamp")
    return df


def iter_predictions(conn: sqlite3.Connection, symbol: str,
                     start: date, end: date, model: str | None = None) -> Iterator[sqlite3.Row]:
    """Stream predictions for one symbol rather than materialising all of them."""
    sql = ("SELECT id, symbol, market, current_price, predicted_close, confidence, "
           "model_name, created_at FROM predictions "
           "WHERE symbol = ? AND DATE(created_at) BETWEEN ? AND ?")
    params: list = [symbol, start.isoformat(), end.isoformat()]
    if model:
        sql += " AND model_name = ?"
        params.append(model)
    sql += " ORDER BY created_at, id"

    cur = conn.execute(sql, params)
    while True:
        batch = cur.fetchmany(500)
        if not batch:
            return
        yield from batch


def build_bearish_lookup(conn: sqlite3.Connection, symbol: str,
                         start: date, end: date, model: str) -> dict[datetime, float]:
    """{floor_hour -> confidence} for the model that drove entries."""
    lookup: dict[datetime, float] = {}
    for pred in iter_predictions(conn, symbol, start, end, model):
        ts = _parse_ts(pred["created_at"])
        if ts is None or pred["confidence"] is None:
            continue
        lookup[ts.replace(minute=0, second=0, microsecond=0)] = float(pred["confidence"])
    return lookup


def active_model_for(conn: sqlite3.Connection, symbol: str,
                     start: date, end: date) -> str | None:
    """
    Which model drove trading over the range.

    Derived from the predictions actually recorded, not from settings.json. In
    dual-model periods both models produced predictions but only active_model
    traded, so this returns the model linked to real BUY trades when that is
    determinable, else the most frequent one.
    """
    row = conn.execute(
        "SELECT p.model_name, COUNT(*) AS n FROM trades t "
        "JOIN predictions p ON p.id = t.prediction_id "
        "WHERE t.symbol = ? AND t.direction = 'BUY' "
        "AND DATE(t.opened_at) BETWEEN ? AND ? "
        "GROUP BY p.model_name ORDER BY n DESC LIMIT 1",
        (symbol, start.isoformat(), end.isoformat()),
    ).fetchone()
    if row:
        return row["model_name"]

    row = conn.execute(
        "SELECT model_name, COUNT(*) AS n FROM predictions "
        "WHERE symbol = ? AND DATE(created_at) BETWEEN ? AND ? "
        "GROUP BY model_name ORDER BY n DESC LIMIT 1",
        (symbol, start.isoformat(), end.isoformat()),
    ).fetchone()
    return row["model_name"] if row else None


def replay_symbol(conn: sqlite3.Connection,
                  market: str,
                  symbol: str,
                  start: date,
                  end: date,
                  settings: dict) -> list[SimulatedTrade]:
    """
    Simulate one symbol over [start, end].

    Entry mirrors trading_engine.process_prediction: confidence gate, minimum
    predicted move, one open position at a time, and the 1h post-close cooldown.
    Entry price is the prediction's recorded current_price, which is the live
    spot the engine actually filled at.
    """
    random.seed(SEED)
    np.random.seed(SEED)

    walk_candles, bearish_threshold = _import_walk_candles()

    df = load_candles(conn, market, symbol, start, end)
    if df.empty:
        logger.info("%s: no candles in range, skipping", symbol)
        return []

    model = active_model_for(conn, symbol, start, end)
    if model is None:
        logger.info("%s: no predictions in range, skipping", symbol)
        return []

    min_conf = float(settings["min_confidence"])
    min_move = float(settings["min_predicted_move_pct"])
    sl_pct = float(settings["stop_loss_pct"])
    tp_pct = float(settings["take_profit_pct"])

    bearish = build_bearish_lookup(conn, symbol, start, end, model)

    trades: list[SimulatedTrade] = []
    blocked_until: datetime | None = None

    for pred in iter_predictions(conn, symbol, start, end, model):
        ts = _parse_ts(pred["created_at"])
        if ts is None or pred["confidence"] is None:
            continue
        if blocked_until is not None and ts < blocked_until:
            continue

        confidence = float(pred["confidence"])
        if confidence < min_conf:
            continue

        entry_price = float(pred["current_price"] or 0.0)
        pred_close = float(pred["predicted_close"] or 0.0)
        if entry_price <= 0:
            continue

        if abs(pred_close - entry_price) / entry_price < min_move:
            continue

        future = df[df.index > ts]
        if future.empty:
            continue

        walk = walk_candles(
            future,
            entry_price,
            sl_abs=entry_price * (1 - sl_pct),
            tp_abs=entry_price * (1 + tp_pct),
            bearish_preds=bearish,
        )

        exit_price = float(walk["exit_price"])
        reason = walk["exit_reason"]
        if reason == "max_hold":
            reason = "horizon"

        trade = SimulatedTrade(
            symbol=symbol,
            market=market,
            entry_ts=ts,
            entry_price=entry_price,
            exit_ts=walk["exit_ts"],
            exit_price=exit_price,
            exit_reason=reason,
            return_pct=(exit_price - entry_price) / entry_price,
            entry_prediction_id=int(pred["id"]),
            entry_confidence=confidence,
            entry_model=pred["model_name"],
        )
        trades.append(trade)

        exit_ts = walk["exit_ts"] or df.index[-1].to_pydatetime()
        blocked_until = exit_ts + timedelta(seconds=COOLDOWN_SECONDS)

    # Re-read prediction timestamps from the DB rather than from the trade.
    # Deriving them from the trade would compare entry_ts against itself and
    # make the linkage assertion vacuous.
    if trades:
        ids = {t.entry_prediction_id for t in trades}
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, created_at FROM predictions WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        prediction_times = {int(r["id"]): _parse_ts(r["created_at"]) for r in rows}

        for trade in trades:
            assert_entry_side_linkage(trade, prediction_times)

    trades.sort(key=SimulatedTrade.sort_key)
    logger.info("%s [%s]: %d simulated trades", symbol, model, len(trades))
    return trades


def replay(conn: sqlite3.Connection,
           symbols: list[str],
           start: date,
           end: date,
           settings: dict) -> list[SimulatedTrade]:
    """Replay each symbol independently. Deterministic ordering by design."""
    logger.info("Replay RNG seed = %d (no RNG on this path; seeded for safety)", SEED)

    all_trades: list[SimulatedTrade] = []
    for symbol in sorted(symbols):
        market = market_for_symbol(conn, symbol)
        if market is None:
            logger.warning("%s: symbol not present in snapshot, skipping", symbol)
            continue
        all_trades.extend(replay_symbol(conn, market, symbol, start, end, settings))

    all_trades.sort(key=SimulatedTrade.sort_key)
    return all_trades


def market_for_symbol(conn: sqlite3.Connection, symbol: str) -> str | None:
    row = conn.execute(
        "SELECT market FROM candles WHERE symbol = ? LIMIT 1", (symbol,)
    ).fetchone()
    return row["market"] if row else None
