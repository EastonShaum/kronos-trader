"""
Conditional exit fidelity.

End-to-end match rate measures cascade length, not simulation quality: one
divergent exit shifts the cooldown, which shifts the next entry, which shifts
everything after it. This module removes that coupling by FORCING each
simulated trade to start from a recorded live entry - same symbol, same
timestamp, same entry price - and then evaluating only the exit.

Entry logic needs config stability; exit fidelity does not, provided the
threshold in force at the time is known. So this runs across all recorded
SL/TP exits rather than only the clean-config window.
"""
from __future__ import annotations

import logging
import random
import sqlite3
from dataclasses import dataclass

import pandas as pd

from backtest.fill_data import SL_PCT, FillObservation, tp_pct_at
from backtest.fill_models import Bar, FillModel
from backtest.replay import _import_walk_candles, _parse_ts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExitComparison:
    symbol: str
    entry_ts: object
    exit_type: str                 # live exit type: SL | TP
    live_exit_ts: object
    live_fill: float
    sim_exit_ts: object | None
    sim_fill: float
    sim_reason: str
    reason_match: bool
    exit_ts_delta_hours: float | None
    fill_delta: float              # sim_fill - live_fill, relative to entry
    pnl_delta: float               # (sim - live) gross P&L at live quantity

    @property
    def favours_sim(self) -> bool:
        return self.pnl_delta > 0


def _bars_for(conn: sqlite3.Connection, market: str, symbol: str) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT timestamp, open, high, low, close FROM candles "
        "WHERE market = ? AND symbol = ? ORDER BY timestamp",
        (market, symbol),
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [{"timestamp": _parse_ts(r["timestamp"]), "open": r["open"], "high": r["high"],
          "low": r["low"], "close": r["close"]} for r in rows]
    ).set_index("timestamp")


def _quantity_for(conn: sqlite3.Connection, obs: FillObservation) -> float:
    row = conn.execute(
        "SELECT quantity FROM trades WHERE symbol = ? AND direction = 'BUY' "
        "AND opened_at <= ? ORDER BY opened_at DESC LIMIT 1",
        (obs.symbol, obs.entry_ts.strftime("%Y-%m-%d %H:%M:%S.%f")),
    ).fetchone()
    return float(row["quantity"]) if row and row["quantity"] else 0.0


def evaluate(conn: sqlite3.Connection,
             observations: list[FillObservation],
             fill_model: FillModel,
             rng: random.Random | None = None) -> list[ExitComparison]:
    """
    Force entries to the recorded ones, simulate only the exit.

    _walk_candles detects the trigger bar; fill_model prices it. The bearish
    signal path is disabled here on purpose - forcing the entry isolates the
    threshold mechanism, and a bearish flip would exit on a rule the fill
    model has nothing to say about.
    """
    walk_candles, _ = _import_walk_candles()
    bars_cache: dict[tuple[str, str], pd.DataFrame] = {}
    results: list[ExitComparison] = []

    for obs in observations:
        key = (obs.market, obs.symbol)
        if key not in bars_cache:
            bars_cache[key] = _bars_for(conn, obs.market, obs.symbol)
        df = bars_cache[key]
        if df.empty:
            continue

        future = df[df.index > obs.entry_ts]
        if future.empty:
            continue

        sl_abs = obs.entry_price * (1 - SL_PCT)
        tp_abs = obs.entry_price * (1 + tp_pct_at(obs.entry_ts))

        walk = walk_candles(future, obs.entry_price, sl_abs=sl_abs, tp_abs=tp_abs,
                            bearish_preds=None)

        reason = walk["exit_reason"]
        sim_exit_ts = walk["exit_ts"]

        if reason == "take_profit":
            trigger, threshold_price = "TP", tp_abs
        elif reason == "stop_loss":
            trigger, threshold_price = "SL", sl_abs
        else:
            # Walk never triggered a threshold: max_hold / no_candles.
            trigger, threshold_price = None, None

        if trigger is None:
            sim_fill = float(walk["exit_price"])
        else:
            bar = None
            if sim_exit_ts is not None and sim_exit_ts in df.index:
                row = df.loc[sim_exit_ts]
                bar = Bar(float(row["open"]), float(row["high"]),
                          float(row["low"]), float(row["close"]))
            sim_fill = fill_model.price(trigger, threshold_price, bar, rng)

        qty = _quantity_for(conn, obs)
        live_pnl = (obs.fill_price - obs.entry_price) * qty
        sim_pnl = (sim_fill - obs.entry_price) * qty

        delta_hours = None
        if sim_exit_ts is not None:
            delta_hours = (sim_exit_ts - obs.exit_ts).total_seconds() / 3600.0

        sim_reason_label = {"take_profit": "TP", "stop_loss": "SL"}.get(reason, reason)

        results.append(ExitComparison(
            symbol=obs.symbol, entry_ts=obs.entry_ts, exit_type=obs.exit_type,
            live_exit_ts=obs.exit_ts, live_fill=obs.fill_price,
            sim_exit_ts=sim_exit_ts, sim_fill=sim_fill,
            sim_reason=sim_reason_label,
            reason_match=(sim_reason_label == obs.exit_type),
            exit_ts_delta_hours=delta_hours,
            fill_delta=(sim_fill - obs.fill_price) / obs.entry_price,
            pnl_delta=sim_pnl - live_pnl,
        ))

    return results


def summarise(results: list[ExitComparison]) -> dict:
    if not results:
        return {"n": 0}

    deltas = [r.pnl_delta for r in results]
    fill_deltas = [r.fill_delta for r in results]
    ts_deltas = [r.exit_ts_delta_hours for r in results
                 if r.exit_ts_delta_hours is not None]

    ordered = sorted(deltas)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

    return {
        "n": len(results),
        "reason_match": sum(1 for r in results if r.reason_match),
        "reason_match_pct": sum(1 for r in results if r.reason_match) / len(results) * 100,
        "favours_sim": sum(1 for r in results if r.pnl_delta > 0),
        "favours_live": sum(1 for r in results if r.pnl_delta < 0),
        "ties": sum(1 for r in results if r.pnl_delta == 0),
        "median_pnl_delta": median,
        "total_pnl_delta": sum(deltas),
        "mean_fill_delta_pp": (sum(fill_deltas) / len(fill_deltas)) * 100,
        "median_exit_ts_delta_h": sorted(ts_deltas)[len(ts_deltas) // 2] if ts_deltas else None,
    }
