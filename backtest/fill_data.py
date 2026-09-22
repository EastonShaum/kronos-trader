"""
Empirical fill dataset.

Extracts every historical stop-loss and take-profit exit with the threshold
that was in force at the time, the price it actually filled at, and the
context needed to judge why the two differ.

The core quantity is `overshoot`: the signed relative distance the fill sat
PAST its threshold, in the direction the price was travelling.

    SL: overshoot = (threshold - fill) / threshold     > 0 means a worse fill
    TP: overshoot = (fill - threshold) / threshold     > 0 means a better fill

Both are positive when the fill landed further along than the threshold, which
is what the live 3-minute sampling produces and what ExactFill assumes never
happens.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from backtest.replay import _parse_ts

logger = logging.getLogger(__name__)

# Stop-loss has no recorded change anywhere in audit_recommendations.
SL_PCT = 0.03

# take_profit_pct 6.0 -> 3.0, applied 2026-09-03 18:32:55 (audit_recommendations id 13).
TP_CHANGE_TS = datetime(2026, 9, 3, 18, 32, 55)
TP_PCT_BEFORE = 0.06
TP_PCT_AFTER = 0.03

ATR_LOOKBACK = 14


def tp_pct_at(ts: datetime) -> float:
    return TP_PCT_BEFORE if ts < TP_CHANGE_TS else TP_PCT_AFTER


@dataclass(frozen=True)
class FillObservation:
    symbol: str
    market: str
    exit_type: str            # SL | TP
    exit_route: str           # price_monitor | prediction_cycle
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    threshold_pct: float
    threshold_price: float
    fill_price: float
    overshoot: float          # signed, relative, positive = past the threshold
    hour_utc: int
    crossed_session: bool     # exit bar followed a >1h gap in the candle series
    bar_open: float | None
    bar_high: float | None
    bar_low: float | None
    bar_close: float | None
    outside_bar: bool         # Pattern C - fill not inside its own bar's range
    atr_pct: float | None     # ATR14 before entry, relative to entry price

    def as_row(self) -> dict:
        row = asdict(self)
        row["entry_ts"] = self.entry_ts.isoformat()
        row["exit_ts"] = self.exit_ts.isoformat()
        return row


def _exit_type(notes: str | None) -> str | None:
    if not notes:
        return None
    if "stop_loss" in notes:
        return "SL"
    if "take_profit" in notes:
        return "TP"
    return None


def _load_bars(conn: sqlite3.Connection, market: str, symbol: str) -> list[tuple]:
    rows = conn.execute(
        "SELECT timestamp, open, high, low, close FROM candles "
        "WHERE market = ? AND symbol = ? ORDER BY timestamp",
        (market, symbol),
    ).fetchall()
    return [(_parse_ts(r["timestamp"]), r["open"], r["high"], r["low"], r["close"])
            for r in rows]


def _atr_pct(bars: list[tuple], entry_ts: datetime, entry_price: float) -> float | None:
    """ATR14 over the bars preceding entry, as a fraction of entry price."""
    prior = [b for b in bars if b[0] is not None and b[0] <= entry_ts]
    if len(prior) < ATR_LOOKBACK + 1 or entry_price <= 0:
        return None
    window = prior[-(ATR_LOOKBACK + 1):]
    trs = []
    for i in range(1, len(window)):
        _, _, high, low, _ = window[i]
        prev_close = window[i - 1][4]
        if None in (high, low, prev_close):
            continue
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    if not trs:
        return None
    return (sum(trs) / len(trs)) / entry_price


def build_dataset(conn: sqlite3.Connection) -> tuple[list[FillObservation], dict]:
    """
    Every SL/TP exit in history, with its threshold and fill context.

    Returns (observations, report). Nothing is silently dropped: every
    exclusion is counted in the report.
    """
    sells = conn.execute(
        "SELECT t.id, t.portfolio_id, t.symbol, t.price, t.prediction_id, "
        "t.opened_at, t.notes, p.market "
        "FROM trades t JOIN portfolios p ON p.id = t.portfolio_id "
        "WHERE t.direction = 'SELL' ORDER BY t.opened_at"
    ).fetchall()

    bars_cache: dict[tuple[str, str], list[tuple]] = {}
    observations: list[FillObservation] = []
    report = {
        "sells_scanned": len(sells),
        "not_threshold_exit": 0,
        "no_paired_buy": 0,
        "no_bar_at_exit": 0,
        "outside_bar_excluded": 0,
        "kept": 0,
    }

    for sell in sells:
        etype = _exit_type(sell["notes"])
        if etype is None:
            report["not_threshold_exit"] += 1
            continue

        buy = conn.execute(
            "SELECT price, opened_at FROM trades WHERE portfolio_id = ? AND symbol = ? "
            "AND direction = 'BUY' AND opened_at <= ? ORDER BY opened_at DESC LIMIT 1",
            (sell["portfolio_id"], sell["symbol"], sell["opened_at"]),
        ).fetchone()
        if buy is None or not buy["price"]:
            report["no_paired_buy"] += 1
            continue

        entry_ts = _parse_ts(buy["opened_at"])
        exit_ts = _parse_ts(sell["opened_at"])
        entry_price = float(buy["price"])
        fill_price = float(sell["price"] or 0.0)
        if entry_ts is None or exit_ts is None or entry_price <= 0 or fill_price <= 0:
            report["no_paired_buy"] += 1
            continue

        market = str(sell["market"]).lower()
        key = (market, sell["symbol"])
        if key not in bars_cache:
            bars_cache[key] = _load_bars(conn, market, sell["symbol"])
        bars = bars_cache[key]

        bar_ts = exit_ts.replace(minute=0, second=0, microsecond=0)
        bar = next((b for b in bars if b[0] == bar_ts), None)
        if bar is None:
            report["no_bar_at_exit"] += 1
            continue

        _, b_open, b_high, b_low, b_close = bar
        outside = not (float(b_low) <= fill_price <= float(b_high))

        if etype == "SL":
            threshold_pct = SL_PCT
            threshold_price = entry_price * (1 - SL_PCT)
            overshoot = (threshold_price - fill_price) / threshold_price
        else:
            threshold_pct = tp_pct_at(exit_ts)
            threshold_price = entry_price * (1 + threshold_pct)
            overshoot = (fill_price - threshold_price) / threshold_price

        idx = next((i for i, b in enumerate(bars) if b[0] == bar_ts), None)
        crossed = False
        if idx is not None and idx > 0:
            crossed = (bar_ts - bars[idx - 1][0]) > timedelta(hours=1)

        observations.append(FillObservation(
            symbol=sell["symbol"], market=market, exit_type=etype,
            exit_route="price_monitor" if sell["prediction_id"] == 0 else "prediction_cycle",
            entry_ts=entry_ts, entry_price=entry_price, exit_ts=exit_ts,
            threshold_pct=threshold_pct, threshold_price=threshold_price,
            fill_price=fill_price, overshoot=overshoot,
            hour_utc=exit_ts.hour, crossed_session=crossed,
            bar_open=b_open, bar_high=b_high, bar_low=b_low, bar_close=b_close,
            outside_bar=outside,
            atr_pct=_atr_pct(bars, entry_ts, entry_price),
        ))
        report["kept"] += 1

    report["outside_bar_excluded"] = sum(1 for o in observations if o.outside_bar)
    return observations, report


def fittable(observations: list[FillObservation]) -> list[FillObservation]:
    """
    Observations usable for fitting.

    Pattern C fills - those outside their own bar's OHLC range - are excluded.
    They come from a price series the candle table does not contain, so no
    fill model built on candles can reproduce them.
    """
    return [o for o in observations if not o.outside_bar]
