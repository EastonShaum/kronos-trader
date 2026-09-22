"""
Reconciliation report.

Compares simulated trades against the recorded live trades over the same
period. The point is to find out whether the simulator can reproduce history
at all - a low match rate is a finding to report, not a number to tune.

P&L comparison note: position sizing is portfolio-level and out of scope for
this patch, so simulated P&L is evaluated as gross P&L at the matched live
trade's quantity. That isolates exit timing and exit price - what the replay
actually models - from capital allocation and cost-model drift, neither of
which this patch reproduces.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from backtest import replay as replay_mod
from backtest.replay import SimulatedTrade, _parse_ts
from backtest.settings_resolver import (
    ModelBoundaryError,
    SettingsUnavailableError,
    resolve,
)
from backtest.snapshot import open_readonly

logger = logging.getLogger("backtest.reconcile")

ONE_CANDLE = timedelta(hours=1)
SCHEDULER_STATUS_URL = "http://localhost:8000/api/scheduler/status"
PNL_EPSILON = 1e-6


@dataclass(frozen=True)
class LiveTrade:
    symbol: str
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    exit_reason: str
    quantity: float
    gross_pnl: float
    net_pnl: float | None
    entry_prediction_id: int | None
    exit_path: str            # prediction_cycle | price_monitor


def _parse_exit_reason(notes: str | None) -> str:
    match = re.search(r"Closed:\s*(\w+)", notes or "")
    return match.group(1) if match else "unknown"


def load_live_trades(conn: sqlite3.Connection, symbol: str,
                     start: date, end: date) -> list[LiveTrade]:
    """
    Pair each SELL with the most recent prior BUY for the same portfolio+symbol.

    Mirrors diagnostics._get_entry_prediction. The BUY is the entry side; its
    prediction_id is the entry prediction. The SELL's prediction_id is the exit
    prediction and is deliberately not used for entry attribution.
    """
    sells = conn.execute(
        "SELECT id, portfolio_id, symbol, price, quantity, realized_pnl, gross_pnl, "
        "net_pnl, prediction_id, opened_at, notes FROM trades "
        "WHERE symbol = ? AND direction = 'SELL' "
        "AND DATE(opened_at) BETWEEN ? AND ? ORDER BY opened_at",
        (symbol, start.isoformat(), end.isoformat()),
    ).fetchall()

    live: list[LiveTrade] = []
    for sell in sells:
        buy = conn.execute(
            "SELECT id, price, quantity, prediction_id, opened_at FROM trades "
            "WHERE portfolio_id = ? AND symbol = ? AND direction = 'BUY' "
            "AND opened_at <= ? ORDER BY opened_at DESC LIMIT 1",
            (sell["portfolio_id"], symbol, sell["opened_at"]),
        ).fetchone()
        if buy is None:
            continue

        entry_ts = _parse_ts(buy["opened_at"])
        exit_ts = _parse_ts(sell["opened_at"])
        if entry_ts is None or exit_ts is None:
            continue

        entry_price = float(buy["price"] or 0.0)
        exit_price = float(sell["price"] or 0.0)
        qty = float(buy["quantity"] or 0.0)
        if entry_price <= 0 or qty <= 0:
            continue

        gross = sell["gross_pnl"]
        if gross is None:
            gross = sell["realized_pnl"]
        gross = float(gross or 0.0)

        live.append(LiveTrade(
            symbol=symbol,
            entry_ts=entry_ts,
            entry_price=entry_price,
            exit_ts=exit_ts,
            exit_price=exit_price,
            exit_reason=_parse_exit_reason(sell["notes"]),
            quantity=qty,
            gross_pnl=gross,
            net_pnl=float(sell["net_pnl"]) if sell["net_pnl"] is not None else None,
            entry_prediction_id=buy["prediction_id"],
            exit_path="price_monitor" if sell["prediction_id"] == 0 else "prediction_cycle",
        ))
    return live


@dataclass
class Divergence:
    symbol: str
    entry_ts: datetime
    live_reason: str
    sim_reason: str
    live_exit_ts: datetime
    sim_exit_ts: datetime | None
    live_pnl: float
    sim_pnl: float
    reasons: list[str]

    @property
    def abs_pnl_diff(self) -> float:
        return abs(self.live_pnl - self.sim_pnl)


@dataclass
class SymbolResult:
    symbol: str
    matched: int = 0
    divergent: int = 0
    missing: int = 0
    spurious: int = 0
    live_pnl: float = 0.0
    sim_pnl: float = 0.0
    divergences: list[Divergence] = None

    def __post_init__(self):
        if self.divergences is None:
            self.divergences = []

    @property
    def n_live(self) -> int:
        return self.matched + self.divergent + self.missing

    @property
    def match_rate(self) -> float:
        total = self.matched + self.divergent + self.missing + self.spurious
        return (self.matched / total * 100.0) if total else 0.0


def compare(symbol: str, sim: list[SimulatedTrade], live: list[LiveTrade],
            tolerance: float) -> SymbolResult:
    """Pair on entry timestamp within one candle, then classify."""
    result = SymbolResult(symbol=symbol)
    unmatched_sim = list(sim)

    for lt in live:
        result.live_pnl += lt.gross_pnl

        candidates = [
            (abs(st.entry_ts - lt.entry_ts), st) for st in unmatched_sim
            if abs(st.entry_ts - lt.entry_ts) <= ONE_CANDLE
        ]
        if not candidates:
            result.missing += 1
            continue

        candidates.sort(key=lambda pair: pair[0])
        st = candidates[0][1]
        unmatched_sim.remove(st)

        # Gross P&L at the live trade's quantity - see module docstring.
        sim_pnl = (st.exit_price - st.entry_price) * lt.quantity
        result.sim_pnl += sim_pnl

        problems: list[str] = []
        if st.exit_reason != lt.exit_reason:
            problems.append(f"exit_reason {lt.exit_reason}->{st.exit_reason}")
        if st.exit_ts is None or abs(st.exit_ts - lt.exit_ts) > ONE_CANDLE:
            problems.append("exit_ts")

        diff = abs(sim_pnl - lt.gross_pnl)
        if abs(lt.gross_pnl) < PNL_EPSILON:
            if diff > PNL_EPSILON:
                problems.append("pnl")
        elif diff / abs(lt.gross_pnl) > tolerance:
            problems.append(f"pnl {diff / abs(lt.gross_pnl) * 100:.1f}%")

        if problems:
            result.divergent += 1
            result.divergences.append(Divergence(
                symbol=symbol, entry_ts=lt.entry_ts,
                live_reason=lt.exit_reason, sim_reason=st.exit_reason,
                live_exit_ts=lt.exit_ts, sim_exit_ts=st.exit_ts,
                live_pnl=lt.gross_pnl, sim_pnl=sim_pnl, reasons=problems,
            ))
        else:
            result.matched += 1

    result.spurious = len(unmatched_sim)
    return result


def check_contention(ignore: bool = False) -> None:
    """
    Refuse to start while a prediction cycle is running.

    There is no prediction-cycle lock in the codebase. The closest real signal
    is main._active_predictions, exposed via /api/scheduler/status. If the API
    is unreachable this cannot be determined, which is reported rather than
    assumed safe.
    """
    if ignore:
        logger.warning("Contention check skipped (--ignore-contention)")
        return

    try:
        with urllib.request.urlopen(SCHEDULER_STATUS_URL, timeout=3) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        logger.warning(
            "Could not reach %s (%s) - cannot confirm whether a prediction "
            "cycle is active. Proceeding.", SCHEDULER_STATUS_URL, exc
        )
        return

    active = payload.get("active_predictions") or []
    if active:
        print(
            f"Prediction cycle active for: {', '.join(active)}.\n"
            f"This harness is CPU-bound and the Pi has limited headroom. "
            f"Re-run once the cycle finishes, or pass --ignore-contention.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    logger.info("No active prediction cycle - proceeding.")


def render(results: list[SymbolResult], tolerance: float) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 74)
    add("RECONCILIATION REPORT")
    add("=" * 74)
    add(f"{'symbol':<10}{'live':>6}{'sim':>6}{'match':>7}{'div':>6}"
        f"{'miss':>6}{'spur':>6}{'rate':>8}{'livePnL':>11}{'simPnL':>11}")
    add("-" * 74)

    tot = SymbolResult(symbol="TOTAL")
    for res in results:
        add(f"{res.symbol:<10}{res.n_live:>6}"
            f"{res.matched + res.divergent + res.spurious:>6}"
            f"{res.matched:>7}{res.divergent:>6}{res.missing:>6}{res.spurious:>6}"
            f"{res.match_rate:>7.1f}%{res.live_pnl:>11.2f}{res.sim_pnl:>11.2f}")
        tot.matched += res.matched
        tot.divergent += res.divergent
        tot.missing += res.missing
        tot.spurious += res.spurious
        tot.live_pnl += res.live_pnl
        tot.sim_pnl += res.sim_pnl
        tot.divergences.extend(res.divergences)

    add("-" * 74)
    add(f"{'TOTAL':<10}{tot.n_live:>6}"
        f"{tot.matched + tot.divergent + tot.spurious:>6}"
        f"{tot.matched:>7}{tot.divergent:>6}{tot.missing:>6}{tot.spurious:>6}"
        f"{tot.match_rate:>7.1f}%{tot.live_pnl:>11.2f}{tot.sim_pnl:>11.2f}")
    add("")
    add(f"Aggregate match rate : {tot.match_rate:.1f}%  "
        f"(tolerance {tolerance * 100:.2f}% relative on per-trade gross P&L)")
    add(f"Recorded gross P&L   : {tot.live_pnl:.2f}")
    add(f"Simulated gross P&L  : {tot.sim_pnl:.2f}")
    add(f"Difference           : {tot.sim_pnl - tot.live_pnl:+.2f}")
    add("")

    worst = sorted(tot.divergences, key=lambda d: d.abs_pnl_diff, reverse=True)[:5]
    if worst:
        add("WORST 5 DIVERGENCES BY ABSOLUTE P&L DIFFERENCE")
        add("-" * 74)
        for i, d in enumerate(worst, 1):
            sim_exit = d.sim_exit_ts.isoformat(sep=" ") if d.sim_exit_ts else "none"
            add(f"{i}. {d.symbol} entry {d.entry_ts.isoformat(sep=' ')}")
            add(f"   live: {d.live_reason:<15} exit {d.live_exit_ts.isoformat(sep=' ')} "
                f"pnl {d.live_pnl:+.2f}")
            add(f"   sim : {d.sim_reason:<15} exit {sim_exit} pnl {d.sim_pnl:+.2f}")
            add(f"   diff {d.abs_pnl_diff:.2f} | {', '.join(d.reasons)}")
    else:
        add("No divergences.")

    add("")
    add("Match rate is a finding, not a pass condition.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.reconcile",
        description="Reconcile replayed trades against recorded live trades.",
    )
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", required=True,
                        help="comma-separated, e.g. AMD,SIDU")
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--settings", type=Path,
                        help="JSON file of settings known to be in effect over "
                             "the range (required: position_size_pct, "
                             "min_confidence, stop_loss_pct, min_predicted_move_pct)")
    parser.add_argument("--allow-model-boundary", action="store_true")
    parser.add_argument("--ignore-contention", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s - %(message)s",
    )

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    operator_settings = None
    if args.settings:
        operator_settings = json.loads(args.settings.read_text())

    check_contention(args.ignore_contention)

    conn = open_readonly(args.snapshot)
    try:
        try:
            resolved = resolve(conn, start, end, operator_settings,
                               args.allow_model_boundary)
        except (ModelBoundaryError, SettingsUnavailableError) as exc:
            print(f"\nREFUSED: {exc}\n", file=sys.stderr)
            return 3

        sim_trades = replay_mod.replay(conn, symbols, start, end, resolved.values)
        by_symbol: dict[str, list[SimulatedTrade]] = {s: [] for s in symbols}
        for trade in sim_trades:
            by_symbol.setdefault(trade.symbol, []).append(trade)

        results = [
            compare(symbol, by_symbol.get(symbol, []),
                    load_live_trades(conn, symbol, start, end), args.tolerance)
            for symbol in sorted(symbols)
        ]
        print(render(results, args.tolerance))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
