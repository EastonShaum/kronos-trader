"""
Re-run the measurements on the corrected exit simulator.

    venv/bin/python backend/scripts/rerun_diagnostics.py [--sims 1000] [--seed 0]

Read-only on trading data: opens the live DB with a `mode=ro` URI and never
writes to it. Output goes to backend/reports/diagnostics_<date>.md.

Three sections:
  1. Stop-fill validation - live SL fills vs simulate_exit
  2. Random-entry benchmark, compounding, segmented by config era
  3. Per-symbol expected value with block-bootstrap CIs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
from bisect import bisect_right
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from cost_model import apply_costs  # noqa: E402
from exit_sim import simulate_exit  # noqa: E402
from stats_bootstrap import block_bootstrap_ci  # noqa: E402

logger = logging.getLogger("rerun_diagnostics")

LIVE_DB = BACKEND / "kronos_trader.db"
REPORTS = BACKEND / "reports"

SL_PCT = 0.03
TP_CHANGE_TS = datetime(2026, 9, 3, 18, 32, 55)   # audit_recommendations id 13
TP_PCT_BEFORE = 0.06
TP_PCT_AFTER = 0.03

POSITION_SIZE_PCT = 0.15
COOLDOWN = timedelta(hours=1)
MAX_HOLD_BARS = 72
MIN_N_FOR_EV = 15


def tp_pct_at(ts: datetime) -> float:
    return TP_PCT_BEFORE if ts < TP_CHANGE_TS else TP_PCT_AFTER


def parse_ts(raw) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(raw).strip(), fmt)
        except ValueError:
            continue
    return None


def open_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ── Config eras ─────────────────────────────────────────────────────────────

def derive_eras(conn: sqlite3.Connection) -> list[dict]:
    """
    Eras keyed on the model that actually DROVE ENTRIES, plus the TP change.

    NOT on which models emitted predictions. During dual-model mode both models
    predicted every cycle, but trading_engine.process_prediction only acts when
    `pred_model == active_model`, so only one model ever generated entries.
    Keying on the emitting set would pool two entirely different entry
    generators into one "mini+small" era - see verify_era_purity(), which
    asserts the split is real rather than assumed.

    Boundaries are the recorded active_model changes in audit_recommendations
    plus the recorded take-profit change. settings.json is gitignored and has
    never been committed, so these records are the only recoverable history.

    Boundaries are TIMESTAMPS, not dates. Both recorded config changes landed
    mid-day (active_model at 23:13, take-profit at 18:32), so a date-granularity
    boundary puts that day's earlier trades in the wrong era - which is exactly
    what verify_era_purity() catches.
    """
    model_changes = conn.execute(
        "SELECT applied_at FROM audit_recommendations "
        "WHERE param = 'active_model' AND status = 'applied' AND applied_at IS NOT NULL "
        "ORDER BY applied_at"
    ).fetchall()

    bounds = {parse_ts(r["applied_at"]) for r in model_changes}
    bounds.discard(None)
    bounds.add(TP_CHANGE_TS)

    span = conn.execute(
        "SELECT MIN(opened_at) AS lo, MAX(opened_at) AS hi "
        "FROM trades WHERE direction = 'BUY'"
    ).fetchone()
    first, last = parse_ts(span["lo"]), parse_ts(span["hi"])
    end_sentinel = last + timedelta(seconds=1)

    ordered = sorted(b for b in bounds if first < b < end_sentinel)

    eras: list[dict] = []
    start = first
    for boundary in ordered + [end_sentinel]:
        if start < boundary:
            eras.append({"start": start, "end": boundary})   # half-open [start, end)
        start = boundary

    for era in eras:
        era["tp_pct"] = tp_pct_at(era["start"])
        era["models"] = "?"
        era["name"] = f"{era['start']:%Y-%m-%d}..{era['end']:%Y-%m-%d}"
    return eras


def verify_era_purity(conn: sqlite3.Connection, eras: list[dict]) -> dict:
    """
    Check that each era's entries were driven by exactly one model, and that
    era entry counts sum to the total number of BUY trades.

    This is the check that the dual-mode split is real. If an era contains
    entries from both models, the boundary is wrong and the era is pooling
    two different entry generators.
    """
    rows = conn.execute(
        "SELECT b.opened_at, p.model_name FROM trades b "
        "JOIN predictions p ON p.id = b.prediction_id "
        "WHERE b.direction = 'BUY' ORDER BY b.opened_at"
    ).fetchall()

    total_buys = conn.execute(
        "SELECT COUNT(*) AS n FROM trades WHERE direction = 'BUY'"
    ).fetchone()["n"]

    for era in eras:
        era["_models_seen"] = {}

    unassigned = 0
    for row in rows:
        ts = parse_ts(row["opened_at"])
        era = era_for(eras, ts) if ts else None
        if era is None:
            unassigned += 1
            continue
        era["_models_seen"][row["model_name"]] = \
            era["_models_seen"].get(row["model_name"], 0) + 1

    impure = []
    assigned = 0
    for era in eras:
        seen = era["_models_seen"]
        n = sum(seen.values())
        assigned += n
        era["n_entries"] = n
        era["models"] = "+".join(sorted(seen)) if seen else "-"
        era["name"] = (f"{era['start']:%Y-%m-%d %H:%M}..{era['end']:%Y-%m-%d %H:%M} "
                       f"[{era['models']}] TP={era['tp_pct']:.0%}")
        if len(seen) > 1:
            impure.append({"era": era["name"], "models": dict(seen)})

    return {
        "total_buys": total_buys,
        "linked_entries": len(rows),
        "assigned_to_eras": assigned,
        "unassigned": unassigned,
        "sums_correctly": assigned + unassigned == len(rows) == total_buys,
        "impure_eras": impure,
        "all_pure": not impure,
    }


SQL_TS = "%Y-%m-%d %H:%M:%S.%f"


def era_for(eras: list[dict], ts: datetime) -> dict | None:
    """Half-open [start, end) so a boundary timestamp lands in exactly one era."""
    for era in eras:
        if era["start"] <= ts < era["end"]:
            return era
    return None


# ── Candle access ───────────────────────────────────────────────────────────

class CandleStore:
    """Per-symbol bar lists plus a sorted timestamp index for O(log n) slicing."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._bars: dict[str, list[tuple]] = {}
        self._ts: dict[str, list[datetime]] = {}
        self._market: dict[str, str] = {}

    def market(self, symbol: str) -> str | None:
        if symbol not in self._market:
            row = self.conn.execute(
                "SELECT market FROM candles WHERE symbol = ? LIMIT 1", (symbol,)
            ).fetchone()
            self._market[symbol] = row["market"] if row else None
        return self._market[symbol]

    def bars(self, symbol: str) -> list[tuple]:
        if symbol not in self._bars:
            rows = self.conn.execute(
                "SELECT timestamp, open, high, low, close FROM candles "
                "WHERE symbol = ? ORDER BY timestamp", (symbol,)
            ).fetchall()
            bars = [(parse_ts(r["timestamp"]), r["open"], r["high"], r["low"], r["close"])
                    for r in rows]
            bars = [b for b in bars if b[0] is not None]
            self._bars[symbol] = bars
            self._ts[symbol] = [b[0] for b in bars]
        return self._bars[symbol]

    def after(self, symbol: str, ts: datetime, limit: int | None = None) -> list[tuple]:
        bars = self.bars(symbol)
        idx = bisect_right(self._ts[symbol], ts)
        return bars[idx:idx + limit] if limit else bars[idx:]

    def bar_at(self, symbol: str, ts: datetime) -> tuple | None:
        bars = self.bars(symbol)
        key = ts.replace(minute=0, second=0, microsecond=0)
        i = bisect_right(self._ts[symbol], key) - 1
        if 0 <= i < len(bars) and bars[i][0] == key:
            return bars[i]
        return None


# ── 1. Stop-fill validation ─────────────────────────────────────────────────

def stop_fill_validation(conn: sqlite3.Connection, store: CandleStore) -> dict:
    """Compare each live stop-loss fill against what simulate_exit would give."""
    sells = conn.execute(
        "SELECT t.id, t.portfolio_id, t.symbol, t.price, t.prediction_id, "
        "t.opened_at, t.notes FROM trades t "
        "WHERE t.direction = 'SELL' AND t.notes LIKE '%stop_loss%' "
        "ORDER BY t.opened_at"
    ).fetchall()

    matched, unreproducible, no_sim = [], [], 0

    for sell in sells:
        buy = conn.execute(
            "SELECT price, opened_at FROM trades WHERE portfolio_id = ? AND symbol = ? "
            "AND direction = 'BUY' AND opened_at <= ? ORDER BY opened_at DESC LIMIT 1",
            (sell["portfolio_id"], sell["symbol"], sell["opened_at"]),
        ).fetchone()
        if not buy or not buy["price"]:
            continue

        entry_ts, exit_ts = parse_ts(buy["opened_at"]), parse_ts(sell["opened_at"])
        entry_price, live_fill = float(buy["price"]), float(sell["price"])
        if not entry_ts or not exit_ts or entry_price <= 0 or live_fill <= 0:
            continue

        live_ret = (live_fill - entry_price) / entry_price
        bar = store.bar_at(sell["symbol"], exit_ts)

        record = {
            "symbol": sell["symbol"], "entry_ts": entry_ts, "exit_ts": exit_ts,
            "entry_price": entry_price, "live_fill": live_fill, "live_ret": live_ret,
            "market": store.market(sell["symbol"]) or "unknown",
            "route": "price_monitor" if sell["prediction_id"] == 0 else "prediction_cycle",
        }

        # Pattern C: the live fill is not inside its own backfilled bar, so no
        # candle-based simulator can reach it. Reported, never "fixed".
        if bar is not None and not (float(bar[3]) <= live_fill <= float(bar[2])):
            record["bar_low"], record["bar_high"] = float(bar[3]), float(bar[2])
            unreproducible.append(record)
            continue

        result = simulate_exit(
            entry_price, store.after(sell["symbol"], entry_ts, MAX_HOLD_BARS),
            SL_PCT, tp_pct_at(entry_ts), max_bars=MAX_HOLD_BARS,
        )
        if result is None or result.reason != "stop_loss":
            no_sim += 1
            continue

        sim_ret = (result.exit_price - entry_price) / entry_price
        record.update({
            "sim_fill": result.exit_price, "sim_ret": sim_ret,
            "gapped": result.gapped, "diff_pp": (sim_ret - live_ret) * 100,
        })
        matched.append(record)

    live_rets = np.array([r["live_ret"] for r in matched])
    sim_rets = np.array([r["sim_ret"] for r in matched])
    diffs = np.array([r["diff_pp"] for r in matched])

    # Gaps are a market-structure effect: equities close overnight and at
    # weekends, crypto never does. Pooling the two hides that.
    by_market: dict[str, dict] = {}
    for r in matched:
        b = by_market.setdefault(r["market"], {"n": 0, "gapped": 0, "diffs": []})
        b["n"] += 1
        b["gapped"] += int(r["gapped"])
        b["diffs"].append(r["diff_pp"])
    for b in by_market.values():
        b["gapped_pct"] = b["gapped"] / b["n"] * 100 if b["n"] else 0.0
        b["mean_diff_pp"] = float(np.mean(b["diffs"])) if b["diffs"] else 0.0

    return {
        "by_market": by_market,
        "n_stop_exits": len(sells),
        "n_compared": len(matched),
        "n_unreproducible": len(unreproducible),
        "n_no_sim_stop": no_sim,
        "unreproducible": unreproducible,
        "live_mean_pct": float(live_rets.mean() * 100) if len(live_rets) else 0.0,
        "live_median_pct": float(np.median(live_rets) * 100) if len(live_rets) else 0.0,
        "sim_mean_pct": float(sim_rets.mean() * 100) if len(sim_rets) else 0.0,
        "sim_median_pct": float(np.median(sim_rets) * 100) if len(sim_rets) else 0.0,
        "live_pct_worse_than_sl": float((live_rets < -SL_PCT).mean() * 100) if len(live_rets) else 0.0,
        "sim_pct_worse_than_sl": float((sim_rets < -SL_PCT).mean() * 100) if len(sim_rets) else 0.0,
        "sim_gapped_pct": float(np.mean([r["gapped"] for r in matched]) * 100) if matched else 0.0,
        "mean_diff_pp": float(diffs.mean()) if len(diffs) else 0.0,
        "median_diff_pp": float(np.median(diffs)) if len(diffs) else 0.0,
    }


# ── 2. Random-entry benchmark, compounding ──────────────────────────────────

def _is_tradeable(ts: datetime, market: str) -> bool:
    if market == "crypto":
        return True
    return ts.weekday() < 5 and 14 <= ts.hour < 21


def _live_era_pnl(conn: sqlite3.Connection, eras: list[dict]) -> dict[str, dict]:
    """Live realised net P&L and trade counts per era, by entry time."""
    rows = conn.execute(
        "SELECT s.symbol, s.net_pnl, s.realized_pnl, s.opened_at AS exit_at, "
        "p.market, b.opened_at AS entry_at, b.price AS entry_price, s.price AS exit_price "
        "FROM trades s JOIN portfolios p ON p.id = s.portfolio_id "
        "JOIN trades b ON b.id = ("
        "  SELECT id FROM trades x WHERE x.portfolio_id = s.portfolio_id "
        "  AND x.symbol = s.symbol AND x.direction = 'BUY' "
        "  AND x.opened_at <= s.opened_at ORDER BY x.opened_at DESC LIMIT 1) "
        "WHERE s.direction = 'SELL' AND s.status = 'CLOSED'"
    ).fetchall()

    out: dict[str, dict] = {e["name"]: {"pnl": 0.0, "n": 0, "per_symbol": {}} for e in eras}
    for row in rows:
        entry_ts = parse_ts(row["entry_at"])
        if entry_ts is None:
            continue
        era = era_for(eras, entry_ts)
        if era is None:
            continue
        pnl = row["net_pnl"] if row["net_pnl"] is not None else row["realized_pnl"]
        pnl = float(pnl or 0.0)
        entry_price = float(row["entry_price"] or 0.0)
        exit_price = float(row["exit_price"] or 0.0)

        bucket = out[era["name"]]
        bucket["pnl"] += pnl
        bucket["n"] += 1
        if entry_price > 0:
            ret_pct = (exit_price - entry_price) / entry_price * 100
            bucket["per_symbol"].setdefault(row["symbol"], []).append(ret_pct)
    return out


def _run_portfolio(candidates: list[tuple], starting: float, era: dict,
                   store: CandleStore, market_of: dict[str, str]) -> float:
    """
    Walk one set of entries through simulate_exit with compounding capital.

    Shared by the random arm and the sim-live arm so the only difference
    between them is WHICH entries are taken. Sizing, cooldown, exit rule and
    cost treatment are identical by construction.

    Returns total net P&L.
    """
    cash = starting
    pending: list[tuple[datetime, float]] = []   # (exit_ts, proceeds)
    # One position per symbol at a time, plus the live 1h post-close cooldown -
    # both collapse into "not available until".
    busy_until: dict[str, datetime] = {}
    realised = 0.0

    for entry_ts, symbol, entry_price in candidates:
        # Settle exits that completed before this entry.
        still = []
        for exit_ts, proceeds in pending:
            if exit_ts <= entry_ts:
                cash += proceeds
            else:
                still.append((exit_ts, proceeds))
        pending = still

        if entry_ts < busy_until.get(symbol, entry_ts):
            continue

        size = cash * POSITION_SIZE_PCT
        if size < 1.0:
            continue
        qty = size / entry_price
        cash -= size

        bars = store.after(symbol, entry_ts, MAX_HOLD_BARS)
        result = simulate_exit(entry_price, bars, SL_PCT, era["tp_pct"],
                               max_bars=MAX_HOLD_BARS)
        if result is None:
            cash += size
            continue

        costs = apply_costs(
            market=market_of[symbol], symbol=symbol, side="buy",
            entry_price=entry_price, exit_price=result.exit_price,
            position_size=qty,
        )
        # Net, so this is comparable with the live net_pnl it is scored against.
        realised += costs["net_pnl"]
        proceeds = qty * result.exit_price - (
            costs["entry_fee"] + costs["exit_fee"]
            + costs["entry_slippage"] + costs["exit_slippage"]
        )
        pending.append((result.ts, proceeds))
        busy_until[symbol] = result.ts + COOLDOWN

    return realised


def _live_entries(conn: sqlite3.Connection, era: dict) -> list[tuple]:
    """Recorded BUY entries inside an era, as (ts, symbol, entry_price)."""
    rows = conn.execute(
        "SELECT symbol, price, opened_at FROM trades "
        "WHERE direction = 'BUY' AND opened_at >= ? AND opened_at < ? "
        "ORDER BY opened_at",
        (era["start"].strftime(SQL_TS), era["end"].strftime(SQL_TS)),
    ).fetchall()
    out = []
    for row in rows:
        ts = parse_ts(row["opened_at"])
        price = float(row["price"] or 0.0)
        if ts and price > 0:
            out.append((ts, row["symbol"], price))
    return out


def random_entry_benchmark(conn: sqlite3.Connection, store: CandleStore,
                           eras: list[dict], live: dict[str, dict],
                           n_sims: int, seed: int) -> list[dict]:
    """
    Random-entry Monte Carlo with COMPOUNDING position sizing.

    Each simulated trade is sized at 15% of the cash available at that moment,
    and cash is returned at exit. The previous implementation set cash once and
    never decremented it, so every trade was a flat notional and the random arm
    neither compounded nor decayed - not comparable with the live strategy,
    which does both.

    Run independently per era. Never pooled across a model or threshold change.
    """
    results = []

    for era in eras:
        counts = conn.execute(
            "SELECT b.symbol, p.market, COUNT(*) AS n "
            "FROM trades s JOIN portfolios p ON p.id = s.portfolio_id "
            "JOIN trades b ON b.id = ("
            "  SELECT id FROM trades x WHERE x.portfolio_id = s.portfolio_id "
            "  AND x.symbol = s.symbol AND x.direction = 'BUY' "
            "  AND x.opened_at <= s.opened_at ORDER BY x.opened_at DESC LIMIT 1) "
            "WHERE s.direction = 'SELL' AND s.status = 'CLOSED' "
            "AND b.opened_at >= ? AND b.opened_at < ? GROUP BY b.symbol",
            (era["start"].strftime(SQL_TS), era["end"].strftime(SQL_TS)),
        ).fetchall()

        plan = [(r["symbol"], str(r["market"]).lower(), r["n"]) for r in counts]
        live_bucket = live.get(era["name"], {"pnl": 0.0, "n": 0})

        if not plan or live_bucket["n"] == 0:
            results.append({"era": era["name"], "skipped": "no live trades in era"})
            continue

        # Eligible entry bars inside the era, per symbol.
        pool: dict[str, list[tuple]] = {}
        for symbol, market, _n in plan:
            bars = [
                (i, b) for i, b in enumerate(store.bars(symbol))
                if era["start"] <= b[0] < era["end"]
                and _is_tradeable(b[0], market)
                and b[4] is not None and float(b[4]) > 0
            ]
            if len(bars) >= 2:
                pool[symbol] = bars

        plan = [(s, m, n) for s, m, n in plan if s in pool]
        if not plan:
            results.append({"era": era["name"], "skipped": "no eligible candles"})
            continue

        market_of = {s: m for s, m, _n in plan}
        starting = 1000.0 * len({m for _s, m, _n in plan})
        rng = random.Random(seed)
        totals = []

        for _ in range(n_sims):
            candidates = []
            for symbol, _market, n in plan:
                for _k in range(n):
                    _idx, bar = rng.choice(pool[symbol])
                    candidates.append((bar[0], symbol, float(bar[4])))
            candidates.sort(key=lambda c: c[0])

            totals.append(_run_portfolio(candidates, starting, era, store, market_of))

        # Third arm: the LIVE entry timestamps, replayed through the same exit
        # simulator, sizing, cooldown and costs as the random arm. Deterministic,
        # so it runs once.
        live_candidates = [
            (ts, sym, px) for ts, sym, px in _live_entries(conn, era)
            if sym in market_of
        ]
        sim_live_pnl = _run_portfolio(live_candidates, starting, era, store, market_of)

        arr = np.array(totals)
        live_pnl = live_bucket["pnl"]
        results.append({
            "era": era["name"],
            "n_sims": n_sims,
            "live_pnl": live_pnl,
            "live_n_trades": live_bucket["n"],
            "sim_live_pnl": sim_live_pnl,
            "n_live_entries": len(live_candidates),
            "random_mean": float(arr.mean()),
            "random_median": float(np.median(arr)),
            "random_p5": float(np.percentile(arr, 5)),
            "random_p95": float(np.percentile(arr, 95)),
            "live_percentile": float((arr <= live_pnl).mean() * 100),
            "p_value": float((arr >= live_pnl).mean()),
            # Entry skill: sim-live vs random. Both use simulated exits, so the
            # only thing that differs is WHICH bars were entered on.
            "entry_skill_percentile": float((arr <= sim_live_pnl).mean() * 100),
            "entry_skill_p": float((arr >= sim_live_pnl).mean()),
            # Execution cost: live vs sim-live. Both use the live entries, so the
            # only thing that differs is how the exits actually filled.
            "execution_cost": live_pnl - sim_live_pnl,
            "starting_capital": starting,
        })
    return results


# ── 3. Per-symbol EV ────────────────────────────────────────────────────────

def per_symbol_ev(live: dict[str, dict], seed: int) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for era_name, bucket in live.items():
        rows = []
        for symbol, rets in sorted(bucket.get("per_symbol", {}).items()):
            n = len(rets)
            row = {"symbol": symbol, "n": n, "mean_pct": float(np.mean(rets))}
            if n < MIN_N_FOR_EV:
                row["status"] = "insufficient"
            else:
                ci = block_bootstrap_ci(rets, reps=5000, seed=seed)
                row.update({
                    "status": "ok",
                    "ci_low": ci["lower"], "ci_high": ci["upper"],
                    "block_size": ci["block_size"], "method": ci["method_used"],
                })
            rows.append(row)
        out[era_name] = rows
    return out


# ── Report ──────────────────────────────────────────────────────────────────

def render(stop_fill: dict, bench: list[dict], ev: dict, eras: list[dict],
           purity: dict, n_sims: int, seed: int) -> str:
    L: list[str] = []
    a = L.append

    a(f"# Diagnostics re-run — {date.today().isoformat()}")
    a("")
    a(f"Exit simulation: `backend/exit_sim.py` (ported from vectorbt "
      f"`get_stop_price_nb`). Bootstrap: vendored `arch`. "
      f"{n_sims} simulations, seed {seed}. Read-only on the live DB.")
    a("")

    a("## What changed vs previous numbers")
    a("")
    a("| Area | Before | Now |")
    a("|---|---|---|")
    a("| Stop/TP fill | Always filled exactly at the level | Gap rule: fills at `open` "
      "when the bar opened through the level |")
    a("| Same-bar SL+TP | Take-profit won | **Stop-loss wins** (vectorbt order) |")
    a("| Exit implementations | 2 independent copies (`_walk_candles`, `_simulate_exit`) | "
      "1 shared `simulate_exit`, both are now adapters |")
    a("| Random-arm sizing | Flat notional; `cash` set once, never decremented | "
      "**15% of available cash, compounding**, with cash returned at exit |")
    a("| Random-arm TP | Hardcoded 6% for all history | Per-era threshold (6% → 3% on 2026-09-03) |")
    a("| Segmentation | Pooled across all history | Split per model/threshold era, never pooled |")
    a("| Bootstrap | IID resample, duplicated in 2 modules | Block bootstrap via vendored `arch` |")
    a("| Benchmark arms | random vs live only | **Third arm: live entries + simulated exits**, "
      "separating entry skill from execution cost |")
    a("| Era keying | Models that emitted predictions | **Model that drove entries** "
      "(dual-mode split at the active_model flip) |")
    a("| `apply_costs` | Long-sided gross P&L regardless of `side` | Side-aware; "
      "`side=\'buy\'` unchanged, so live is bit-identical |")
    a("")
    a("The first four all pushed in the same direction: they made the simulator "
      "look better than reality, or the random benchmark look better than it should. "
      "Numbers below are expected to move against the simulator.")
    a("")

    a("## Config eras")
    a("")
    a("Boundaries are the recorded `active_model` changes in "
      "`audit_recommendations` plus the recorded take-profit change, at "
      "timestamp granularity (both landed mid-day). `settings.json` is "
      "gitignored and has never been committed, so these records are the only "
      "recoverable history. Membership is then verified against entry linkage "
      "below rather than assumed.")
    a("")
    a("| Era | Entry model | TP | Entries |")
    a("|---|---|---|---|")
    for era in eras:
        a(f"| {era['start']:%Y-%m-%d %H:%M} .. {era['end']:%Y-%m-%d %H:%M} "
          f"| {era['models']} | {era['tp_pct']:.0%} | {era.get('n_entries', 0)} |")
    a("")
    a("**Eras are keyed on the model that drove entries, not the model set that "
      "emitted predictions.** In dual-model mode both models predicted every "
      "cycle, but `process_prediction` only acts when `pred_model == "
      "active_model`, so only one model ever generated entries. Keying on the "
      "emitting set would have pooled two different entry generators into a "
      "single `mini+small` era.")
    a("")
    a(f"- Entry-model purity: **{'PASS' if purity['all_pure'] else 'FAIL'}** — "
      f"{'every era has exactly one entry model' if purity['all_pure'] else 'see below'}")
    a(f"- Era entry counts sum to total: **{'PASS' if purity['sums_correctly'] else 'FAIL'}** — "
      f"{purity['assigned_to_eras']} assigned + {purity['unassigned']} unassigned "
      f"= {purity['linked_entries']} linked = {purity['total_buys']} total BUY trades")
    if purity["impure_eras"]:
        a("")
        a("| Impure era | Models present |")
        a("|---|---|")
        for row in purity["impure_eras"]:
            a(f"| {row['era']} | {row['models']} |")
    a("")

    a("## 1. Stop-fill validation")
    a("")
    a(f"- Live stop-loss exits found: **{stop_fill['n_stop_exits']}**")
    a(f"- Compared against `simulate_exit`: **{stop_fill['n_compared']}**")
    a(f"- Unreproducible (fill outside its own bar): **{stop_fill['n_unreproducible']}**")
    a(f"- Simulator did not produce a stop at all: **{stop_fill['n_no_sim_stop']}**")
    a("")
    a("| Metric | Live | Simulated |")
    a("|---|---|---|")
    a(f"| Mean stop return | {stop_fill['live_mean_pct']:.3f}% | {stop_fill['sim_mean_pct']:.3f}% |")
    a(f"| Median stop return | {stop_fill['live_median_pct']:.3f}% | {stop_fill['sim_median_pct']:.3f}% |")
    a(f"| Worse than −3.00% | {stop_fill['live_pct_worse_than_sl']:.1f}% | "
      f"{stop_fill['sim_pct_worse_than_sl']:.1f}% |")
    a("")
    a(f"Simulated fills flagged as gapped: **{stop_fill['sim_gapped_pct']:.1f}%**. "
      f"Mean sim−live difference: **{stop_fill['mean_diff_pp']:+.3f}pp** "
      f"(median {stop_fill['median_diff_pp']:+.3f}pp). "
      f"A positive number means the simulator still reports a better fill than reality.")
    a("")
    a("### Gap rate by market")
    a("")
    a("Equities close overnight and at weekends; crypto never does. Pooling the "
      "two hides a real structural difference in how often price can jump a level.")
    a("")
    a("| Market | n | Gapped | Mean sim−live |")
    a("|---|---|---|---|")
    for market, b in sorted(stop_fill["by_market"].items()):
        a(f"| {market} | {b['n']} | {b['gapped_pct']:.1f}% | {b['mean_diff_pp']:+.3f}pp |")
    a("")

    if stop_fill["unreproducible"]:
        a("### Unreproducible fills (not fixed, not excluded silently)")
        a("")
        a("These filled outside their own backfilled bar's range. `price_monitor` "
          "samples live Alpaca IEX bars while `candles` is written by a separate "
          "backfill, so the price the exit actually filled at is not in the candle "
          "table. No candle-based simulator can reach them.")
        a("")
        a("| Symbol | Exit | Fill | Bar low | Bar high | Route |")
        a("|---|---|---|---|---|---|")
        for r in stop_fill["unreproducible"]:
            a(f"| {r['symbol']} | {r['exit_ts']:%Y-%m-%d %H:%M} | {r['live_fill']:.4f} | "
              f"{r['bar_low']:.4f} | {r['bar_high']:.4f} | {r['route']} |")
        a("")

    a("## 2. Three-arm benchmark (compounding, per era)")
    a("")
    a("Three arms, differing in exactly one thing each:")
    a("")
    a("| Arm | Entries | Exits |")
    a("|---|---|---|")
    a("| random | random bars | `simulate_exit` |")
    a("| sim-live | **live entry timestamps** | `simulate_exit` |")
    a("| live | live entry timestamps | what actually filled |")
    a("")
    a("So **sim-live vs random isolates entry selection** (exits identical), and "
      "**live vs sim-live isolates execution** (entries identical). Sizing, "
      "cooldown and costs are shared by construction — all three run through the "
      "same `_run_portfolio`.")
    a("")
    a("**Caveat on execution cost.** The sim-live arm exits via `simulate_exit`, "
      "which section 1 shows is still optimistic by roughly +0.5pp per stop even "
      "after the gap rule. So part of every `execution cost` figure is the "
      "simulator being too generous rather than live execution being bad. Treat "
      "it as an upper bound on the true execution drag, not a measurement of it.")
    a("")
    for row in bench:
        a(f"### {row['era']}")
        a("")
        if "skipped" in row:
            a(f"Skipped — {row['skipped']}.")
            a("")
            continue
        a(f"- Live net P&L: **{row['live_pnl']:+.2f}** over {row['live_n_trades']} trades")
        a(f"- Sim-live net P&L: **{row['sim_live_pnl']:+.2f}** "
          f"over {row['n_live_entries']} replayed entries")
        a(f"- Random mean: {row['random_mean']:+.2f} | median: {row['random_median']:+.2f} "
          f"| p5–p95: {row['random_p5']:+.2f} … {row['random_p95']:+.2f}")
        a("")
        a(f"- **Entry skill — sim-live percentile vs random: {row['entry_skill_percentile']:.1f}** "
          f"(p = {row['entry_skill_p']:.4f})")
        a(f"- **Execution cost — live minus sim-live: {row['execution_cost']:+.2f}** "
          f"(upper bound — see caveat below)")
        a(f"- Live percentile vs random (both effects combined): "
          f"{row['live_percentile']:.1f} (p = {row['p_value']:.4f})")
        a(f"- Starting capital: {row['starting_capital']:.0f}")
        a("")

    a("## 3. Per-symbol expected value")
    a("")
    a(f"Trade-level return % per symbol, block-bootstrap 95% CI. "
      f"Symbols with n < {MIN_N_FOR_EV} are flagged insufficient and get no CI.")
    a("")
    for era_name, rows in ev.items():
        if not rows:
            continue
        a(f"### {era_name}")
        a("")
        a("| Symbol | n | Mean % | 95% CI | Block | Method |")
        a("|---|---|---|---|---|---|")
        for r in rows:
            if r["status"] == "insufficient":
                a(f"| {r['symbol']} | {r['n']} | {r['mean_pct']:+.3f} | insufficient | — | — |")
            else:
                a(f"| {r['symbol']} | {r['n']} | {r['mean_pct']:+.3f} | "
                  f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['block_size']} | {r['method']} |")
        a("")

    a("---")
    a("")
    a("Nothing here was tuned to match recorded history. Gaps between simulated "
      "and live results are reported as found.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--db", type=Path, default=LIVE_DB)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    conn = open_readonly(args.db)
    try:
        store = CandleStore(conn)
        eras = derive_eras(conn)
        purity = verify_era_purity(conn, eras)
        logger.info("Derived %d config eras | purity=%s sums=%s",
                    len(eras), purity["all_pure"], purity["sums_correctly"])
        if not purity["all_pure"]:
            logger.warning("Impure eras (entries from >1 model): %s", purity["impure_eras"])

        logger.info("1/3 stop-fill validation...")
        stop_fill = stop_fill_validation(conn, store)

        logger.info("2/3 random-entry benchmark (%d sims x %d eras)...",
                    args.sims, len(eras))
        live = _live_era_pnl(conn, eras)
        bench = random_entry_benchmark(conn, store, eras, live, args.sims, args.seed)

        logger.info("3/3 per-symbol EV...")
        ev = per_symbol_ev(live, args.seed)

        REPORTS.mkdir(parents=True, exist_ok=True)
        out = REPORTS / f"diagnostics_{date.today().isoformat()}.md"
        out.write_text(render(stop_fill, bench, ev, eras, purity, args.sims, args.seed))
        logger.info("Report written: %s", out)

        # JSON sidecar for /api/diagnostics/three-arm. The dashboard reads this
        # rather than recomputing: 1000 simulations per era is far too slow to
        # serve synchronously.
        sidecar = REPORTS / f"diagnostics_{date.today().isoformat()}.json"
        sidecar.write_text(json.dumps({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "n_sims": args.sims,
            "seed": args.seed,
            "eras": [
                {"start": e["start"].isoformat(), "end": e["end"].isoformat(),
                 "models": e["models"], "tp_pct": e["tp_pct"],
                 "n_entries": e.get("n_entries", 0)}
                for e in eras
            ],
            "purity": {k: v for k, v in purity.items() if k != "impure_eras"},
            "stop_fill": {k: v for k, v in stop_fill.items()
                          if k not in ("unreproducible",)},
            "n_unreproducible": stop_fill.get("n_unreproducible", 0),
            "three_arm": bench,
            "per_symbol_ev": ev,
        }, indent=2, default=str))
        logger.info("JSON sidecar written: %s", sidecar)
        print(out)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
