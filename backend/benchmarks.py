"""
Benchmark engine for Kronos Trader.

Computes three comparisons against the live strategy:
  3a. Buy & Hold       — allocate at first trade, hold to now
  3b. Random Entry     — Monte Carlo N=1000, same trade count/symbols/exit rules
  3c. Inverse Signal   — short side of every actual signal, same entry timestamp

All benchmarks use:
  - Identical starting capital and position sizing as the live strategy
  - The same cost model (via cost_model.apply_costs)
  - The same date window as the live trade history

Statistics computed for every strategy:
  net_total_return, net_cagr, sharpe (hourly returns, annualised sqrt(8760)),
  max_drawdown, longest_drawdown_hours, win_rate, profit_factor,
  avg_net_pnl_per_trade, bootstrap_95ci_mean_return (10k resamples),
  breakdown by market / symbol / confidence bucket.
"""
import logging
import math
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from database import SessionLocal, Candle, Trade, Portfolio, Prediction, MarketType, TradeStatus, TradeDirection
from cost_model import apply_costs
from exit_sim import simulate_exit

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Exit rule constants (must match live trading_engine.py)
# -------------------------------------------------------
STOP_LOSS_PCT   = 0.03
TAKE_PROFIT_PCT = 0.06
MAX_HOLD_HOURS  = 72   # maximum bars to hold before forced exit

# Annualisation factor: hourly returns → annual (8760 hours / year)
ANNUALISE = math.sqrt(8760)

# Monte Carlo iterations
MONTE_CARLO_N = 1000

# US market hours (Eastern) — for stock entry filtering
MARKET_OPEN_HOUR  = 9   # 09:30 ET but we use hourly bars → 09:00–16:00 ET
MARKET_OPEN_MIN   = 30
MARKET_CLOSE_HOUR = 16


# -------------------------------------------------------
# DB helpers
# -------------------------------------------------------

def _get_candles_df(db, market: str, symbol: str) -> pd.DataFrame:
    """Load all stored candles for a symbol as a DataFrame indexed by timestamp."""
    rows = (
        db.query(Candle)
        .filter(Candle.market == market, Candle.symbol == symbol)
        .order_by(Candle.timestamp)
        .all()
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([
        {"timestamp": r.timestamp, "open": r.open, "high": r.high,
         "low": r.low, "close": r.close, "volume": r.volume}
        for r in rows
    ]).set_index("timestamp")
    return df


def _get_closed_trades(db) -> list:
    """All closed SELL trades with realized_pnl, joined to portfolio market."""
    return (
        db.query(Trade, Portfolio)
        .join(Portfolio, Trade.portfolio_id == Portfolio.id)
        .filter(
            Trade.status    == TradeStatus.CLOSED,
            Trade.direction == TradeDirection.SELL,
        )
        .order_by(Trade.closed_at)
        .all()
    )


def _get_portfolio_start(db, market: str) -> tuple[float, datetime]:
    """Return (starting_balance, first_trade_time) for a market."""
    port = db.query(Portfolio).filter_by(market=MarketType(market)).first()
    if not port:
        return 0.0, datetime.utcnow()
    first = (
        db.query(Trade)
        .filter(Trade.portfolio_id == port.id)
        .order_by(Trade.opened_at)
        .first()
    )
    first_ts = first.opened_at if first else port.created_at
    return port.starting_balance, first_ts


# -------------------------------------------------------
# Statistics
# -------------------------------------------------------

def _compute_stats(
    equity_curve: list[tuple[datetime, float]],   # (timestamp, equity)
    trades_net_pnl: list[float],
    starting_balance: float,
    label: str,
) -> dict:
    """
    Compute the full statistics suite for a strategy or benchmark.
    equity_curve must be sorted by timestamp.
    """
    if not equity_curve or starting_balance <= 0:
        return {"label": label, "insufficient_data": True}

    ts_list, eq_list = zip(*equity_curve)
    eq_arr  = np.array(eq_list, dtype=float)
    ts_arr  = list(ts_list)

    # Hourly returns — align to 1h grid by resampling
    eq_series = pd.Series(eq_arr, index=pd.DatetimeIndex(ts_arr))
    hourly    = eq_series.resample("1h").last().ffill()
    h_returns = hourly.pct_change().dropna()

    # Net total return
    final_equity     = eq_arr[-1]
    net_total_return = (final_equity - starting_balance) / starting_balance

    # Net CAGR
    first_ts = ts_arr[0]
    last_ts  = ts_arr[-1]
    hours    = max((last_ts - first_ts).total_seconds() / 3600, 1)
    years    = hours / 8760
    net_cagr = (1 + net_total_return) ** (1 / years) - 1 if years > 0 and final_equity > 0 else 0.0

    # Sharpe — annualised using sqrt(8760) factor; risk-free rate assumed 0
    sharpe = float(h_returns.mean() / h_returns.std() * ANNUALISE) if h_returns.std() > 0 else 0.0

    # Max drawdown and longest drawdown duration
    peak       = eq_arr[0]
    max_dd     = 0.0
    dd_start   = ts_arr[0]
    longest_dd = timedelta(0)
    in_dd      = False
    dd_start_ts = ts_arr[0]

    for i, (ts, eq) in enumerate(zip(ts_arr, eq_arr)):
        if eq > peak:
            if in_dd:
                dd_dur = ts - dd_start_ts
                if dd_dur > longest_dd:
                    longest_dd = dd_dur
            peak   = eq
            in_dd  = False
        else:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
            if not in_dd:
                in_dd = True
                dd_start_ts = ts

    if in_dd:
        dd_dur = ts_arr[-1] - dd_start_ts
        if dd_dur > longest_dd:
            longest_dd = dd_dur

    # Trade-level stats
    win_pnl  = [p for p in trades_net_pnl if p >= 0]
    loss_pnl = [p for p in trades_net_pnl if p < 0]
    n_trades = len(trades_net_pnl)
    win_rate = len(win_pnl) / n_trades if n_trades else 0.0

    gross_profit = sum(win_pnl)
    gross_loss   = abs(sum(loss_pnl))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )
    avg_net_pnl  = float(np.mean(trades_net_pnl)) if trades_net_pnl else 0.0

    # Bootstrap 95% CI on mean net trade return (10,000 resamples)
    ci_low, ci_high = _bootstrap_ci(trades_net_pnl, n_resamples=10_000)

    return {
        "label":                    label,
        "starting_balance":         round(starting_balance, 2),
        "final_equity":             round(float(final_equity), 2),
        "net_total_return":         round(net_total_return * 100, 4),   # percent
        "net_cagr":                 round(net_cagr * 100, 4),           # percent
        "sharpe":                   round(sharpe, 4),
        "sharpe_note":              "Computed on hourly returns, annualised with sqrt(8760)",
        "max_drawdown_pct":         round(max_dd * 100, 4),
        "longest_drawdown_hours":   round(longest_dd.total_seconds() / 3600, 1),
        "n_trades":                 n_trades,
        "win_rate":                 round(win_rate * 100, 2),
        "profit_factor":            round(profit_factor, 4) if profit_factor != float("inf") else None,
        "avg_net_pnl_per_trade":    round(avg_net_pnl, 4),
        "bootstrap_95ci_mean_return": {
            "low":  round(ci_low,  4),
            "high": round(ci_high, 4),
            "note": "10,000 resamples of mean net trade return (USD)",
        },
        "equity_curve": [
            {"time": t.isoformat(), "equity": round(float(e), 4)}
            for t, e in zip(ts_arr[::max(1, len(ts_arr)//500)], eq_arr[::max(1, len(eq_arr)//500)])
        ],  # down-sample to ≤500 points for API response size
    }


def _bootstrap_ci(
    values: list[float], n_resamples: int = 10_000, confidence: float = 0.95
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
             for _ in range(n_resamples)]
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def _confidence_bucket(conf: Optional[float]) -> str:
    if conf is None:
        return "unknown"
    pct = conf * 100
    if pct < 80:
        return "<80%"
    elif pct < 90:
        return "80-90%"
    return ">=90%"


# -------------------------------------------------------
# 3a. Buy & Hold
# -------------------------------------------------------

def run_buy_and_hold(db) -> dict:
    """
    For each market: allocate starting_balance equally across traded symbols.
    Hold from first trade to now. One entry fee per symbol, no exit.
    """
    all_results: dict[str, dict] = {}

    for market in ["crypto", "stocks"]:
        starting, first_ts = _get_portfolio_start(db, market)
        if starting <= 0 or first_ts is None:
            continue

        # Find symbols actually traded in this market
        port = db.query(Portfolio).filter_by(market=MarketType(market)).first()
        if not port:
            continue
        symbols = list({
            t.symbol for t in db.query(Trade).filter_by(portfolio_id=port.id).all()
        })
        if not symbols:
            continue

        alloc_per_sym = starting / len(symbols)
        now = datetime.utcnow()

        # Build a combined equity curve from first_ts to now at hourly resolution
        # Each symbol contributes independently; sum them
        sym_curves: dict[str, pd.Series] = {}

        for symbol in symbols:
            df = _get_candles_df(db, market, symbol)
            if df.empty:
                continue

            # Entry cost
            entry_price = float(df.loc[df.index >= first_ts, "close"].iloc[0]) if not df.loc[df.index >= first_ts].empty else float(df["close"].iloc[-1])
            quantity = alloc_per_sym / entry_price

            costs = apply_costs(
                market=market, symbol=symbol, side="buy",
                entry_price=entry_price, exit_price=entry_price,
                position_size=quantity,
            )
            net_start_value = alloc_per_sym - costs["entry_fee"] - costs["entry_slippage"]
            adj_qty = net_start_value / entry_price

            # Equity = adj_qty * close_price at each bar
            mask = df.index >= first_ts
            sym_eq = df.loc[mask, "close"] * adj_qty
            sym_curves[symbol] = sym_eq

        if not sym_curves:
            continue

        combined = pd.concat(sym_curves.values(), axis=1).ffill().sum(axis=1)
        # Trim to common range
        combined = combined[combined.index <= now]

        eq_curve = [(idx.to_pydatetime(), float(v)) for idx, v in combined.items()]
        trades_pnl = [float(combined.iloc[-1]) - starting]  # single "trade" = entire period

        all_results[market] = _compute_stats(eq_curve, trades_pnl, starting, f"buy_hold_{market}")

    # Merge into a combined result
    if not all_results:
        return {"error": "Insufficient candle data for buy-and-hold"}

    # Combine equity curves across markets
    total_starting = sum(v.get("starting_balance", 0) for v in all_results.values())
    combined_eq_curves = []
    for mkt_stats in all_results.values():
        for pt in mkt_stats.get("equity_curve", []):
            combined_eq_curves.append(pt)
    combined_eq_curves.sort(key=lambda x: x["time"])

    return {
        "label":            "buy_hold",
        "by_market":        all_results,
        "total_starting":   total_starting,
        "equity_curve":     combined_eq_curves,
    }


# -------------------------------------------------------
# Candle-based exit simulation (for 3b and 3c)
# -------------------------------------------------------

def _simulate_exit(
    candles_df: pd.DataFrame,
    entry_ts: datetime,
    entry_price: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    is_short: bool = False,
) -> Optional[tuple[float, datetime]]:
    """
    Thin adapter over exit_sim.simulate_exit, the single shared exit simulator.

    All price-trigger logic now lives in exit_sim. This keeps the existing
    (exit_price, exit_ts) return shape so call sites are unchanged.

    NOTE: MAX_HOLD_HOURS is still applied as a BAR COUNT, not hours, exactly as
    before this refactor. For stocks (~7 bars/day) "72h" is really ~10 trading
    days. Pre-existing defect, deliberately not changed here.
    """
    future = candles_df[candles_df.index > entry_ts].head(MAX_HOLD_HOURS)
    if future.empty:
        return None

    bars = [
        (
            ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            row["open"], row["high"], row["low"], row["close"],
        )
        for ts, row in future.iterrows()
    ]

    result = simulate_exit(
        entry_price,
        bars,
        sl_pct=stop_loss_pct,
        tp_pct=take_profit_pct,
        max_bars=len(bars),
        is_short=is_short,
    )
    if result is None:
        return None
    return result.exit_price, result.ts


# -------------------------------------------------------
# 3b. Random Entry (Monte Carlo)
# -------------------------------------------------------

def _is_tradeable_hour(ts: datetime, market: str) -> bool:
    """Return True if timestamp is a valid entry for this market."""
    if market == "crypto":
        return True
    # Stocks: Mon–Fri, 09:30–16:00 ET (approximate with UTC offset -5/-4)
    # Use simple weekday + hour filter; UTC 14:30–21:00 ≈ ET 09:30–16:00
    if ts.weekday() >= 5:
        return False
    # Approximate ET market hours in UTC (ignoring DST for simplicity)
    return 14 <= ts.hour < 21


def run_monte_carlo(db, n_iterations: int = MONTE_CARLO_N) -> dict:
    """
    N=1000 random-entry Monte Carlo.
    Returns distribution of net total returns + percentile/p-value of live strategy.
    Memory-efficient: accumulates only final equity per iteration, not full curves.
    """
    trade_pairs = _get_closed_trades(db)
    if not trade_pairs:
        return {"error": "No closed trades"}

    # Build per-market/symbol trade counts and live net P&L
    sym_counts: dict[str, dict[str, int]] = {}
    live_net_pnl: list[float] = []

    for trade, port in trade_pairs:
        mkt = port.market.value if hasattr(port.market, "value") else str(port.market)
        sym_counts.setdefault(mkt, {})
        sym_counts[mkt][trade.symbol] = sym_counts[mkt].get(trade.symbol, 0) + 1
        pnl = trade.net_pnl if trade.net_pnl is not None else (trade.realized_pnl or 0)
        live_net_pnl.append(pnl)

    live_net_total = sum(live_net_pnl)

    # Load candles for all relevant symbols
    candles: dict[str, dict[str, pd.DataFrame]] = {}
    tradeable_ts: dict[str, dict[str, list[datetime]]] = {}

    for market, sym_map in sym_counts.items():
        candles[market]      = {}
        tradeable_ts[market] = {}
        for symbol in sym_map:
            df = _get_candles_df(db, market, symbol)
            if df.empty:
                continue
            candles[market][symbol] = df
            tradeable_ts[market][symbol] = [
                idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                for idx in df.index
                if _is_tradeable_hour(
                    idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx, market
                )
            ]

    # Starting balance across all markets
    total_starting = 0.0
    for market in sym_counts:
        s, _ = _get_portfolio_start(db, market)
        total_starting += s

    if total_starting <= 0:
        return {"error": "No portfolio balance"}

    position_size_pct = 0.15  # default; matches live engine

    # Run iterations — accumulate only final net P&L (memory-efficient)
    iteration_totals: list[float] = []

    for iteration in range(n_iterations):
        iter_pnl = 0.0

        for market, sym_map in sym_counts.items():
            s, _ = _get_portfolio_start(db, market)
            cash = s

            for symbol, count in sym_map.items():
                df = candles.get(market, {}).get(symbol)
                valid_ts = tradeable_ts.get(market, {}).get(symbol, [])
                if df is None or len(valid_ts) < 2:
                    continue

                for _ in range(count):
                    entry_ts = random.choice(valid_ts)
                    # Entry price = close of chosen candle
                    try:
                        entry_price = float(df.loc[entry_ts, "close"])
                    except KeyError:
                        continue

                    trade_value = cash * position_size_pct
                    qty = trade_value / entry_price if entry_price > 0 else 0
                    if qty <= 0:
                        continue

                    result = _simulate_exit(df, entry_ts, entry_price, STOP_LOSS_PCT, TAKE_PROFIT_PCT)
                    if result is None:
                        continue

                    exit_price, _ = result
                    costs = apply_costs(
                        market=market, symbol=symbol, side="buy",
                        entry_price=entry_price, exit_price=exit_price,
                        position_size=qty,
                    )
                    iter_pnl += costs["net_pnl"]

        iteration_totals.append(iter_pnl)

        if (iteration + 1) % 100 == 0:
            logger.debug(f"[monte_carlo] {iteration + 1}/{n_iterations} iterations done")

    if not iteration_totals:
        return {"error": "Monte Carlo produced no results"}

    arr = np.array(iteration_totals)
    percentile = float(np.mean(arr <= live_net_total) * 100)
    p_value    = float(np.mean(arr >= live_net_total))

    return {
        "label":               "random_entry_monte_carlo",
        "n_iterations":        n_iterations,
        "live_net_total":      round(live_net_total, 4),
        "random_mean":         round(float(np.mean(arr)), 4),
        "random_median":       round(float(np.median(arr)), 4),
        "random_std":          round(float(np.std(arr)), 4),
        "random_p5":           round(float(np.percentile(arr, 5)), 4),
        "random_p95":          round(float(np.percentile(arr, 95)), 4),
        "live_percentile":     round(percentile, 2),
        "empirical_p_value":   round(p_value, 4),
        "interpretation":      (
            f"Live strategy net P&L (${live_net_total:.2f}) ranks at the "
            f"{percentile:.1f}th percentile of {n_iterations} random-entry runs. "
            f"p-value = {p_value:.4f} (fraction of random runs >= live result)."
        ),
        # Down-sample histogram to 50 bins for API efficiency
        "histogram": {
            "bins":   [round(x, 4) for x in np.linspace(float(arr.min()), float(arr.max()), 51)[:-1].tolist()],
            "counts": np.histogram(arr, bins=50)[0].tolist(),
            "live_value": round(live_net_total, 4),
        },
    }


# -------------------------------------------------------
# 3c. Inverse Signal
# -------------------------------------------------------

def run_inverse_signal(db) -> dict:
    """
    For every actual closed trade: simulate the short side.
    Same entry timestamp, mirrored stop (+3%) and take-profit (-6%).
    Exit evaluated against stored candles — not a sign flip.
    """
    trade_pairs = _get_closed_trades(db)
    if not trade_pairs:
        return {"error": "No closed trades"}

    # Pre-load candles
    candles_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for trade, port in trade_pairs:
        mkt = port.market.value if hasattr(port.market, "value") else str(port.market)
        key = (mkt, trade.symbol)
        if key not in candles_cache:
            candles_cache[key] = _get_candles_df(db, mkt, trade.symbol)

    inverse_pnl: list[float] = []
    excluded = 0
    total_starting = 0.0
    seen_markets: set = set()

    equity_events: list[tuple[datetime, float]] = []

    for trade, port in trade_pairs:
        mkt = port.market.value if hasattr(port.market, "value") else str(port.market)
        if mkt not in seen_markets:
            s, _ = _get_portfolio_start(db, mkt)
            total_starting += s
            seen_markets.add(mkt)

        df = candles_cache.get((mkt, trade.symbol), pd.DataFrame())
        if df.empty:
            excluded += 1
            continue

        # Entry: same open timestamp as the actual trade
        entry_ts = trade.opened_at
        try:
            # Find closest candle at or after entry_ts
            eligible = df[df.index >= entry_ts]
            if eligible.empty:
                excluded += 1
                continue
            entry_price = float(eligible.iloc[0]["close"])
        except Exception:
            excluded += 1
            continue

        result = _simulate_exit(
            df, entry_ts, entry_price,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
            is_short=True,
        )

        if result is None:
            excluded += 1
            continue

        exit_price, exit_ts = result
        qty = (trade.total_value or 0) / entry_price if entry_price > 0 else 0
        if qty <= 0:
            excluded += 1
            continue

        # Short P&L: profit when price falls
        gross_pnl = (entry_price - exit_price) * qty
        costs = apply_costs(
            market=mkt, symbol=trade.symbol, side="buy",
            entry_price=entry_price, exit_price=exit_price,
            position_size=qty,
        )
        # For shorts, fees/slippage are still costs
        net = gross_pnl - costs["entry_fee"] - costs["exit_fee"] - costs["entry_slippage"] - costs["exit_slippage"]
        inverse_pnl.append(net)
        equity_events.append((exit_ts, net))

    if not inverse_pnl:
        return {"error": "No inverse trades could be simulated", "excluded": excluded}

    # Build equity curve from starting balance
    equity_events.sort(key=lambda x: x[0])
    eq = total_starting
    eq_curve: list[tuple[datetime, float]] = [(
        equity_events[0][0] - timedelta(hours=1) if equity_events else datetime.utcnow(),
        eq
    )]
    for ts, pnl in equity_events:
        eq += pnl
        eq_curve.append((ts, eq))

    stats = _compute_stats(eq_curve, inverse_pnl, total_starting, "inverse_signal")
    stats["excluded_count"] = excluded
    stats["excluded_note"]  = "Trades excluded due to missing candles at entry/exit"
    return stats


# -------------------------------------------------------
# Live strategy stats
# -------------------------------------------------------

def run_live_stats(db) -> dict:
    """Compute full statistics for the live strategy using stored cost fields."""
    trade_pairs = _get_closed_trades(db)
    if not trade_pairs:
        return {"error": "No closed trades"}

    total_starting = 0.0
    seen_markets: set = set()
    equity_events: list[tuple[datetime, float]] = []
    pnl_list: list[float] = []

    by_market:  dict[str, list[float]] = {}
    by_symbol:  dict[str, list[float]] = {}
    by_conf_bkt: dict[str, list[float]] = {}

    for trade, port in trade_pairs:
        mkt = port.market.value if hasattr(port.market, "value") else str(port.market)
        if mkt not in seen_markets:
            s, _ = _get_portfolio_start(db, mkt)
            total_starting += s
            seen_markets.add(mkt)

        net = trade.net_pnl if trade.net_pnl is not None else (trade.realized_pnl or 0)
        gross = trade.gross_pnl if trade.gross_pnl is not None else (trade.realized_pnl or 0)

        ts = trade.closed_at or trade.opened_at
        pnl_list.append(net)
        equity_events.append((ts, net))

        by_market.setdefault(mkt, []).append(net)
        by_symbol.setdefault(trade.symbol, []).append(net)

        # Confidence bucket from the ENTRY prediction (BUY trade's linked prediction),
        # not the EXIT prediction (which is what sell trade.prediction_id points to).
        conf = None
        entry_buy = (
            db.query(Trade)
            .filter(
                Trade.portfolio_id == trade.portfolio_id,
                Trade.symbol       == trade.symbol,
                Trade.direction    == TradeDirection.BUY,
                Trade.opened_at    <= trade.opened_at,
            )
            .order_by(Trade.opened_at.desc())
            .first()
        )
        if entry_buy and entry_buy.prediction_id:
            entry_pred = db.query(Prediction).get(entry_buy.prediction_id)
            if entry_pred:
                conf = entry_pred.confidence
        bucket = _confidence_bucket(conf)
        by_conf_bkt.setdefault(bucket, []).append(net)

    # Build equity curve
    equity_events.sort(key=lambda x: x[0])
    eq = total_starting
    eq_curve: list[tuple[datetime, float]] = [
        (equity_events[0][0] - timedelta(hours=1), eq)
    ] if equity_events else []
    for ts, pnl in equity_events:
        eq += pnl
        eq_curve.append((ts, eq))

    stats = _compute_stats(eq_curve, pnl_list, total_starting, "live_strategy")

    # Gross vs net comparison
    gross_list = [
        (t.gross_pnl if t.gross_pnl is not None else t.realized_pnl or 0)
        for t, _ in trade_pairs
    ]
    total_gross = sum(gross_list)
    total_net   = sum(pnl_list)
    stats["gross_net_comparison"] = {
        "total_gross_pnl": round(total_gross, 4),
        "total_net_pnl":   round(total_net, 4),
        "total_cost_drag": round(total_gross - total_net, 4),
        "cost_drag_pct":   round((total_gross - total_net) / abs(total_gross) * 100, 2)
            if total_gross != 0 else 0.0,
    }

    # Breakdowns
    def _bucket_stats(pnl_by_key: dict[str, list[float]]) -> dict:
        return {
            k: {
                "n_trades":           len(v),
                "total_net_pnl":      round(sum(v), 4),
                "avg_net_pnl":        round(float(np.mean(v)), 4) if v else 0,
                "win_rate":           round(sum(1 for x in v if x >= 0) / len(v) * 100, 2) if v else 0,
            }
            for k, v in pnl_by_key.items()
        }

    stats["by_market"]          = _bucket_stats(by_market)
    stats["by_symbol"]          = _bucket_stats(by_symbol)
    stats["by_confidence_bucket"] = _bucket_stats(by_conf_bkt)
    stats["confidence_note"] = (
        "Buckets: <80% (just above 75% threshold), 80-90%, >=90%. "
        "Monotonically increasing win_rate/avg_pnl would confirm the confidence signal carries information."
    )

    return stats


# -------------------------------------------------------
# Coverage check for /api/benchmarks/status
# -------------------------------------------------------

def check_coverage(db) -> dict:
    """
    Return whether stored candles are sufficient to trust benchmark results.
    Threshold: ≥99% for crypto, ≥95% for stocks.
    """
    from candle_backfill import _symbols_from_trades, _earliest_trade_ts, _coverage_report
    symbol_map = _symbols_from_trades(db)
    now = datetime.utcnow()
    results: dict = {}
    all_ok = True

    for market, symbols in symbol_map.items():
        threshold = 0.99 if market == "crypto" else 0.95
        results[market] = {}
        for symbol in symbols:
            earliest = _earliest_trade_ts(db, market, symbol)
            # actual count
            from database import Candle as CandleModel
            count = db.query(CandleModel).filter(
                CandleModel.market == market,
                CandleModel.symbol == symbol,
            ).count()

            if earliest:
                expected = int((now - earliest).total_seconds() / 3600) + 1
                if market == "stocks":
                    # Roughly 252 trading days * 6.5 hours = 1638 bars/year
                    elapsed_days = (now - earliest).days
                    expected_trading = max(1, int(elapsed_days * 252 / 365 * 6.5))
                    coverage = min(1.0, count / expected_trading)
                else:
                    coverage = min(1.0, count / expected) if expected > 0 else 0.0
            else:
                coverage = 0.0

            ok = coverage >= threshold
            if not ok:
                all_ok = False

            results[market][symbol] = {
                "candle_count": count,
                "coverage_pct": round(coverage * 100, 1),
                "threshold_pct": round(threshold * 100, 1),
                "ok": ok,
            }

    return {
        "sufficient": all_ok,
        "by_symbol":  results,
        "checked_at": now.isoformat(),
    }


# -------------------------------------------------------
# Master runner
# -------------------------------------------------------

def run_all_benchmarks(db) -> dict:
    """
    Run all benchmark computations and return the combined result dict.
    Monte Carlo is the most expensive — chunked to avoid OOM on Pi.
    """
    logger.info("[benchmarks] Starting full benchmark run...")
    now = datetime.utcnow()

    logger.info("[benchmarks] Computing live strategy stats...")
    live = run_live_stats(db)

    logger.info("[benchmarks] Computing buy-and-hold...")
    buy_hold = run_buy_and_hold(db)

    logger.info("[benchmarks] Running Monte Carlo (N=1000)...")
    mc = run_monte_carlo(db, n_iterations=MONTE_CARLO_N)

    logger.info("[benchmarks] Running inverse signal simulation...")
    inverse = run_inverse_signal(db)

    logger.info("[benchmarks] All benchmarks complete.")

    return {
        "computed_at":     now.isoformat(),
        "live_strategy":   live,
        "buy_hold":        buy_hold,
        "monte_carlo":     mc,
        "inverse_signal":  inverse,
    }