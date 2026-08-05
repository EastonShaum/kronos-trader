"""
Diagnostic engine for Kronos Trader.

Phase 2 of the measurement layer — two analyses:
  A. MFE/MAE Analysis: for every closed trade, walk forward through actual holding-period
     candles to find Maximum Favorable Excursion (best unrealised profit) and Maximum
     Adverse Excursion (worst unrealised loss). Shows how much "room" the exits leave.

  B. Exit Policy Replay: simulate 5 alternative exit policies on the exact same entries,
     compare net P&L against the live baseline.

     Policies:
       1. baseline        — live result (actual stored net_pnl, no candle walk needed)
       2. doubled_fixed   — SL=-6%, TP=+12%, max 72h
       3. extended_hold   — SL=-3%, TP=+6%, max 168h (7 days)
       4. horizon_only    — no SL/TP, exit at 24h (model's prediction horizon)
       5. atr_based       — SL=1.5×ATR14, TP=2×ATR14, max 72h

DESIGN NOTE — single price-walk implementation
----------------------------------------------
`_walk_candles()` is the single shared candle-walk used by both MFE/MAE and the
exit replay.  MFE/MAE passes the actual holding window (entry → actual close) with
sl_abs=None/tp_abs=None.  Exit-replay passes the policy window with SL/TP levels.

Because the live strategy exits on 'bearish_signal' (58% of trades), the actual
holding windows are often 1–5 h long, far shorter than the 72h replay window.
This means:
  - MFE/MAE shows the excursion DURING the actual hold (what actually happened)
  - Replay policies show what WOULD HAVE happened under different exit rules
These measure different things on purpose; any trade-level difference between
MFE/MAE breach and replay stop-loss is a 'bearish_signal saved this' case,
which is reported explicitly in the cross-validation output.
"""
import os
import re
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from database import SessionLocal, Candle, Trade, Portfolio, Prediction, TradeStatus, TradeDirection
from cost_model import apply_costs

# ── Constants (must match live trading_engine.py) ──────────────────────────
LIVE_SL_PCT  = 0.03
LIVE_TP_PCT  = 0.06
LIVE_MAX_H   = 72
MIN_CONFIDENCE    = 0.75
# A prediction is "bearish" (triggers close of a long) when confidence <= this.
# Mirrors trading_engine.py: `elif confidence <= (1 - MIN_CONFIDENCE)`.
BEARISH_THRESHOLD = 1.0 - MIN_CONFIDENCE   # 0.25

ATR_LOOKBACK   = 14
ATR_SL_MULT    = 1.5
ATR_TP_MULT    = 2.0

EXTENDED_MAX_H = 168
HORIZON_H      = 24


# ── DB helpers ──────────────────────────────────────────────────────────────

def _get_candles_df(db, market: str, symbol: str) -> pd.DataFrame:
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
         "low": r.low, "close": r.close}
        for r in rows
    ]).set_index("timestamp")
    return df


def _get_closed_trades(db):
    return (
        db.query(Trade, Portfolio)
        .join(Portfolio, Trade.portfolio_id == Portfolio.id)
        .filter(
            Trade.status    == TradeStatus.CLOSED,
            Trade.direction == TradeDirection.SELL,
        )
        .order_by(Trade.opened_at)
        .all()
    )


def _get_entry_prediction(db, sell_trade: Trade) -> Optional[Prediction]:
    buy = (
        db.query(Trade)
        .filter(
            Trade.portfolio_id == sell_trade.portfolio_id,
            Trade.symbol       == sell_trade.symbol,
            Trade.direction    == TradeDirection.BUY,
            Trade.opened_at    <= sell_trade.opened_at,
        )
        .order_by(Trade.opened_at.desc())
        .first()
    )
    if not buy or not buy.prediction_id:
        return None
    return db.query(Prediction).get(buy.prediction_id)


def _parse_exit_reason(notes: str) -> str:
    """Extract exit reason from trade notes, e.g. 'Closed: bearish_signal ...' → 'bearish_signal'."""
    m = re.search(r"Closed:\s*(\w+)", notes or "")
    return m.group(1) if m else "unknown"


# ── Shared candle-walk implementation ───────────────────────────────────────

def _walk_candles(
    candle_window: pd.DataFrame,
    entry_price: float,
    sl_abs: Optional[float],    # absolute SL price; None = no stop-loss
    tp_abs: Optional[float],    # absolute TP price; None = no take-profit
    bearish_preds: Optional[dict] = None,
    # {floor_hour_datetime → confidence_float} for this symbol/model.
    # None or empty → signal-flip check disabled (used by MFE/MAE).
) -> dict:
    """
    Single shared candle-walk used by both MFE/MAE and exit-replay.

    Priority order per bar (mirrors live trading_engine._maybe_close_position):
      1. Take-profit  (intra-candle high vs tp_abs)
      2. Stop-loss    (intra-candle low  vs sl_abs)
      3. Bearish-flip (Prediction.confidence <= BEARISH_THRESHOLD for this hour)
         → exits at candle close, same as live engine using current_price

    If bearish_preds is provided, the walk counts candles where no prediction
    was available; this count is returned as 'signal_gaps'.

    Returns dict:
        exit_price, exit_ts, exit_reason
            ('take_profit'|'stop_loss'|'bearish_signal'|'max_hold'|'no_candles')
        mfe_pct  — max(high - entry) / entry × 100 across the whole window
        mae_pct  — max(entry - low)  / entry × 100 across the whole window
        signal_gaps  — bars where bearish_preds had no entry for that hour (int)
    """
    if candle_window.empty:
        return {
            "exit_price": entry_price, "exit_ts": None, "exit_reason": "no_candles",
            "mfe_pct": 0.0, "mae_pct": 0.0, "signal_gaps": 0,
        }

    mfe_pct     = 0.0
    mae_pct     = 0.0
    signal_gaps = 0
    check_signal = bearish_preds is not None

    for ts, row in candle_window.iterrows():
        high  = float(row["high"])
        low   = float(row["low"])
        close = float(row["close"])
        ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

        # Track excursions across the full intra-bar range before any exit
        mfe_pct = max(mfe_pct, (high - entry_price) / entry_price * 100)
        mae_pct = max(mae_pct, (entry_price - low)  / entry_price * 100)

        # 1. Take-profit (price trigger, intra-candle high)
        if tp_abs is not None and high >= tp_abs:
            return {"exit_price": tp_abs, "exit_ts": ts_dt, "exit_reason": "take_profit",
                    "mfe_pct": mfe_pct, "mae_pct": mae_pct, "signal_gaps": signal_gaps}

        # 2. Stop-loss (price trigger, intra-candle low)
        if sl_abs is not None and low <= sl_abs:
            return {"exit_price": sl_abs, "exit_ts": ts_dt, "exit_reason": "stop_loss",
                    "mfe_pct": mfe_pct, "mae_pct": mae_pct, "signal_gaps": signal_gaps}

        # 3. Bearish-signal check (hourly, after price triggers)
        if check_signal:
            hour_key = ts_dt.replace(minute=0, second=0, microsecond=0)
            conf = bearish_preds.get(hour_key)
            if conf is None:
                signal_gaps += 1
            elif conf <= BEARISH_THRESHOLD:
                return {"exit_price": close, "exit_ts": ts_dt, "exit_reason": "bearish_signal",
                        "mfe_pct": mfe_pct, "mae_pct": mae_pct, "signal_gaps": signal_gaps}

    last_ts    = candle_window.index[-1]
    last_ts_dt = last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts
    return {
        "exit_price": float(candle_window.iloc[-1]["close"]),
        "exit_ts": last_ts_dt,
        "exit_reason": "max_hold",
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "signal_gaps": signal_gaps,
    }


def _compute_atr(pre_candles: pd.DataFrame, period: int = ATR_LOOKBACK) -> Optional[float]:
    if len(pre_candles) < period:
        return None
    recent = pre_candles.tail(period)
    closes = recent["close"].values
    highs  = recent["high"].values
    lows   = recent["low"].values
    tr = [max(highs[i] - lows[i],
              abs(highs[i] - closes[i - 1]),
              abs(lows[i]  - closes[i - 1]))
          for i in range(1, len(recent))]
    return float(np.mean(tr)) if tr else None


def _policy_net(mkt, symbol, entry_price, qty, exit_price) -> float:
    costs = apply_costs(
        market=mkt, symbol=symbol, side="buy",
        entry_price=entry_price, exit_price=exit_price,
        position_size=qty,
    )
    gross = (exit_price - entry_price) * qty
    return gross - costs["entry_fee"] - costs["exit_fee"] - costs["entry_slippage"] - costs["exit_slippage"]


def _load_bearish_lookup(
    db,
    symbols: list,
    start_ts: datetime,
    end_ts: datetime,
    model_name: str,
) -> dict:
    """
    Load Prediction rows for a specific model and build:
        {symbol: {floor_hour_datetime: confidence}}

    The key is `created_at` floored to the hour so it aligns with hourly
    candle timestamps.  If multiple predictions exist for the same symbol-hour
    (e.g. scheduler ran twice), the most-recently-created one wins — same as
    the live engine which processes predictions sequentially and the last write
    of `_maybe_close_position` governs the outcome.

    Only the model that drove each trade's entry fires bearish-signal closes
    in the live engine (inactive-model predictions return early from
    process_prediction before the close check).  Callers pass the entry
    trade's model_name so lookups are model-scoped.
    """
    preds = (
        db.query(Prediction)
        .filter(
            Prediction.symbol.in_(symbols),
            Prediction.model_name == model_name,
            Prediction.created_at >= start_ts,
            Prediction.created_at <= end_ts,
        )
        .order_by(Prediction.symbol, Prediction.created_at)
        .all()
    )

    lookup: dict[str, dict[datetime, float]] = {}
    for pred in preds:
        hour_key = pred.created_at.replace(minute=0, second=0, microsecond=0)
        sym = pred.symbol
        if sym not in lookup:
            lookup[sym] = {}
        # Most-recent within hour wins (overwrite earlier)
        if pred.confidence is not None:
            lookup[sym][hour_key] = pred.confidence

    return lookup


# ── A. MFE / MAE Analysis ───────────────────────────────────────────────────

def run_mfe_mae(db) -> dict:
    """
    For every closed trade, compute MFE and MAE over the actual holding period.
    Uses _walk_candles() on the (entry_ts, exit_ts] window with no SL/TP triggers.
    """
    trade_pairs = _get_closed_trades(db)
    if not trade_pairs:
        return {"error": "No closed trades"}

    candles_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for trade, port in trade_pairs:
        mkt = port.market.value if hasattr(port.market, "value") else str(port.market)
        key = (mkt, trade.symbol)
        if key not in candles_cache:
            candles_cache[key] = _get_candles_df(db, mkt, trade.symbol)

    rows = []
    excluded = 0

    for sell, port in trade_pairs:
        mkt = port.market.value if hasattr(port.market, "value") else str(port.market)
        df  = candles_cache.get((mkt, sell.symbol), pd.DataFrame())
        if df.empty:
            excluded += 1
            continue

        buy = (
            db.query(Trade)
            .filter(
                Trade.portfolio_id == sell.portfolio_id,
                Trade.symbol       == sell.symbol,
                Trade.direction    == TradeDirection.BUY,
                Trade.opened_at    <= sell.opened_at,
            )
            .order_by(Trade.opened_at.desc())
            .first()
        )
        if not buy:
            excluded += 1
            continue

        entry_ts    = buy.opened_at
        exit_ts     = sell.closed_at or sell.opened_at
        entry_price = float(buy.price)
        exit_price  = float(sell.price)
        qty         = float(buy.quantity)

        if entry_price <= 0:
            excluded += 1
            continue

        # Actual holding window: candles strictly after entry, up to and including exit
        hold_window = df[(df.index > entry_ts) & (df.index <= exit_ts)]
        if hold_window.empty:
            excluded += 1
            continue

        walk = _walk_candles(hold_window, entry_price, sl_abs=None, tp_abs=None)
        mfe_pct  = walk["mfe_pct"]
        mae_pct  = walk["mae_pct"]
        actual_pct = (exit_price - entry_price) / entry_price * 100
        efficiency = (actual_pct / mfe_pct) if mfe_pct > 0 else None
        hold_hours = (exit_ts - entry_ts).total_seconds() / 3600
        net = sell.net_pnl if sell.net_pnl is not None else (sell.realized_pnl or 0)
        actual_exit_reason = _parse_exit_reason(sell.notes)

        rows.append({
            "trade_id":    sell.id,
            "symbol":      sell.symbol,
            "market":      mkt,
            "opened_at":   str(entry_ts)[:19],
            "hold_hours":  round(hold_hours, 1),
            "hold_bars":   len(hold_window),
            "entry_price": round(entry_price, 6),
            "exit_price":  round(exit_price, 6),
            "actual_pct":  round(actual_pct, 3),
            "mfe_pct":     round(mfe_pct, 3),
            "mae_pct":     round(mae_pct, 3),
            "efficiency":  round(efficiency, 3) if efficiency is not None else None,
            "net_pnl":     round(net, 4),
            "actual_exit_reason": actual_exit_reason,
        })

    if not rows:
        return {"error": "No trades with candle data", "excluded": excluded}

    mfe_vals = [r["mfe_pct"] for r in rows]
    mae_vals = [r["mae_pct"] for r in rows]
    eff_vals = [r["efficiency"] for r in rows if r["efficiency"] is not None]

    def _mfe_bucket(mfe: float) -> str:
        if mfe < 1:  return "<1%"
        if mfe < 3:  return "1-3%"
        if mfe < 6:  return "3-6%"
        if mfe < 10: return "6-10%"
        return ">=10%"

    mfe_dist: dict[str, int] = {}
    for r in rows:
        mfe_dist[_mfe_bucket(r["mfe_pct"])] = mfe_dist.get(_mfe_bucket(r["mfe_pct"]), 0) + 1

    winners = [r for r in rows if r["net_pnl"] >= 0]
    losers  = [r for r in rows if r["net_pnl"] < 0]

    def _smean(vals):
        return round(float(np.mean(vals)), 3) if vals else None

    mae_breach_count = sum(1 for v in mae_vals if v >= LIVE_SL_PCT * 100)
    mfe_above_tp     = sum(1 for v in mfe_vals if v > LIVE_TP_PCT * 100)

    return {
        "total_trades":   len(rows),
        "excluded":       excluded,
        "summary": {
            "avg_mfe_pct":       _smean(mfe_vals),
            "median_mfe_pct":    round(float(np.median(mfe_vals)), 3),
            "avg_mae_pct":       _smean(mae_vals),
            "median_mae_pct":    round(float(np.median(mae_vals)), 3),
            "avg_efficiency":    _smean(eff_vals),
            "median_efficiency": round(float(np.median(eff_vals)), 3) if eff_vals else None,
            "pct_mfe_gt_tp":     round(mfe_above_tp / len(mfe_vals) * 100, 1),
            "pct_mae_ge_sl":     round(mae_breach_count / len(mae_vals) * 100, 1),
        },
        "winners": {
            "n": len(winners),
            "avg_mfe_pct":    _smean([r["mfe_pct"] for r in winners]),
            "avg_mae_pct":    _smean([r["mae_pct"] for r in winners]),
            "avg_efficiency": _smean([r["efficiency"] for r in winners if r["efficiency"] is not None]),
        },
        "losers": {
            "n": len(losers),
            "avg_mfe_pct":    _smean([r["mfe_pct"] for r in losers]),
            "avg_mae_pct":    _smean([r["mae_pct"] for r in losers]),
            "avg_efficiency": _smean([r["efficiency"] for r in losers if r["efficiency"] is not None]),
        },
        "mfe_distribution": mfe_dist,
        "interpretation": (
            f"avg MFE {_smean(mfe_vals):.1f}% vs TP gate {LIVE_TP_PCT*100:.0f}%: "
            f"{mfe_above_tp}/{len(mfe_vals)} trades exceeded TP level in raw price. "
            f"avg MAE {_smean(mae_vals):.1f}% vs SL gate {LIVE_SL_PCT*100:.0f}%: "
            f"{mae_breach_count}/{len(mae_vals)} trades dipped to or below SL level "
            f"before actual exit."
        ),
        "trades": rows,
    }


# ── B. Exit Policy Replay ───────────────────────────────────────────────────

# Baseline uses actual trade.net_pnl — no candle walk, exact live reproduction.
# All other policies use _walk_candles with buy.quantity as position size.
_POLICIES = {
    "baseline":      {"desc": "Live results (actual stored net_pnl)"},
    "doubled_fixed": {"sl": 0.06,        "tp": 0.12,        "max_h": LIVE_MAX_H,   "desc": "SL=-6%, TP=+12%, max 72h"},
    "extended_hold": {"sl": LIVE_SL_PCT, "tp": LIVE_TP_PCT, "max_h": EXTENDED_MAX_H, "desc": "SL=-3%, TP=+6%, max 168h"},
    "horizon_only":  {"sl": None,        "tp": None,        "max_h": HORIZON_H,    "desc": "No SL/TP, exit at 24h"},
    "atr_based":     {"max_h": LIVE_MAX_H, "desc": f"SL=1.5×ATR{ATR_LOOKBACK}, TP=2×ATR{ATR_LOOKBACK}, max {LIVE_MAX_H}h"},
}


def run_exit_replay(db) -> dict:
    """
    Replay 5 exit policies on all real historical entries.

    Baseline: reads actual trade.net_pnl (no candle walk) — exact live reproduction.
    Alternatives: walk candles from BUY opened_at using buy.quantity, with:
      - Price triggers first (intra-candle high/low for SL/TP)
      - Bearish-signal check after price triggers (same threshold as live engine:
        confidence <= BEARISH_THRESHOLD = 0.25), using real historical Prediction rows
        filtered to the same model that drove each trade's entry.
    """
    trade_pairs = _get_closed_trades(db)
    if not trade_pairs:
        return {"error": "No closed trades"}

    # ── Collect metadata for bearish-lookup pre-load ─────────────────────────
    symbols = list({sell.symbol for sell, _ in trade_pairs})
    all_ts  = [buy_ts for sell, _ in trade_pairs
               for buy_ts in [sell.opened_at] if buy_ts]
    if not all_ts:
        return {"error": "No trade timestamps"}
    from datetime import timedelta
    replay_start = min(all_ts) - timedelta(hours=1)
    replay_end   = max(t for sell, _ in trade_pairs
                       for t in [sell.closed_at or sell.opened_at]) + timedelta(hours=EXTENDED_MAX_H + 1)

    # Build one lookup per model (only "small" and "mini" exist)
    bearish_lookups: dict[str, dict] = {}
    for model in ("small", "mini"):
        bearish_lookups[model] = _load_bearish_lookup(
            db, symbols, replay_start, replay_end, model
        )

    # ── Load candles ─────────────────────────────────────────────────────────
    candles_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for sell, port in trade_pairs:
        mkt = port.market.value if hasattr(port.market, "value") else str(port.market)
        key = (mkt, sell.symbol)
        if key not in candles_cache:
            candles_cache[key] = _get_candles_df(db, mkt, sell.symbol)

    policy_pnl:      dict[str, list[float]]    = {p: [] for p in _POLICIES}
    policy_wins:     dict[str, int]            = {p: 0  for p in _POLICIES}
    policy_exits:    dict[str, dict[str, int]] = {p: {} for p in _POLICIES}
    policy_sig_gaps: dict[str, int]            = {p: 0  for p in _POLICIES}

    excluded = 0

    for sell, port in trade_pairs:
        mkt = port.market.value if hasattr(port.market, "value") else str(port.market)
        df  = candles_cache.get((mkt, sell.symbol), pd.DataFrame())

        buy = (
            db.query(Trade)
            .filter(
                Trade.portfolio_id == sell.portfolio_id,
                Trade.symbol       == sell.symbol,
                Trade.direction    == TradeDirection.BUY,
                Trade.opened_at    <= sell.opened_at,
            )
            .order_by(Trade.opened_at.desc())
            .first()
        )

        # ── Baseline: actual stored result, always available ─────────────────
        actual_net    = sell.net_pnl if sell.net_pnl is not None else (sell.realized_pnl or 0)
        actual_reason = _parse_exit_reason(sell.notes)
        policy_pnl["baseline"].append(actual_net)
        if actual_net >= 0:
            policy_wins["baseline"] += 1
        policy_exits["baseline"][actual_reason] = \
            policy_exits["baseline"].get(actual_reason, 0) + 1

        # ── Alternative policies need candles + paired BUY ───────────────────
        if df.empty or not buy:
            excluded += 1
            continue

        entry_ts    = buy.opened_at
        entry_price = float(buy.price)
        qty         = float(buy.quantity)

        if entry_price <= 0 or qty <= 0:
            excluded += 1
            continue

        future_all  = df[df.index > entry_ts]
        pre_candles = df[df.index <= entry_ts]

        # Determine which model drove this entry (filters bearish lookup)
        entry_pred = db.query(Prediction).get(buy.prediction_id) if buy.prediction_id else None
        active_model = entry_pred.model_name if entry_pred else "small"
        sym_bearish = bearish_lookups.get(active_model, {}).get(sell.symbol, {})

        for policy_name, cfg in _POLICIES.items():
            if policy_name == "baseline":
                continue

            if future_all.empty:
                continue

            if policy_name == "atr_based":
                atr = _compute_atr(pre_candles)
                sl_abs_p = (entry_price - ATR_SL_MULT * atr) if atr else entry_price * (1 - LIVE_SL_PCT)
                tp_abs_p = (entry_price + ATR_TP_MULT * atr) if atr else entry_price * (1 + LIVE_TP_PCT)
            else:
                sl_pct   = cfg["sl"]
                tp_pct   = cfg["tp"]
                sl_abs_p = entry_price * (1 - sl_pct) if sl_pct is not None else None
                tp_abs_p = entry_price * (1 + tp_pct) if tp_pct is not None else None

            window = future_all.head(cfg["max_h"])
            walk   = _walk_candles(window, entry_price, sl_abs_p, tp_abs_p,
                                   bearish_preds=sym_bearish)
            ep     = walk["exit_price"]
            reason = walk["exit_reason"]
            policy_sig_gaps[policy_name] += walk["signal_gaps"]

            net = _policy_net(mkt, sell.symbol, entry_price, qty, ep)
            policy_pnl[policy_name].append(net)
            if net >= 0:
                policy_wins[policy_name] += 1
            policy_exits[policy_name][reason] = \
                policy_exits[policy_name].get(reason, 0) + 1

    baseline_total = sum(policy_pnl["baseline"])

    results = {}
    for policy_name, pnl_list in policy_pnl.items():
        if not pnl_list:
            results[policy_name] = {"error": "No trades"}
            continue
        total = sum(pnl_list)
        n     = len(pnl_list)
        wins  = policy_wins[policy_name]
        delta = total - baseline_total
        results[policy_name] = {
            "description":       _POLICIES[policy_name]["desc"],
            "n_trades":          n,
            "total_net_pnl":     round(total, 4),
            "avg_net_pnl":       round(total / n, 4),
            "win_rate_pct":      round(wins / n * 100, 2),
            "vs_baseline_pnl":   round(delta, 4),
            "vs_baseline_pct":   round(delta / abs(baseline_total) * 100, 2) if baseline_total != 0 else 0.0,
            "exit_reasons":      policy_exits[policy_name],
            "signal_gap_bars":   policy_sig_gaps.get(policy_name, 0),
        }

    ranked = sorted(
        [(k, v["total_net_pnl"]) for k, v in results.items() if "total_net_pnl" in v],
        key=lambda x: x[1], reverse=True,
    )

    baseline_bearish_pct = round(
        policy_exits["baseline"].get("bearish_signal", 0) / len(policy_pnl["baseline"]) * 100, 1
    ) if policy_pnl["baseline"] else 0.0

    return {
        "total_trades":          len(policy_pnl["baseline"]),
        "excluded_alt_policies": excluded,
        "baseline_bearish_pct":  baseline_bearish_pct,
        "policies":              results,
        "ranking":               [{"policy": k, "total_net_pnl": v} for k, v in ranked],
        "interpretation": (
            "Baseline: actual stored net_pnl (exact live reproduction). "
            "Alternatives: _walk_candles with buy.quantity; bearish-signal check uses "
            f"real Prediction rows (threshold: confidence <= {BEARISH_THRESHOLD}), "
            "filtered to the model that drove each trade's entry, after price triggers. "
            "signal_gap_bars counts hourly walk steps with no Prediction available."
        ),
    }


# ── Confidence diagnostic (for API endpoint) ───────────────────────────────

def run_confidence_stats(db) -> dict:
    """Per-confidence-bucket breakdown using corrected entry prediction linkage."""
    trade_pairs = _get_closed_trades(db)
    if not trade_pairs:
        return {"error": "No closed trades"}

    def _bucket(conf: Optional[float]) -> str:
        if conf is None:
            return "unknown"
        pct = conf * 100
        if pct < 80:  return "<80%"
        if pct < 90:  return "80-90%"
        return ">=90%"

    buckets: dict[str, list[float]] = {}
    no_entry_pred = 0

    for trade, port in trade_pairs:
        net = trade.net_pnl if trade.net_pnl is not None else (trade.realized_pnl or 0)
        entry_pred = _get_entry_prediction(db, trade)
        conf = entry_pred.confidence if entry_pred else None
        if conf is None:
            no_entry_pred += 1
        buckets.setdefault(_bucket(conf), []).append(net)

    result = {}
    for b, pnls in buckets.items():
        n = len(pnls)
        result[b] = {
            "n_trades":      n,
            "total_net_pnl": round(sum(pnls), 4),
            "avg_net_pnl":   round(float(np.mean(pnls)), 4),
            "win_rate_pct":  round(sum(1 for x in pnls if x >= 0) / n * 100, 2),
        }

    return {
        "total_trades":  len(trade_pairs),
        "no_entry_pred": no_entry_pred,
        "by_bucket":     result,
        "note": (
            "Confidence bucketed using the ENTRY prediction (BUY trade's linked prediction). "
            "Corrected from prior bug where SELL trade's prediction_id (exit prediction) was used."
        ),
    }


# ── Lag analysis ───────────────────────────────────────────────────────────

def _build_candle_index(db) -> dict:
    """Return {(market_str, symbol): [(datetime, close), ...]} sorted by timestamp."""
    from collections import defaultdict
    rows = (
        db.query(Candle.market, Candle.symbol, Candle.timestamp, Candle.close)
        .order_by(Candle.market, Candle.symbol, Candle.timestamp)
        .all()
    )
    idx: dict = defaultdict(list)
    for market, symbol, ts, close in rows:
        # ORM returns proper datetime; coerce just in case
        if not isinstance(ts, datetime):
            ts = datetime.fromisoformat(str(ts))
        mkt_str = market.value if hasattr(market, "value") else str(market)
        idx[(mkt_str, symbol)].append((ts, float(close) if close is not None else 0.0))
    return dict(idx)


def _nearest_candle_before(idx: dict, market: str, symbol: str, ts: datetime):
    """Binary search for (timestamp, close) of the last candle at or before ts."""
    lst = idx.get((market, symbol), [])
    if not lst:
        return None
    lo, hi = 0, len(lst) - 1
    result = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if lst[mid][0] <= ts:
            result = lst[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def run_lag_analysis(db) -> dict:
    """
    Measures three latency components for every prediction that triggered a BUY trade.

    1. decision_lag_s  — Prediction.created_at → BUY Trade.opened_at
       Captures: inference time + trading-engine decision + DB write.

    2. price_staleness_s — age of the most recent candle at prediction creation time.
       For hourly crypto candles this is typically 0–60 min.
       For stocks after market close it can be many hours (overnight gap).

    3. batch_position_effect — within a scheduler run (same UTC minute), does being
       later in the batch correlate with worse P&L?

    Correlations are Pearson r; |r| < 0.05 is treated as negligible.
    """
    from collections import defaultdict

    # Preload BUY trades with their entry predictions
    buy_rows = (
        db.query(Trade, Prediction)
        .join(Prediction, Trade.prediction_id == Prediction.id)
        .filter(
            Trade.direction == TradeDirection.BUY,
            Trade.prediction_id.isnot(None),
        )
        .order_by(Trade.opened_at)
        .all()
    )
    if not buy_rows:
        return {"error": "No BUY trades with linked predictions found"}

    # Preload SELL trades indexed by (portfolio_id, symbol) → sorted list
    sell_rows = (
        db.query(Trade)
        .filter(Trade.direction == TradeDirection.SELL, Trade.status == TradeStatus.CLOSED)
        .order_by(Trade.opened_at)
        .all()
    )
    sell_idx: dict = defaultdict(list)
    for s in sell_rows:
        sell_idx[(s.portfolio_id, s.symbol)].append(s)

    candle_idx = _build_candle_index(db)

    records = []
    for buy, pred in buy_rows:
        decision_lag_s = (buy.opened_at - pred.created_at).total_seconds()

        mkt_str = pred.market.value if hasattr(pred.market, "value") else str(pred.market)
        candle = _nearest_candle_before(candle_idx, mkt_str, pred.symbol, pred.created_at)
        staleness_s = None
        price_gap_pct = None
        if candle is not None:
            staleness_s = (pred.created_at - candle[0]).total_seconds()
            if candle[1]:
                price_gap_pct = abs(pred.current_price - candle[1]) / candle[1] * 100

        # Find the earliest matching SELL after this BUY
        candidates = sell_idx.get((buy.portfolio_id, buy.symbol), [])
        sell = next((s for s in candidates if s.opened_at >= buy.opened_at), None)
        net_pnl = sell.net_pnl if sell and sell.net_pnl is not None else None
        hold_h = (sell.opened_at - buy.opened_at).total_seconds() / 3600 if sell else None

        records.append({
            "buy_id": buy.id,
            "pred_id": pred.id,
            "market": mkt_str,
            "symbol": pred.symbol,
            "model": pred.model_name,
            "confidence": pred.confidence,
            "pred_created_at": pred.created_at,
            "buy_opened_at": buy.opened_at,
            "decision_lag_s": decision_lag_s,
            "staleness_s": staleness_s,
            "price_gap_pct": price_gap_pct,
            "net_pnl": net_pnl,
            "hold_h": hold_h,
        })

    df = pd.DataFrame(records)

    # ── Phase 0: timestamp precision check ─────────────────────────────────
    has_us = bool((df["pred_created_at"].dt.microsecond != 0).any() |
                  (df["buy_opened_at"].dt.microsecond != 0).any())
    neg_lag_n = int((df["decision_lag_s"] < 0).sum())

    # ── Phase 2: batch-position effect ─────────────────────────────────────
    df["pred_minute"] = df["pred_created_at"].dt.floor("min")
    batch_counts = df.groupby("pred_minute")["buy_id"].transform("count")
    df["batch_rank"] = df.groupby("pred_minute")["pred_created_at"].rank(method="first")
    batched = df[batch_counts > 1]
    batch_corr = None
    if len(batched) > 5 and batched["net_pnl"].notna().sum() > 5:
        batch_corr = round(
            float(batched[["batch_rank", "net_pnl"]].dropna().corr().iloc[0, 1]), 4
        )

    # ── Phase 3: lag vs P&L correlations ───────────────────────────────────
    valid_lag = df[df["net_pnl"].notna() & df["decision_lag_s"].notna()]
    lag_corr = None
    if len(valid_lag) > 5:
        lag_corr = round(float(valid_lag[["decision_lag_s", "net_pnl"]].corr().iloc[0, 1]), 4)

    valid_stale = df[df["net_pnl"].notna() & df["staleness_s"].notna()]
    stale_corr = None
    if len(valid_stale) > 5:
        stale_corr = round(float(valid_stale[["staleness_s", "net_pnl"]].corr().iloc[0, 1]), 4)

    # ── Per-market summary ──────────────────────────────────────────────────
    by_market = {}
    for mkt, grp in df.groupby("market"):
        lag_v = grp["decision_lag_s"].dropna()
        stale_v = grp["staleness_s"].dropna()
        by_market[mkt] = {
            "n_trades": len(grp),
            "avg_decision_lag_s":    round(float(lag_v.mean()),            2) if len(lag_v) else None,
            "median_decision_lag_s": round(float(lag_v.median()),          2) if len(lag_v) else None,
            "p95_decision_lag_s":    round(float(lag_v.quantile(0.95)),    2) if len(lag_v) else None,
            "avg_staleness_s":       round(float(stale_v.mean()),          2) if len(stale_v) else None,
            "median_staleness_s":    round(float(stale_v.median()),        2) if len(stale_v) else None,
            "p95_staleness_s":       round(float(stale_v.quantile(0.95)), 2) if len(stale_v) else None,
            "candles_matched":       int(grp["staleness_s"].notna().sum()),
        }

    # ── Top lag outliers (above P95 across all markets) ────────────────────
    outliers = []
    if len(valid_lag) > 0:
        p95 = float(valid_lag["decision_lag_s"].quantile(0.95))
        out = valid_lag[valid_lag["decision_lag_s"] > p95].sort_values(
            "decision_lag_s", ascending=False
        ).head(10)
        for _, r in out.iterrows():
            outliers.append({
                "buy_id":          int(r["buy_id"]),
                "symbol":          r["symbol"],
                "market":          r["market"],
                "decision_lag_s":  round(float(r["decision_lag_s"]), 1),
                "net_pnl":         round(float(r["net_pnl"]), 2) if r["net_pnl"] is not None else None,
            })

    # ── Interpretation ──────────────────────────────────────────────────────
    interp_parts = []
    if lag_corr is not None:
        if abs(lag_corr) < 0.05:
            interp_parts.append("Decision lag has negligible correlation with P&L.")
        elif lag_corr < 0:
            interp_parts.append(
                f"Higher decision lag weakly predicts lower P&L (r={lag_corr})."
            )
        else:
            interp_parts.append(
                f"Higher decision lag unexpectedly predicts higher P&L (r={lag_corr}) — possibly spurious."
            )
    if stale_corr is not None:
        if abs(stale_corr) < 0.05:
            interp_parts.append("Price staleness has negligible correlation with P&L.")
        elif stale_corr < 0:
            interp_parts.append(
                f"Staler data at entry weakly predicts lower P&L (r={stale_corr})."
            )
        else:
            interp_parts.append(
                f"Staler data at entry unexpectedly predicts higher P&L (r={stale_corr})."
            )
    if batch_corr is not None and abs(batch_corr) >= 0.05:
        interp_parts.append(
            f"Batch position has {'negative' if batch_corr < 0 else 'positive'} "
            f"correlation with P&L (r={batch_corr}); "
            f"{'later in batch → worse' if batch_corr < 0 else 'later in batch → better'}."
        )
    interp_parts.append(
        "NOTE: Trade.price = Prediction.current_price = last candle close at data-fetch "
        "time (Option A). For stocks after market hours, entry price can be many hours stale."
    )

    return {
        "total_buy_trades":      len(df),
        "has_microsecond_precision": has_us,
        "negative_lag_trades":   neg_lag_n,
        "by_market":             by_market,
        "lag_vs_pnl_corr":       lag_corr,
        "staleness_vs_pnl_corr": stale_corr,
        "batch_order": {
            "n_batched_trades":  len(batched),
            "rank_vs_pnl_corr":  batch_corr,
        },
        "lag_outliers": outliers,
        "interpretation": " ".join(interp_parts) if interp_parts else "Insufficient data.",
    }


if __name__ == "__main__":
    db = SessionLocal()
    try:
        print("\n=== MFE / MAE ANALYSIS (actual holding window) ===\n")
        mfe = run_mfe_mae(db)
        if "error" in mfe:
            print(f"  Error: {mfe['error']}")
        else:
            s = mfe["summary"]
            print(f"  Trades: {mfe['total_trades']} (excluded: {mfe['excluded']})")
            print(f"  avg MFE: {s['avg_mfe_pct']}%   median MFE: {s['median_mfe_pct']}%")
            print(f"  avg MAE: {s['avg_mae_pct']}%   median MAE: {s['median_mae_pct']}%")
            print(f"  avg efficiency: {s['avg_efficiency']}   median: {s['median_efficiency']}")
            print(f"  MFE > TP ({LIVE_TP_PCT*100:.0f}%): {s['pct_mfe_gt_tp']}% of trades")
            print(f"  MAE >= SL ({LIVE_SL_PCT*100:.0f}%): {s['pct_mae_ge_sl']}% of trades (breached SL level during hold)")
            print(f"  MFE distribution: {mfe['mfe_distribution']}")
            print(f"  → {mfe['interpretation']}")

        print("\n=== EXIT POLICY REPLAY (with bearish-signal exits) ===\n")
        replay = run_exit_replay(db)
        if "error" in replay:
            print(f"  Error: {replay['error']}")
        else:
            print(f"  Trades (baseline): {replay['total_trades']}   "
                  f"Excluded from alt policies: {replay['excluded_alt_policies']}")
            print(f"  Baseline bearish_signal rate: {replay['baseline_bearish_pct']}%  "
                  f"(185/318 = 58.2% — reference)")
            print()
            print(f"  {'Policy':<20}  {'Total P&L':>10}  {'Win%':>6}  {'vs baseline':>12}  Description")
            for r in replay["ranking"]:
                pn = r["policy"]
                d  = replay["policies"][pn]
                if "error" in d: continue
                print(f"  {pn:<20}  ${d['total_net_pnl']:>9.2f}  {d['win_rate_pct']:>5.1f}%  "
                      f"${d['vs_baseline_pnl']:>+10.2f}  {d['description']}")
            print(f"\n  Exit reasons per policy (bearish_signal now included):")
            n_base = replay["total_trades"]
            for pn, d in replay["policies"].items():
                if "error" in d: continue
                reasons = d["exit_reasons"]
                n = d["n_trades"]
                bs_pct = round(reasons.get("bearish_signal", 0) / n * 100, 1) if n else 0
                gaps = d.get("signal_gap_bars", 0)
                print(f"    {pn:<20}: {reasons}  "
                      f"[bearish={bs_pct}%  signal_gaps={gaps}]")

        print("\n=== CONFIDENCE BREAKDOWN (corrected entry prediction) ===\n")
        conf = run_confidence_stats(db)
        if "error" in conf:
            print(f"  Error: {conf['error']}")
        else:
            for b, stats in sorted(conf["by_bucket"].items()):
                print(f"  {b:<8}  n={stats['n_trades']:>3}  win={stats['win_rate_pct']:>5.1f}%  "
                      f"avg_pnl=${stats['avg_net_pnl']:>+7.2f}  total=${stats['total_net_pnl']:>+9.2f}")

        print("\n=== PREDICTION-TO-EXECUTION LAG ANALYSIS ===\n")
        lag = run_lag_analysis(db)
        if "error" in lag:
            print(f"  Error: {lag['error']}")
        else:
            print(f"  Total BUY trades analysed: {lag['total_buy_trades']}")
            print(f"  Microsecond precision: {lag['has_microsecond_precision']}  "
                  f"  Negative-lag trades (clock skew): {lag['negative_lag_trades']}")
            print()
            print(f"  {'Market':<8}  {'n':>4}  {'med lag s':>9}  {'p95 lag s':>9}  "
                  f"{'med stale s':>11}  {'candles matched':>15}")
            for mkt, s in sorted(lag["by_market"].items()):
                print(f"  {mkt:<8}  {s['n_trades']:>4}  "
                      f"{str(s['median_decision_lag_s']):>9}  "
                      f"{str(s['p95_decision_lag_s']):>9}  "
                      f"{str(s['median_staleness_s']):>11}  "
                      f"{s['candles_matched']:>15}")
            print()
            print(f"  Lag vs P&L correlation (r):       {lag['lag_vs_pnl_corr']}")
            print(f"  Staleness vs P&L correlation (r): {lag['staleness_vs_pnl_corr']}")
            bo = lag["batch_order"]
            print(f"  Batch-position effect ({bo['n_batched_trades']} batched trades): "
                  f"r={bo['rank_vs_pnl_corr']}")
            if lag["lag_outliers"]:
                print(f"\n  Top lag outliers (>P95):")
                for o in lag["lag_outliers"][:5]:
                    pnl_str = f"${o['net_pnl']:+.2f}" if o["net_pnl"] is not None else "n/a"
                    print(f"    buy_id={o['buy_id']}  {o['symbol']:<12}  "
                          f"lag={o['decision_lag_s']:.0f}s  pnl={pnl_str}")
            print(f"\n  → {lag['interpretation']}")
    finally:
        db.close()
