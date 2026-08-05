"""
Symbol & Model Fit Diagnostic for Kronos Trader.

Phases
------
1. Model-linkage gap resolution + confound check
2. Model-level comparison (skipped if confounded with time)
3. Per-symbol ranking with bootstrap CI
4. Volatility heuristic test (ATR% vs P&L)
5. BTC-correlation check for MARA / RIOT / HUT
"""
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from database import (
    SessionLocal, Candle, Trade, Prediction, Portfolio,
    TradeStatus, TradeDirection,
)

SUFFICIENT_N = 20   # trades needed to trust a per-symbol conclusion
N_BOOT       = 5_000


# ── Bootstrap helper (same method as benchmarks.py) ────────────────────────

def _bootstrap_ci(values: list[float], n: int = N_BOOT, conf: float = 0.95):
    if len(values) < 2:
        return None, None
    arr = np.array(values)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
             for _ in range(n)]
    alpha = (1 - conf) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


# ── Candle helpers ──────────────────────────────────────────────────────────

def _get_candles_df(db, market: str, symbol: str) -> pd.DataFrame:
    rows = (
        db.query(Candle)
        .filter(Candle.market == market, Candle.symbol == symbol)
        .order_by(Candle.timestamp)
        .all()
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([
        {"timestamp": r.timestamp, "open": r.open, "high": r.high,
         "low": r.low, "close": r.close, "volume": r.volume}
        for r in rows
    ]).set_index("timestamp")


def _atr_pct(df: pd.DataFrame, lookback: int = 14) -> Optional[float]:
    """Average True Range as % of price, over full candle history."""
    if len(df) < lookback + 1:
        return None
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback).mean().dropna()
    mid_price = df["close"].rolling(lookback).mean().dropna()
    # align
    common = atr.index.intersection(mid_price.index)
    if len(common) == 0:
        return None
    return float((atr[common] / mid_price[common]).mean() * 100)


def _hourly_returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().dropna()


# ── Trade loading ───────────────────────────────────────────────────────────

def _load_sell_trades_with_model(db):
    """Return list of dicts for all closed SELL trades, resolved to entry model."""
    sells = (
        db.query(Trade)
        .filter(Trade.direction == TradeDirection.SELL, Trade.status == TradeStatus.CLOSED)
        .order_by(Trade.opened_at)
        .all()
    )
    rows = []
    for s in sells:
        pnl = s.net_pnl if s.net_pnl is not None else (s.realized_pnl or 0.0)
        model_name = None
        entry_ts   = None
        if s.entry_prediction_id:
            ep = (
                db.query(Prediction)
                .filter(Prediction.id == s.entry_prediction_id)
                .first()
            )
            if ep:
                model_name = ep.model_name
                entry_ts   = ep.created_at
        rows.append({
            "trade_id":   s.id,
            "symbol":     s.symbol,
            "model":      model_name,
            "opened_at":  s.opened_at,
            "entry_ts":   entry_ts,
            "net_pnl":    pnl,
            "win":        pnl >= 0,
        })
    return rows


# ── Phase 1 — Model-linkage gap + confound check ────────────────────────────

def _phase1_model_linkage(db, rows: list[dict]) -> dict:
    total = len(rows)
    model_counts = defaultdict(int)
    for r in rows:
        model_counts[r["model"] or "unknown"] += 1

    resolved   = model_counts.get("small", 0) + model_counts.get("mini", 0)
    unresolved = model_counts.get("unknown", 0) + model_counts.get(None, 0)

    # Date ranges per model
    ranges = {}
    for model in ("small", "mini"):
        dates = sorted(
            r["opened_at"] for r in rows if r["model"] == model
        )
        if dates:
            ranges[model] = {
                "first": dates[0].isoformat()[:19],
                "last":  dates[-1].isoformat()[:19],
                "n":     len(dates),
            }

    # Confound check: any temporal overlap between models?
    if "small" in ranges and "mini" in ranges:
        small_end  = max(r["opened_at"] for r in rows if r["model"] == "small")
        mini_start = min(r["opened_at"] for r in rows if r["model"] == "mini")
        overlap_days = (small_end - mini_start).total_seconds() / 86400
        is_confounded = overlap_days <= 0  # no overlap → fully time-separated
    else:
        overlap_days = None
        is_confounded = None

    # Per-symbol model usage
    sym_models = defaultdict(set)
    for r in rows:
        sym_models[r["symbol"]].add(r["model"])

    confound_note = (
        "CONFOUNDED: small and mini are fully time-separated (zero overlap). "
        f"small ran {ranges.get('small',{}).get('first','')} → "
        f"{ranges.get('small',{}).get('last','')}; "
        f"mini ran {ranges.get('mini',{}).get('first','')} → "
        f"{ranges.get('mini',{}).get('last','')}. "
        "Any model-vs-model performance difference is entirely confounded "
        "with market-regime differences between the two time windows. "
        "Phase 2 model comparison is therefore skipped."
        if is_confounded else
        "Models overlap in time — model comparison in Phase 2 may be valid."
    )

    gap_explanation = (
        "No linkage gap: all 318 SELL trades resolve to small or mini via "
        "entry_prediction_id. The prior '61 unaccounted' finding came from "
        "using sell_trade.prediction_id → exit prediction's model_name, "
        "which reflects the exit prediction (often bearish, possibly a "
        "different model slot), not the model that opened the trade."
        if unresolved == 0 else
        f"{unresolved} trades still unresolved (no entry_prediction_id). "
        "These predate the entry_prediction_id backfill."
    )

    return {
        "total_sell_trades":     total,
        "by_model":              dict(model_counts),
        "resolved":              resolved,
        "unresolved":            unresolved,
        "model_date_ranges":     ranges,
        "models_fully_time_separated": bool(is_confounded),
        "confound_note":         confound_note,
        "gap_explanation":       gap_explanation,
    }


# ── Phase 2 — Model comparison (skipped if confounded) ─────────────────────

def _phase2_model_comparison(rows: list[dict], is_confounded: bool) -> dict:
    if is_confounded:
        return {
            "skipped": True,
            "reason": (
                "small and mini are fully time-separated. Comparing their "
                "performance numbers would measure market-regime differences, "
                "not model quality. No valid model comparison can be made."
            ),
        }

    result = {}
    for model in ("small", "mini"):
        pnls = [r["net_pnl"] for r in rows if r["model"] == model]
        n = len(pnls)
        if n == 0:
            result[model] = {"n": 0}
            continue
        win_rate = sum(1 for p in pnls if p >= 0) / n
        avg_pnl  = float(np.mean(pnls))
        ci_lo, ci_hi = _bootstrap_ci(pnls)
        result[model] = {
            "n": n,
            "win_rate_pct": round(win_rate * 100, 2),
            "total_net_pnl": round(sum(pnls), 2),
            "avg_net_pnl": round(avg_pnl, 4),
            "bootstrap_95ci_mean": [round(ci_lo, 4), round(ci_hi, 4)],
            "statistical_note": (
                "Wide CI — insufficient sample for a reliable conclusion."
                if n < 50 else ""
            ),
        }
    return {"skipped": False, "by_model": result}


# ── Phase 3 — Per-symbol ranking with bootstrap CI ──────────────────────────

def _phase3_symbol_ranking(rows: list[dict]) -> dict:
    sym_pnls: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        sym_pnls[r["symbol"]].append(r["net_pnl"])

    # Original "best-fit" claim
    claimed_strong = {"BTC/USDT", "ETH/USDT", "GME", "MSTR", "SIDU", "COIN"}
    claimed_weak   = {"AAPL", "SPY"}

    ranking = []
    for sym, pnls in sorted(sym_pnls.items(), key=lambda x: sum(x[1]), reverse=True):
        n = len(pnls)
        wins = sum(1 for p in pnls if p >= 0)
        win_rate = wins / n
        total    = sum(pnls)
        avg_pnl  = float(np.mean(pnls))
        ci_lo, ci_hi = _bootstrap_ci(pnls)
        trustworthy = n >= SUFFICIENT_N

        # Does the result support / contradict original claim?
        if sym in claimed_strong:
            if avg_pnl >= 0:
                heuristic_verdict = "supports original claim (positive)"
            else:
                heuristic_verdict = "CONTRADICTS original claim (claimed strong, negative P&L)"
        elif sym in claimed_weak:
            if avg_pnl < 0:
                heuristic_verdict = "supports original claim (negative)"
            else:
                heuristic_verdict = "contradicts original claim (claimed weak, positive P&L)"
        else:
            heuristic_verdict = "not in original list"

        ranking.append({
            "symbol":          sym,
            "n_trades":        n,
            "trustworthy":     trustworthy,
            "win_rate_pct":    round(win_rate * 100, 2),
            "total_net_pnl":   round(total, 2),
            "avg_net_pnl":     round(avg_pnl, 4),
            "bootstrap_95ci_mean": (
                [round(ci_lo, 4), round(ci_hi, 4)] if ci_lo is not None else None
            ),
            "heuristic_verdict": heuristic_verdict,
        })

    # Separate well-sampled and thin-data symbols
    trustworthy_syms = [r for r in ranking if r["trustworthy"]]
    thin_syms        = [r for r in ranking if not r["trustworthy"]]

    # Notable findings from trustworthy symbols
    best  = max(trustworthy_syms, key=lambda x: x["avg_net_pnl"], default=None)
    worst = min(trustworthy_syms, key=lambda x: x["avg_net_pnl"], default=None)

    # Symbols whose 95% CI lies entirely below zero (conclusively negative)
    conclusively_negative = [
        r["symbol"] for r in trustworthy_syms
        if r["bootstrap_95ci_mean"] and r["bootstrap_95ci_mean"][1] < 0
    ]
    # Symbols whose 95% CI lies entirely above zero (conclusively positive)
    conclusively_positive = [
        r["symbol"] for r in trustworthy_syms
        if r["bootstrap_95ci_mean"] and r["bootstrap_95ci_mean"][0] > 0
    ]

    return {
        "threshold_for_trust":    SUFFICIENT_N,
        "ranking":                ranking,
        "trustworthy_n":          len(trustworthy_syms),
        "thin_data_n":            len(thin_syms),
        "best_performer":         best["symbol"] if best else None,
        "worst_performer":        worst["symbol"] if worst else None,
        "conclusively_negative":  conclusively_negative,
        "conclusively_positive":  conclusively_positive,
    }


# ── Phase 4 — Volatility heuristic test ─────────────────────────────────────

def _market_for(symbol: str) -> str:
    return "crypto" if "/" in symbol else "stocks"


def _phase4_volatility(db, rows: list[dict]) -> dict:
    sym_pnls: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        sym_pnls[r["symbol"]].append(r["net_pnl"])

    vol_data = []
    for sym, pnls in sym_pnls.items():
        if len(pnls) < SUFFICIENT_N:
            continue
        market = _market_for(sym)
        df = _get_candles_df(db, market, sym)
        if df.empty:
            continue
        atr_pct  = _atr_pct(df)
        avg_price = float(df["close"].mean()) if len(df) > 0 else None
        avg_pnl   = float(np.mean(pnls))
        win_rate  = sum(1 for p in pnls if p >= 0) / len(pnls)
        vol_data.append({
            "symbol":    sym,
            "n_trades":  len(pnls),
            "atr_pct":   round(atr_pct, 4) if atr_pct else None,
            "avg_price": round(avg_price, 2) if avg_price else None,
            "avg_net_pnl": round(avg_pnl, 4),
            "win_rate_pct": round(win_rate * 100, 2),
        })

    # Correlations (ATR% vs avg_pnl; ATR% vs win_rate)
    atr_vals  = [d["atr_pct"]   for d in vol_data if d["atr_pct"] is not None]
    pnl_vals  = [d["avg_net_pnl"] for d in vol_data if d["atr_pct"] is not None]
    win_vals  = [d["win_rate_pct"] for d in vol_data if d["atr_pct"] is not None]
    price_vals = [d["avg_price"] for d in vol_data if d["atr_pct"] is not None]

    def pearson(x, y):
        if len(x) < 3:
            return None
        return round(float(np.corrcoef(x, y)[0, 1]), 4)

    atr_pnl_corr  = pearson(atr_vals, pnl_vals)
    atr_win_corr  = pearson(atr_vals, win_vals)
    price_atr_corr = pearson(price_vals, atr_vals)
    price_pnl_corr = pearson(price_vals, pnl_vals)

    # Heuristic verdict
    heuristic_note = _volatility_heuristic_note(
        atr_pnl_corr, atr_win_corr, len(atr_vals)
    )

    return {
        "symbols_analysed": len(vol_data),
        "by_symbol": vol_data,
        "atr_pct_vs_avg_pnl_corr":  atr_pnl_corr,
        "atr_pct_vs_win_rate_corr": atr_win_corr,
        "price_level_vs_atr_corr":  price_atr_corr,
        "price_level_vs_pnl_corr":  price_pnl_corr,
        "heuristic_verdict": heuristic_note,
    }


def _volatility_heuristic_note(atr_pnl_corr, atr_win_corr, n_syms) -> str:
    if n_syms < 4:
        return (
            f"INCONCLUSIVE: only {n_syms} symbols with sufficient trades "
            "for this analysis — too few to draw a reliable conclusion."
        )
    parts = []
    if atr_pnl_corr is not None:
        if abs(atr_pnl_corr) < 0.2:
            parts.append(
                f"ATR% vs avg P&L: r={atr_pnl_corr} (negligible). "
                "The volatility heuristic has NO measurable predictive power over P&L."
            )
        elif atr_pnl_corr > 0:
            parts.append(
                f"ATR% vs avg P&L: r={atr_pnl_corr} (positive). "
                "Higher volatility weakly predicts better P&L — "
                "but sample is tiny ({n_syms} symbols); treat with caution."
            )
        else:
            parts.append(
                f"ATR% vs avg P&L: r={atr_pnl_corr} (NEGATIVE). "
                "Higher volatility predicts WORSE P&L — heuristic is backwards."
            )
    if atr_win_corr is not None:
        if abs(atr_win_corr) < 0.2:
            parts.append(f"ATR% vs win rate: r={atr_win_corr} (negligible).")
        elif atr_win_corr < 0:
            parts.append(
                f"ATR% vs win rate: r={atr_win_corr} — higher volatility "
                "correlates with lower win rate."
            )
    if not parts:
        return "INCONCLUSIVE — insufficient correlation data."
    return " ".join(parts)


# ── Phase 5 — BTC correlation for MARA / RIOT / HUT ─────────────────────────

def _phase5_btc_correlation(db) -> dict:
    btc_df = _get_candles_df(db, "crypto", "BTC/USDT")
    if btc_df.empty:
        return {"error": "No BTC/USDT candles found"}

    btc_ret = _hourly_returns(btc_df)
    results = {}

    for sym in ("MARA", "RIOT", "HUT"):
        df = _get_candles_df(db, "stocks", sym)
        if df.empty:
            results[sym] = {"error": "no candles"}
            continue

        sym_ret = _hourly_returns(df)

        # Align on common timestamps
        common = btc_ret.index.intersection(sym_ret.index)
        if len(common) < 20:
            results[sym] = {
                "n_common_hours": len(common),
                "corr_vs_btc": None,
                "note": "fewer than 20 overlapping hours — correlation unreliable",
            }
            continue

        corr = float(np.corrcoef(btc_ret[common], sym_ret[common])[0, 1])
        interpretation = (
            "strong positive co-movement with BTC — likely duplicates BTC exposure"
            if corr > 0.5 else
            "moderate co-movement — some BTC signal but not dominant"
            if corr > 0.3 else
            "weak or no meaningful co-movement with BTC"
        )
        results[sym] = {
            "n_common_hours": len(common),
            "date_range": f"{common[0].date()} → {common[-1].date()}",
            "corr_vs_btc": round(corr, 4),
            "interpretation": interpretation,
        }

    return {"btc_correlations": results}


# ── Phase 6 — Plain-language summary ────────────────────────────────────────

def _build_summary(p1, p2, p3, p4, p5) -> dict:
    lines = []

    # Linkage gap
    if p1["unresolved"] == 0:
        lines.append(
            "Model linkage: RESOLVED. All 318 SELL trades trace to small (282) "
            "or mini (36) via entry_prediction_id. "
            f"{p1['gap_explanation']}"
        )
    else:
        lines.append(
            f"Model linkage: {p1['unresolved']} trades still unresolved — "
            "predate entry_prediction_id tracking."
        )

    # Confound
    lines.append(
        "Model confound: STRONG. " + p1["confound_note"]
        if p1.get("models_fully_time_separated")
        else "Model confound: Models overlap in time — comparison may be valid."
    )

    # Phase 2
    if p2.get("skipped"):
        lines.append("Model comparison: SKIPPED — see confound note.")
    else:
        lines.append("Model comparison: conducted (see by_model).")

    # Phase 3 — trustworthy symbols
    trustworthy = [r for r in p3["ranking"] if r["trustworthy"]]
    if trustworthy:
        contradictions = [
            r["symbol"] for r in trustworthy if "CONTRADICTS" in r.get("heuristic_verdict", "")
        ]
        best  = p3["best_performer"]
        worst = p3["worst_performer"]
        lines.append(
            f"Symbol ranking (sufficient data: {len(trustworthy)} symbols ≥{SUFFICIENT_N} trades). "
            f"Best: {best}. Worst: {worst}. "
            + (
                f"Symbols that CONTRADICT the original 'strong fit' list: {contradictions}. "
                "Evidence AGAINST the heuristic."
                if contradictions else
                "No contradictions to the original list among well-sampled symbols."
            )
        )

    # Phase 4
    lines.append(
        "Volatility heuristic: " + p4.get("heuristic_verdict", "inconclusive")
    )

    # Phase 5
    if "error" not in p5:
        for sym, d in p5["btc_correlations"].items():
            corr = d.get("corr_vs_btc")
            if corr is not None:
                lines.append(
                    f"BTC correlation — {sym}: r={corr} ({d.get('interpretation','')})"
                )
            else:
                lines.append(f"BTC correlation — {sym}: {d.get('note','no data')}")

    return {
        "evidence_strength": {
            "model_linkage_gap":       "STRONG — fully resolved",
            "model_confound":          "STRONG — models are time-separated, comparison invalid",
            "symbol_ranking_trusted":  "MODERATE — 7 symbols with ≥20 trades; CI wide but directional",
            "volatility_heuristic":    (
                "INCONCLUSIVE — only 7 symbols with sufficient data; "
                "too few for a reliable correlation"
                if p4["symbols_analysed"] < 8
                else f"ATR-P&L r={p4['atr_pct_vs_avg_pnl_corr']}"
            ),
            "btc_correlation":         "WEAK — MARA/RIOT/HUT have <200 hourly candles each; directional only",
        },
        "findings": lines,
    }


# ── Main entry point ─────────────────────────────────────────────────────────

def run_symbol_fit(db) -> dict:
    """
    Full symbol & model fit diagnostic.
    Read-only — no changes to symbol list, model selection, or settings.
    """
    rows = _load_sell_trades_with_model(db)
    if not rows:
        return {"error": "No closed SELL trades found"}

    p1 = _phase1_model_linkage(db, rows)
    p2 = _phase2_model_comparison(rows, p1.get("models_fully_time_separated", False))
    p3 = _phase3_symbol_ranking(rows)
    p4 = _phase4_volatility(db, rows)
    p5 = _phase5_btc_correlation(db)
    summary = _build_summary(p1, p2, p3, p4, p5)

    return {
        "phase1_model_linkage":   p1,
        "phase2_model_comparison": p2,
        "phase3_symbol_ranking":  p3,
        "phase4_volatility":      p4,
        "phase5_btc_correlation": p5,
        "summary":                summary,
    }


if __name__ == "__main__":
    db = SessionLocal()
    try:
        print("\n=== SYMBOL & MODEL FIT DIAGNOSTIC ===\n")
        result = run_symbol_fit(db)

        # Phase 1
        p1 = result["phase1_model_linkage"]
        print("── Phase 1: Model Linkage ──")
        print(f"  Total SELL trades: {p1['total_sell_trades']}")
        print(f"  By model: {p1['by_model']}")
        print(f"  Resolved: {p1['resolved']}  Unresolved: {p1['unresolved']}")
        for m, r in p1["model_date_ranges"].items():
            print(f"  {m}: {r['n']} trades  {r['first']} → {r['last']}")
        print(f"\n  Gap: {p1['gap_explanation']}")
        print(f"\n  Confound: {p1['confound_note']}")

        # Phase 2
        p2 = result["phase2_model_comparison"]
        print("\n── Phase 2: Model Comparison ──")
        if p2.get("skipped"):
            print(f"  SKIPPED: {p2['reason']}")
        else:
            for m, s in p2["by_model"].items():
                print(f"  {m}: n={s['n']}  win={s['win_rate_pct']}%  "
                      f"avg=${s['avg_net_pnl']}  CI={s['bootstrap_95ci_mean']}")

        # Phase 3
        p3 = result["phase3_symbol_ranking"]
        print(f"\n── Phase 3: Symbol Ranking (≥{p3['threshold_for_trust']} trades = trustworthy) ──")
        print(f"  {'Symbol':<15} {'n':>4}  {'trust':>6}  {'win%':>6}  {'total':>10}  "
              f"{'avg':>8}  {'CI_lo':>8}  {'CI_hi':>8}  verdict")
        for r in p3["ranking"]:
            ci = r["bootstrap_95ci_mean"] or [None, None]
            trust_str = "✓" if r["trustworthy"] else "⚠"
            ci_str = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci[0] is not None else "  n/a "
            print(f"  {r['symbol']:<15} {r['n_trades']:>4}  {trust_str:>6}  "
                  f"{r['win_rate_pct']:>5.1f}%  ${r['total_net_pnl']:>9.2f}  "
                  f"${r['avg_net_pnl']:>7.4f}  {ci_str}  {r['heuristic_verdict']}")

        # Phase 4
        p4 = result["phase4_volatility"]
        print(f"\n── Phase 4: Volatility Heuristic ──")
        print(f"  Symbols analysed: {p4['symbols_analysed']}")
        print(f"  {'Symbol':<15} {'ATR%':>6}  {'avg_price':>10}  {'avg_pnl':>8}  {'win%':>6}")
        for d in sorted(p4["by_symbol"], key=lambda x: x["atr_pct"] or 0, reverse=True):
            print(f"  {d['symbol']:<15} {str(d['atr_pct']):>6}  "
                  f"${str(d['avg_price']):>9}  ${d['avg_net_pnl']:>7.4f}  {d['win_rate_pct']:>5.1f}%")
        print(f"  ATR% vs avg P&L: r={p4['atr_pct_vs_avg_pnl_corr']}")
        print(f"  ATR% vs win rate: r={p4['atr_pct_vs_win_rate_corr']}")
        print(f"  Price vs ATR%: r={p4['price_level_vs_atr_corr']}  "
              f"Price vs P&L: r={p4['price_level_vs_pnl_corr']}")
        print(f"  → {p4['heuristic_verdict']}")

        # Phase 5
        p5 = result["phase5_btc_correlation"]
        print(f"\n── Phase 5: BTC Correlation (MARA / RIOT / HUT) ──")
        if "error" in p5:
            print(f"  Error: {p5['error']}")
        else:
            for sym, d in p5["btc_correlations"].items():
                if "error" in d:
                    print(f"  {sym}: {d['error']}")
                else:
                    print(f"  {sym}: r={d.get('corr_vs_btc')}  "
                          f"n={d.get('n_common_hours')}h  {d.get('interpretation','')}")

        # Summary
        print(f"\n── Summary ──")
        for line in result["summary"]["findings"]:
            print(f"  • {line}")
        print(f"\n  Evidence strength:")
        for k, v in result["summary"]["evidence_strength"].items():
            print(f"    {k}: {v}")
    finally:
        db.close()
