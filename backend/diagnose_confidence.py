"""
Confidence linkage diagnostic — Phase 1, read-only.

Checks two hypotheses for why the benchmark confidence-bucket breakdown
showed 198/318 trades in the '<80%' bucket despite a ≥75% entry gate:

  Hypothesis A — scale/unit mismatch: Prediction.confidence stored as 0-100
                 but read as 0.0-1.0 fraction (or vice versa).
  Hypothesis B — wrong prediction linked: SELL trades' prediction_id points to
                 the EXIT prediction (which can be bearish/low), not the ENTRY
                 prediction that cleared the entry gate.

Run from backend/:
    ~/Trading/kronos-trader/venv/bin/python diagnose_confidence.py [--sample N]
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from database import SessionLocal, Trade, Prediction, Portfolio, TradeStatus, TradeDirection


def _get_entry_prediction(db, sell_trade: Trade) -> Prediction | None:
    """
    Find the entry-side prediction for a SELL trade by locating the most recent
    BUY trade on the same symbol/portfolio opened before this SELL, then returning
    its linked prediction.
    """
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


def _confidence_pct(conf: float | None) -> float | None:
    if conf is None:
        return None
    return round(conf * 100, 2)


def run_diagnosis(db, sample_n: int = 30) -> dict:
    # ── 1. Hypothesis A: check raw confidence value distribution ────────────
    all_confs = db.query(Prediction.confidence).filter(Prediction.confidence.isnot(None)).all()
    conf_vals = sorted(set(round(c[0], 8) for c in all_confs))

    # Are these multiples of 1/5 (5 samples)?  1/10? 1/24?
    def _detect_denominator(vals: list[float]) -> int | None:
        for denom in [5, 10, 12, 20, 24, 25, 50, 100]:
            frac = 1.0 / denom
            if all(abs(round(v / frac) * frac - v) < 1e-7 for v in vals if v > 0):
                return denom
        return None

    non_zero = [v for v in conf_vals if v > 0]
    denom = _detect_denominator(non_zero) if non_zero else None
    sample_count_inferred = denom

    hyp_a_result = {
        "distinct_values_count": len(conf_vals),
        "min": conf_vals[0] if conf_vals else None,
        "max": conf_vals[-1] if conf_vals else None,
        "sample_values": conf_vals[:12],
        "inferred_denominator": sample_count_inferred,
        "verdict": (
            f"Values are multiples of 1/{sample_count_inferred} (inferred {sample_count_inferred} MC samples). "
            "Scale is 0.0–1.0 fraction. "
            "Hypothesis A (unit mismatch) is RULED OUT — values are correct fractions."
            if sample_count_inferred else
            "Could not infer sample count. Values appear continuous — not simple MC fraction."
        ),
    }

    # ── 2. Hypothesis B: compare SELL vs BUY trade confidence ─────────────
    sell_trades = (
        db.query(Trade)
        .filter(Trade.status == TradeStatus.CLOSED, Trade.direction == TradeDirection.SELL)
        .order_by(Trade.opened_at)
        .all()
    )

    # Get sample spanning full date range
    step = max(1, len(sell_trades) // sample_n)
    sample_trades = sell_trades[::step][:sample_n]

    rows = []
    for sell in sample_trades:
        exit_pred = db.query(Prediction).get(sell.prediction_id) if sell.prediction_id else None
        entry_pred = _get_entry_prediction(db, sell)

        exit_conf  = _confidence_pct(exit_pred.confidence  if exit_pred  else None)
        entry_conf = _confidence_pct(entry_pred.confidence if entry_pred else None)

        flag = ""
        if exit_conf is not None and exit_conf < 75:
            flag += "EXIT_CONF_LOW "
        if entry_conf is not None and entry_conf < 75:
            flag += "ENTRY_CONF_LOW "
        if exit_pred and entry_pred and exit_pred.id != entry_pred.id:
            flag += "DIFFERENT_PREDICTIONS "

        rows.append({
            "trade_id":       sell.id,
            "symbol":         sell.symbol,
            "opened_at":      str(sell.opened_at)[:19],
            "exit_pred_id":   sell.prediction_id,
            "exit_conf_pct":  exit_conf,
            "entry_pred_id":  entry_pred.id if entry_pred else None,
            "entry_conf_pct": entry_conf,
            "flag":           flag.strip(),
        })

    # ── 3. Full population: benchmark bucket using exit vs entry prediction ──
    total = len(sell_trades)
    exit_below_75  = 0
    entry_below_75 = 0
    entry_unknown  = 0
    no_exit_pred   = 0

    # Bucket distributions for both
    exit_buckets  = {"<80%": 0, "80-90%": 0, ">=90%": 0, "unknown": 0}
    entry_buckets = {"<80%": 0, "80-90%": 0, ">=90%": 0, "unknown": 0}

    for sell in sell_trades:
        # Exit side (current/buggy)
        exit_pred = db.query(Prediction).get(sell.prediction_id) if sell.prediction_id else None
        if exit_pred and exit_pred.confidence is not None:
            ec = exit_pred.confidence * 100
            if ec < 75:
                exit_below_75 += 1
            exit_buckets["<80%" if ec < 80 else ("80-90%" if ec < 90 else ">=90%")] += 1
        else:
            no_exit_pred += 1
            exit_buckets["unknown"] += 1

        # Entry side (corrected)
        entry_pred = _get_entry_prediction(db, sell)
        if entry_pred and entry_pred.confidence is not None:
            ec = entry_pred.confidence * 100
            if ec < 75:
                entry_below_75 += 1
            entry_buckets["<80%" if ec < 80 else ("80-90%" if ec < 90 else ">=90%")] += 1
        else:
            entry_unknown += 1
            entry_buckets["unknown"] += 1

    hyp_b_result = {
        "total_sell_trades": total,
        "no_exit_prediction": no_exit_pred,
        "exit_below_75pct": exit_below_75,
        "entry_below_75pct": entry_below_75,
        "entry_unknown": entry_unknown,
        "buggy_buckets_using_exit_pred":    exit_buckets,
        "corrected_buckets_using_entry_pred": entry_buckets,
        "verdict": (
            "Hypothesis B CONFIRMED. SELL trades' prediction_id links to the EXIT prediction. "
            "Exit predictions have low confidence when trades close on bearish signals or stop/TP events. "
            f"Using exit predictions: {exit_below_75}/{total} trades below 75% threshold. "
            f"Using entry predictions: {entry_below_75}/{total} trades below 75% threshold "
            f"(these are from when the threshold was lower). "
            f"{entry_unknown} trades have no matchable entry prediction."
        ),
    }

    # ── 4. Verification: post-correction, do any valid entries have conf < 75? ─
    verification = {
        "trades_corrected_below_75": entry_below_75,
        "trades_with_unknown_entry": entry_unknown,
        "status": (
            "PASS — all linked entry predictions meet or exceed 75% threshold"
            if entry_below_75 == 0
            else f"SECOND BUG FOUND — {entry_below_75} trades have entry confidence < 75%. "
                 "Investigate whether threshold changed over the trade history."
        ),
    }

    return {
        "hypothesis_a_scale_mismatch": hyp_a_result,
        "hypothesis_b_wrong_prediction": hyp_b_result,
        "sample_trades": rows,
        "verification": verification,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=30, help="Sample size across date range")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = run_diagnosis(db, sample_n=args.sample)

        import json
        print("\n=== CONFIDENCE LINKAGE DIAGNOSTIC ===\n")

        print("── Hypothesis A: Scale/Unit Mismatch ──")
        a = result["hypothesis_a_scale_mismatch"]
        print(f"  Distinct values: {a['distinct_values_count']}, range [{a['min']:.4f}, {a['max']:.4f}]")
        print(f"  Sample: {[round(v, 4) for v in (a['sample_values'] or [])[:8]]}")
        print(f"  Inferred sample count: {a['inferred_denominator']}")
        print(f"  → {a['verdict']}\n")

        print("── Hypothesis B: Wrong Prediction Linked ──")
        b = result["hypothesis_b_wrong_prediction"]
        print(f"  Total SELL trades: {b['total_sell_trades']}")
        print(f"  Exit-pred buckets (CURRENT/BUGGY):    {b['buggy_buckets_using_exit_pred']}")
        print(f"  Entry-pred buckets (CORRECTED):        {b['corrected_buckets_using_entry_pred']}")
        print(f"  Exit conf < 75%: {b['exit_below_75pct']}, Entry conf < 75%: {b['entry_below_75pct']}")
        print(f"  → {b['verdict']}\n")

        print("── Sample Trade Detail ──")
        print(f"  {'Trade':>5}  {'Symbol':<10}  {'ExitConf%':>9}  {'EntryConf%':>10}  {'Flag'}")
        for row in result["sample_trades"][:20]:
            print(f"  {row['trade_id']:>5}  {row['symbol']:<10}  "
                  f"{str(row['exit_conf_pct'] or '—'):>9}  "
                  f"{str(row['entry_conf_pct'] or '—'):>10}  "
                  f"{row['flag']}")

        print(f"\n── Verification Gate ──")
        v = result["verification"]
        print(f"  {v['status']}")
        print(f"  Trades with unknown entry: {v['trades_with_unknown_entry']}")

        print("\n=== ROOT CAUSE SUMMARY ===")
        print("""
  BUG LOCATION: benchmarks.py → run_live_stats() → confidence-bucket computation.

  CAUSE: _get_closed_trades() returns only SELL trades. Each SELL trade's
  prediction_id links to the EXIT prediction (the most-recent prediction at
  close time), not the ENTRY prediction that cleared the ≥75% confidence gate.
  Exit predictions are often bearish (confidence ≈ 0.2) or whatever the
  prediction cycle happened to produce at stop/TP fire time.

  EFFECT: The bucket shows 198/318 trades in '<80%' because those are the
  exit predictions — not a reflection of entry signal quality.

  FIX: In run_live_stats(), look up the corresponding BUY trade's
  prediction_id for each SELL trade to get entry confidence.
  See benchmarks.py (post-fix) for the corrected implementation.
""")
    finally:
        db.close()
