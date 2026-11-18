"""
Probabilistic-output experiment.

QUESTION
--------
The live engine computes `confidence` as the fraction of bars in a single
AVERAGED forecast path that sit above the current price
(kronos_engine.py:103). The Monte Carlo samples are averaged away inside
Kronos before that statistic is taken, so the model's own uncertainty never
reaches the trading logic. Measured over 22,051 scored predictions that
statistic carries no information: accuracy is flat across its range and
corr(confidence, correct) = -0.019.

This script computes the quantity the Kronos paper actually recommends - a
directional PROBABILITY across independent sample paths - and tests whether
it is informative where the current statistic is not.

    p_up = fraction of N independent sample paths whose final close is above
           the current price

METHOD
------
Kronos averages over the sample axis inside auto_regressive_inference, so
sample_count>1 cannot return per-sample paths. Instead this calls
predict_batch with the SAME window repeated N times at sample_count=1.
Sampling is stochastic (T=1.0, top_p=0.9), so each batch element is an
independent draw. No library modification, no duplicated normalisation.

Verified on BTC/USDT: 8 repeats produce 8 distinct paths with a final-close
dispersion of ~1.4% of price.

COST
----
About 15s per path on a Raspberry Pi 5, and batching gives no speedup because
inference is CPU-bound. N=8 costs ~2 minutes per evaluated point. Results are
appended to a JSONL file after every point, so an interrupted run is still
usable and a rerun resumes where it stopped.

READ-ONLY on trading data. Writes only its own output file.

    venv/bin/python backend/experiments/prob_signal.py --points 200 --samples 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parent / "Kronos"))

from kronos_engine import _infer_freq, load_model  # noqa: E402

logger = logging.getLogger("prob_signal")

LIVE_DB = BACKEND / "kronos_trader.db"
SCHEDULER_URL = "http://localhost:8000/api/scheduler/status"

HORIZON = 24
LOOKBACK = 400
# Thresholds the live engine trades, so the outcome columns are directly
# comparable with what the system actually does.
SL_PCT = 0.03
TP_PCT = 0.03


def open_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def check_contention(ignore: bool) -> None:
    """Refuse to add CPU load while a prediction cycle is running."""
    if ignore:
        logger.warning("Contention check skipped (--ignore-contention)")
        return
    try:
        import urllib.request
        with urllib.request.urlopen(SCHEDULER_URL, timeout=3) as r:
            active = json.loads(r.read().decode()).get("active_predictions") or []
    except Exception as exc:
        logger.warning("Could not reach scheduler (%s); proceeding.", exc)
        return
    if active:
        print(f"Prediction cycle active for {active}. This experiment is CPU-bound "
              f"and would contend with it. Wait, or pass --ignore-contention.",
              file=sys.stderr)
        raise SystemExit(2)


def load_candles(conn, symbol: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT timestamp, open, high, low, close, volume FROM candles "
        "WHERE symbol = ? ORDER BY timestamp", (symbol,)
    ).fetchall()


def build_points(conn, symbols: list[str], n_points: int, seed: int) -> list[dict]:
    """
    Candidate evaluation points, stratified evenly across symbols.

    A point needs LOOKBACK bars behind it and HORIZON bars ahead so the
    outcome is measurable.
    """
    rng = random.Random(seed)
    per_symbol = max(1, n_points // max(1, len(symbols)))
    points = []
    for sym in symbols:
        rows = load_candles(conn, sym)
        lo, hi = LOOKBACK, len(rows) - HORIZON - 1
        if hi <= lo:
            logger.warning("%s: only %d candles, skipping", sym, len(rows))
            continue
        idxs = rng.sample(range(lo, hi), min(per_symbol, hi - lo))
        for i in sorted(idxs):
            points.append({"symbol": sym, "idx": i,
                           "anchor": str(rows[i]["timestamp"])})
    rng.shuffle(points)
    return points[:n_points]


def evaluate_point(predictor, rows, idx: int, n_samples: int) -> dict | None:
    """Run N independent paths for one point and score them against the outcome."""
    win = rows[idx - LOOKBACK: idx + 1]
    fut = rows[idx + 1: idx + 1 + HORIZON]
    if len(win) < 50 or len(fut) < HORIZON:
        return None

    df = pd.DataFrame([{
        "timestamps": pd.to_datetime(r["timestamp"]),
        "open": r["open"], "high": r["high"], "low": r["low"],
        "close": r["close"], "volume": r["volume"] or 0.0,
    } for r in win]).set_index("timestamps")
    if df[["open", "high", "low", "close"]].isnull().values.any():
        return None

    cur = float(df["close"].iloc[-1])
    if cur <= 0:
        return None

    freq = _infer_freq(df.index)
    x_ts = pd.Series(df.index)
    y_ts = pd.Series(pd.date_range(start=df.index[-1], periods=HORIZON + 1, freq=freq)[1:])
    dfi = df[["open", "high", "low", "close", "volume"]].ffill().reset_index(drop=True)

    paths = predictor.predict_batch(
        df_list=[dfi] * n_samples,
        x_timestamp_list=[x_ts] * n_samples,
        y_timestamp_list=[y_ts] * n_samples,
        pred_len=HORIZON, T=1.0, top_p=0.9, sample_count=1, verbose=False,
    )

    closes = np.array([p["close"].values for p in paths], dtype=float)   # (N, HORIZON)
    highs = np.array([p["high"].values for p in paths], dtype=float)
    lows = np.array([p["low"].values for p in paths], dtype=float)
    finals = closes[:, -1]

    # --- the candidate signal: a real directional probability -------------
    p_up = float((finals > cur).mean())
    # Probability the take-profit is touched before the stop, per path. This is
    # closer to what the engine actually needs than direction alone.
    tp_lvl, sl_lvl = cur * (1 + TP_PCT), cur * (1 - SL_PCT)
    tp_first = 0
    for j in range(len(paths)):
        hit_tp = np.argmax(highs[j] >= tp_lvl) if (highs[j] >= tp_lvl).any() else None
        hit_sl = np.argmax(lows[j] <= sl_lvl) if (lows[j] <= sl_lvl).any() else None
        if hit_tp is not None and (hit_sl is None or hit_tp < hit_sl):
            tp_first += 1
    p_tp_first = tp_first / len(paths)

    # --- the incumbent statistic, reproduced exactly ----------------------
    mean_path = closes.mean(axis=0)
    legacy_confidence = float((mean_path > cur).mean())

    # --- what actually happened -------------------------------------------
    actual_close = float(fut[-1]["close"])
    fut_high = max(float(r["high"]) for r in fut)
    fut_low = min(float(r["low"]) for r in fut)
    actual_up = actual_close > cur
    # Which level was touched first in reality
    real_tp_first = None
    for r in fut:
        if float(r["low"]) <= sl_lvl:
            real_tp_first = False
            break
        if float(r["high"]) >= tp_lvl:
            real_tp_first = True
            break

    return {
        # symbol is filled in by the caller, which knows it
        "anchor": str(rows[idx]["timestamp"]),
        "current_price": cur,
        "n_samples": n_samples,
        "p_up": p_up,
        "p_tp_first": p_tp_first,
        "legacy_confidence": legacy_confidence,
        "sample_sd_pct": float(finals.std() / cur * 100),
        "pred_mean_final": float(finals.mean()),
        "actual_close": actual_close,
        "actual_up": bool(actual_up),
        "actual_ret_pct": (actual_close - cur) / cur * 100,
        "actual_max_ret_pct": (fut_high - cur) / cur * 100,
        "actual_min_ret_pct": (fut_low - cur) / cur * 100,
        "real_tp_first": real_tp_first,
        "correct_p_up": bool((p_up > 0.5) == actual_up),
        "correct_legacy": bool((legacy_confidence > 0.5) == actual_up),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=200)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--model", default="small", choices=["small", "mini"])
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT,NVDA,AMD,MARA,HUT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=BACKEND / "experiments" / "prob_signal_results.jsonl")
    ap.add_argument("--ignore-contention", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    check_contention(args.ignore_contention)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    conn = open_readonly(LIVE_DB)

    # Resume: skip points already recorded.
    done = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["symbol"], r["anchor"]))
            except Exception:
                continue
        logger.info("Resuming: %d points already recorded", len(done))

    points = build_points(conn, symbols, args.points, args.seed)
    todo = [p for p in points if (p["symbol"], p["anchor"]) not in done]
    logger.info("%d points selected, %d still to do (%d samples each, model=%s)",
                len(points), len(todo), args.samples, args.model)
    if not todo:
        logger.info("Nothing to do.")
        return 0

    est = len(todo) * args.samples * 15 / 3600
    logger.info("Rough estimate: %.1f hours at ~15s per path", est)

    predictor = load_model(args.model)
    cache: dict[str, list] = {}
    t0 = time.time()

    with args.out.open("a") as fh:
        for i, pt in enumerate(todo, start=1):
            sym = pt["symbol"]
            if sym not in cache:
                cache[sym] = load_candles(conn, sym)
            try:
                rec = evaluate_point(predictor, cache[sym], pt["idx"], args.samples)
            except Exception as exc:
                logger.warning("%s @ %s failed: %s", sym, pt["anchor"], exc)
                continue
            if rec is None:
                continue
            rec["symbol"] = sym
            rec["model"] = args.model
            rec["evaluated_at"] = datetime.now(timezone.utc).isoformat()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 5 == 0 or i == 1:
                rate = (time.time() - t0) / i
                logger.info("%d/%d  p_up=%.3f legacy=%.3f actual_up=%s  "
                            "(%.0fs/pt, ~%.1fh left)",
                            i, len(todo), rec["p_up"], rec["legacy_confidence"],
                            rec["actual_up"], rate, rate * (len(todo) - i) / 3600)

    conn.close()
    logger.info("Done. Results: %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
