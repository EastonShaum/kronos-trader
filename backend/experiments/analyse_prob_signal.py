"""
Analyse the probabilistic-output experiment.

Tests whether `p_up` (fraction of independent sample paths ending above entry)
carries directional information that `legacy_confidence` (fraction of bars in
the averaged path above entry) does not.

The comparison is PAIRED - both signals are computed on the same points from
the same model runs - which is far more powerful than comparing either against
a fixed baseline, and matters because the sample here is small and expensive.

    venv/bin/python backend/experiments/analyse_prob_signal.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from stats_bootstrap import block_bootstrap_ci  # noqa: E402

DEFAULT_IN = BACKEND / "experiments" / "prob_signal_results.jsonl"


def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"No results at {path}", file=sys.stderr)
        raise SystemExit(1)
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def calib(recs, key, label, edges):
    print(f"\n  {label}")
    print(f"    {'bucket':>12}{'n':>6}{'pred':>8}{'actual up%':>12}{'lift':>8}")
    print("    " + "-" * 46)
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        sub = [r for r in recs if lo <= r[key] < hi]
        if len(sub) < 5:
            continue
        up = np.mean([r["actual_up"] for r in sub]) * 100
        pred = np.mean([r[key] for r in sub]) * 100
        rows.append((f"{lo:.2f}-{hi:.2f}", len(sub), pred, up))
        print(f"    {rows[-1][0]:>12}{len(sub):>6}{pred:>8.1f}{up:>12.1f}{up-50:>+8.1f}")
    if len(rows) >= 2:
        x = np.array([r[2] for r in rows]); y = np.array([r[3] for r in rows])
        w = np.array([r[1] for r in rows])
        # Weighted correlation between predicted and realised frequency. A real
        # probability should be positively, monotonically related.
        xm, ym = np.average(x, weights=w), np.average(y, weights=w)
        cov = np.average((x-xm)*(y-ym), weights=w)
        sx = np.sqrt(np.average((x-xm)**2, weights=w))
        sy = np.sqrt(np.average((y-ym)**2, weights=w))
        r = cov/(sx*sy) if sx > 0 and sy > 0 else float("nan")
        print(f"    weighted calibration corr: {r:+.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    args = ap.parse_args(argv)

    recs = load(args.input)
    n = len(recs)
    print(f"=== PROBABILISTIC-OUTPUT EXPERIMENT — {n} evaluated points ===")
    if n < 20:
        print("Too few points to say anything yet.")
        return 0
    by_sym = {}
    for r in recs:
        by_sym[r["symbol"]] = by_sym.get(r["symbol"], 0) + 1
    print(f"  symbols: {by_sym}")
    print(f"  samples per point: {recs[0]['n_samples']}  model: {recs[0].get('model')}")
    print(f"  mean sample dispersion: {np.mean([r['sample_sd_pct'] for r in recs]):.3f}% of price")
    print(f"  base rate (actual up):  {np.mean([r['actual_up'] for r in recs])*100:.1f}%")

    # --- headline: paired accuracy of the two signals ---------------------
    cp = np.array([1.0 if r["correct_p_up"] else 0.0 for r in recs])
    cl = np.array([1.0 if r["correct_legacy"] else 0.0 for r in recs])

    print("\n=== DIRECTIONAL ACCURACY (same points, same model runs) ===")
    for label, arr in (("p_up (sample paths)", cp), ("legacy confidence", cl)):
        ci = block_bootstrap_ci(arr, block_size=1, reps=4000, seed=0)
        band = "[%.1f, %.1f]" % (ci["lower"]*100, ci["upper"]*100)
        star = " *" if ci["lower"]*100 > 50 else ""
        print(f"  {label:<22}{arr.mean()*100:>7.2f}%  95% CI {band}{star}")
    print("  * = CI excludes 50%")

    # McNemar: of the points where they disagree, who is right more often?
    b = int(np.sum((cp == 1) & (cl == 0)))   # p_up right, legacy wrong
    c = int(np.sum((cp == 0) & (cl == 1)))   # legacy right, p_up wrong
    print(f"\n  disagreements: p_up right/legacy wrong = {b}, "
          f"legacy right/p_up wrong = {c}")
    if b + c > 0:
        pv = sps.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
        print(f"  McNemar exact p = {pv:.4f}  "
              f"-> {'p_up is better' if b > c and pv < 0.05 else 'legacy is better' if c > b and pv < 0.05 else 'no detectable difference'}")

    # --- calibration ------------------------------------------------------
    print("\n=== CALIBRATION: does a higher score mean a higher chance of going up? ===")
    calib(recs, "p_up", "p_up (fraction of sample paths ending above entry)",
          [0, .13, .38, .63, .88, 1.01])
    calib(recs, "legacy_confidence", "legacy_confidence (bars above entry on averaged path)",
          [0, .1, .25, .5, .75, .9, 1.01])

    # --- high-conviction subset ------------------------------------------
    print("\n=== HIGH-CONVICTION SUBSETS (what selective trading would actually use) ===")
    print(f"  {'filter':<34}{'n':>5}{'up%':>8}{'mean ret%':>11}")
    print("  " + "-" * 58)
    tests = [
        ("p_up >= 0.875",            lambda r: r["p_up"] >= 0.875),
        ("p_up <= 0.125",            lambda r: r["p_up"] <= 0.125),
        ("p_tp_first >= 0.75",       lambda r: r["p_tp_first"] >= 0.75),
        ("legacy >= 0.75 (live gate)", lambda r: r["legacy_confidence"] >= 0.75),
        ("legacy >= 0.90",           lambda r: r["legacy_confidence"] >= 0.90),
    ]
    for label, fn in tests:
        sub = [r for r in recs if fn(r)]
        if len(sub) < 5:
            print(f"  {label:<34}{len(sub):>5}      (too few)")
            continue
        up = np.mean([r["actual_up"] for r in sub]) * 100
        ret = np.mean([r["actual_ret_pct"] for r in sub])
        print(f"  {label:<34}{len(sub):>5}{up:>8.1f}{ret:>+11.3f}")

    # --- does p_tp_first predict the real race? --------------------------
    have = [r for r in recs if r["real_tp_first"] is not None]
    if len(have) >= 20:
        pt = np.array([1.0 if (r["p_tp_first"] > 0.5) == r["real_tp_first"] else 0.0
                       for r in have])
        ci = block_bootstrap_ci(pt, block_size=1, reps=4000, seed=0)
        print(f"\n=== p_tp_first predicting which level was touched first ===")
        print(f"  n={len(have)}  accuracy {pt.mean()*100:.2f}%  "
              f"95% CI [{ci['lower']*100:.1f}, {ci['upper']*100:.1f}]")
        print(f"  (base rate TP-first in reality: "
              f"{np.mean([r['real_tp_first'] for r in have])*100:.1f}%)")

    print("\n  Reminder: this is a screen, not a verdict. At this sample size only a")
    print("  large effect is detectable. A null result here does not prove there is")
    print("  no signal - it bounds how big one could be.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
