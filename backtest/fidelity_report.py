"""
Conditional exit-fidelity report: ExactFill vs EmpiricalFill.

    python -m backtest.fidelity_report --snapshot <file>.db [--draws 200]

Reports the sign test, the moving-block bootstrap CI on median per-trade P&L
delta, and the aggregate simulated-vs-live P&L across draws.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from backtest.exit_fidelity import evaluate, summarise
from backtest.fill_data import build_dataset, fittable
from backtest.fill_models import ExactFill, WorstCaseFill, build_empirical, make_rng
from backtest.overshoot_fit import correlations, fit
from backtest.snapshot import open_readonly
from backtest.stats import moving_block_bootstrap_median, sign_test

logger = logging.getLogger("backtest.fidelity")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backtest.fidelity_report")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=200,
                        help="stochastic draws for EmpiricalFill")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--clamp-to-bar", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    conn = open_readonly(args.snapshot)
    try:
        observations, report = build_dataset(conn)
        good = fittable(observations)

        print("=" * 74)
        print("EMPIRICAL FILL DATASET")
        print("=" * 74)
        for key, val in report.items():
            print(f"  {key:24s} {val}")
        print(f"  {'fittable':24s} {len(good)}")
        print()

        sl_fit, tp_fit = fit(good, "SL"), fit(good, "TP")
        print("FITTED OVERSHOOT (pooled across symbols)")
        print("-" * 74)
        for f in (sl_fit, tp_fit):
            print(f"  {f.summary()}")
            if f.tail:
                print(f"    tail events: " + ", ".join(f"{x * 100:.1f}pp" for x in f.tail))
        print()

        print("CANDIDATE CONDITIONING VARIABLES (measured, not used)")
        print("-" * 74)
        for etype in ("SL", "TP"):
            corr = correlations(good, etype)
            routes = ", ".join(
                f"{k}: n={v['n']} med={v['median'] * 100:.3f}pp"
                for k, v in corr.get("by_route", {}).items()
            )
            print(f"  {etype}: ATR r={corr.get('atr_pearson_r'):+.3f} | "
                  f"session crossed med={corr.get('session_crossed_median', 0) * 100:.3f}pp "
                  f"(n={corr.get('session_crossed_n')}) vs same "
                  f"{corr.get('session_same_median', 0) * 100:.3f}pp "
                  f"(n={corr.get('session_same_n')})")
            print(f"      route -> {routes}")
        print()

        # --- ExactFill (deterministic null model) --------------------------
        exact = evaluate(conn, good, ExactFill(), rng=None)
        exact_deltas = [r.pnl_delta for r in exact]
        exact_sum = summarise(exact)

        worst = evaluate(conn, good, WorstCaseFill(), rng=None)
        worst_sum = summarise(worst)

        # --- EmpiricalFill across draws ------------------------------------
        empirical = build_empirical(good, clamp_to_bar=args.clamp_to_bar)
        print(f"EmpiricalFill seed={args.seed} draws={args.draws} "
              f"clamp_to_bar={args.clamp_to_bar}")
        print()

        draw_totals: list[float] = []
        pooled_deltas: list[float] = []
        per_draw_medians: list[float] = []
        per_draw_sign: list[dict] = []
        first_draw_deltas: list[float] = []
        empirical.reset_counters()

        for draw in range(args.draws):
            rng = make_rng(args.seed, draw)
            res = evaluate(conn, good, empirical, rng=rng)
            deltas = [r.pnl_delta for r in res]
            draw_totals.append(sum(deltas))
            per_draw_medians.append(float(np.median(deltas)))
            pooled_deltas.extend(deltas)
            # Sign test is computed PER DRAW. Pooling 200 draws of the same 165
            # trades would inflate n by 200x through pseudo-replication and make
            # any p-value meaningless.
            per_draw_sign.append(sign_test(deltas))
            if draw == 0:
                first_draw_deltas = deltas

        oob_rate = (empirical.out_of_bar_draws / empirical.total_draws * 100
                    if empirical.total_draws else 0.0)

        # --- Report ---------------------------------------------------------
        print("=" * 74)
        print("CONDITIONAL EXIT FIDELITY (entries forced to recorded live entries)")
        print("=" * 74)
        print(f"  trades evaluated        : {exact_sum['n']}")
        print(f"  exit-reason match       : {exact_sum['reason_match']}/{exact_sum['n']} "
              f"({exact_sum['reason_match_pct']:.1f}%)")
        print(f"  median exit_ts delta    : {exact_sum['median_exit_ts_delta_h']:+.2f} h")
        print()
        print(f"{'model':<14}{'favours sim':>13}{'favours live':>14}"
              f"{'median dP&L':>13}{'total dP&L':>13}")
        print("-" * 74)
        print(f"{'ExactFill':<14}{exact_sum['favours_sim']:>13}{exact_sum['favours_live']:>14}"
              f"{exact_sum['median_pnl_delta']:>13.4f}{exact_sum['total_pnl_delta']:>13.2f}")
        print(f"{'EmpiricalFill':<14}"
              f"{sum(1 for d in pooled_deltas if d > 0):>13}"
              f"{sum(1 for d in pooled_deltas if d < 0):>14}"
              f"{float(np.median(pooled_deltas)):>13.4f}"
              f"{float(np.mean(draw_totals)):>13.2f}")
        print(f"{'WorstCaseFill':<14}{worst_sum['favours_sim']:>13}{worst_sum['favours_live']:>14}"
              f"{worst_sum['median_pnl_delta']:>13.4f}{worst_sum['total_pnl_delta']:>13.2f}")
        print()
        print(f"  EmpiricalFill out-of-bar draws: {oob_rate:.1f}% "
              f"({empirical.out_of_bar_draws}/{empirical.total_draws})")
        print()

        # --- Task 5.1 sign test --------------------------------------------
        print("=" * 74)
        print("TASK 5.1 - SIGN TEST on per-trade (sim P&L - live P&L)")
        print("=" * 74)
        st_exact = sign_test(exact_deltas)
        print(f"  {'ExactFill':<14} n={st_exact['n']:<6} +{st_exact['positive']:<6} "
              f"-{st_exact['negative']:<6} {st_exact['pct_favouring_sim']:.1f}% favour sim  "
              f"p={st_exact['p_value']:.3g}  [{st_exact['skew']}]")

        shares = np.array([s["pct_favouring_sim"] for s in per_draw_sign])
        pvals = np.array([s["p_value"] for s in per_draw_sign])
        sig = float(np.mean(pvals < 0.05)) * 100
        print(f"  {'EmpiricalFill':<14} per-draw over {args.draws} draws (n={st_exact['n']} each):")
        print(f"                 favour sim: median {np.median(shares):.1f}%  "
              f"p5-p95 [{np.percentile(shares, 5):.1f}%, {np.percentile(shares, 95):.1f}%]")
        print(f"                 p<0.05 in {sig:.1f}% of draws  "
              f"(median p={np.median(pvals):.3g})")
        print()
        print("  Note: the EmpiricalFill sign test is computed per draw, not pooled.")
        print("  Pooling would multiply n by the draw count through pseudo-replication.")
        print()

        # --- Task 5.2 bootstrap CI -----------------------------------------
        print("=" * 74)
        print("TASK 5.2 - MOVING-BLOCK BOOTSTRAP CI on median P&L delta")
        print("=" * 74)
        # Both bootstraps run on per-trade deltas so the two rows are comparable.
        # EmpiricalFill uses draw 0; the across-draw spread is reported below it.
        bs_exact = moving_block_bootstrap_median(exact_deltas, block=args.block,
                                                 seed=args.seed)
        bs_emp = moving_block_bootstrap_median(first_draw_deltas, block=args.block,
                                               seed=args.seed)
        for label, bs in (("ExactFill", bs_exact), ("EmpiricalFill", bs_emp)):
            print(f"  {label:<14} median={bs['point_median']:+.4f}  "
                  f"95% CI [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]  "
                  f"block={bs['block']}  straddles_zero={bs['straddles_zero']}")
        print(f"  {'':14} (EmpiricalFill row is draw 0, per-trade deltas - same "
              f"basis as ExactFill)")
        pdm = np.array(per_draw_medians)
        print(f"  across {args.draws} draws, per-draw median dP&L: "
              f"median {np.median(pdm):+.4f}  "
              f"p5-p95 [{np.percentile(pdm, 5):+.4f}, {np.percentile(pdm, 95):+.4f}]")
        print()

        # --- Task 5.3 aggregate distribution -------------------------------
        print("=" * 74)
        print("TASK 5.3 - AGGREGATE SIM vs LIVE P&L ACROSS DRAWS")
        print("=" * 74)
        arr = np.array(draw_totals)
        print(f"  live total (reference)  : 0.00  (delta measured against live)")
        print(f"  ExactFill delta         : {exact_sum['total_pnl_delta']:+.2f}")
        print(f"  EmpiricalFill delta     : mean {arr.mean():+.2f}  "
              f"median {np.median(arr):+.2f}")
        print(f"    p5 / p95              : {np.percentile(arr, 5):+.2f} / "
              f"{np.percentile(arr, 95):+.2f}")
        print(f"    min / max             : {arr.min():+.2f} / {arr.max():+.2f}")
        inside = np.percentile(arr, 5) <= 0 <= np.percentile(arr, 95)
        print(f"  live sits inside p5-p95 : {inside}")
        print()
        print("  Reminder: the fill distribution was fitted in Task 2 and then used.")
        print("  It has not been adjusted to improve any number on this page.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
