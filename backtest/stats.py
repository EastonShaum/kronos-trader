"""
Directionality statistics.

The pass criterion is not "the simulator is accurate", it is "the simulator's
errors stop favouring the simulator". These measure that.

Bootstrap note: the patch brief refers to a moving-block approach "already
established" for overlapping-sample work. There isn't one - the only bootstrap
in the codebase is benchmarks._bootstrap_ci, which is an IID resample. A
moving-block bootstrap is implemented here because per-trade deltas are
serially dependent (consecutive trades share symbols, regimes and volatility),
and an IID resample would understate the interval.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import stats as sps

logger = logging.getLogger(__name__)

DEFAULT_BLOCK = 5
DEFAULT_RESAMPLES = 10_000


def sign_test(deltas: list[float]) -> dict:
    """
    Exact two-sided binomial test on the sign of per-trade (sim - live).

    H0: a divergence is equally likely to favour either side. Rejection with
    a positive skew is the bug this patch exists to remove.
    """
    arr = np.asarray(deltas, dtype=float)
    pos = int(np.sum(arr > 0))
    neg = int(np.sum(arr < 0))
    ties = int(np.sum(arr == 0))
    n = pos + neg

    if n == 0:
        return {"n": 0, "positive": pos, "negative": neg, "ties": ties,
                "p_value": 1.0, "skew": "no non-tied observations"}

    result = sps.binomtest(pos, n, p=0.5, alternative="two-sided")
    share = pos / n
    return {
        "n": n,
        "positive": pos,
        "negative": neg,
        "ties": ties,
        "pct_favouring_sim": share * 100,
        "p_value": float(result.pvalue),
        "skew": ("favours simulator" if share > 0.5 else
                 "favours live" if share < 0.5 else "balanced"),
    }


def moving_block_bootstrap_median(values: list[float],
                                  block: int = DEFAULT_BLOCK,
                                  n_resamples: int = DEFAULT_RESAMPLES,
                                  confidence: float = 0.95,
                                  seed: int = 0) -> dict:
    """
    Moving-block bootstrap CI on the median, preserving local dependence.

    Blocks of consecutive observations are resampled with replacement, so
    within-block serial correlation survives into each resample.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0}
    block = max(1, min(block, n))

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    max_start = n - block

    medians = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block] for s in starts])[:n]
        medians[i] = np.median(sample)

    alpha = (1 - confidence) / 2
    return {
        "n": n,
        "block": block,
        "n_resamples": n_resamples,
        "seed": seed,
        "point_median": float(np.median(arr)),
        "ci_low": float(np.quantile(medians, alpha)),
        "ci_high": float(np.quantile(medians, 1 - alpha)),
        "confidence": confidence,
        "straddles_zero": bool(np.quantile(medians, alpha) <= 0 <= np.quantile(medians, 1 - alpha)),
    }
