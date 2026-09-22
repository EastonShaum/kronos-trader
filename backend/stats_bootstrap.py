"""
Project-facing bootstrap wrappers.

Import from here, never from `vendor/` directly. The vendored arch code is an
implementation detail and its layout may change on re-vendor.

BLOCK SIZE AND OVERLAPPING SAMPLES
----------------------------------
Automatic block-length selection assumes the series is a plain time series
whose dependence the data itself reveals. That holds for TRADE-LEVEL series
(one P&L per closed trade).

It does NOT hold for PREDICTION-LEVEL series with overlapping 24h horizons.
Consecutive hourly predictions share 23 of 24 hours of outcome, so successive
observations are mechanically correlated by construction, not by market
dynamics. Auto-selection reads that as weak dependence and returns a block far
too small, producing intervals that are much too narrow.

    Callers working on prediction-level series with a 24h horizon MUST pass
    block_size >= 24 explicitly. Do not rely on auto-selection there.
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vendor.arch_bootstrap.bootstrap import (  # noqa: E402
    SPA,
    CircularBlockBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
    optimal_block_length,
)
from vendor.arch_bootstrap.utility.exceptions import PerformanceWarning  # noqa: E402

_KINDS = {
    "moving": MovingBlockBootstrap,
    "circular": CircularBlockBootstrap,
    "stationary": StationaryBootstrap,
}

# optimal_block_length returns a frame with both columns; pick per bootstrap kind.
_BLOCK_COLUMN = {
    "moving": "circular",
    "circular": "circular",
    "stationary": "stationary",
}


# optimal_block_length scans autocorrelations out to ceil(sqrt(n)) + max(5, log10(n))
# lags, which indexes past the end of the array for very short series. Below this
# many observations there is no usable dependence signal anyway, so fall back to 1.
MIN_N_FOR_AUTO_BLOCK = 16


def resolve_block_size(values, kind: str = "moving") -> int:
    """
    Auto block length via optimal_block_length, ceiled and clamped to >= 1.

    Returns 1 when the series is too short to estimate a block length, which
    makes the bootstrap IID. Callers can see this in the returned `block_size`.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < MIN_N_FOR_AUTO_BLOCK:
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        # A constant or near-constant series divides by zero variance inside
        # optimal_block_length; the non-finite result is handled below.
        with np.errstate(invalid="ignore", divide="ignore"):
            try:
                table = optimal_block_length(arr)
            except (ValueError, IndexError):
                return 1

    column = _BLOCK_COLUMN.get(kind, "circular")
    raw = float(np.asarray(table[column]).ravel()[0])
    if not math.isfinite(raw):
        return 1
    return max(1, int(math.ceil(raw)))


def block_bootstrap_ci(values,
                       block_size: int | None = None,
                       reps: int = 5000,
                       alpha: float = 0.05,
                       method: str = "bca",
                       seed: int | None = None,
                       kind: str = "moving") -> dict:
    """
    Block-bootstrap confidence interval on the mean.

    Falls back from `bca` to `percentile` when bca fails, which it can at small
    n (the jackknife acceleration term degenerates). The method actually used
    is reported in `method_used` rather than hidden.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)

    if n == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"),
                "n": 0, "block_size": 0, "reps": reps,
                "method_used": "none", "kind": kind}

    mean = float(np.mean(arr))
    if n == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "n": 1,
                "block_size": 1, "reps": reps, "method_used": "degenerate",
                "kind": kind}

    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {kind!r}")

    if block_size is None:
        block_size = resolve_block_size(arr, kind)
    block_size = max(1, min(int(block_size), n))

    size = 1.0 - alpha
    method_used = method

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        bs = _KINDS[kind](block_size, arr, seed=seed)
        try:
            ci = bs.conf_int(np.mean, reps=reps, method=method, size=size)
        except Exception:
            # bca can fail on degenerate/near-constant samples.
            bs = _KINDS[kind](block_size, arr, seed=seed)
            ci = bs.conf_int(np.mean, reps=reps, method="percentile", size=size)
            method_used = "percentile"

    flat = np.asarray(ci, dtype=float).ravel()
    return {
        "mean": mean,
        "lower": float(flat[0]),
        "upper": float(flat[1]),
        "n": n,
        "block_size": int(block_size),
        "reps": int(reps),
        "method_used": method_used,
        "kind": kind,
    }


def spa_test(benchmark_losses,
             model_losses,
             block_size: int,
             reps: int = 5000,
             seed: int | None = None) -> dict:
    """
    Hansen's Superior Predictive Ability test.

    Answers "does ANY of the candidate models beat the benchmark, after
    accounting for the fact that we looked at many of them". Inputs are LOSSES
    (lower is better), not returns - negate P&L before passing it in.

    `block_size` is required here on purpose: SPA is normally run on
    prediction-level loss series, where auto-selection is not safe. See the
    module docstring.

    Not yet wired into any caller. Exposed for later use.
    """
    bench = np.asarray(benchmark_losses, dtype=float).ravel()
    models = np.asarray(model_losses, dtype=float)
    if models.ndim == 1:
        models = models.reshape(-1, 1)

    if bench.shape[0] != models.shape[0]:
        raise ValueError(
            f"benchmark has {bench.shape[0]} obs, models have {models.shape[0]}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        spa = SPA(bench, models, block_size=int(block_size), reps=reps, seed=seed)
        spa.compute()

    return {
        "pvalue_lower": float(spa.pvalues["lower"]),
        "pvalue_consistent": float(spa.pvalues["consistent"]),
        "pvalue_upper": float(spa.pvalues["upper"]),
        "n": int(bench.shape[0]),
        "n_models": int(models.shape[1]),
        "block_size": int(block_size),
        "reps": int(reps),
    }
