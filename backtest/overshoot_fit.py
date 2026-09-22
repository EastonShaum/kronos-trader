"""
Overshoot distribution fit.

133 SL and 42 TP observations across 8+ symbols is too thin for per-symbol
fits, so SL and TP are each POOLED across symbols. Eight separate fits off
~16 observations each would look precise and mean nothing.

SL and TP are fitted separately - they are different mechanisms and are not
symmetric.

The tail is modelled as a mixture rather than a single distribution. A -37%
stop is gap risk: the price was never available at the threshold. A -3.2% stop
is sampling latency: the monitor checked 3 minutes late. Fitting one
distribution across both either erases gap risk or makes every stop look
catastrophic. So: a body distribution, plus a separately reported empirical
rate and magnitude for tail events.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

from backtest.fill_data import FillObservation

logger = logging.getLogger(__name__)

# Body/tail split. Overshoot beyond this multiple of the threshold is treated
# as a gap event rather than sampling latency.
TAIL_MULTIPLE = 1.0


@dataclass
class OvershootFit:
    exit_type: str
    n: int
    body: list[float] = field(default_factory=list)
    tail: list[float] = field(default_factory=list)
    tail_rate: float = 0.0
    threshold_pct_used: float = 0.0

    # Empirical summary of the body - the ground truth.
    median: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    stdev: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.exit_type}: n={self.n} body={len(self.body)} tail={len(self.tail)} "
            f"tail_rate={self.tail_rate:.1%} | median={self.median * 100:.3f}pp "
            f"IQR=[{self.q1 * 100:.3f}, {self.q3 * 100:.3f}]pp "
            f"p90={self.p90 * 100:.3f}pp max_body={self.maximum * 100:.3f}pp"
        )


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def fit(observations: list[FillObservation], exit_type: str) -> OvershootFit:
    """Pool across symbols, split body from tail, summarise empirically."""
    subset = [o for o in observations if o.exit_type == exit_type]
    if not subset:
        return OvershootFit(exit_type=exit_type, n=0)

    # Tail cutoff is relative to the threshold actually in force.
    body: list[float] = []
    tail: list[float] = []
    for obs in subset:
        if obs.overshoot > TAIL_MULTIPLE * obs.threshold_pct:
            tail.append(obs.overshoot)
        else:
            body.append(obs.overshoot)

    result = OvershootFit(
        exit_type=exit_type,
        n=len(subset),
        body=sorted(body),
        tail=sorted(tail),
        tail_rate=len(tail) / len(subset),
        threshold_pct_used=statistics.median([o.threshold_pct for o in subset]),
    )

    if body:
        result.median = _pct(body, 0.50)
        result.q1 = _pct(body, 0.25)
        result.q3 = _pct(body, 0.75)
        result.p90 = _pct(body, 0.90)
        result.p99 = _pct(body, 0.99)
        result.mean = statistics.fmean(body)
        result.stdev = statistics.stdev(body) if len(body) > 1 else 0.0
        result.minimum = min(body)
        result.maximum = max(body)

    return result


def correlations(observations: list[FillObservation], exit_type: str) -> dict:
    """
    Candidate conditioning variables, checked but deliberately not used.

    At ~130 SL / ~40 TP observations, conditioning the fill model on ATR or
    session boundary would be fitting noise. Reported so a later patch with
    more data can revisit.
    """
    subset = [o for o in observations if o.exit_type == exit_type]
    out: dict = {"exit_type": exit_type, "n": len(subset)}

    with_atr = [(o.atr_pct, o.overshoot) for o in subset if o.atr_pct is not None]
    if len(with_atr) > 2:
        xs = [a for a, _ in with_atr]
        ys = [b for _, b in with_atr]
        try:
            out["atr_pearson_r"] = statistics.correlation(xs, ys)
        except (statistics.StatisticsError, ValueError):
            out["atr_pearson_r"] = None
        out["atr_n"] = len(with_atr)

    crossed = [o.overshoot for o in subset if o.crossed_session]
    same = [o.overshoot for o in subset if not o.crossed_session]
    out["session_crossed_n"] = len(crossed)
    out["session_same_n"] = len(same)
    if crossed:
        out["session_crossed_median"] = _pct(crossed, 0.50)
    if same:
        out["session_same_median"] = _pct(same, 0.50)

    by_route: dict[str, list[float]] = {}
    for obs in subset:
        by_route.setdefault(obs.exit_route, []).append(obs.overshoot)
    out["by_route"] = {
        route: {"n": len(vals), "median": _pct(vals, 0.50)}
        for route, vals in by_route.items()
    }
    return out
