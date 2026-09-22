"""
Single source of truth for simulating a price-based exit from OHLC candles.

Ported from vectorbt's `get_stop_price_nb` (and its call site in
`simulate_from_signal_func_nb`), vectorbt/portfolio/nb.py.

    Source:    https://github.com/polakowo/vectorbt
    Commit:    a3c0f40979648585b4581fc4694fd6e92b3fffca
    License:   Apache License 2.0 with Commons Clause
    Copyright: (c) 2021 Oleg Polakow

    "The Software is provided to you by the Licensor under the License, as
    defined below, subject to the following condition. Without limiting other
    conditions in the License, the grant of rights under the License will not
    include, and the License does not grant to you, the right to Sell the
    Software."

    This is a semantic re-implementation in pure Python/numpy for diagnostics
    use. No vectorbt code is vendored and the library is not a dependency.

SEMANTICS
---------
Level placement depends on direction:

    long   SL = entry * (1 - sl_pct)  [below]   TP = entry * (1 + tp_pct)  [above]
    short  SL = entry * (1 + sl_pct)  [above]   TP = entry * (1 - tp_pct)  [below]

vectorbt picks the same branches via the sign of `position_now` combined with
`hit_below` (nb.py lines 1875-1888); this is that logic with the direction made
explicit rather than inferred from a position size.

Per level, by which way it sits relative to entry:

    downward level L:  if open <= L            -> fill at open
                       elif low <= L <= high   -> fill at L
    upward   level L:  if L <= open            -> fill at open
                       elif low <= L <= high   -> fill at L

Stop-loss is evaluated first, so WHEN BOTH LEVELS FALL INSIDE THE SAME BAR,
THE STOP WINS. That mirrors vectorbt, where the take-profit branch runs only
when the stop-loss branch returned NaN. It is the conservative choice: from
OHLC alone there is no way to know which level the price reached first.

NaN handling, also from vectorbt, applied BEFORE any level check:
    open is NaN -> use close
    low  is NaN -> use min(open, close)
    high is NaN -> use max(open, close)

The `gapped` flag uses a STRICT comparison (open < L downward, open > L upward)
while the fill decision keeps <= / >=. At exact equality the fill price is the
same either way, so calling it a gap would inflate gap counts in the stop-fill
report without changing any price.

ASSUMPTION - THE ENTRY BAR IS NOT CHECKED
-----------------------------------------
Stops are evaluated starting from the bar AFTER entry. The entry bar's
intra-bar path is unknowable from OHLC: knowing that bar's high and low says
nothing about whether price reached the stop before or after the entry fill.
Checking it would manufacture exits that may never have been reachable.
Callers must pass candles beginning strictly after the entry timestamp.

This module handles PRICE-BASED exits only. Signal-based exits (the bearish
confidence flip) stay in the callers - they depend on prediction data this
function has no access to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

__all__ = ["ExitResult", "simulate_exit"]


@dataclass(frozen=True)
class ExitResult:
    ts: Any
    exit_price: float
    reason: str              # stop_loss | take_profit | max_bars
    bars_held: int
    gapped: bool


def _is_nan(value) -> bool:
    try:
        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _clean_bar(open_, high, low, close) -> tuple[float, float, float, float]:
    """vectorbt's NaN fallbacks, nb.py lines 2090-2095."""
    close = float(close)
    open_ = close if _is_nan(open_) else float(open_)
    low = min(open_, close) if _is_nan(low) else float(low)
    high = max(open_, close) if _is_nan(high) else float(high)
    return open_, high, low, close


def _check_level(level: float, downward: bool,
                 open_: float, high: float, low: float) -> tuple[float, bool] | None:
    """
    Resolve one level against one cleaned bar.

    Returns (fill_price, gapped) or None if the level was not reached. The
    gapped flag is strict so that open == level is not counted as a gap.
    """
    if downward:
        if open_ <= level:
            return open_, open_ < level
        if low <= level <= high:
            return level, False
    else:
        if level <= open_:
            return open_, open_ > level
        if low <= level <= high:
            return level, False
    return None


def simulate_exit(entry_price: float,
                  candles: Iterable[tuple],
                  sl_pct: float | None,
                  tp_pct: float | None,
                  max_bars: int | None = None,
                  is_short: bool = False) -> ExitResult | None:
    """
    Walk candles from the bar after entry and return the first price-based exit.

    Parameters
    ----------
    entry_price : fill price of the entry
    candles     : iterable of (ts, open, high, low, close), starting the bar
                  AFTER entry, in ascending time order
    sl_pct      : stop-loss as a non-negative fraction (0.03 = 3% adverse move);
                  None disables
    tp_pct      : take-profit as a non-negative fraction (0.06 = 6% favourable
                  move); None disables
    max_bars    : force an exit at this bar's close if no level was hit;
                  None means walk to the end and return None if nothing hit
    is_short    : place the levels on the opposite sides of entry

    Returns
    -------
    ExitResult, or None if no level was hit and max_bars was not reached.
    `exit_price` is a PRICE, not a P&L - callers hold the sign convention and
    must compute short P&L as (entry - exit), not (exit - entry).
    """
    if entry_price is None or entry_price <= 0:
        return None

    # vectorbt raises on a negative stop; a fraction >= 1 would put the level at
    # or below zero, which is not a price.
    if sl_pct is not None:
        if sl_pct < 0:
            raise ValueError(f"sl_pct must be >= 0, got {sl_pct}")
        if not is_short and sl_pct >= 1:
            raise ValueError(f"long sl_pct must be < 1, got {sl_pct}")
    if tp_pct is not None:
        if tp_pct < 0:
            raise ValueError(f"tp_pct must be >= 0, got {tp_pct}")
        if is_short and tp_pct >= 1:
            raise ValueError(f"short tp_pct must be < 1, got {tp_pct}")

    if is_short:
        sl_level = entry_price * (1 + sl_pct) if sl_pct is not None else None
        tp_level = entry_price * (1 - tp_pct) if tp_pct is not None else None
        sl_downward, tp_downward = False, True
    else:
        sl_level = entry_price * (1 - sl_pct) if sl_pct is not None else None
        tp_level = entry_price * (1 + tp_pct) if tp_pct is not None else None
        sl_downward, tp_downward = True, False

    bars_held = 0
    last = None

    for row in candles:
        ts, raw_open, raw_high, raw_low, raw_close = row
        if _is_nan(raw_close):
            continue

        # NaN fallbacks resolve before any level is checked, both directions.
        open_, high, low, close = _clean_bar(raw_open, raw_high, raw_low, raw_close)
        bars_held += 1
        last = (ts, close)

        # 1. Stop-loss first - wins ties within a bar.
        if sl_level is not None:
            hit = _check_level(sl_level, sl_downward, open_, high, low)
            if hit is not None:
                return ExitResult(ts, hit[0], "stop_loss", bars_held, hit[1])

        # 2. Take-profit only if the stop did not fire.
        if tp_level is not None:
            hit = _check_level(tp_level, tp_downward, open_, high, low)
            if hit is not None:
                return ExitResult(ts, hit[0], "take_profit", bars_held, hit[1])

        if max_bars is not None and bars_held >= max_bars:
            return ExitResult(ts, close, "max_bars", bars_held, False)

    if max_bars is not None and last is not None:
        # Ran out of candles before max_bars; exit at the last close available.
        return ExitResult(last[0], last[1], "max_bars", bars_held, False)

    return None
