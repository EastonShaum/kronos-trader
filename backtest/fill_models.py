"""
Pluggable fill models.

_walk_candles() stays the trigger detector: it answers "did the threshold
trigger on this bar". These answer the separate question "at what price".
Live code is not aware this module exists.

Clamping note
-------------
EmpiricalFill does NOT clamp the sampled fill into the bar's OHLC range by
default. Clamping would be physically tidier but it truncates the fitted
distribution from the pessimistic side only, quietly reintroducing the very
optimism this patch exists to remove. The unclamped draw is faithful to the
fit; `clamp_to_bar=True` is available so the alternative can be measured
rather than assumed, and the out-of-bar rate is tracked either way.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from backtest.overshoot_fit import OvershootFit

logger = logging.getLogger(__name__)

DEFAULT_SEED = 0


@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float


class FillModel:
    """price(trigger_type, threshold_price, bar, rng) -> fill_price"""

    name = "base"
    stochastic = False

    def price(self, trigger_type: str, threshold_price: float,
              bar: Bar | None, rng: random.Random | None) -> float:
        raise NotImplementedError


class ExactFill(FillModel):
    """
    Fills at the threshold. Reproduces Patch 1 behaviour exactly.

    Retained as the null model: this is the hypothesis that the simulator's
    fills are achievable, which the data says happens about 4% of the time.
    """

    name = "exact"
    stochastic = False

    def price(self, trigger_type, threshold_price, bar=None, rng=None) -> float:
        return threshold_price


class WorstCaseFill(FillModel):
    """Fills at the bar extreme. Pessimistic bound, not a prediction."""

    name = "worst_case"
    stochastic = False

    def price(self, trigger_type, threshold_price, bar=None, rng=None) -> float:
        if bar is None:
            return threshold_price
        return float(bar.low) if trigger_type == "SL" else float(bar.high)


@dataclass
class EmpiricalFill(FillModel):
    """
    Samples overshoot from the fitted body/tail mixture.

    Draw procedure per fill:
      with probability tail_rate  -> resample an observed tail overshoot
      otherwise                   -> resample an observed body overshoot
    Both arms resample the empirical values directly rather than a parametric
    approximation, so the fitted shape is reproduced without imposing one.
    """

    sl_fit: OvershootFit
    tp_fit: OvershootFit
    clamp_to_bar: bool = False
    name: str = "empirical"
    stochastic: bool = True
    out_of_bar_draws: int = field(default=0, init=False)
    total_draws: int = field(default=0, init=False)

    def _fit_for(self, trigger_type: str) -> OvershootFit:
        return self.sl_fit if trigger_type == "SL" else self.tp_fit

    def sample_overshoot(self, trigger_type: str, rng: random.Random) -> float:
        fit = self._fit_for(trigger_type)
        if fit.tail and rng.random() < fit.tail_rate:
            return rng.choice(fit.tail)
        if fit.body:
            return rng.choice(fit.body)
        return 0.0

    def price(self, trigger_type, threshold_price, bar=None, rng=None) -> float:
        if rng is None:
            raise ValueError("EmpiricalFill requires an explicit rng")

        overshoot = self.sample_overshoot(trigger_type, rng)
        # Overshoot is positive past the threshold in the direction of travel:
        # down for a stop, up for a take-profit.
        if trigger_type == "SL":
            fill = threshold_price * (1 - overshoot)
        else:
            fill = threshold_price * (1 + overshoot)

        self.total_draws += 1
        if bar is not None and not (float(bar.low) <= fill <= float(bar.high)):
            self.out_of_bar_draws += 1
            if self.clamp_to_bar:
                fill = min(max(fill, float(bar.low)), float(bar.high))

        return fill

    def reset_counters(self) -> None:
        self.out_of_bar_draws = 0
        self.total_draws = 0


def build_empirical(observations, clamp_to_bar: bool = False) -> EmpiricalFill:
    """Fit from fittable observations and return a ready EmpiricalFill."""
    from backtest.overshoot_fit import fit as fit_overshoot

    model = EmpiricalFill(
        sl_fit=fit_overshoot(observations, "SL"),
        tp_fit=fit_overshoot(observations, "TP"),
        clamp_to_bar=clamp_to_bar,
    )
    logger.info("EmpiricalFill built | %s", model.sl_fit.summary())
    logger.info("EmpiricalFill built | %s", model.tp_fit.summary())
    return model


def make_rng(seed: int = DEFAULT_SEED, draw: int = 0) -> random.Random:
    """
    Independent, reproducible stream per draw.

    Seeded from arithmetic on ints only - no reliance on hash(), which is not
    a stable cross-process seed source. Same (seed, draw) always yields the
    same sequence.
    """
    return random.Random(seed * 1_000_003 + draw)
