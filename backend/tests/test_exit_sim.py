"""One test per rule ported from vectorbt's get_stop_price_nb."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exit_sim import simulate_exit  # noqa: E402

ENTRY = 100.0
SL_PCT = 0.03      # level 97.0
TP_PCT = 0.06      # level 106.0


def test_gap_down_through_stop_fills_at_open():
    """open <= SL means the level was passed before the bar traded."""
    candles = [("t1", 94.0, 95.0, 93.0, 94.5)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.reason == "stop_loss"
    assert result.exit_price == 94.0
    assert result.gapped is True
    # The whole point: a gap fills worse than the -3% the stop nominally promises.
    assert (result.exit_price - ENTRY) / ENTRY < -SL_PCT


def test_normal_stop_touch_fills_at_the_level():
    candles = [("t1", 99.0, 99.5, 96.5, 98.0)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.reason == "stop_loss"
    assert result.exit_price == pytest.approx(97.0)
    assert result.gapped is False
    assert result.bars_held == 1


def test_gap_up_through_take_profit_fills_at_open():
    candles = [("t1", 108.0, 109.0, 107.5, 108.5)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.reason == "take_profit"
    assert result.exit_price == 108.0
    assert result.gapped is True


def test_normal_take_profit_touch_fills_at_the_level():
    candles = [("t1", 101.0, 106.5, 100.5, 105.0)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.reason == "take_profit"
    assert result.exit_price == pytest.approx(106.0)
    assert result.gapped is False


def test_stop_wins_when_both_levels_inside_one_bar():
    """
    vectorbt evaluates take-profit only when stop-loss returned NaN, so the
    stop wins the tie. OHLC cannot tell us which level came first.
    """
    candles = [("t1", 100.0, 107.0, 96.0, 101.0)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.reason == "stop_loss"
    assert result.exit_price == pytest.approx(97.0)


def test_neither_level_hit_returns_none():
    candles = [
        ("t1", 100.0, 101.0, 99.0, 100.5),
        ("t2", 100.5, 102.0, 99.5, 101.0),
    ]
    assert simulate_exit(ENTRY, candles, SL_PCT, TP_PCT) is None


def test_max_bars_forces_exit_at_close():
    candles = [
        ("t1", 100.0, 101.0, 99.0, 100.5),
        ("t2", 100.5, 102.0, 99.5, 101.0),
        ("t3", 101.0, 102.5, 100.0, 102.2),
    ]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT, max_bars=2)

    assert result.reason == "max_bars"
    assert result.ts == "t2"
    assert result.exit_price == pytest.approx(101.0)
    assert result.bars_held == 2


def test_nan_open_falls_back_to_close():
    """open is NaN -> use close. Close 96.0 is below the 97.0 stop, so gapped."""
    candles = [("t1", float("nan"), 98.0, 95.0, 96.0)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.reason == "stop_loss"
    assert result.exit_price == 96.0
    assert result.gapped is True


def test_nan_high_and_low_fall_back_to_open_close_range():
    """low -> min(open, close), high -> max(open, close). Range [98, 99] misses both."""
    candles = [("t1", 99.0, float("nan"), float("nan"), 98.0)]
    assert simulate_exit(ENTRY, candles, SL_PCT, TP_PCT) is None


def test_nan_low_only_still_allows_stop_via_open_close_range():
    candles = [("t1", 99.0, 99.5, float("nan"), 96.5)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.reason == "stop_loss"
    assert result.exit_price == pytest.approx(97.0)


def test_bar_with_nan_close_is_skipped():
    candles = [
        ("t1", float("nan"), float("nan"), float("nan"), float("nan")),
        ("t2", 99.0, 99.5, 96.5, 98.0),
    ]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.ts == "t2"
    assert result.bars_held == 1, "the unusable bar must not count as held"


def test_disabled_levels_are_ignored():
    candles = [("t1", 90.0, 91.0, 89.0, 90.5)]
    assert simulate_exit(ENTRY, candles, None, TP_PCT) is None

    result = simulate_exit(ENTRY, candles, SL_PCT, None)
    assert result.reason == "stop_loss"


def test_multi_bar_walk_reports_bars_held():
    candles = [
        ("t1", 100.0, 101.0, 99.0, 100.5),
        ("t2", 100.5, 102.0, 99.5, 101.0),
        ("t3", 101.0, 101.5, 96.0, 97.5),
    ]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.ts == "t3"
    assert result.bars_held == 3
    assert result.reason == "stop_loss"


def test_invalid_entry_price_returns_none():
    assert simulate_exit(0.0, [("t1", 1.0, 1.0, 1.0, 1.0)], SL_PCT, TP_PCT) is None


# --- gapped flag at exact equality ----------------------------------------

def test_open_exactly_at_stop_is_not_flagged_as_gap():
    """
    open == SL fills at the same price either way, so flagging it a gap would
    inflate gap counts in the stop-fill report without changing any price.
    """
    candles = [("t1", 97.0, 98.0, 96.0, 97.5)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.reason == "stop_loss"
    assert result.exit_price == pytest.approx(97.0)
    assert result.gapped is False


def test_open_exactly_at_take_profit_is_not_flagged_as_gap():
    candles = [("t1", 106.0, 107.0, 105.0, 106.5)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.reason == "take_profit"
    assert result.exit_price == pytest.approx(106.0)
    assert result.gapped is False


def test_open_strictly_through_stop_is_flagged_as_gap():
    candles = [("t1", 96.99, 97.5, 96.0, 97.0)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT)

    assert result.gapped is True
    assert result.exit_price == pytest.approx(96.99)


# --- input validation ------------------------------------------------------

@pytest.mark.parametrize("sl,tp", [(-0.01, TP_PCT), (SL_PCT, -0.01)])
def test_negative_fractions_are_rejected(sl, tp):
    with pytest.raises(ValueError, match=">= 0"):
        simulate_exit(ENTRY, [("t1", 100.0, 101.0, 99.0, 100.0)], sl, tp)


def test_short_take_profit_of_one_or_more_is_rejected():
    """A short TP of 100% would put the level at zero, which is not a price."""
    with pytest.raises(ValueError, match="short tp_pct must be < 1"):
        simulate_exit(ENTRY, [("t1", 100.0, 101.0, 99.0, 100.0)],
                      SL_PCT, 1.0, is_short=True)


def test_long_stop_of_one_or_more_is_rejected():
    with pytest.raises(ValueError, match="long sl_pct must be < 1"):
        simulate_exit(ENTRY, [("t1", 100.0, 101.0, 99.0, 100.0)], 1.0, TP_PCT)


def test_short_take_profit_just_under_one_is_allowed():
    result = simulate_exit(ENTRY, [("t1", 100.0, 101.0, 0.5, 0.9)],
                           SL_PCT, 0.99, is_short=True)
    assert result.reason == "take_profit"
    assert result.exit_price == pytest.approx(1.0)


# --- short side (ported from get_stop_price_nb position_now < 0) -----------

def test_short_levels_sit_on_opposite_sides_of_entry():
    """Short: SL above entry (103.0), TP below entry (94.0)."""
    rising = [("t1", 101.0, 103.5, 100.5, 103.0)]
    result = simulate_exit(ENTRY, rising, SL_PCT, TP_PCT, is_short=True)
    assert result.reason == "stop_loss"
    assert result.exit_price == pytest.approx(103.0)

    falling = [("t1", 99.0, 99.5, 93.5, 94.0)]
    result = simulate_exit(ENTRY, falling, SL_PCT, TP_PCT, is_short=True)
    assert result.reason == "take_profit"
    assert result.exit_price == pytest.approx(94.0)


def test_short_gap_up_through_stop_fills_at_open():
    candles = [("t1", 105.0, 106.0, 104.5, 105.5)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT, is_short=True)

    assert result.reason == "stop_loss"
    assert result.exit_price == 105.0
    assert result.gapped is True
    # Adverse for a short, i.e. worse than the nominal +3% stop.
    assert (ENTRY - result.exit_price) / ENTRY < -SL_PCT


def test_short_gap_down_through_take_profit_fills_at_open():
    candles = [("t1", 92.0, 93.0, 91.0, 92.5)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT, is_short=True)

    assert result.reason == "take_profit"
    assert result.exit_price == 92.0
    assert result.gapped is True
    assert (ENTRY - result.exit_price) / ENTRY > TP_PCT


def test_short_stop_wins_when_both_levels_inside_one_bar():
    candles = [("t1", 100.0, 104.0, 93.0, 99.0)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT, is_short=True)

    assert result.reason == "stop_loss"
    assert result.exit_price == pytest.approx(103.0)


def test_short_nan_open_falls_back_to_close_before_level_checks():
    candles = [("t1", float("nan"), 104.0, 102.0, 103.5)]
    result = simulate_exit(ENTRY, candles, SL_PCT, TP_PCT, is_short=True)

    assert result.reason == "stop_loss"
    assert result.exit_price == 103.5
    assert result.gapped is True


def test_long_and_short_are_not_symmetric_in_reason():
    """The same rising bar stops a short but takes profit on a long."""
    candles = [("t1", 101.0, 107.0, 100.5, 106.5)]

    assert simulate_exit(ENTRY, candles, SL_PCT, TP_PCT).reason == "take_profit"
    assert simulate_exit(ENTRY, candles, SL_PCT, TP_PCT, is_short=True).reason == "stop_loss"
