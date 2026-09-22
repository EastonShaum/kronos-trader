"""
Golden test pinning live P&L arithmetic for side='buy'.

PROVENANCE: every expected value below was produced by the PRE-REFACTOR
apply_costs (commit 8825799, before _gross_pnl was extracted), not by the
current implementation. They are a fixed external reference, not a snapshot
of what the code happens to do now.

WHY THIS EXISTS: the live close path in trading_engine._close_position calls
apply_costs with side='buy', and the numbers it returns are written to
trades.gross_pnl / net_pnl and become the system's record of its own
performance. The side-aware refactor claimed to leave that path bit-identical.
This test enforces that claim instead of asserting it.

IF THIS TEST FAILS: do not regenerate the expected values from the current
code. A failure means live P&L arithmetic changed. Either the change was
unintended and should be reverted, or it was deliberate and the trades table
now contains rows computed two different ways - which needs a migration plan,
not a test update.

Sanity anchor: the 'stocks_loss_real' case uses the entry, exit and quantity
of trade 814 (MARA, 2026-09-15) and reproduces its recorded net_pnl of
-3.23893 exactly.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cost_model  # noqa: E402
from cost_model import apply_costs  # noqa: E402


def _pinned_config(fee_bps, slip_bps, enabled=True, overrides=None):
    """
    Fixed cost config.

    apply_costs normally reads settings.json at call time, so without pinning
    this the golden values would drift whenever the live cost settings change
    and the test would be measuring settings rather than arithmetic.
    """
    return {
        "enabled": enabled,
        "model_version": "v1",
        "crypto": {"fee_bps_per_side": fee_bps, "slippage_bps_per_side": slip_bps},
        "stocks": {"fee_bps_per_side": fee_bps, "slippage_bps_per_side": slip_bps},
        "dynamic_slippage": {"enabled": False, "k": 0.25},
        "symbol_slippage_override_bps": overrides or {},
    }


# (market, symbol, entry, exit, qty, fee_bps, slip_bps, enabled, overrides, expected)
GOLDEN = [
    ("crypto_profit",
     "crypto", "BTC/USDT", 100.0, 103.0, 1.5, 40, 5, True, None,
     {"gross_pnl": 4.5, "entry_fee": 0.6, "exit_fee": 0.618,
      "entry_slippage": 0.075, "exit_slippage": 0.07725,
      "net_pnl": 3.12975, "cost_model_ver": "v1"}),

    ("crypto_loss",
     "crypto", "BTC/USDT", 50000.0, 48500.0, 0.02, 40, 5, True, None,
     {"gross_pnl": -30.0, "entry_fee": 4.0, "exit_fee": 3.88,
      "entry_slippage": 0.5, "exit_slippage": 0.485,
      "net_pnl": -38.865, "cost_model_ver": "v1"}),

    ("stocks_profit",
     "stocks", "NVDA", 450.25, 463.75, 0.333, 0, 10, True, None,
     {"gross_pnl": 4.4955, "entry_fee": 0.0, "exit_fee": 0.0,
      "entry_slippage": 0.149933, "exit_slippage": 0.154429,
      "net_pnl": 4.191138, "cost_model_ver": "v1"}),

    # Real trade 814: MARA, 2026-09-15, recorded net_pnl -3.23893.
    ("stocks_loss_real",
     "stocks", "MARA", 11.36, 10.985, 8.15143076878535, 0, 10, True, None,
     {"gross_pnl": -3.056787, "entry_fee": 0.0, "exit_fee": 0.0,
      "entry_slippage": 0.0926, "exit_slippage": 0.089543,
      "net_pnl": -3.23893, "cost_model_ver": "v1"}),

    # Zero gross, non-zero cost: slippage alone must still produce a loss.
    ("breakeven",
     "stocks", "SPY", 620.5, 620.5, 1.0, 0, 10, True, None,
     {"gross_pnl": 0.0, "entry_fee": 0.0, "exit_fee": 0.0,
      "entry_slippage": 0.6205, "exit_slippage": 0.6205,
      "net_pnl": -1.241, "cost_model_ver": "v1"}),

    ("symbol_override",
     "stocks", "SIDU", 4.25, 4.05, 35.0, 0, 10, True, {"SIDU": 50},
     {"gross_pnl": -7.0, "entry_fee": 0.0, "exit_fee": 0.0,
      "entry_slippage": 0.74375, "exit_slippage": 0.70875,
      "net_pnl": -8.4525, "cost_model_ver": "v1"}),

    # The disabled branch computes gross separately and must stay in step.
    ("costs_disabled",
     "crypto", "ETH/USDT", 3000.0, 2910.0, 0.5, 40, 5, False, None,
     {"gross_pnl": -45.0, "entry_fee": 0.0, "exit_fee": 0.0,
      "entry_slippage": 0.0, "exit_slippage": 0.0,
      "net_pnl": -45.0, "cost_model_ver": "disabled"}),

    ("high_fee_tier",
     "crypto", "BTC/USDT", 92000.0, 94760.0, 0.0125, 60, 8, True, None,
     {"gross_pnl": 34.5, "entry_fee": 6.9, "exit_fee": 7.107,
      "entry_slippage": 0.92, "exit_slippage": 0.9476,
      "net_pnl": 18.6254, "cost_model_ver": "v1"}),
]


@pytest.mark.parametrize(
    "name,market,symbol,entry,exit_price,qty,fee_bps,slip_bps,enabled,overrides,expected",
    GOLDEN,
    ids=[case[0] for case in GOLDEN],
)
def test_buy_side_pnl_is_bit_identical_to_pre_refactor(
    monkeypatch, name, market, symbol, entry, exit_price, qty,
    fee_bps, slip_bps, enabled, overrides, expected,
):
    monkeypatch.setattr(
        cost_model, "_load_cost_config",
        lambda: _pinned_config(fee_bps, slip_bps, enabled, overrides),
    )

    result = apply_costs(
        market=market, symbol=symbol, side="buy",
        entry_price=entry, exit_price=exit_price, position_size=qty,
    )

    # Exact equality, not approx: apply_costs rounds to 6dp, so these are
    # stable, and "close enough" is not the property under test.
    assert result == expected, (
        f"{name}: live buy-side P&L arithmetic changed.\n"
        f"  expected (pre-refactor): {expected}\n"
        f"  got (current):           {result}\n"
        f"Do not regenerate these values - see module docstring."
    )


def test_sell_side_is_the_negation_of_buy_side_gross(monkeypatch):
    """
    The one behaviour the refactor was meant to add.

    Shorts invert gross P&L; fees and slippage are charged on notional and
    must stay identical between the two sides.
    """
    monkeypatch.setattr(
        cost_model, "_load_cost_config", lambda: _pinned_config(40, 5)
    )
    kwargs = dict(market="crypto", symbol="BTC/USDT",
                  entry_price=100.0, exit_price=103.0, position_size=1.5)

    long = apply_costs(side="buy", **kwargs)
    short = apply_costs(side="sell", **kwargs)

    assert long["gross_pnl"] == 4.5
    assert short["gross_pnl"] == -4.5
    for field in ("entry_fee", "exit_fee", "entry_slippage", "exit_slippage"):
        assert short[field] == long[field], f"{field} must not depend on side"
