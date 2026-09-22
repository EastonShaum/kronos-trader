"""
Cost model for paper trading.

Applies realistic fees and adversely-directional slippage to trade P&L.
Slippage is NOT symmetric: buys fill above reference, sells fill below.

Default rates (Binance.US spot taker as of 2024-Q4):
  - Tier 0: 0.60% taker (60 bps) — use 40 bps as conservative estimate after
    small BNB discount. Adjust via settings.json if your account is on a
    higher tier or benefiting from promotions.

Stocks via Alpaca paper are commission-free but have real spread/impact slippage.
10 bps is optimistic for liquid names (AAPL/SPY); illiquid names (SIDU, GME)
can easily be 30-100 bps. Add per-symbol overrides via settings.json.
"""
import json
import logging
import pathlib

logger = logging.getLogger(__name__)

SETTINGS_FILE = pathlib.Path(__file__).parent / "settings.json"

# Fallback defaults — overridden by settings.json
_DEFAULT_COSTS = {
    "enabled": True,
    "model_version": "v1",
    "crypto": {"fee_bps_per_side": 40, "slippage_bps_per_side": 5},
    "stocks": {"fee_bps_per_side": 0,  "slippage_bps_per_side": 10},
    "dynamic_slippage": {
        "enabled": False,
        "k": 0.25,
    },
    # Per-symbol slippage overrides (bps): e.g. {"SIDU": 50, "GME": 80}
    "symbol_slippage_override_bps": {},
}

# Per-symbol fee overrides (bps) for non-standard arrangements
_SYMBOL_FEE_OVERRIDES: dict = {}


def _load_cost_config() -> dict:
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE) as f:
                s = json.load(f)
            cfg = s.get("costs", {})
            # Merge with defaults so missing keys always resolve
            merged = dict(_DEFAULT_COSTS)
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            return merged
    except Exception as e:
        logger.warning(f"[cost_model] Could not load settings: {e}")
    return dict(_DEFAULT_COSTS)


def _bps_to_frac(bps: float) -> float:
    return bps / 10_000.0


def _gross_pnl(side: str, entry_price: float, exit_price: float,
               position_size: float) -> float:
    """
    Gross P&L for the side the position was opened on.

    'buy'  (long)  profits when price rises: exit - entry
    'sell' (short) profits when price falls: entry - exit

    Every caller in the project passes side='buy', so this is behaviour-
    preserving for all of them, including the live close path in
    trading_engine._close_position.
    """
    if side == "sell":
        return (entry_price - exit_price) * position_size
    return (exit_price - entry_price) * position_size


def _dynamic_slippage_bps(k: float, candle: dict | None) -> float | None:
    """Compute range-based slippage from fill candle OHLC, or None if unavailable."""
    if candle is None:
        return None
    try:
        h = float(candle["high"])
        lo = float(candle["low"])
        c = float(candle["close"])
        if c <= 0:
            return None
        return k * ((h - lo) / c) * 10_000.0
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def apply_costs(
    market: str,
    symbol: str,
    side: str,         # 'buy' or 'sell' (perspective of the opening trade)
    entry_price: float,
    exit_price: float,
    position_size: float,  # quantity in base units
    entry_candle: dict | None = None,
    exit_candle: dict | None = None,
) -> dict:
    """
    Compute gross and net P&L for a closed long trade.

    Parameters
    ----------
    market        : 'crypto' | 'stocks' | 'forex'
    symbol        : e.g. 'BTC/USDT', 'AAPL'
    side          : 'buy' (long) — shorts not traded in paper mode
    entry_price   : fill price on open
    exit_price    : fill price on close
    position_size : number of units held
    entry_candle  : dict with open/high/low/close keys for the entry bar (optional)
    exit_candle   : dict with open/high/low/close keys for the exit bar (optional)

    Returns
    -------
    dict with keys:
        gross_pnl, entry_fee, exit_fee,
        entry_slippage, exit_slippage, net_pnl,
        cost_model_ver
    """
    cfg = _load_cost_config()

    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    if not cfg.get("enabled", True):
        gross = _gross_pnl(side, entry_price, exit_price, position_size)
        return {
            "gross_pnl":      round(gross, 6),
            "entry_fee":      0.0,
            "exit_fee":       0.0,
            "entry_slippage": 0.0,
            "exit_slippage":  0.0,
            "net_pnl":        round(gross, 6),
            "cost_model_ver": "disabled",
        }

    version = cfg.get("model_version", "v1")
    market_cfg = cfg.get(market, cfg.get("crypto"))  # fallback to crypto rates

    fee_bps = float(market_cfg.get("fee_bps_per_side", 40))
    slip_bps_base = float(market_cfg.get("slippage_bps_per_side", 5))

    # Per-symbol slippage override
    sym_overrides = cfg.get("symbol_slippage_override_bps", {})
    if symbol in sym_overrides:
        slip_bps_base = float(sym_overrides[symbol])

    # Dynamic slippage (range-based) replaces the fixed rate when enabled
    dyn_cfg = cfg.get("dynamic_slippage", {})
    if dyn_cfg.get("enabled", False):
        k = float(dyn_cfg.get("k", 0.25))
        entry_dyn = _dynamic_slippage_bps(k, entry_candle)
        exit_dyn  = _dynamic_slippage_bps(k, exit_candle)
        entry_slip_bps = entry_dyn if entry_dyn is not None else slip_bps_base
        exit_slip_bps  = exit_dyn  if exit_dyn  is not None else slip_bps_base
    else:
        entry_slip_bps = slip_bps_base
        exit_slip_bps  = slip_bps_base

    entry_notional = entry_price * position_size
    exit_notional  = exit_price  * position_size

    # Fees: paid on notional each side
    entry_fee = entry_notional * _bps_to_frac(fee_bps)
    exit_fee  = exit_notional  * _bps_to_frac(fee_bps)

    # Adverse slippage: buys fill high, sells fill low
    # Modelled as cost proportional to the notional (conservative, deterministic)
    entry_slippage = entry_notional * _bps_to_frac(entry_slip_bps)
    exit_slippage  = exit_notional  * _bps_to_frac(exit_slip_bps)

    gross_pnl = _gross_pnl(side, entry_price, exit_price, position_size)
    # Fees and slippage are charged on notional and are direction-agnostic.
    total_cost = entry_fee + exit_fee + entry_slippage + exit_slippage
    net_pnl    = gross_pnl - total_cost

    return {
        "gross_pnl":      round(gross_pnl,      6),
        "entry_fee":      round(entry_fee,       6),
        "exit_fee":       round(exit_fee,        6),
        "entry_slippage": round(entry_slippage,  6),
        "exit_slippage":  round(exit_slippage,   6),
        "net_pnl":        round(net_pnl,         6),
        "cost_model_ver": version,
    }