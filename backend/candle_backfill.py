"""
Candle backfill — fetch and persist historical hourly OHLCV for all traded symbols.

CLI usage:
    ~/Trading/kronos-trader/venv/bin/python candle_backfill.py [--symbols SYMBOL ...]

Backfills from (earliest trade timestamp – 400 candles) through now.
Upserts on UNIQUE(market, symbol, timestamp) so re-runs are safe.
Prints a per-symbol coverage report with gaps.

Also exposed as run_candle_backfill() for APScheduler.
"""
import argparse
import logging
import os
import sys
import time
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import ccxt
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

# Allow running from repo root or backend/ directory
sys.path.insert(0, os.path.dirname(__file__))

from database import SessionLocal, Candle, Trade, Portfolio, MarketType, engine
from sqlalchemy import text

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
LOOKBACK_EXTRA = 400       # candles before earliest trade
CANDLE_HOURS   = 1         # hourly
RATE_LIMIT_SLEEP = 0.25    # seconds between pages
MAX_RETRIES      = 5
BACKOFF_BASE     = 2.0     # exponential backoff multiplier


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_BASE,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


def _upsert_candles(db, rows: list[dict]):
    """
    Insert-or-ignore candle rows.
    Uses raw SQL for efficiency on large batches; SQLite INSERT OR IGNORE
    is safe on the UNIQUE(market, symbol, timestamp) constraint.
    """
    if not rows:
        return
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT OR IGNORE INTO candles
                    (market, symbol, timestamp, open, high, low, close, volume)
                VALUES
                    (:market, :symbol, :timestamp, :open, :high, :low, :close, :volume)
            """),
            rows,
        )
        conn.commit()


# -------------------------------------------------------
# Crypto — Binance US
# -------------------------------------------------------

def _fetch_crypto_candles(
    symbol: str, since_dt: datetime, until_dt: datetime
) -> list[dict]:
    """Page through Binance US hourly candles with retry and rate limiting."""
    exchange = ccxt.binanceus({"enableRateLimit": True})
    since_ms = int(since_dt.timestamp() * 1000)
    until_ms  = int(until_dt.timestamp() * 1000)
    rows: list[dict] = []
    limit = 1000  # Binance max per request

    while since_ms < until_ms:
        attempt = 0
        raw = None
        while attempt < MAX_RETRIES:
            try:
                raw = exchange.fetch_ohlcv(
                    symbol, timeframe="1h", since=since_ms, limit=limit
                )
                break
            except Exception as e:
                sleep = BACKOFF_BASE ** attempt
                logger.warning(f"[crypto] {symbol} fetch error ({e}), retry in {sleep}s")
                time.sleep(sleep)
                attempt += 1

        if not raw:
            logger.error(f"[crypto] {symbol} failed after {MAX_RETRIES} retries, stopping")
            break

        for ohlcv in raw:
            ts_ms, o, h, lo, c, v = ohlcv
            if ts_ms > until_ms:
                break
            rows.append({
                "market":    "crypto",
                "symbol":    symbol,
                "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).replace(tzinfo=None),
                "open":   o, "high": h, "low": lo, "close": c, "volume": v,
            })

        if len(raw) < limit:
            break
        since_ms = raw[-1][0] + 3_600_000  # next hour
        time.sleep(RATE_LIMIT_SLEEP)

    return rows


# -------------------------------------------------------
# Stocks — Alpaca (IEX feed, free tier)
# -------------------------------------------------------

def _fetch_stock_candles(
    symbol: str, since_dt: datetime, until_dt: datetime
) -> list[dict]:
    """Page through Alpaca hourly bars for a single stock symbol."""
    data_url = "https://data.alpaca.markets"
    headers = {
        "APCA-API-KEY-ID":     os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET", ""),
    }
    rows: list[dict] = []
    # Alpaca uses RFC3339 timestamps
    start_str = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str   = until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    page_token: Optional[str] = None

    with _make_session() as session:
        while True:
            params = {
                "timeframe": "1Hour",
                "feed":      "iex",
                "start":     start_str,
                "end":       end_str,
                "limit":     1000,
            }
            if page_token:
                params["page_token"] = page_token

            attempt = 0
            resp = None
            while attempt < MAX_RETRIES:
                try:
                    r = session.get(
                        f"{data_url}/v2/stocks/{symbol}/bars",
                        headers=headers,
                        params=params,
                        timeout=30,
                    )
                    if r.status_code == 429:
                        sleep = BACKOFF_BASE ** attempt
                        logger.warning(f"[stocks] {symbol} rate-limited, retry in {sleep}s")
                        time.sleep(sleep)
                        attempt += 1
                        continue
                    r.raise_for_status()
                    resp = r.json()
                    break
                except Exception as e:
                    sleep = BACKOFF_BASE ** attempt
                    logger.warning(f"[stocks] {symbol} fetch error ({e}), retry in {sleep}s")
                    time.sleep(sleep)
                    attempt += 1

            if resp is None:
                logger.error(f"[stocks] {symbol} failed after {MAX_RETRIES} retries, stopping")
                break

            for bar in resp.get("bars", []):
                ts = pd.to_datetime(bar["t"]).to_pydatetime().replace(tzinfo=None)
                rows.append({
                    "market":    "stocks",
                    "symbol":    symbol,
                    "timestamp": ts,
                    "open":   bar["o"],
                    "high":   bar["h"],
                    "low":    bar["l"],
                    "close":  bar["c"],
                    "volume": bar["v"],
                })

            page_token = resp.get("next_page_token")
            if not page_token:
                break
            time.sleep(RATE_LIMIT_SLEEP)

    return rows


# -------------------------------------------------------
# Coverage report
# -------------------------------------------------------

def _coverage_report(db, symbol: str, market: str, since_dt: datetime, until_dt: datetime):
    """
    Print expected vs actual candle count and list gaps.
    Crypto gaps are warnings; stock gaps (nights/weekends) are informational.
    """
    candles = (
        db.query(Candle)
        .filter(
            Candle.market    == market,
            Candle.symbol    == symbol,
            Candle.timestamp >= since_dt,
            Candle.timestamp <= until_dt,
        )
        .order_by(Candle.timestamp)
        .all()
    )

    if not candles:
        print(f"  [{symbol}] No candles stored — fetch may have failed")
        return

    actual = len(candles)
    ts_set = {c.timestamp for c in candles}
    ts_list = sorted(ts_set)

    # Expected = every hour in the window (crypto) or just report actual gaps
    total_hours = int((until_dt - since_dt).total_seconds() // 3600) + 1

    gaps: list[datetime] = []
    current = ts_list[0]
    for ts in ts_list[1:]:
        expected_next = current + timedelta(hours=1)
        if ts > expected_next:
            gap_count = int((ts - expected_next).total_seconds() // 3600)
            for i in range(gap_count):
                gaps.append(expected_next + timedelta(hours=i))
        current = ts

    if market == "crypto":
        coverage_pct = actual / total_hours * 100 if total_hours else 0
        status = "OK" if coverage_pct >= 99 else "WARN"
        print(f"  [{symbol}] crypto  actual={actual} / expected≈{total_hours}  "
              f"coverage={coverage_pct:.1f}%  [{status}]")
        if gaps:
            sample = gaps[:10]
            print(f"    WARN: {len(gaps)} gap(s). First 10: {[str(g) for g in sample]}")
    else:
        # Stocks — gaps for nights/weekends/holidays are expected
        print(f"  [{symbol}] stocks  actual={actual} bars stored  "
              f"(gaps incl. nights/weekends/holidays not counted as errors)")
        # Only flag unexpectedly large gaps (>72h) within market window
        big_gaps = [(g, e) for g, e in zip(ts_list, ts_list[1:])
                    if (e - g).total_seconds() > 72 * 3600]
        if big_gaps:
            print(f"    INFO: {len(big_gaps)} unusually large gap(s) (>72h):")
            for g, e in big_gaps[:5]:
                print(f"      {g} → {e}")


# -------------------------------------------------------
# Per-symbol backfill dispatcher
# -------------------------------------------------------

def _backfill_symbol(db, market: str, symbol: str, since_dt: datetime, until_dt: datetime):
    logger.info(f"Backfilling {market}/{symbol}  {since_dt.date()} → {until_dt.date()}")

    if market == "crypto":
        rows = _fetch_crypto_candles(symbol, since_dt, until_dt)
    elif market == "stocks":
        rows = _fetch_stock_candles(symbol, since_dt, until_dt)
    else:
        logger.warning(f"No candle fetcher for market '{market}' — skipping {symbol}")
        return

    _upsert_candles(db, rows)
    logger.info(f"  Upserted {len(rows)} candles for {symbol}")


# -------------------------------------------------------
# Append-only incremental update (for APScheduler)
# -------------------------------------------------------

def _append_recent(db, market: str, symbol: str):
    """Fetch the last ~48h of candles to stay current."""
    until_dt = datetime.utcnow()
    since_dt = until_dt - timedelta(hours=48)
    _backfill_symbol(db, market, symbol, since_dt, until_dt)


# -------------------------------------------------------
# Discover symbols from trades table
# -------------------------------------------------------

def _symbols_from_trades(db) -> dict[str, list[str]]:
    """
    Return {market: [symbol, ...]} for every market/symbol pair in the trades table.
    Excludes forex (no candle fetcher).
    """
    rows = (
        db.query(Trade.symbol, Portfolio.market)
        .join(Portfolio, Trade.portfolio_id == Portfolio.id)
        .filter(Portfolio.market.in_([MarketType.CRYPTO, MarketType.STOCKS]))
        .distinct()
        .all()
    )
    result: dict[str, list[str]] = {"crypto": [], "stocks": []}
    for symbol, market in rows:
        mkt = market.value if hasattr(market, "value") else market
        if mkt in result and symbol not in result[mkt]:
            result[mkt].append(symbol)
    return result


def _earliest_trade_ts(db, market: str, symbol: str) -> Optional[datetime]:
    from sqlalchemy import func
    port = db.query(Portfolio).filter_by(market=MarketType(market)).first()
    if not port:
        return None
    row = (
        db.query(func.min(Trade.opened_at))
        .filter(Trade.portfolio_id == port.id, Trade.symbol == symbol)
        .scalar()
    )
    return row


# -------------------------------------------------------
# Main backfill runner
# -------------------------------------------------------

def run_candle_backfill(
    symbols_override: Optional[dict[str, list[str]]] = None,
    incremental: bool = False,
):
    """
    Full backfill (or incremental append) of candles for all traded symbols.

    Parameters
    ----------
    symbols_override : if provided, use these instead of auto-detecting from trades
    incremental      : if True, only fetch the last 48h (for APScheduler hourly job)
    """
    db = SessionLocal()
    try:
        symbol_map = symbols_override or _symbols_from_trades(db)
        until_dt = datetime.utcnow()

        for market, symbols in symbol_map.items():
            for symbol in symbols:
                if incremental:
                    _append_recent(db, market, symbol)
                    continue

                earliest = _earliest_trade_ts(db, market, symbol)
                if earliest:
                    since_dt = earliest - timedelta(hours=LOOKBACK_EXTRA)
                else:
                    since_dt = until_dt - timedelta(days=90)  # fallback: 90 days

                _backfill_symbol(db, market, symbol, since_dt, until_dt)

        if not incremental:
            print("\n=== Candle Coverage Report ===")
            until_dt = datetime.utcnow()
            for market, symbols in symbol_map.items():
                for symbol in symbols:
                    earliest = _earliest_trade_ts(db, market, symbol)
                    since_dt = (
                        earliest - timedelta(hours=LOOKBACK_EXTRA)
                        if earliest else until_dt - timedelta(days=90)
                    )
                    _coverage_report(db, symbol, market, since_dt, until_dt)
            print("================================\n")

    finally:
        db.close()


# -------------------------------------------------------
# Historical cost backfill — recompute costs for all closed trades
# -------------------------------------------------------

def backfill_trade_costs():
    """
    One-off: populate gross_pnl/net_pnl/cost columns on all closed trades
    that lack them.  Safe to re-run — skips trades already having net_pnl set.
    """
    import importlib
    cost_model = importlib.import_module("cost_model")

    db = SessionLocal()
    try:
        trades = (
            db.query(Trade)
            .join(Portfolio, Trade.portfolio_id == Portfolio.id)
            .filter(
                Trade.status == "closed",
                Trade.net_pnl.is_(None),
                Trade.realized_pnl.isnot(None),
            )
            .all()
        )

        logger.info(f"[cost_backfill] Processing {len(trades)} historical closed trades...")

        portfolios: dict[int, Portfolio] = {}
        for t in trades:
            if t.portfolio_id not in portfolios:
                portfolios[t.portfolio_id] = db.query(Portfolio).get(t.portfolio_id)

        updated = 0
        for t in trades:
            port = portfolios.get(t.portfolio_id)
            if not port:
                continue
            market = port.market.value if hasattr(port.market, "value") else str(port.market)

            # For historical trades we don't have separate entry/exit prices;
            # realized_pnl = (exit_price - avg_price) * qty so we infer exit_price
            # from total_value / quantity (close trade stores proceeds as total_value)
            if t.direction and hasattr(t.direction, "value"):
                direction = t.direction.value
            else:
                direction = str(t.direction)

            if direction != "sell" or not t.quantity or t.quantity <= 0:
                continue  # only closing SELL trades carry realized P&L

            exit_price = t.price  # stored at close time
            # entry_price recovered from realized_pnl and quantity
            # realized_pnl = (exit_price - entry_price) * qty
            entry_price = exit_price - (t.realized_pnl / t.quantity)
            if entry_price <= 0:
                continue

            costs = cost_model.apply_costs(
                market        = market,
                symbol        = t.symbol,
                side          = "buy",
                entry_price   = entry_price,
                exit_price    = exit_price,
                position_size = t.quantity,
            )

            t.gross_pnl      = costs["gross_pnl"]
            t.entry_fee      = costs["entry_fee"]
            t.exit_fee       = costs["exit_fee"]
            t.entry_slippage = costs["entry_slippage"]
            t.exit_slippage  = costs["exit_slippage"]
            t.net_pnl        = costs["net_pnl"]
            t.cost_model_ver = costs["cost_model_ver"]
            updated += 1

            if updated % 50 == 0:
                db.commit()
                logger.info(f"[cost_backfill] {updated} trades updated so far...")

        db.commit()
        logger.info(f"[cost_backfill] Done — {updated} trades updated.")
        return updated

    finally:
        db.close()


# -------------------------------------------------------
# CLI entry point
# -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Candle backfill for Kronos Trader")
    parser.add_argument(
        "--symbols", nargs="*",
        help="Specific symbols to backfill (e.g. BTC/USDT AAPL). Default: all from trades table."
    )
    parser.add_argument(
        "--market", choices=["crypto", "stocks"],
        help="Market to backfill (required if --symbols is given)."
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Only fetch last 48h (for scheduled updates)."
    )
    parser.add_argument(
        "--cost-backfill", action="store_true",
        help="Also run the historical cost backfill for closed trades."
    )
    args = parser.parse_args()

    override = None
    if args.symbols:
        if not args.market:
            parser.error("--market is required when --symbols is specified")
        override = {args.market: args.symbols}

    run_candle_backfill(symbols_override=override, incremental=args.incremental)

    if args.cost_backfill:
        print("\n=== Cost Backfill ===")
        n = backfill_trade_costs()
        print(f"Updated {n} trades with net_pnl and cost columns.")
        print("====================\n")