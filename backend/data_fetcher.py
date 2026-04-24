"""
Market Data Fetcher
Handles OHLCV data for Crypto (Binance), Stocks (Alpaca), Forex (OANDA)
"""
import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import ccxt
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30  # seconds

def _make_session() -> requests.Session:
    """Requests session with automatic retry on transient connection errors."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,           # waits 1s, 2s, 4s between attempts
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


# -------------------------------------------------------
# Crypto — Binance via ccxt (no API key needed for public data)
# -------------------------------------------------------
class CryptoDataFetcher:
    SYMBOLS = ["BTC/USDT", "ETH/USDT"]
    DISPLAY_SYMBOLS = {"BTC/USDT": "BTC", "ETH/USDT": "ETH"}

    def __init__(self):
        self.exchange = ccxt.binanceus({
            "apiKey":    os.getenv("BINANCE_API_KEY", ""),
            "secret":    os.getenv("BINANCE_SECRET", ""),
            "enableRateLimit": True,
        })

    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.set_index("timestamp")

    def get_ticker(self, symbol: str) -> dict:
        t = self.exchange.fetch_ticker(symbol)
        return {
            "symbol":     self.DISPLAY_SYMBOLS.get(symbol, symbol),
            "price":      t["last"],
            "change_24h": t["percentage"],
            "volume_24h": t["quoteVolume"],
        }

    def get_all_tickers(self) -> list:
        return [self.get_ticker(s) for s in self.SYMBOLS]


# -------------------------------------------------------
# Stocks — Alpaca (paper trading free tier)
# -------------------------------------------------------
class StockDataFetcher:
    SYMBOLS = ["AAPL", "TSLA", "NVDA", "SPY"]
    BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    DATA_URL = "https://data.alpaca.markets"

    def __init__(self):
        self.headers = {
            "APCA-API-KEY-ID":     os.getenv("ALPACA_API_KEY", ""),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET", ""),
        }

    def get_ohlcv(self, symbol: str, timeframe: str = "1Hour", limit: int = 500) -> pd.DataFrame:
        end   = datetime.utcnow()
        start = end - timedelta(hours=limit)
        url = (
            f"{self.DATA_URL}/v2/stocks/{symbol}/bars"
            f"?timeframe={timeframe}&feed=iex"
            f"&start={start.isoformat()}Z"
            f"&end={end.isoformat()}Z"
            f"&limit={limit}"
        )
        with _make_session() as s:
            resp = s.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        bars = resp.json().get("bars", [])
        df = pd.DataFrame(bars)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["t"])
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return df[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp")

    def get_ticker(self, symbol: str) -> dict:
        url = f"{self.DATA_URL}/v2/stocks/{symbol}/trades/latest"
        with _make_session() as s:
            resp = s.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        trade = resp.json().get("trade", {})
        return {
            "symbol":     symbol,
            "price":      trade.get("p", 0.0),
            "change_24h": None,
            "volume_24h": None,
        }

    def get_all_tickers(self) -> list:
        results = []
        for s in self.SYMBOLS:
            try:
                results.append(self.get_ticker(s))
            except Exception as e:
                logger.warning(f"Failed ticker for {s}: {e}")
        return results


# -------------------------------------------------------
# Forex — OANDA practice account
# -------------------------------------------------------
class ForexDataFetcher:
    INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
    BASE_URL = os.getenv("OANDA_BASE_URL", "https://api-fxpractice.oanda.com")

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {os.getenv('OANDA_API_KEY', '')}",
            "Content-Type":  "application/json",
        }
        self.account_id = os.getenv("OANDA_ACCOUNT_ID", "")

    def get_ohlcv(self, instrument: str, granularity: str = "H1", count: int = 500, limit: int = 500) -> pd.DataFrame:
        url = (
            f"{self.BASE_URL}/v3/instruments/{instrument}/candles"
            f"?granularity={granularity}&count={count}&price=M"
        )
        with _make_session() as s:
            resp = s.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        candles = resp.json().get("candles", [])
        rows = []
        for c in candles:
            mid = c["mid"]
            rows.append({
                "timestamp": pd.to_datetime(c["time"]),
                "open":   float(mid["o"]),
                "high":   float(mid["h"]),
                "low":    float(mid["l"]),
                "close":  float(mid["c"]),
                "volume": float(c.get("volume", 0)),
            })
        df = pd.DataFrame(rows)
        return df.set_index("timestamp") if not df.empty else pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )

    def get_ticker(self, instrument: str) -> dict:
        url = f"{self.BASE_URL}/v3/instruments/{instrument}/candles?granularity=M1&count=1&price=M"
        with _make_session() as s:
            resp = s.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        candles = resp.json().get("candles", [])
        price = float(candles[-1]["mid"]["c"]) if candles else 0.0
        return {
            "symbol":     instrument.replace("_", "/"),
            "price":      price,
            "change_24h": None,
            "volume_24h": None,
        }

    def get_all_tickers(self) -> list:
        results = []
        for inst in self.INSTRUMENTS:
            try:
                results.append(self.get_ticker(inst))
            except Exception as e:
                logger.warning(f"Failed ticker for {inst}: {e}")
        return results


# -------------------------------------------------------
# Unified fetcher
# -------------------------------------------------------
class MarketDataFetcher:
    def __init__(self):
        self.crypto = CryptoDataFetcher()
        self.stocks = StockDataFetcher()
        self.forex  = ForexDataFetcher()

    def get_ohlcv(self, market: str, symbol: str, limit: int = 500) -> pd.DataFrame:
        if market == "crypto":
            return self.crypto.get_ohlcv(symbol, limit=limit)
        elif market == "stocks":
            return self.stocks.get_ohlcv(symbol, limit=limit)
        elif market == "forex":
            return self.forex.get_ohlcv(symbol, limit=limit)
        raise ValueError(f"Unknown market: {market}")

    def get_all_prices(self) -> dict:
        return {
            "crypto": self.crypto.get_all_tickers(),
            "stocks": self.stocks.get_all_tickers(),
            #"forex":  self.forex.get_all_tickers(),
        }