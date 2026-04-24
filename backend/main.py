"""
Kronos Trader — FastAPI Backend
All API routes + APScheduler for automated predictions
"""
import os
import logging
import asyncio
import pathlib
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager
import pytz

import secrets
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from database import init_db, get_db, Portfolio, Trade, Prediction, PriceSnapshot, MarketType

# -------------------------------------------------------
# Basic Auth middleware (disabled — comment back in to re-enable)
# -------------------------------------------------------
# DASHBOARD_USER     = os.getenv("DASHBOARD_USER", "kronos")
# DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
#
# class BasicAuthMiddleware:
#     def __init__(self, app):
#         self.app = app
#
#     async def __call__(self, scope, receive, send):
#         if scope["type"] not in ("http", "websocket"):
#             await self.app(scope, receive, send)
#             return
#
#         # Skip auth if no password is configured
#         if not DASHBOARD_PASSWORD:
#             await self.app(scope, receive, send)
#             return
#
#         # Always allow health check
#         if scope.get("path") == "/api/health":
#             await self.app(scope, receive, send)
#             return
#
#         request = Request(scope, receive)
#         auth = request.headers.get("Authorization", "")
#
#         if auth.startswith("Basic "):
#             import base64
#             try:
#                 decoded = base64.b64decode(auth[6:]).decode("utf-8")
#                 username, _, password = decoded.partition(":")
#                 user_ok = secrets.compare_digest(username, DASHBOARD_USER)
#                 pass_ok = secrets.compare_digest(password, DASHBOARD_PASSWORD)
#                 if user_ok and pass_ok:
#                     await self.app(scope, receive, send)
#                     return
#             except Exception:
#                 pass
#
#         # Reject — send WWW-Authenticate to trigger browser login prompt
#         response = Response(
#             content="Unauthorized",
#             status_code=401,
#             headers={"WWW-Authenticate": 'Basic realm="Kronos Trader"'},
#         )
#         await response(scope, receive, send)
from data_fetcher import MarketDataFetcher
from kronos_engine import run_prediction, run_dual_prediction
from trading_engine import PaperTradingEngine

_log_file = pathlib.Path(__file__).parent / "kronos.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(_log_file)),
    ],
)
logger = logging.getLogger(__name__)

# Symbols to track per market
CRYPTO_SYMBOLS  = ["BTC/USDT", "ETH/USDT"]
STOCK_SYMBOLS   = ["AAPL", "TSLA", "NVDA", "SPY"]
FOREX_SYMBOLS   = ["EUR_USD", "GBP_USD", "USD_JPY"]
#FOREX_SYMBOLS   = []

fetcher  = MarketDataFetcher()
trader   = PaperTradingEngine()
scheduler = AsyncIOScheduler()

# Markets currently running a prediction cycle — checked by the dashboard
# to keep the trigger button disabled until the run completes.
_active_predictions: set = set()


# -------------------------------------------------------
# Market hours helper
# -------------------------------------------------------
_ET = pytz.timezone("America/New_York")

def is_market_open() -> bool:
    """Return True if US stock market is currently open (Mon–Fri 09:30–16:00 ET)."""
    now = datetime.now(_ET)
    if now.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now < market_close


# -------------------------------------------------------
# Scheduler jobs
# -------------------------------------------------------
async def run_prediction_cycle(market: str, symbols: list, triggered_by: str = "scheduler"):
    """Run Kronos predictions for all symbols in a market.
    Blocking calls (data fetch + model inference) are offloaded to a thread
    pool via asyncio.to_thread so the event loop stays responsive.
    """
    _active_predictions.add(market)
    from database import SessionLocal
    db = SessionLocal()
    try:
        cfg          = load_settings()
        model_cfg    = cfg.get("models", {})
        active_model = model_cfg.get("active_model", "small")
        dual_mode    = model_cfg.get("dual_model_mode", True)

        for symbol in symbols:
            try:
                logger.info(f"🔄 Predicting {symbol} ({market})...")

                # Offload blocking HTTP fetch to thread pool
                df = await asyncio.to_thread(fetcher.get_ohlcv, market, symbol, 500)
                if df.empty or len(df) < 50:
                    logger.warning(f"Not enough data for {symbol}, skipping.")
                    continue

                horizon     = int(os.getenv("PREDICTION_HORIZON", 24))
                sample_cnt  = int(os.getenv("SAMPLE_COUNT", 5))

                if dual_mode:
                    # Run both models; process_prediction will only trade for active_model
                    dual = await asyncio.to_thread(
                        run_dual_prediction,
                        df, symbol, horizon, sample_cnt, triggered_by,
                    )
                    trader.process_prediction(dual["small"], market, db, active_model)
                    trader.process_prediction(dual["mini"],  market, db, active_model)
                else:
                    result = await asyncio.to_thread(
                        run_prediction,
                        df, symbol, horizon, sample_cnt, triggered_by, active_model,
                    )
                    trader.process_prediction(result, market, db, active_model)

            except Exception as e:
                logger.error(f"Error predicting {symbol}: {e}", exc_info=True)
    finally:
        _active_predictions.discard(market)
        db.close()


async def update_prices_job():
    """Refresh latest prices every 5 minutes."""
    from database import SessionLocal, PriceSnapshot, MarketType
    db = SessionLocal()
    try:
        all_prices = fetcher.get_all_prices()
        for market, prices in all_prices.items():
            market_type = MarketType(market)
            trader.update_prices(prices, market, db)
            for p in prices:
                snap = PriceSnapshot(
                    symbol=p["symbol"],
                    market=market_type,
                    price=p["price"] or 0.0,
                    change_24h=p.get("change_24h"),
                    volume_24h=p.get("volume_24h"),
                )
                db.add(snap)
        db.commit()
    except Exception as e:
        logger.error(f"Price update failed: {e}", exc_info=True)
    finally:
        db.close()


# -------------------------------------------------------
# App lifecycle
# -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("✅ Database ready.")

    interval_min = int(os.getenv("PREDICTION_INTERVAL_MINUTES", 60))

    # Price updates every 5 minutes
    scheduler.add_job(update_prices_job, IntervalTrigger(minutes=5), id="price_update", misfire_grace_time=60)

    # Predictions per market — load symbols from settings at run time
    def _active_syms(market, defaults):
        cfg = load_settings()
        disabled = set(cfg.get("disabled_symbols", []))
        return [s for s in cfg.get("symbols", {}).get(market, defaults) if s not in disabled]

    async def crypto_job():
        syms = _active_syms("crypto", CRYPTO_SYMBOLS)
        logger.info(f"[scheduler] crypto_job starting, symbols={syms}")
        try:
            await run_prediction_cycle("crypto", syms)
        except Exception as e:
            logger.error(f"[scheduler] crypto_job unhandled error: {e}", exc_info=True)

    async def stocks_job():
        if not is_market_open():
            logger.info("[scheduler] Skipping stocks prediction — market is closed.")
            return
        syms = _active_syms("stocks", STOCK_SYMBOLS)
        logger.info(f"[scheduler] stocks_job starting, symbols={syms}")
        try:
            await run_prediction_cycle("stocks", syms)
        except Exception as e:
            logger.error(f"[scheduler] stocks_job unhandled error: {e}", exc_info=True)

    async def forex_job():
        syms = _active_syms("forex", FOREX_SYMBOLS)
        if not syms:
            return
        logger.info(f"[scheduler] forex_job starting, symbols={syms}")
        try:
            await run_prediction_cycle("forex", syms)
        except Exception as e:
            logger.error(f"[scheduler] forex_job unhandled error: {e}", exc_info=True)

    # Stagger prediction jobs so they never fire at the same second as price_update.
    # misfire_grace_time=300 means APScheduler will still run the job if it's up to
    # 5 minutes late (prevents "missed" drops when another job is in the executor).
    now = datetime.now(pytz.UTC)
    scheduler.add_job(
        crypto_job, IntervalTrigger(minutes=interval_min, start_date=now + timedelta(minutes=2)),
        id="predict_crypto", misfire_grace_time=300,
    )
    scheduler.add_job(
        stocks_job, IntervalTrigger(minutes=interval_min, start_date=now + timedelta(minutes=3)),
        id="predict_stocks", misfire_grace_time=300,
    )
    scheduler.add_job(
        forex_job,  IntervalTrigger(minutes=interval_min, start_date=now + timedelta(minutes=4)),
        id="predict_forex",  misfire_grace_time=300,
    )

    scheduler.start()
    logger.info(f"✅ Scheduler started (predictions every {interval_min} min).")
    yield
    scheduler.shutdown()


app = FastAPI(title="Kronos Trader API", version="1.0.0", lifespan=lifespan)

# app.add_middleware(BasicAuthMiddleware)  # disabled — see commented block above
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
# Routes — Portfolio & Overview
# -------------------------------------------------------
@app.get("/api/overview")
def get_overview(db: Session = Depends(get_db)):
    """Dashboard summary: all 3 portfolios."""
    markets   = ["crypto", "stocks", "forex"]
    summaries = [trader.get_portfolio_summary(m, db) for m in markets]
    total_equity  = sum(s.get("equity", 0) for s in summaries)
    total_start   = sum(s.get("starting_balance", 0) for s in summaries)
    total_pnl     = total_equity - total_start
    return {
        "portfolios":    summaries,
        "total_equity":  round(total_equity, 2),
        "total_starting": round(total_start, 2),
        "total_pnl":     round(total_pnl, 2),
        "total_pnl_pct": round((total_pnl / total_start) * 100, 2) if total_start else 0,
    }


@app.get("/api/portfolio/{market}")
def get_portfolio(market: str, db: Session = Depends(get_db)):
    if market not in ["crypto", "stocks", "forex"]:
        raise HTTPException(400, "Invalid market")
    return trader.get_portfolio_summary(market, db)


# -------------------------------------------------------
# Routes — Trades
# -------------------------------------------------------
@app.get("/api/trades")
def get_trades(
    market: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    q = db.query(Trade)
    if market:
        try:
            mt = MarketType(market)
            portfolio = db.query(Portfolio).filter_by(market=mt).first()
            if portfolio:
                q = q.filter(Trade.portfolio_id == portfolio.id)
        except Exception:
            pass
    trades = q.order_by(Trade.opened_at.desc()).limit(limit).all()
    return [
        {
            "id":           t.id,
            "symbol":       t.symbol,
            "direction":    t.direction.value,
            "quantity":     round(t.quantity, 6),
            "price":        round(t.price, 4),
            "total_value":  round(t.total_value, 2),
            "status":       t.status.value,
            "realized_pnl": round(t.realized_pnl or 0, 2),
            "opened_at":    t.opened_at.isoformat(),
            "closed_at":    t.closed_at.isoformat() if t.closed_at else None,
            "notes":        t.notes,
        }
        for t in trades
    ]


# -------------------------------------------------------
# Routes — Predictions
# -------------------------------------------------------
@app.get("/api/predictions")
def get_predictions(
    symbol: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    q = db.query(Prediction)
    if symbol:
        q = q.filter(Prediction.symbol == symbol)
    preds = q.order_by(Prediction.created_at.desc()).limit(limit).all()
    return [
        {
            "id":               p.id,
            "symbol":           p.symbol,
            "market":           p.market.value,
            "current_price":    round(p.current_price, 4),
            "predicted_close":  round(p.predicted_close, 4),
            "predicted_high":   round(p.predicted_high, 4),
            "predicted_low":    round(p.predicted_low, 4),
            "confidence":       round((p.confidence or 0) * 100, 1),
            "horizon_hours":    p.horizon_hours,
            "triggered_by":     p.triggered_by,
            "model_name":       p.model_name or "small",
            "raw_output":       p.raw_output,
            "created_at":       p.created_at.isoformat(),
        }
        for p in preds
    ]


# -------------------------------------------------------
# Routes — Manual trigger
# -------------------------------------------------------
class TriggerRequest(BaseModel):
    market:  Optional[str] = None   # 'crypto' | 'stocks' | 'forex' | None = all
    symbol:  Optional[str] = None   # specific symbol or None = all in market

@app.post("/api/predict/trigger")
async def manual_trigger(req: TriggerRequest, background_tasks: BackgroundTasks):
    """Manually trigger a prediction run from the dashboard."""
    cfg      = load_settings()
    disabled = set(cfg.get("disabled_symbols", []))
    syms_cfg = cfg.get("symbols", {})
    targets = {
        "crypto": [s for s in syms_cfg.get("crypto", CRYPTO_SYMBOLS) if s not in disabled],
        "stocks": [s for s in syms_cfg.get("stocks", STOCK_SYMBOLS)  if s not in disabled],
        "forex":  [s for s in syms_cfg.get("forex",  FOREX_SYMBOLS)  if s not in disabled],
    }

    if req.market and req.market not in targets:
        raise HTTPException(400, "Invalid market")

    markets_to_run = {req.market: list(targets[req.market])} if req.market else {k: list(v) for k, v in targets.items()}

    if req.symbol:
        sym_upper = req.symbol.upper()
        for mkt in markets_to_run:
            # Try to match against configured symbols first
            filtered = [s for s in markets_to_run[mkt] if sym_upper in s.upper()]
            if filtered:
                markets_to_run[mkt] = filtered
            else:
                # Symbol not in configured list — run it anyway as a one-off
                markets_to_run[mkt] = [req.symbol.upper()]

    for mkt, syms in markets_to_run.items():
        if syms:
            background_tasks.add_task(run_prediction_cycle, mkt, syms, "manual")

    return {"status": "triggered", "markets": list(markets_to_run.keys()), "symbols": {k: v for k, v in markets_to_run.items() if v}}


# -------------------------------------------------------
# Routes — Price Snapshots
# -------------------------------------------------------
@app.get("/api/prices")
def get_prices(db: Session = Depends(get_db)):
    """Latest price snapshot for all tracked symbols."""
    results = {}
    for market in ["crypto", "stocks", "forex"]:
        mt = MarketType(market)
        # Get latest snapshot per symbol
        latest = (
            db.query(PriceSnapshot)
            .filter(PriceSnapshot.market == mt)
            .order_by(PriceSnapshot.fetched_at.desc())
            .limit(20)
            .all()
        )
        seen = {}
        for snap in latest:
            if snap.symbol not in seen:
                seen[snap.symbol] = {
                    "symbol":     snap.symbol,
                    "price":      round(snap.price, 4),
                    "change_24h": round(snap.change_24h or 0, 2),
                    "fetched_at": snap.fetched_at.isoformat(),
                }
        results[market] = list(seen.values())
    return results


# -------------------------------------------------------
# Routes — Scheduler status
# -------------------------------------------------------
@app.get("/api/scheduler/status")
def scheduler_status():
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id":       job.id,
            "next_run": str(job.next_run_time) if job.next_run_time else None,
        })
    return {
        "running":            scheduler.running,
        "jobs":               jobs,
        "active_predictions": list(_active_predictions),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/notifications/test")
async def test_notification():
    """Send a test Discord embed to verify the webhook is configured."""
    from notifications import send_test_message
    ok = await send_test_message()
    return {
        "success": ok,
        "message": "Test message sent — check your Discord channel." if ok
                   else "Failed. Check DISCORD_WEBHOOK_URL and ensure discord_enabled is true in Settings.",
    }


@app.get("/api/logs")
def get_logs(lines: int = 100):
    """Return the last N lines of the server log file."""
    try:
        with open(_log_file) as f:
            all_lines = f.readlines()
        return {"lines": [l.rstrip() for l in all_lines[-lines:]]}
    except FileNotFoundError:
        return {"lines": []}


@app.get("/api/system")
def get_system_stats(db: Session = Depends(get_db)):
    """Raspberry Pi hardware stats — never crashes, returns null for unreadable values."""
    import psutil

    def _read(fn):
        try:
            return fn()
        except Exception:
            return None

    # CPU temperature
    def _cpu_temp():
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000, 1)

    # Uptime
    def _uptime():
        with open("/proc/uptime") as f:
            seconds = float(f.read().split()[0])
        return round(seconds / 3600, 1)

    # RAM
    def _ram():
        m = psutil.virtual_memory()
        return {
            "ram_used":  round(m.used  / 1024 / 1024, 1),
            "ram_total": round(m.total / 1024 / 1024, 1),
            "ram_pct":   round(m.percent, 1),
        }

    # Disk
    def _disk():
        d = psutil.disk_usage("/")
        return {
            "disk_used":  round(d.used  / 1024 / 1024 / 1024, 2),
            "disk_total": round(d.total / 1024 / 1024 / 1024, 2),
            "disk_pct":   round(d.percent, 1),
        }

    # Find uvicorn PID
    def _pid():
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                if any("uvicorn" in c for c in (proc.info["cmdline"] or [])):
                    return proc.info["pid"]
            except Exception:
                pass
        return None

    # Predictions today
    def _preds_today():
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        return db.query(Prediction).filter(Prediction.created_at >= today).count()

    # Minutes since last prediction
    def _last_pred_ago():
        last = db.query(Prediction).order_by(Prediction.created_at.desc()).first()
        if not last:
            return None
        delta = datetime.utcnow() - last.created_at
        return round(delta.total_seconds() / 60, 1)

    ram  = _read(_ram)  or {}
    disk = _read(_disk) or {}

    return {
        "cpu_temp":              _read(_cpu_temp),
        "cpu_usage":             _read(lambda: round(psutil.cpu_percent(interval=0.2), 1)),
        "ram_used":              ram.get("ram_used"),
        "ram_total":             ram.get("ram_total"),
        "ram_pct":               ram.get("ram_pct"),
        "disk_used":             disk.get("disk_used"),
        "disk_total":            disk.get("disk_total"),
        "disk_pct":              disk.get("disk_pct"),
        "uptime_hours":          _read(_uptime),
        "kronos_pid":            _read(_pid),
        "prediction_count_today": _read(_preds_today),
        "last_prediction_ago":   _read(_last_pred_ago),
        "timestamp":             datetime.utcnow().isoformat(),
    }


from fastapi import Body
import json

# -------------------------------------------------------
# Settings — stored in a simple JSON file
# -------------------------------------------------------
SETTINGS_FILE = pathlib.Path(__file__).parent / "settings.json"

DEFAULT_SETTINGS = {
    "risk": {
        "position_size_pct":      0.15,
        "min_confidence":         0.75,
        "stop_loss_pct":          0.03,
        "take_profit_pct":        0.06,
        "min_predicted_move_pct": 0.02,
    },
    "prediction": {
        "interval_minutes":  60,
        "horizon_hours":     24,
        "sample_count":      5,
        "lookback_candles":  400,
    },
    "symbols": {
        "crypto": ["BTC/USDT", "ETH/USDT"],
        "stocks": ["AAPL", "TSLA", "NVDA", "SPY"],
        "forex":  [],
    },
    "disabled_symbols": [],
    "notifications": {
        "discord_enabled":     False,
        "discord_webhook_url": "",
    },
    "models": {
        "active_model":    "small",
        "dual_model_mode": True,
    },
}

def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return DEFAULT_SETTINGS.copy()

def save_settings(s: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)
    import trading_engine as te
    te.POSITION_SIZE_PCT      = s["risk"]["position_size_pct"]
    te.MIN_CONFIDENCE         = s["risk"]["min_confidence"]
    te.STOP_LOSS_PCT          = s["risk"]["stop_loss_pct"]
    te.TAKE_PROFIT_PCT        = s["risk"]["take_profit_pct"]
    te.MIN_PREDICTED_MOVE_PCT = s["risk"].get("min_predicted_move_pct", 0.02)
    # Ensure models block always has defaults
    if "models" not in s:
        s["models"] = {"active_model": "small", "dual_model_mode": True}


@app.get("/api/settings")
def get_settings():
    return load_settings()


@app.post("/api/settings")
def update_settings(settings: dict = Body(...)):
    save_settings(settings)
    new_interval = settings["prediction"]["interval_minutes"]
    for job_id in ["predict_crypto", "predict_stocks", "predict_forex"]:
        job = scheduler.get_job(job_id)
        if job:
            scheduler.reschedule_job(
                job_id,
                trigger=IntervalTrigger(minutes=new_interval)
            )
    return {"status": "saved", "settings": settings}


# -------------------------------------------------------
# Chart Data — live candles + prediction overlay + indicators
# -------------------------------------------------------
@app.get("/api/chart/{market}/{symbol}")
def get_chart_data(market: str, symbol: str, limit: int = 100):
    try:
        import pandas as pd
        import pandas_ta as ta

        df = fetcher.get_ohlcv(market, symbol.replace("-", "/"), limit=limit)
        if df.empty:
            raise HTTPException(404, "No data found")

        df = df.reset_index()

        # Build timestamps list once — shared by candles + indicators
        timestamps = [str(row.get("timestamp", row.name)) for _, row in df.iterrows()]

        candles = []
        for i, (_, row) in enumerate(df.iterrows()):
            candles.append({
                "time":   timestamps[i],
                "open":   round(float(row["open"]),  4),
                "high":   round(float(row["high"]),  4),
                "low":    round(float(row["low"]),   4),
                "close":  round(float(row["close"]), 4),
                "volume": round(float(row.get("volume", 0)), 2),
            })

        # --- RSI (14) ---
        close       = df["close"].astype(float)
        rsi_series  = ta.rsi(close, length=14)
        rsi_data    = []
        latest_rsi  = None
        for i, val in enumerate(rsi_series):
            if pd.notna(val):
                v = round(float(val), 2)
                rsi_data.append({"time": timestamps[i], "value": v})
                latest_rsi = v

        rsi_condition = (
            "overbought" if latest_rsi is not None and latest_rsi > 70 else
            "oversold"   if latest_rsi is not None and latest_rsi < 30 else
            "neutral"
        )

        # --- MACD (12, 26, 9) ---
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        macd_data          = []
        latest_macd_signal = None
        if macd_df is not None and not macd_df.empty:
            mcol = next((c for c in macd_df.columns if c.startswith("MACD_")),  None)
            hcol = next((c for c in macd_df.columns if c.startswith("MACDh_")), None)
            scol = next((c for c in macd_df.columns if c.startswith("MACDs_")), None)
            if mcol and hcol and scol:
                for i in range(len(timestamps)):
                    m = macd_df[mcol].iloc[i]
                    h = macd_df[hcol].iloc[i]
                    s = macd_df[scol].iloc[i]
                    if pd.notna(m) and pd.notna(h) and pd.notna(s):
                        macd_data.append({
                            "time":      timestamps[i],
                            "macd":      round(float(m), 4),
                            "signal":    round(float(s), 4),
                            "histogram": round(float(h), 4),
                        })
                if macd_data:
                    last = macd_data[-1]
                    latest_macd_signal = "bullish" if last["macd"] > last["signal"] else "bearish"

        # --- Prediction overlay ---
        from database import SessionLocal, Prediction
        db2 = SessionLocal()
        try:
            sym_key = symbol.replace("-", "/")
            pred = (
                db2.query(Prediction)
                .filter(Prediction.symbol == sym_key)
                .order_by(Prediction.created_at.desc())
                .first()
            )
            prediction_overlay = None
            if pred and pred.raw_output:
                raw    = json.loads(pred.raw_output)
                times  = list(raw["open"].keys())
                opens  = list(raw["open"].values())
                highs  = list(raw["high"].values())
                lows   = list(raw["low"].values())
                closes = list(raw["close"].values())
                prediction_overlay = {
                    "created_at": pred.created_at.isoformat(),
                    "confidence": round((pred.confidence or 0) * 100, 1),
                    "candles": [
                        {
                            "time":  times[i],
                            "open":  round(float(opens[i]),  4),
                            "high":  round(float(highs[i]),  4),
                            "low":   round(float(lows[i]),   4),
                            "close": round(float(closes[i]), 4),
                        }
                        for i in range(len(times))
                    ],
                }
        finally:
            db2.close()

        return {
            "symbol":             symbol,
            "market":             market,
            "candles":            candles,
            "prediction":         prediction_overlay,
            # Indicators
            "rsi":                rsi_data,
            "macd":               macd_data,
            "latest_rsi":         latest_rsi,
            "rsi_condition":      rsi_condition,
            "latest_macd_signal": latest_macd_signal,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# -------------------------------------------------------
# Prediction Accuracy
# -------------------------------------------------------
@app.get("/api/accuracy")
def get_accuracy(hours: int = 24, db: Session = Depends(get_db)):
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    preds  = (
        db.query(Prediction)
        .filter(Prediction.created_at >= cutoff)
        .order_by(Prediction.created_at.asc())
        .all()
    )

    results = []
    price_cache = {}
    for p in preds:
        sym = p.symbol
        if sym not in price_cache:
            try:
                snap = (
                    db.query(PriceSnapshot)
                    .filter(PriceSnapshot.symbol.contains(sym.split("/")[0]))
                    .order_by(PriceSnapshot.fetched_at.desc())
                    .first()
                )
                price_cache[sym] = snap.price if snap else None
            except Exception:
                price_cache[sym] = None

        actual_price = price_cache.get(sym)
        if actual_price is None:
            continue

        pred_error_pct = None
        if p.predicted_close and actual_price:
            pred_error_pct = round(
                ((p.predicted_close - actual_price) / actual_price) * 100, 2
            )

        direction_correct = None
        if p.predicted_close and actual_price and p.current_price:
            pred_went_up   = p.predicted_close > p.current_price
            actual_went_up = actual_price > p.current_price
            direction_correct = (pred_went_up == actual_went_up)

        results.append({
            "id":                  p.id,
            "symbol":              p.symbol,
            "market":              p.market.value,
            "predicted_at":        p.created_at.isoformat(),
            "price_at_prediction": round(p.current_price, 4),
            "predicted_close":     round(p.predicted_close, 4),
            "actual_price":        round(actual_price, 4),
            "error_pct":           pred_error_pct,
            "direction_correct":   direction_correct,
            "confidence":          round((p.confidence or 0) * 100, 1),
        })

    total             = len(results)
    correct_direction = sum(1 for r in results if r["direction_correct"] is True)
    avg_error         = (
        sum(abs(r["error_pct"]) for r in results if r["error_pct"] is not None) / total
        if total > 0 else 0
    )

    return {
        "predictions":        results,
        "total":              total,
        "direction_accuracy": round((correct_direction / total) * 100, 1) if total > 0 else 0,
        "avg_error_pct":      round(avg_error, 2),
        "hours":              hours,
    }

# -------------------------------------------------------
# Equity Curve
# -------------------------------------------------------
@app.get("/api/equity-curve")
def get_equity_curve(
    market: Optional[str] = None,
    days: int = 30,
    db: Session = Depends(get_db),
):
    from database import Portfolio, Trade, Position, MarketType, TradeStatus

    # Cutoff: days=0 means all time
    cutoff = datetime.utcnow() - timedelta(days=days) if days > 0 else datetime(2000, 1, 1)

    if market and market != "all":
        try:
            market_list = [MarketType(market)]
        except ValueError:
            raise HTTPException(400, "Invalid market")
    else:
        market_list = [MarketType.CRYPTO, MarketType.STOCKS, MarketType.FOREX]

    # Load portfolios that exist
    portfolios: dict = {}
    for mkt in market_list:
        p = db.query(Portfolio).filter_by(market=mkt).first()
        if p:
            portfolios[mkt] = p

    if not portfolios:
        return {"points": [], "starting_balance": 0, "max_drawdown": 0,
                "best_trade": 0, "worst_trade": 0}

    total_starting = sum(p.starting_balance for p in portfolios.values())

    # Gather all closed trades in the window, tagged with their market
    all_closed: list = []
    best_trade_val  = None
    worst_trade_val = None

    for mkt, portfolio in portfolios.items():
        trades = (
            db.query(Trade)
            .filter(
                Trade.portfolio_id == portfolio.id,
                Trade.status       == TradeStatus.CLOSED,
                Trade.closed_at    != None,
                Trade.closed_at    >= cutoff,
            )
            .all()
        )
        for t in trades:
            all_closed.append((t, mkt))
            pnl = t.realized_pnl or 0
            if best_trade_val is None or pnl > best_trade_val:
                best_trade_val = pnl
            if worst_trade_val is None or pnl < worst_trade_val:
                worst_trade_val = pnl

    # Sort all closed trades chronologically
    all_closed.sort(key=lambda x: x[0].closed_at)

    # Running realized PnL per market
    running_pnl: dict = {mkt: 0.0 for mkt in portfolios}

    # Build the curve
    points = []

    # Anchor point at the earliest portfolio creation
    earliest_ts = min(p.created_at for p in portfolios.values())
    points.append({
        "time":    earliest_ts.isoformat() + "Z",
        "equity":  round(total_starting, 2),
        "pnl":     0.0,
        "pnl_pct": 0.0,
    })

    for trade, mkt in all_closed:
        running_pnl[mkt] += trade.realized_pnl or 0
        total_equity = sum(
            portfolios[m].starting_balance + running_pnl[m]
            for m in portfolios
        )
        total_pnl = total_equity - total_starting
        points.append({
            "time":    trade.closed_at.isoformat() + "Z",
            "equity":  round(total_equity, 2),
            "pnl":     round(total_pnl, 2),
            "pnl_pct": round((total_pnl / total_starting) * 100, 2) if total_starting else 0,
        })

    # Final "now" point using live portfolio values (cash + open position mark-to-market)
    live_equity = 0.0
    for mkt, portfolio in portfolios.items():
        positions = db.query(Position).filter(
            Position.portfolio_id == portfolio.id,
            Position.quantity > 0,
        ).all()
        live_equity += portfolio.cash_balance + sum(
            pos.current_price * pos.quantity for pos in positions
        )
    live_pnl = live_equity - total_starting
    points.append({
        "time":    datetime.utcnow().isoformat() + "Z",
        "equity":  round(live_equity, 2),
        "pnl":     round(live_pnl, 2),
        "pnl_pct": round((live_pnl / total_starting) * 100, 2) if total_starting else 0,
    })

    # Max drawdown: peak-to-trough percentage
    max_dd = 0.0
    peak   = points[0]["equity"]
    for pt in points:
        if pt["equity"] > peak:
            peak = pt["equity"]
        if peak > 0:
            dd = (peak - pt["equity"]) / peak * 100
            if dd > max_dd:
                max_dd = dd

    return {
        "points":           points,
        "starting_balance": round(total_starting, 2),
        "max_drawdown":     round(max_dd, 2),
        "best_trade":       round(best_trade_val, 2) if best_trade_val is not None else 0,
        "worst_trade":      round(worst_trade_val, 2) if worst_trade_val is not None else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib

STATIC_DIR = pathlib.Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_dashboard():
        return FileResponse(str(STATIC_DIR / "index.html"))
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib

STATIC_DIR = pathlib.Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_dashboard():
        return FileResponse(str(STATIC_DIR / "index.html"))
