"""
Discord Webhook Notifications for Kronos Trader.
Sends rich embeds when paper trades are opened or closed.

All public functions are fire-and-forget (sync) — they never raise
exceptions or block the trading engine. send_test_message() is the
only awaitable, used by the test API route.
"""
import asyncio
import json
import logging
import os
import pathlib
import threading
from datetime import datetime, timezone

import aiohttp

logger = logging.getLogger(__name__)

_SETTINGS_FILE = pathlib.Path(__file__).parent / "settings.json"

# Embed colours (decimal) — match the dashboard CSS variables
_COLOR_GREEN = 0x00E5A0   # --success (buy / profitable close)
_COLOR_RED   = 0xFF3B5C   # --danger  (loss close)
_COLOR_BLUE  = 0x0088FF   # --accent2 (info / test)


# -------------------------------------------------------
# Config helpers
# -------------------------------------------------------

def _load_discord_config() -> tuple:
    """
    Return (webhook_url: str, enabled: bool).
    Settings tab values in settings.json take priority; env vars are fallback.
    """
    try:
        if _SETTINGS_FILE.exists():
            s = json.loads(_SETTINGS_FILE.read_text())
            notif = s.get("notifications", {})
            url     = notif.get("discord_webhook_url") or os.getenv("DISCORD_WEBHOOK_URL", "")
            enabled = notif.get("discord_enabled",
                                os.getenv("DISCORD_NOTIFICATIONS_ENABLED", "false").lower() == "true")
            return url.strip(), bool(enabled)
    except Exception:
        pass
    return (
        os.getenv("DISCORD_WEBHOOK_URL", "").strip(),
        os.getenv("DISCORD_NOTIFICATIONS_ENABLED", "false").lower() == "true",
    )


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _footer() -> dict:
    return {"text": f"Kronos Trader  •  Raspberry Pi  |  {_now_str()}"}


# -------------------------------------------------------
# Core HTTP call
# -------------------------------------------------------

async def _post(payload: dict) -> bool:
    """POST a webhook payload to Discord. Returns True on success."""
    url, enabled = _load_discord_config()
    if not enabled or not url:
        return False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    logger.warning(f"Discord webhook {resp.status}: {body[:200]}")
                    return False
                return True
    except Exception as exc:
        logger.warning(f"Discord notification failed: {exc}")
        return False


def _fire(coro) -> None:
    """
    Submit a coroutine as a daemon background thread so the trading
    engine is never blocked or crashed by a slow / failed Discord call.
    """
    def _run():
        try:
            asyncio.run(coro)
        except Exception as exc:
            logger.warning(f"Discord background task error: {exc}")

    threading.Thread(target=_run, daemon=True).start()


# -------------------------------------------------------
# Public notification functions  (sync, fire-and-forget)
# -------------------------------------------------------

def send_trade_opened(
    symbol: str,
    market: str,
    direction: str,
    quantity: float,
    price: float,
    trade_value: float,
    confidence: float,  # 0-1 float
) -> None:
    """Send a 'trade opened' embed. Non-blocking."""
    async def _send():
        conf_pct = confidence * 100 if confidence <= 1.0 else confidence
        payload = {
            "embeds": [{
                "title":       f"📈  Trade Opened — {symbol}",
                "color":       _COLOR_GREEN,
                "description": f"**{market.upper()}**  ·  {direction.upper()}",
                "fields": [
                    {"name": "Symbol",      "value": symbol,                  "inline": True},
                    {"name": "Direction",   "value": direction.upper(),        "inline": True},
                    {"name": "Confidence",  "value": f"{conf_pct:.1f}% bull", "inline": True},
                    {"name": "Entry Price", "value": f"${price:,.4f}",        "inline": True},
                    {"name": "Quantity",    "value": f"{quantity:.6f}",       "inline": True},
                    {"name": "Trade Value", "value": f"${trade_value:,.2f}",  "inline": True},
                ],
                "footer":    _footer(),
                "timestamp": _iso_now(),
            }]
        }
        await _post(payload)

    _fire(_send())


def send_trade_closed(
    symbol: str,
    market: str,
    quantity: float,
    entry_price: float,
    exit_price: float,
    realized_pnl: float,
    reason: str,
) -> None:
    """Send a 'trade closed' embed. Non-blocking."""
    async def _send():
        pnl_pct  = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0
        pnl_str  = f"+${realized_pnl:,.2f}" if realized_pnl >= 0 else f"-${abs(realized_pnl):,.2f}"
        pct_str  = f"+{pnl_pct:.2f}%" if pnl_pct >= 0 else f"{pnl_pct:.2f}%"
        color    = _COLOR_GREEN if realized_pnl >= 0 else _COLOR_RED

        payload = {
            "embeds": [{
                "title":       f"📉  Trade Closed — {symbol}",
                "color":       color,
                "description": f"**{pnl_str}** ({pct_str})",
                "fields": [
                    {"name": "Symbol",       "value": symbol,                 "inline": True},
                    {"name": "Market",       "value": market.upper(),         "inline": True},
                    {"name": "Reason",       "value": reason,                 "inline": True},
                    {"name": "Entry Price",  "value": f"${entry_price:,.4f}", "inline": True},
                    {"name": "Exit Price",   "value": f"${exit_price:,.4f}",  "inline": True},
                    {"name": "Quantity",     "value": f"{quantity:.6f}",      "inline": True},
                    {"name": "Realized PnL", "value": pnl_str,               "inline": True},
                    {"name": "Return",       "value": pct_str,               "inline": True},
                ],
                "footer":    _footer(),
                "timestamp": _iso_now(),
            }]
        }
        await _post(payload)

    _fire(_send())


def send_prediction_summary(predictions: list) -> None:
    """
    Send a summary embed for a completed prediction cycle.
    predictions: list of dicts with keys symbol, confidence, current_price.
    Non-blocking.
    """
    if not predictions:
        return

    async def _send():
        lines = []
        for p in predictions[:10]:   # Discord description limit: ~2000 chars
            conf = p.get("confidence", 0)
            if conf <= 1.0:
                conf *= 100
            arrow = "▲" if conf >= 55 else "▼"
            lines.append(
                f"`{p['symbol']:<12}` {arrow} **{conf:.1f}%** bull"
                f"  @  ${p.get('current_price', 0):,.4f}"
            )
        if len(predictions) > 10:
            lines.append(f"*… and {len(predictions) - 10} more*")

        payload = {
            "embeds": [{
                "title":       f"🤖  Kronos Prediction Run — {len(predictions)} symbol(s)",
                "color":       _COLOR_BLUE,
                "description": "\n".join(lines),
                "footer":      _footer(),
                "timestamp":   _iso_now(),
            }]
        }
        await _post(payload)

    _fire(_send())


# -------------------------------------------------------
# Awaitable — used by the test API route
# -------------------------------------------------------

async def send_test_message() -> bool:
    """Send a test embed. Awaitable — call with `await` from an async route."""
    url, _ = _load_discord_config()
    payload = {
        "embeds": [{
            "title":       "✅  Kronos Trader — Test Notification",
            "color":       _COLOR_BLUE,
            "description": "Discord notifications are configured correctly.",
            "fields": [
                {"name": "Status",  "value": "🟢  Connected",     "inline": True},
                {"name": "Source",  "value": "Raspberry Pi 5",    "inline": True},
                {"name": "Webhook", "value": f"`…{url[-20:]}`" if len(url) > 20 else url,
                 "inline": True},
            ],
            "footer":    _footer(),
            "timestamp": _iso_now(),
        }]
    }
    return await _post(payload)