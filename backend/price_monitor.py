"""
Price Monitor — intraday stop-loss / take-profit enforcement.
Runs every 3 minutes between prediction cycles so thresholds are
caught when price actually crosses them, not only at prediction time.
"""
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from database import Position, Portfolio, Prediction, SessionLocal
from data_fetcher import MarketDataFetcher

logger = logging.getLogger(__name__)
_last_run_info = {"last_run": None, "checked": 0, "closed": 0}


def run_price_monitor(trading_engine):
    """
    Check all open positions against current prices every 3 minutes.
    Fires stop-loss and take-profit without running Kronos.
    """
    db: Session = SessionLocal()
    checked = 0
    closed = 0

    try:
        from main import load_settings
        settings = load_settings()
        stop_loss_pct   = settings.get("risk", {}).get("stop_loss_pct",   0.03)
        take_profit_pct = settings.get("risk", {}).get("take_profit_pct", 0.06)

        fetcher = MarketDataFetcher()

        # All open positions across every portfolio
        open_positions = db.query(Position).filter(Position.quantity > 0).all()

        for pos in open_positions:
            portfolio = db.query(Portfolio).filter_by(id=pos.portfolio_id).first()
            if not portfolio:
                continue

            market = portfolio.market.value.lower()
            symbol = pos.symbol

            try:
                df = fetcher.get_ohlcv(market, symbol, limit=2)
                if df is None or df.empty:
                    continue

                current_price = float(df["close"].iloc[-1])
                pos.current_price = current_price
                checked += 1

                pct_change = (current_price - pos.avg_price) / pos.avg_price

                if pct_change <= -stop_loss_pct:
                    logger.info(
                        f"🔴 PRICE MONITOR stop-loss: {symbol} "
                        f"@ {current_price:.4f} ({pct_change*100:.2f}%)"
                    )
                    stub_pred = _make_stub(symbol, market, current_price)
                    trading_engine._close_position(
                        pos, portfolio, stub_pred,
                        current_price,
                        f"stop_loss ({pct_change*100:.1f}%)",
                        db,
                    )
                    db.commit()
                    closed += 1

                elif pct_change >= take_profit_pct:
                    logger.info(
                        f"🟢 PRICE MONITOR take-profit: {symbol} "
                        f"@ {current_price:.4f} ({pct_change*100:.2f}%)"
                    )
                    stub_pred = _make_stub(symbol, market, current_price)
                    trading_engine._close_position(
                        pos, portfolio, stub_pred,
                        current_price,
                        f"take_profit ({pct_change*100:.1f}%)",
                        db,
                    )
                    db.commit()
                    closed += 1

                else:
                    pos.unrealized_pnl = (current_price - pos.avg_price) * pos.quantity

            except Exception as e:
                logger.warning(f"Price monitor error for {symbol}: {e}")
                continue

        db.commit()

    except Exception as e:
        logger.error(f"Price monitor fatal error: {e}", exc_info=True)
    finally:
        _last_run_info.update({
            "last_run": datetime.utcnow().isoformat(),
            "checked":  checked,
            "closed":   closed,
        })
        db.close()
        logger.info(f"Price monitor: checked={checked} closed={closed}")


def _make_stub(symbol: str, market: str, price: float) -> Prediction:
    """Minimal Prediction stub so _close_position has something to reference."""
    from database import MarketType
    return Prediction(
        id              = 0,
        symbol          = symbol,
        market          = MarketType(market),
        current_price   = price,
        predicted_close = price,
        predicted_high  = price,
        predicted_low   = price,
        confidence      = 0,
        horizon_hours   = 24,
    )