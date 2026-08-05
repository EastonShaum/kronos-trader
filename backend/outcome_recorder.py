"""
Outcome Recorder — checks predictions older than 24h and records actual price results.
Called hourly by the scheduler; can also be triggered manually via /api/outcomes/trigger.
"""
import logging
import concurrent.futures
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import Prediction
from data_fetcher import MarketDataFetcher

logger = logging.getLogger(__name__)


def record_outcomes(db: Session) -> int:
    """
    Find predictions older than 24h with no outcome recorded.
    Fetch the actual current price (best proxy for the predicted close)
    and record whether the direction prediction was correct.

    Returns the number of outcomes newly recorded.
    """
    cutoff = datetime.utcnow() - timedelta(hours=24)

    pending = db.query(Prediction).filter(
        Prediction.created_at <= cutoff,
        Prediction.actual_close == None,       # noqa: E711
        Prediction.current_price != None,      # noqa: E711
        Prediction.current_price > 0,
    ).all()

    if not pending:
        logger.info("No pending outcomes to record")
        return 0

    fetcher = MarketDataFetcher()
    recorded = 0
    errors = 0
    price_cache: dict = {}

    for pred in pending:
        key = (pred.market.value if hasattr(pred.market, 'value') else str(pred.market),
               pred.symbol)

        if key not in price_cache:
            market_str = key[0]
            symbol = key[1]
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(fetcher.get_ohlcv, market_str, symbol, 2)
                    df = future.result(timeout=15)
                if df is not None and not df.empty:
                    price_cache[key] = float(df['close'].iloc[-1])
                else:
                    price_cache[key] = None
            except concurrent.futures.TimeoutError:
                logger.warning(f"Timeout fetching price for {symbol} ({market_str}) — skipping")
                price_cache[key] = None
            except Exception as e:
                logger.warning(f"Could not fetch price for {symbol} ({market_str}): {e}")
                price_cache[key] = None

        actual_price = price_cache.get(key)

        if actual_price is None or actual_price <= 0:
            errors += 1
            continue

        try:
            entry_price = pred.current_price
            predicted_close = pred.predicted_close

            actual_pct = ((actual_price - entry_price) / entry_price) * 100
            actual_dir = 'UP' if actual_price > entry_price else 'DOWN'
            predicted_dir = 'UP' if predicted_close > entry_price else 'DOWN'
            correct = (actual_dir == predicted_dir)

            pred.actual_close = round(actual_price, 6)
            pred.actual_direction = actual_dir
            pred.was_correct = correct
            pred.outcome_pct = round(actual_pct, 4)
            pred.outcome_recorded_at = datetime.utcnow()

            recorded += 1
            logger.info(
                f"Outcome recorded: {pred.symbol} [{pred.model_name}] "
                f"predicted={predicted_dir} actual={actual_dir} "
                f"correct={correct} move={actual_pct:.2f}%"
            )
        except Exception as e:
            logger.error(f"Error recording outcome for prediction {pred.id}: {e}")
            errors += 1

    if recorded > 0:
        db.commit()

    logger.info(f"Outcome recorder finished — recorded={recorded} errors={errors}")
    return recorded
