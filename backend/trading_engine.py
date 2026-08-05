"""
Paper Trading Engine
Simulates trades based on Kronos predictions.
Manages portfolios, positions, and PnL tracking.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session

from database import (
    Portfolio, Position, Trade, Prediction,
    MarketType, TradeDirection, TradeStatus, SessionLocal
)
import notifications
from cost_model import apply_costs as _apply_costs

logger = logging.getLogger(__name__)

# How much of available cash to risk per trade (15%)
POSITION_SIZE_PCT = 0.15
# Minimum confidence threshold to take a trade
MIN_CONFIDENCE = 0.75
# Stop loss percentage
STOP_LOSS_PCT = 0.03
# Take profit percentage
TAKE_PROFIT_PCT = 0.06
# Minimum absolute predicted price move to enter a trade (2%)
MIN_PREDICTED_MOVE_PCT = 0.02


class PaperTradingEngine:

    def __init__(self):
        self._cooldowns = {}  # symbol -> datetime of last close

    # ---------------------------------------------------
    # Core: process a new prediction and decide to trade
    # ---------------------------------------------------
    def process_prediction(self, prediction_result: dict, market: str, db: Session,
                           active_model: str = "small") -> dict:
        """
        Given a Kronos prediction, decide whether to open/close a trade.
        Always saves the prediction to DB; only executes trades when
        prediction_result['model_name'] matches active_model.
        Returns a summary of what action was taken.
        """
        symbol     = prediction_result["symbol"]
        confidence = prediction_result["confidence"]
        current    = prediction_result["current_price"]
        pred_close = prediction_result["predicted_close"]
        market_type = MarketType(market)
        pred_model  = prediction_result.get("model_name", "small")

        # Save prediction to DB (always, for both models)
        pred = Prediction(
            symbol          = symbol,
            market          = market_type,
            current_price   = current,
            predicted_high  = prediction_result["predicted_high"],
            predicted_low   = prediction_result["predicted_low"],
            predicted_close = pred_close,
            confidence      = confidence,
            horizon_hours   = prediction_result["horizon_hours"],
            triggered_by    = prediction_result.get("triggered_by", "scheduler"),
            model_name      = pred_model,
            raw_output      = prediction_result.get("raw_output"),
        )
        db.add(pred)
        db.flush()  # get pred.id

        # Only trade based on the active model
        if pred_model != active_model:
            db.commit()
            logger.info(f"[{pred_model}] {symbol} prediction saved (inactive model — no trade)")
            return {"action": "none", "reason": f"inactive_model ({pred_model})"}

        portfolio = db.query(Portfolio).filter_by(market=market_type).first()
        if not portfolio:
            logger.error(f"No portfolio found for market {market}")
            return {"action": "none", "reason": "no_portfolio"}

        # --- Check for existing position to close ---
        existing_pos = db.query(Position).filter_by(
            portfolio_id=portfolio.id, symbol=symbol
        ).first()

        action_taken = {}
        just_closed = False

        if existing_pos and existing_pos.quantity > 0:
            action_taken = self._maybe_close_position(
                existing_pos, portfolio, pred, current, confidence, db
            )
            if action_taken.get("action") == "closed":
                just_closed = True

        # FIX B — if we just closed this cycle, return immediately without re-entering
        if just_closed:
            db.commit()
            self._recalculate_portfolio_pnl(portfolio.id, db)
            return {
                **action_taken,
                "symbol":          symbol,
                "confidence":      confidence,
                "current_price":   current,
                "predicted_close": pred_close,
            }

        # FIX A — cooldown: block re-entry for 1 full prediction cycle after any close
        cooldown_seconds = 3600
        if symbol in self._cooldowns:
            elapsed = (datetime.utcnow() - self._cooldowns[symbol]).total_seconds()
            if elapsed < cooldown_seconds:
                logger.info(
                    f"⏳ {symbol} cooldown active ({int(elapsed//60)}m elapsed, "
                    f"{int((cooldown_seconds-elapsed)//60)}m remaining)"
                )
                db.commit()
                return {"action": "none", "reason": "cooldown"}

        # --- Consider opening a new position ---
        # Gate on minimum predicted price move to filter weak signals
        predicted_move_pct = abs(pred_close - current) / current if current else 0.0

        if confidence >= MIN_CONFIDENCE:
            # FIX C — block duplicate open positions via fresh DB query
            existing = db.query(Position).filter(
                Position.portfolio_id == portfolio.id,
                Position.symbol == symbol,
                Position.quantity > 0
            ).first()
            if existing:
                logger.info(f"⚠️ {symbol} already has open position, skipping new entry")
                db.commit()
                return {"action": "none", "reason": "already_open"}

            # FILTER A — dual-model agreement gate
            try:
                from main import load_settings
                settings = load_settings()
                require_agreement = settings.get("models", {}).get("require_agreement", True)
            except ImportError:
                require_agreement = True

            if require_agreement:
                other_model = 'mini' if pred_model == 'small' else 'small'
                cutoff = datetime.utcnow() - timedelta(minutes=5)
                other_pred = db.query(Prediction).filter(
                    Prediction.symbol == symbol,
                    Prediction.model_name == other_model,
                    Prediction.created_at >= cutoff
                ).order_by(Prediction.created_at.desc()).first()

                if other_pred:
                    other_bullish = other_pred.predicted_close > other_pred.current_price
                    this_bullish  = pred.predicted_close > pred.current_price
                    if other_bullish != this_bullish:
                        logger.info(
                            f"⚖️ {symbol} models disagree "
                            f"({pred_model}={'UP' if this_bullish else 'DOWN'}, "
                            f"{other_model}={'UP' if other_bullish else 'DOWN'}) — skipping"
                        )
                        db.commit()
                        return {"action": "none", "reason": "model_disagreement"}

            # Bullish signal — open position
            if predicted_move_pct < MIN_PREDICTED_MOVE_PCT:
                logger.info(
                    f"Skipping {symbol} — predicted move {predicted_move_pct:.1%} "
                    f"below minimum {MIN_PREDICTED_MOVE_PCT:.1%}"
                )
                action_taken = {"action": "none", "reason": "predicted_move_too_small"}
            else:
                action_taken = self._open_position(
                    symbol, portfolio, pred, current, TradeDirection.BUY, db
                )
        elif confidence <= (1 - MIN_CONFIDENCE):
            # Bearish — skip shorts in paper trading
            logger.info(f"Bearish signal for {symbol} but skipping short (confidence={confidence:.1%})")
            action_taken = {"action": "none", "reason": "bearish_no_short"}
        else:
            action_taken = {
                "action": "none",
                "reason": f"ambiguous_signal ({confidence:.1%})"
            }

        db.commit()

        # Update portfolio PnL
        self._recalculate_portfolio_pnl(portfolio.id, db)

        return {
            **action_taken,
            "symbol":     symbol,
            "confidence": confidence,
            "current_price": current,
            "predicted_close": pred_close,
        }

    # ---------------------------------------------------
    def _open_position(
        self, symbol: str, portfolio: Portfolio,
        pred: Prediction, price: float,
        direction: TradeDirection, db: Session
    ) -> dict:
        # Position sizing: risk POSITION_SIZE_PCT of cash
        trade_value = portfolio.cash_balance * POSITION_SIZE_PCT
        if trade_value < 1.0:
            return {"action": "none", "reason": "insufficient_cash"}

        quantity = trade_value / price
        portfolio.cash_balance -= trade_value

        # Upsert position
        pos = db.query(Position).filter_by(
            portfolio_id=portfolio.id, symbol=symbol
        ).first()
        if pos:
            # Average in
            total_qty   = pos.quantity + quantity
            pos.avg_price = (pos.avg_price * pos.quantity + price * quantity) / total_qty
            pos.quantity  = total_qty
        else:
            pos = Position(
                portfolio_id  = portfolio.id,
                symbol        = symbol,
                quantity      = quantity,
                avg_price     = price,
                current_price = price,
            )
            db.add(pos)

        trade = Trade(
            portfolio_id  = portfolio.id,
            symbol        = symbol,
            direction     = direction,
            quantity      = quantity,
            price         = price,
            total_value   = trade_value,
            status        = TradeStatus.OPEN,
            prediction_id = pred.id,
            notes         = f"Opened by Kronos | confidence={pred.confidence:.1%}",
        )
        db.add(trade)

        logger.info(f"📈 OPENED {direction.value.upper()} {quantity:.6f} {symbol} @ {price:.4f} (${trade_value:.2f})")
        notifications.send_trade_opened(
            symbol=symbol,
            market=portfolio.market.value,
            direction=direction.value,
            quantity=quantity,
            price=price,
            trade_value=trade_value,
            confidence=pred.confidence or 0.0,
        )
        return {"action": "opened", "direction": direction.value, "quantity": quantity, "value": trade_value}

    # ---------------------------------------------------
    def _maybe_close_position(
        self, pos: Position, portfolio: Portfolio,
        pred: Prediction, current_price: float,
        confidence: float, db: Session
    ) -> dict:
        pos.current_price = current_price
        pnl_pct = (current_price - pos.avg_price) / pos.avg_price

        should_close = False
        reason = ""

        if pnl_pct <= -STOP_LOSS_PCT:
            should_close = True
            reason = f"stop_loss ({pnl_pct:.1%})"
        elif pnl_pct >= TAKE_PROFIT_PCT:
            should_close = True
            reason = f"take_profit ({pnl_pct:.1%})"
        elif confidence <= (1 - MIN_CONFIDENCE):
            should_close = True
            reason = f"bearish_signal (confidence={confidence:.1%})"

        if should_close:
            return self._close_position(pos, portfolio, pred, current_price, reason, db)

        pos.unrealized_pnl = (current_price - pos.avg_price) * pos.quantity
        return {"action": "holding", "unrealized_pnl": pos.unrealized_pnl}

    # ---------------------------------------------------
    def _close_position(
        self, pos: Position, portfolio: Portfolio,
        pred: Prediction, price: float, reason: str, db: Session
    ) -> dict:
        realized_pnl = (price - pos.avg_price) * pos.quantity
        proceeds     = price * pos.quantity
        portfolio.cash_balance += proceeds
        portfolio.total_pnl    += realized_pnl

        costs = _apply_costs(
            market         = portfolio.market.value,
            symbol         = pos.symbol,
            side           = "buy",
            entry_price    = pos.avg_price,
            exit_price     = price,
            position_size  = pos.quantity,
        )

        trade = Trade(
            portfolio_id  = portfolio.id,
            symbol        = pos.symbol,
            direction     = TradeDirection.SELL,
            quantity      = pos.quantity,
            price         = price,
            total_value   = proceeds,
            status        = TradeStatus.CLOSED,
            realized_pnl  = realized_pnl,
            gross_pnl     = costs["gross_pnl"],
            entry_fee     = costs["entry_fee"],
            exit_fee      = costs["exit_fee"],
            entry_slippage = costs["entry_slippage"],
            exit_slippage  = costs["exit_slippage"],
            net_pnl       = costs["net_pnl"],
            cost_model_ver = costs["cost_model_ver"],
            prediction_id = pred.id,
            closed_at     = datetime.utcnow(),
            notes         = f"Closed: {reason}",
        )
        db.add(trade)

        # Close the original BUY trade entry
        open_buy = db.query(Trade).filter(
            Trade.portfolio_id == portfolio.id,
            Trade.symbol == pos.symbol,
            Trade.direction == TradeDirection.BUY,
            Trade.status == TradeStatus.OPEN
            ).order_by(Trade.opened_at.desc()).first()
        if open_buy:
            open_buy.status = TradeStatus.CLOSED
            open_buy.closed_at = datetime.utcnow()

        logger.info(f"📉 CLOSED {pos.symbol} @ {price:.4f} | PnL: ${realized_pnl:.2f} | Reason: {reason}")
        notifications.send_trade_closed(
            symbol=pos.symbol,
            market=portfolio.market.value,
            quantity=pos.quantity,
            entry_price=pos.avg_price,
            exit_price=price,
            realized_pnl=realized_pnl,
            reason=reason,
        )

        pos.quantity      = 0.0
        pos.unrealized_pnl = 0.0
        self._cooldowns[pos.symbol] = datetime.utcnow()

        return {"action": "closed", "realized_pnl": realized_pnl, "reason": reason}

    # ---------------------------------------------------
    def update_prices(self, prices: list, market: str, db: Session):
        """Update position prices with latest market data."""
        market_type = MarketType(market)
        portfolio   = db.query(Portfolio).filter_by(market=market_type).first()
        if not portfolio:
            return

        price_map = {p["symbol"]: p["price"] for p in prices}
        positions = db.query(Position).filter(
            Position.portfolio_id == portfolio.id,
            Position.quantity > 0
        ).all()

        for pos in positions:
            sym = pos.symbol.replace("/", "")  # BTC/USDT -> BTCUSDT
            price = price_map.get(pos.symbol) or price_map.get(sym)
            if price:
                pos.current_price  = price
                pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity

        db.commit()

    # ---------------------------------------------------
    def _recalculate_portfolio_pnl(self, portfolio_id: int, db: Session):
        portfolio  = db.query(Portfolio).filter_by(id=portfolio_id).first()
        positions  = db.query(Position).filter(
            Position.portfolio_id == portfolio_id,
            Position.quantity > 0
        ).all()
        unrealized = sum(p.unrealized_pnl or 0 for p in positions)
        equity     = portfolio.cash_balance + sum(
            p.current_price * p.quantity for p in positions
        )
        portfolio.total_pnl = equity - portfolio.starting_balance
        db.commit()

    # ---------------------------------------------------
    def get_portfolio_summary(self, market: str, db: Session) -> dict:
        market_type = MarketType(market)
        portfolio   = db.query(Portfolio).filter_by(market=market_type).first()
        if not portfolio:
            return {}

        positions = db.query(Position).filter(
            Position.portfolio_id == portfolio.id,
            Position.quantity > 0
        ).all()

        equity = portfolio.cash_balance + sum(
            p.current_price * p.quantity for p in positions
        )

        return {
            "market":           market,
            "cash_balance":     round(portfolio.cash_balance, 2),
            "starting_balance": round(portfolio.starting_balance, 2),
            "equity":           round(equity, 2),
            "total_pnl":        round(portfolio.total_pnl, 2),
            "total_pnl_pct":    round((portfolio.total_pnl / portfolio.starting_balance) * 100, 2),
            "positions": [
                {
                    "symbol":        p.symbol,
                    "quantity":      round(p.quantity, 6),
                    "avg_price":     round(p.avg_price, 4),
                    "current_price": round(p.current_price, 4),
                    "unrealized_pnl": round(p.unrealized_pnl or 0, 2),
                }
                for p in positions
            ],
        }
