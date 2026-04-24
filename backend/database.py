"""
Database models for Kronos Trader
"""
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Boolean, ForeignKey, Text, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import enum
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./kronos_trader.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class MarketType(str, enum.Enum):
    CRYPTO = "crypto"
    STOCKS = "stocks"
    FOREX  = "forex"


class TradeDirection(str, enum.Enum):
    BUY  = "buy"
    SELL = "sell"


class TradeStatus(str, enum.Enum):
    OPEN   = "open"
    CLOSED = "closed"


# -------------------------------------------------------
# Portfolio — one per market
# -------------------------------------------------------
class Portfolio(Base):
    __tablename__ = "portfolios"

    id             = Column(Integer, primary_key=True)
    market         = Column(Enum(MarketType), unique=True, nullable=False)
    cash_balance   = Column(Float, nullable=False)          # available USD
    starting_balance = Column(Float, nullable=False)
    total_pnl      = Column(Float, default=0.0)
    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    positions = relationship("Position", back_populates="portfolio")
    trades    = relationship("Trade",    back_populates="portfolio")


# -------------------------------------------------------
# Position — current open holdings per symbol
# -------------------------------------------------------
class Position(Base):
    __tablename__ = "positions"

    id           = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    symbol       = Column(String(20), nullable=False)
    quantity     = Column(Float, default=0.0)
    avg_price    = Column(Float, default=0.0)
    current_price = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="positions")


# -------------------------------------------------------
# Trade — every buy/sell action
# -------------------------------------------------------
class Trade(Base):
    __tablename__ = "trades"

    id           = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    symbol       = Column(String(20), nullable=False)
    direction    = Column(Enum(TradeDirection), nullable=False)
    quantity     = Column(Float, nullable=False)
    price        = Column(Float, nullable=False)
    total_value  = Column(Float, nullable=False)
    status       = Column(Enum(TradeStatus), default=TradeStatus.OPEN)
    realized_pnl = Column(Float, default=0.0)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    opened_at    = Column(DateTime, default=datetime.utcnow)
    closed_at    = Column(DateTime, nullable=True)
    notes        = Column(Text, nullable=True)

    portfolio  = relationship("Portfolio", back_populates="trades")
    prediction = relationship("Prediction")


# -------------------------------------------------------
# Prediction — Kronos model output per run
# -------------------------------------------------------
class Prediction(Base):
    __tablename__ = "predictions"

    id             = Column(Integer, primary_key=True)
    symbol         = Column(String(20), nullable=False)
    market         = Column(Enum(MarketType), nullable=False)
    current_price  = Column(Float, nullable=False)
    predicted_high = Column(Float, nullable=False)
    predicted_low  = Column(Float, nullable=False)
    predicted_close = Column(Float, nullable=False)
    confidence     = Column(Float, nullable=True)        # % chance price goes up
    horizon_hours  = Column(Integer, default=24)
    triggered_by   = Column(String(20), default="scheduler")  # 'scheduler' | 'manual'
    model_name     = Column(String(20), default="small")      # 'small' | 'mini'
    raw_output     = Column(Text, nullable=True)             # JSON candles
    created_at     = Column(DateTime, default=datetime.utcnow)


# -------------------------------------------------------
# PriceSnapshot — latest market prices
# -------------------------------------------------------
class PriceSnapshot(Base):
    __tablename__ = "price_snapshots"

    id         = Column(Integer, primary_key=True)
    symbol     = Column(String(20), nullable=False)
    market     = Column(Enum(MarketType), nullable=False)
    price      = Column(Float, nullable=False)
    volume_24h = Column(Float, nullable=True)
    change_24h = Column(Float, nullable=True)   # percent
    fetched_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables and seed portfolios if they don't exist."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for market, balance_env in [
            (MarketType.CRYPTO, "PAPER_BALANCE_CRYPTO"),
            (MarketType.STOCKS, "PAPER_BALANCE_STOCKS"),
            (MarketType.FOREX,  "PAPER_BALANCE_FOREX"),
        ]:
            existing = db.query(Portfolio).filter_by(market=market).first()
            if not existing:
                bal = float(os.getenv(balance_env, 1000.0))
                db.add(Portfolio(
                    market=market,
                    cash_balance=bal,
                    starting_balance=bal
                ))
        db.commit()
        print("✅ Database initialized with paper trading portfolios.")
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
