"""
Database models for Kronos Trader
"""
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Boolean, ForeignKey, Text, Enum, UniqueConstraint, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import inspect as sa_inspect
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

    # Cost model columns — populated at close time; NULL on pre-migration rows
    gross_pnl       = Column(Float, nullable=True)
    entry_fee       = Column(Float, nullable=True)
    exit_fee        = Column(Float, nullable=True)
    entry_slippage  = Column(Float, nullable=True)
    exit_slippage   = Column(Float, nullable=True)
    net_pnl         = Column(Float, nullable=True)
    cost_model_ver  = Column(String(10), nullable=True)
    # Entry prediction linkage — for SELL trades, points to the BUY trade's prediction_id
    entry_prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)

    portfolio        = relationship("Portfolio", back_populates="trades")
    prediction       = relationship("Prediction", foreign_keys=[prediction_id])
    entry_prediction = relationship("Prediction", foreign_keys=[entry_prediction_id])


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

    # Outcome columns — populated ~24h after prediction
    actual_close         = Column(Float, nullable=True)      # real price 24h later
    actual_direction     = Column(String(4), nullable=True)  # 'UP' or 'DOWN'
    was_correct          = Column(Boolean, nullable=True)    # direction match
    outcome_pct          = Column(Float, nullable=True)      # actual % move
    outcome_recorded_at  = Column(DateTime, nullable=True)


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


# -------------------------------------------------------
# Candle — persisted OHLCV for benchmark reproducibility
# -------------------------------------------------------
class Candle(Base):
    __tablename__ = "candles"

    id        = Column(Integer, primary_key=True)
    market    = Column(String(20), nullable=False)   # 'crypto' | 'stocks'
    symbol    = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)     # UTC, candle open
    open      = Column(Float)
    high      = Column(Float)
    low       = Column(Float)
    close     = Column(Float)
    volume    = Column(Float)

    __table_args__ = (
        UniqueConstraint("market", "symbol", "timestamp", name="uq_candle_market_symbol_ts"),
    )


# -------------------------------------------------------
# AuditRecommendation — AI-generated trading improvement suggestions
# -------------------------------------------------------
class AuditRecommendation(Base):
    __tablename__ = "audit_recommendations"

    id             = Column(Integer, primary_key=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
    audit_type     = Column(String)          # "weekly" | "monthly" | "diagnostic"
    category       = Column(String)          # "risk_param" | "symbol_list" | "model_strategy"
    param          = Column(String)
    current_value  = Column(String)
    proposed_value = Column(String)
    confidence     = Column(String)          # "low" | "medium" | "high" | "out_of_bounds"
    reasoning      = Column(Text)
    status         = Column(String, default="pending")  # pending|approved|rejected|applied
    applied_at     = Column(DateTime, nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    # Origin of the recommendation — distinguishes scheduled audit jobs from
    # one-off diagnostic sessions that use the same approval queue.
    source         = Column(String, default="scheduled_audit", nullable=True)


def _migrate_cost_columns():
    """Idempotent: add cost columns to trades table if they don't exist."""
    insp = sa_inspect(engine)
    existing = {c["name"] for c in insp.get_columns("trades")}
    new_cols = [
        ("gross_pnl",             "REAL"),
        ("entry_fee",             "REAL"),
        ("exit_fee",              "REAL"),
        ("entry_slippage",        "REAL"),
        ("exit_slippage",         "REAL"),
        ("net_pnl",               "REAL"),
        ("cost_model_ver",        "TEXT"),
        ("entry_prediction_id",   "INTEGER"),
    ]
    with engine.connect() as conn:
        for col_name, col_type in new_cols:
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}"))
        conn.commit()


def _migrate_audit_columns():
    """Idempotent: add source column to audit_recommendations if it doesn't exist."""
    insp = sa_inspect(engine)
    try:
        existing = {c["name"] for c in insp.get_columns("audit_recommendations")}
    except Exception:
        return  # table doesn't exist yet; create_all will handle it
    with engine.connect() as conn:
        if "source" not in existing:
            conn.execute(text(
                "ALTER TABLE audit_recommendations ADD COLUMN source TEXT DEFAULT 'scheduled_audit'"
            ))
        conn.commit()


def init_db():
    """Create all tables, run idempotent migrations, and seed portfolios."""
    Base.metadata.create_all(bind=engine)
    _migrate_cost_columns()
    _migrate_audit_columns()
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
