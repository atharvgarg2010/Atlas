"""
database/models/market_data.py
==============================
Project Atlas — Market Data and Indicator Caching Models

Defines the SQLAlchemy schema for the OHLCV warehouse, the pre-calculated
indicator cache, and the symbol sync metadata tracking.
"""

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from database.connection import Base


class SymbolMetadata(Base):
    """
    Tracks the sync status and metadata for a specific symbol.
    Avoids expensive full-table scans when checking cache freshness.
    """
    __tablename__ = "symbol_metadata"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    last_synced: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    first_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    last_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    total_candles: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="ACTIVE")  # ACTIVE, STALE, INVALID


class MarketData(Base):
    """
    Historical OHLCV data.
    """
    __tablename__ = "market_data"
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_market_data_symbol_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MarketIndicators(Base):
    """
    Pre-calculated technical indicators for a given symbol and date.
    Calculated via IndicatorEngine during the daily sync.
    """
    __tablename__ = "market_indicators"
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_market_indicators_symbol_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    
    # Pre-calculated indicators
    ema_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_50: Mapped[float | None] = mapped_column(Float, nullable=True)
    sma_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    rsi_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr_14: Mapped[float | None] = mapped_column(Float, nullable=True)
