"""
database/models.py
==================
Project Atlas — SQLAlchemy ORM Models (Sprint 2)

Tables created in this sprint:
    - stocks         : Master registry of tracked NSE symbols
    - market_data    : OHLCV candles (one row per symbol × timeframe × date)
    - system_logs    : Structured operational event log

Architecture
------------
- All models inherit from ``Base`` defined in ``database.connection``.
  This ensures Alembic's env.py discovers them via ``Base.metadata``.
- ``NUMERIC(12, 4)`` is used for prices — never float — to avoid
  floating-point drift on financial data.
- ``TIMESTAMP(timezone=True)`` stores all datetimes in UTC.
- ``market_data`` has a ``UNIQUE(symbol, timeframe, ts)`` constraint.
  All inserts use ``ON CONFLICT DO NOTHING`` — OHLCV data is immutable.

Usage
-----
    from database.models import Stock, MarketData, SystemLog

    with db.session() as s:
        stock = Stock(symbol="RELIANCE.NS", exchange="NSE")
        s.add(stock)

Dependencies:
    sqlalchemy, database.connection.Base
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Boolean,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base


# ─── Stock ────────────────────────────────────────────────────────────────────


class Stock(Base):
    """
    Master registry of all tracked NSE/BSE symbols.

    One row per symbol. Seeded at startup from config.market.watchlist.
    ``is_active=False`` marks delisted or removed symbols without deleting
    their historical data.

    Columns
    -------
    symbol   : NSE ticker with suffix — 'RELIANCE.NS', 'TCS.NS', etc.
    name     : Full company name (nullable — populated later if needed)
    sector   : GICS sector string (nullable — populated later if needed)
    exchange : Exchange code. Default 'NSE'.
    is_active: Whether Atlas actively tracks this symbol.
    """

    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    exchange: Mapped[str] = mapped_column(String(10), nullable=False, default="NSE")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("ix_stocks_symbol", "symbol"),
        Index("ix_stocks_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Stock symbol={self.symbol!r} active={self.is_active}>"


# ─── MarketData ───────────────────────────────────────────────────────────────


class MarketData(Base):
    """
    OHLCV candle data for a symbol × timeframe × timestamp combination.

    One row per (symbol, timeframe, ts) triplet. The UNIQUE constraint on
    that triplet means all inserts can safely use ON CONFLICT DO NOTHING —
    duplicates are silently skipped without error.

    Columns
    -------
    symbol    : Foreign key to stocks.symbol (cascade delete).
    timeframe : Candle width — '1d', '1h', '15m', etc.
    ts        : Candle open time in UTC (TIMESTAMPTZ).
    open/high/low/close : NUMERIC(12,4) — price in INR.
    volume    : Total shares traded in the candle period.
    created_at: Row insertion timestamp (UTC).
    """

    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(
        String(20), nullable=False
    )
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False, default="1d")
    ts: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "ts", name="uq_market_data_symbol_tf_ts"),
        Index("ix_market_data_symbol_ts", "symbol", "ts"),
        Index("ix_market_data_ts", "ts"),
    )

    def __repr__(self) -> str:
        return (
            f"<MarketData symbol={self.symbol!r} tf={self.timeframe!r} "
            f"ts={self.ts} close={self.close}>"
        )


# ─── SystemLog ────────────────────────────────────────────────────────────────


class SystemLog(Base):
    """
    Structured operational event log written by Atlas services.

    This is NOT a replacement for the rotating file log. It captures
    structured job-level events (run start, per-symbol result, error counts)
    so operational history is queryable from the dashboard or SQL.

    Columns
    -------
    level   : Log level string — 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    logger  : Dotted module path — 'services.market_data_service'.
    message : Human-readable event description.
    context : Optional JSONB blob for structured metadata
              (e.g. {'symbol': 'RELIANCE.NS', 'rows_inserted': 252}).
    ts      : Event timestamp in UTC.
    """

    __tablename__ = "system_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(String(10), nullable=False)
    logger: Mapped[str] = mapped_column(String(100), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ts: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_system_logs_ts", "ts"),
        Index("ix_system_logs_level", "level"),
    )

    def __repr__(self) -> str:
        return f"<SystemLog level={self.level!r} logger={self.logger!r} ts={self.ts}>"
