"""
database/models/factors.py
==========================
Project Atlas — Factor Engine Models

Defines the SQLAlchemy schema for storing historical factor rankings
and raw factor scores.
"""

from datetime import date, datetime

from sqlalchemy import (
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


class FactorRanking(Base):
    """
    Historical record of cross-sectional factor rankings.
    """
    __tablename__ = "factor_rankings"
    __table_args__ = (
        UniqueConstraint("ranking_date", "symbol", name="uq_factor_ranking_date_symbol"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(50), index=True)
    ranking_date: Mapped[date] = mapped_column(Date, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    
    universe_size: Mapped[int] = mapped_column(Integer)
    top_n: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Raw Factor Values (Before Normalization)
    momentum_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    trend_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    rs_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity_raw: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Normalized Factor Scores (0-100)
    momentum_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    trend_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rs_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Final Composite
    composite_score: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
