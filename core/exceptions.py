"""
core/exceptions.py
==================
Project Atlas — Custom Exception Hierarchy

All exceptions raised by Atlas services inherit from AtlasError, making
it trivial to catch any platform-level error at the boundary.

Purpose:
    Provide typed, layered exceptions for every failure domain so that
    callers can handle specific cases without catching bare Exception.

Usage:
    from core.exceptions import MarketDataError

    raise MarketDataError("yfinance returned empty response for RELIANCE.NS")
"""

from __future__ import annotations


class AtlasError(Exception):
    """Base exception for all Project Atlas errors."""


# ─── Configuration ────────────────────────────────────────────────────────────


class ConfigurationError(AtlasError):
    """Raised when configuration is invalid, missing, or inconsistent."""


# ─── Database ─────────────────────────────────────────────────────────────────


class DatabaseError(AtlasError):
    """Raised when a database operation fails."""


class DatabaseConnectionError(DatabaseError):
    """Raised when the database is unreachable or authentication fails."""


# ─── Data Fetching ────────────────────────────────────────────────────────────


class DataFetchError(AtlasError):
    """Raised when external data cannot be retrieved after retries."""


class MarketDataError(DataFetchError):
    """Raised on market data (OHLCV) fetch or parsing failures."""


class NewsDataError(DataFetchError):
    """Raised on news data fetch or parsing failures."""


class DataValidationError(AtlasError):
    """Raised when fetched data fails schema or value validation."""


# ─── Analysis ─────────────────────────────────────────────────────────────────


class IndicatorError(AtlasError):
    """Raised when technical indicator calculation fails."""


class ScoringError(AtlasError):
    """Raised when composite stock scoring computation fails."""


class SignalGenerationError(AtlasError):
    """Raised when signal generation logic encounters an error."""


# ─── Backtesting & Paper Trading ──────────────────────────────────────────────


class BacktestError(AtlasError):
    """Raised when a backtest run fails or produces invalid results."""


class PaperTradingError(AtlasError):
    """Raised when a paper trade cannot be executed or tracked."""


# ─── Scheduler ────────────────────────────────────────────────────────────────


class SchedulerError(AtlasError):
    """Raised when the scheduler encounters an unrecoverable error."""
