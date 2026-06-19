"""
core/types.py
=============
Project Atlas — Shared Type Aliases

Centralises all domain-specific TypeAliases so they can be imported
from a single location across services, repositories, and models.

Purpose:
    Improve readability and type-checker precision throughout the codebase.
    A function signature of (symbol: Symbol, price: Price) -> Score is
    self-documenting and eliminates ambiguous bare `str` / `float` params.

Dependencies:
    None — this module must remain import-free to avoid circular deps.
"""

from __future__ import annotations

from typing import TypeAlias

# ─── Market Primitives ────────────────────────────────────────────────────────

Symbol: TypeAlias = str
"""NSE/BSE ticker symbol as used by yfinance. Example: 'RELIANCE.NS'"""

Price: TypeAlias = float
"""A monetary price in INR."""

Score: TypeAlias = float
"""A composite opportunity score in the range [0.0, 100.0]."""

Confidence: TypeAlias = float
"""A confidence percentage in the range [0.0, 100.0]."""

Sentiment: TypeAlias = str
"""Sentiment classification: 'positive' | 'neutral' | 'negative'"""

Timeframe: TypeAlias = str
"""Candle timeframe string. Examples: '1d', '1h', '15m'"""

# ─── Data Structures ──────────────────────────────────────────────────────────

OHLCV: TypeAlias = dict[str, float | int]
"""
Raw OHLCV candle as a plain dict.

Keys: 'open', 'high', 'low', 'close', 'volume'
"""

IndicatorMap: TypeAlias = dict[str, float | None]
"""
Map of indicator name to computed value for a single candle.

Example: {'rsi_14': 58.2, 'macd': 0.45, 'atr_14': 12.3}
"""

ScoringWeightMap: TypeAlias = dict[str, float]
"""
Map of scoring component name to its weight (must sum to 1.0).

Example: {'technical': 0.40, 'momentum': 0.30, 'news': 0.20, 'volume': 0.10}
"""

ReasoningMap: TypeAlias = dict[str, str | float | bool]
"""
Structured reasoning snapshot attached to a signal.

Example: {'rsi': 58.2, 'macd_bullish': True, 'sentiment': 'positive'}
"""

# ─── Ingestion Result ─────────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class SymbolResult:
    """
    Outcome of processing a single symbol during a market data ingestion run.

    Attributes
    ----------
    symbol:          NSE ticker that was processed.
    rows_fetched:    Raw rows returned by yfinance before validation.
    rows_valid:      Rows that passed all validation rules.
    rows_rejected:   Rows that failed validation (dropped with warning).
    rows_inserted:   Net new rows written to market_data (0 if all duplicates).
    error:           Exception message if the symbol failed entirely, else None.
    """

    symbol: str
    rows_fetched: int = 0
    rows_valid: int = 0
    rows_rejected: int = 0
    rows_inserted: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True if the symbol was processed without a fatal error."""
        return self.error is None


@dataclass
class IngestReport:
    """
    Aggregate result of a full MarketDataService.run() call.

    Attributes
    ----------
    total_symbols:   Number of symbols attempted.
    succeeded:       Symbols with ok=True (even if 0 rows inserted).
    failed:          Symbols that raised an unhandled exception.
    total_fetched:   Sum of rows_fetched across all symbols.
    total_inserted:  Sum of rows_inserted across all symbols (net new rows).
    results:         Per-symbol SymbolResult list for detailed inspection.
    """

    total_symbols: int = 0
    succeeded: int = 0
    failed: int = 0
    total_fetched: int = 0
    total_inserted: int = 0
    results: list[SymbolResult] = field(default_factory=list)

    def summary(self) -> str:
        """Return a one-line human-readable summary for logging."""
        return (
            f"symbols={self.total_symbols} ok={self.succeeded} "
            f"failed={self.failed} fetched={self.total_fetched} "
            f"inserted={self.total_inserted}"
        )
