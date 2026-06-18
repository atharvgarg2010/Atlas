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
