"""
services/indicator_service.py
===============================
Technical Indicator Computation Service — Sprint 2

Purpose:
    Read OHLCV data from market_data table, compute all technical indicators
    using the `ta` library, and persist results to indicators table.

Indicators computed:
    Trend:      SMA(20/50/200), EMA(20/50)
    Momentum:   RSI(14), MACD, MACD Signal, MACD Histogram
    Volatility: ATR(14), Bollinger Bands (upper/middle/lower)
    Volume:     VWAP, Volume SMA(20), Relative Volume
    Returns:    Daily Return %, Weekly Return %
"""

from __future__ import annotations

# TODO (Sprint 2): Implement IndicatorService.
