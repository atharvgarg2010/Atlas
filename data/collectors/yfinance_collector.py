"""
data/collectors/yfinance_collector.py
=======================================
Yahoo Finance Data Collector — Sprint 2

Purpose:
    Fetch OHLCV historical and live data for NSE symbols using yfinance.
    Abstracts all yfinance API details so the rest of the codebase never
    imports yfinance directly.

Inputs:
    symbol:   NSE ticker with .NS suffix (e.g. 'RELIANCE.NS')
    period:   yfinance period string (e.g. '1y', '6mo')
    interval: yfinance interval string (e.g. '1d', '1h')

Outputs:
    List of OHLCV dicts with standardised keys: open/high/low/close/volume/ts

Dependencies:
    yfinance, pandas, core.decorators.retry

Failure Scenarios:
    - Empty DataFrame returned → log WARNING, return []
    - HTTP 429 rate limit → @retry with exponential backoff
    - Symbol not found → log ERROR, return []
    - Network timeout → @retry
"""

from __future__ import annotations

# TODO (Sprint 2): Implement YFinanceCollector.
#
# class YFinanceCollector:
#     @retry(max_attempts=3, backoff_seconds=2.0, exceptions=(Exception,))
#     def fetch_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> list[OHLCV]:
#         """Fetch historical OHLCV data. Returns [] on any failure."""
#
#     def fetch_latest(self, symbol: str) -> OHLCV | None:
#         """Fetch the most recent candle for a symbol."""
