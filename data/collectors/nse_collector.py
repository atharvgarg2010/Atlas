"""
data/collectors/nse_collector.py
==================================
NSE Fallback Data Collector — Sprint 2

Purpose:
    Secondary data source for NSE stocks when yfinance is unavailable.
    Provides end-of-day data directly from NSE India.

Note:
    Used only as fallback. Primary source remains yfinance.
"""

from __future__ import annotations

# TODO (Sprint 2): Evaluate nsepy / jugaad-trader / NSE REST API.
