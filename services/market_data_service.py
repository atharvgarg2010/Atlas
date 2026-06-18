"""
services/market_data_service.py
================================
Market Data Collection Service — Sprint 2

Purpose:
    Orchestrate fetching of OHLCV data from Yahoo Finance for all tracked
    symbols and persist it to the market_data table.

Inputs:
    settings.market.watchlist — list of NSE symbols
    settings.market.history_days — how far back to fetch on first run

Outputs:
    Rows upserted into market_data table.

Dependencies:
    data.collectors.yfinance_collector.YFinanceCollector
    repositories.market_data_repository.MarketDataRepository
    core.decorators.retry, core.decorators.timed

Failure Scenarios:
    - yfinance rate limit → @retry with backoff
    - Empty response for symbol → log WARNING, skip symbol, continue
    - DB write failure → log ERROR, re-raise DatabaseError
"""

from __future__ import annotations

# TODO (Sprint 2): Implement MarketDataService.
#
# class MarketDataService:
#     def __init__(self, db: DatabaseManager, settings: Settings) -> None: ...
#
#     @timed
#     def run(self) -> None:
#         """Fetch and store OHLCV data for all watchlist symbols."""
#
#     def _fetch_symbol(self, symbol: str) -> list[OHLCV]:
#         """Fetch history for a single symbol with retry."""
#
#     def _persist(self, symbol: str, candles: list[OHLCV]) -> int:
#         """Upsert candles into market_data. Returns count inserted."""
