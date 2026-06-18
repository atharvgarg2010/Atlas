"""
repositories/market_data_repository.py
=======================================
Data access layer for the `market_data` table.

Purpose:
    Upsert OHLCV candles, query historical data for a symbol,
    and retrieve the latest available candle per symbol.

Dependencies:
    database.models.MarketData  (added in Sprint 2)

Failure Scenarios:
    - UNIQUE(symbol, timeframe, ts) violation → caught and handled as upsert
    - Large history query → paginate with LIMIT/OFFSET
"""

from __future__ import annotations

# TODO (Sprint 2): Implement when database/models.py is created.
#
# class MarketDataRepository:
#     def upsert_candle(self, symbol: str, timeframe: str, candle: OHLCV) -> None: ...
#     def get_history(self, symbol: str, days: int) -> list[MarketData]: ...
#     def get_latest(self, symbol: str) -> MarketData | None: ...
#     def get_symbols_needing_refresh(self, threshold_minutes: int) -> list[str]: ...
