"""
repositories/market_data_repository.py
=======================================
Project Atlas — OHLCV Market Data Access Layer (Sprint 2)

Purpose
-------
All reads and writes for the ``market_data`` table. The ONLY module
that queries or writes market_data directly.

Duplicate Prevention
--------------------
``bulk_upsert`` uses ``INSERT ... ON CONFLICT (symbol, timeframe, ts) DO NOTHING``.
Duplicate rows are silently skipped — OHLCV data is immutable once stored.
The unique constraint ``uq_market_data_symbol_tf_ts`` in the DB is the final
guarantee; this method is the application-level enforcement.

Delta Fetch Optimisation
------------------------
``get_latest_ts`` returns the most recent ``ts`` stored for a symbol.
MarketDataService uses this to fetch only new data (``latest_ts → today``)
instead of re-downloading 365 days every run.

Dependencies
------------
    database.connection.DatabaseManager
    database.models.MarketData
    core.exceptions.DatabaseError
    core.logging
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.exceptions import DatabaseError
from core.logging import get_logger
from database.connection import DatabaseManager
from database.models import MarketData

logger = get_logger(__name__)

# Maximum rows per INSERT batch (avoids hitting PostgreSQL's parameter limit)
_BATCH_SIZE = 500


class MarketDataRepository:
    """
    Data access layer for the ``market_data`` table.

    Args:
        db: Initialised DatabaseManager singleton.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ── Reads ──────────────────────────────────────────────────────────────────

    def get_latest_ts(self, symbol: str, timeframe: str = "1d") -> datetime | None:
        """
        Return the most recent ``ts`` stored for ``symbol`` + ``timeframe``.

        Returns None if no data exists yet (first run).
        Used by MarketDataService to determine the delta fetch window.

        Args:
            symbol:    NSE ticker — e.g. 'RELIANCE.NS'.
            timeframe: Candle width — e.g. '1d'.
        """
        with self._db.session() as s:
            result = s.execute(
                select(func.max(MarketData.ts)).where(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe,
                )
            ).scalar_one_or_none()
            return result  # datetime | None

    def get_history(
        self,
        symbol: str,
        days: int = 365,
        timeframe: str = "1d",
    ) -> list[MarketData]:
        """
        Return up to ``days`` calendar days of candles for ``symbol``.

        Returns rows ordered by ``ts`` ascending.

        Args:
            symbol:    NSE ticker.
            days:      Number of calendar days of history to return.
            timeframe: Candle width — e.g. '1d'.
        """
        from datetime import timezone, timedelta

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        with self._db.session() as s:
            rows = s.execute(
                select(MarketData)
                .where(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe,
                    MarketData.ts >= cutoff,
                )
                .order_by(MarketData.ts.asc())
            ).scalars().all()
            return list(rows)

    def get_latest_candle(self, symbol: str, timeframe: str = "1d") -> MarketData | None:
        """
        Return the single most recent candle for ``symbol``.

        Args:
            symbol:    NSE ticker.
            timeframe: Candle width.
        """
        with self._db.session() as s:
            return s.execute(
                select(MarketData)
                .where(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe,
                )
                .order_by(MarketData.ts.desc())
                .limit(1)
            ).scalar_one_or_none()

    def count_by_symbol(self, symbol: str, timeframe: str = "1d") -> int:
        """Return total stored candles for a symbol."""
        with self._db.session() as s:
            result = s.execute(
                select(func.count())
                .select_from(MarketData)
                .where(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe,
                )
            ).scalar_one()
            return int(result)

    def count_all(self, timeframe: str = "1d") -> int:
        """Return total stored candles across all symbols."""
        with self._db.session() as s:
            result = s.execute(
                select(func.count())
                .select_from(MarketData)
                .where(MarketData.timeframe == timeframe)
            ).scalar_one()
            return int(result)

    # ── Writes ─────────────────────────────────────────────────────────────────

    def bulk_upsert(
        self,
        records: list[dict],
        timeframe: str = "1d",
    ) -> int:
        """
        Batch-insert OHLCV records, silently skipping duplicates.

        Uses PostgreSQL ``INSERT ... ON CONFLICT (symbol, timeframe, ts) DO NOTHING``.
        Processes records in batches of ``_BATCH_SIZE`` to stay within
        PostgreSQL's parameter limit.

        Args:
            records:   List of dicts from YFinanceCollector / OHLCVProcessor.
                       Each dict must have: symbol, ts, open, high, low, close, volume.
            timeframe: Candle width label stored on every row. Default '1d'.

        Returns:
            Total number of rows actually inserted (excludes skipped duplicates).

        Raises:
            DatabaseError: On unexpected DB failure.
        """
        if not records:
            return 0

        total_inserted = 0

        try:
            # Add timeframe to each record (collector doesn't set it)
            enriched = [
                {
                    "symbol": r["symbol"],
                    "timeframe": timeframe,
                    "ts": r["ts"],
                    "open": r["open"],
                    "high": r["high"],
                    "low": r["low"],
                    "close": r["close"],
                    "volume": r["volume"],
                }
                for r in records
            ]

            # Process in batches to avoid parameter limit
            for i in range(0, len(enriched), _BATCH_SIZE):
                batch = enriched[i : i + _BATCH_SIZE]
                stmt = (
                    pg_insert(MarketData)
                    .values(batch)
                    .on_conflict_do_nothing(
                        index_elements=["symbol", "timeframe", "ts"],
                    )
                )
                with self._db.session() as s:
                    result = s.execute(stmt)
                    inserted = result.rowcount if result.rowcount >= 0 else 0
                    total_inserted += inserted

        except Exception as exc:
            symbol = records[0].get("symbol", "?") if records else "?"
            logger.error(f"[market_data_repo] bulk_upsert failed for {symbol!r}: {exc}")
            raise DatabaseError("Failed to upsert market data records") from exc

        return total_inserted
