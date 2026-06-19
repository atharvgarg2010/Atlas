"""
repositories/stock_repository.py
=================================
Project Atlas — Stock Master Data Access Layer (Sprint 2)

Purpose
-------
All CRUD operations for the ``stocks`` table. This is the ONLY module
that queries or writes to ``stocks`` directly.

Rules
-----
- No business logic lives here — only database operations.
- All callers receive ORM instances or plain Python types (never raw SQL).
- ``upsert`` uses INSERT ... ON CONFLICT(symbol) DO UPDATE to handle
  repeated seeding gracefully.

Dependencies
------------
    database.connection.DatabaseManager
    database.models.Stock
    core.exceptions.DatabaseError
    core.logging
"""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.exceptions import DatabaseError
from core.logging import get_logger
from database.connection import DatabaseManager
from database.models import Stock

logger = get_logger(__name__)


class StockRepository:
    """
    Data access layer for the ``stocks`` table.

    Args:
        db: Initialised DatabaseManager singleton.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ── Reads ──────────────────────────────────────────────────────────────────

    def get_by_symbol(self, symbol: str) -> Stock | None:
        """
        Return the Stock ORM object for ``symbol``, or None if not found.

        Args:
            symbol: NSE ticker — e.g. 'RELIANCE.NS'.
        """
        with self._db.session() as s:
            return s.execute(
                select(Stock).where(Stock.symbol == symbol)
            ).scalar_one_or_none()

    def get_all_active(self) -> list[Stock]:
        """Return all stocks where ``is_active=True``, ordered by symbol."""
        with self._db.session() as s:
            rows = s.execute(
                select(Stock).where(Stock.is_active == True).order_by(Stock.symbol)  # noqa: E712
            ).scalars().all()
            return list(rows)

    def count_active(self) -> int:
        """Return count of active stocks."""
        with self._db.session() as s:
            from sqlalchemy import func
            result = s.execute(
                select(func.count()).select_from(Stock).where(Stock.is_active == True)  # noqa: E712
            ).scalar_one()
            return int(result)

    # ── Writes ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        symbol: str,
        name: str | None = None,
        sector: str | None = None,
        exchange: str = "NSE",
    ) -> Stock:
        """
        Insert a stock or update its name/sector/exchange if it already exists.

        Uses PostgreSQL ``INSERT ... ON CONFLICT (symbol) DO UPDATE`` so
        repeated calls are safe.

        Args:
            symbol:   NSE ticker — e.g. 'RELIANCE.NS'.
            name:     Full company name (optional).
            sector:   GICS sector string (optional).
            exchange: Exchange code. Default 'NSE'.

        Returns:
            The upserted Stock ORM instance.

        Raises:
            DatabaseError: If the insert/update fails for an unexpected reason.
        """
        try:
            stmt = (
                pg_insert(Stock)
                .values(symbol=symbol, name=name, sector=sector, exchange=exchange)
                .on_conflict_do_update(
                    index_elements=["symbol"],
                    set_={"name": name, "sector": sector, "exchange": exchange},
                )
                .returning(Stock)
            )
            with self._db.session() as s:
                result = s.execute(stmt).scalar_one()
                return result
        except Exception as exc:
            logger.error(f"[stock_repo] upsert failed for {symbol!r}: {exc}")
            raise DatabaseError(f"Failed to upsert stock {symbol!r}") from exc

    def seed_watchlist(self, symbols: tuple[str, ...] | list[str]) -> int:
        """
        Bulk-insert all watchlist symbols.

        Uses ``INSERT ... ON CONFLICT DO NOTHING`` so existing rows are not
        overwritten. Safe to call multiple times (idempotent).

        Args:
            symbols: Iterable of NSE ticker strings.

        Returns:
            Number of rows actually inserted (0 if all already exist).

        Raises:
            DatabaseError: On unexpected DB failure.
        """
        if not symbols:
            return 0

        try:
            values = [{"symbol": s, "exchange": "NSE"} for s in symbols]
            stmt = pg_insert(Stock).values(values).on_conflict_do_nothing(
                index_elements=["symbol"]
            )
            with self._db.session() as s:
                result = s.execute(stmt)
                inserted = result.rowcount if result.rowcount >= 0 else 0
                logger.info(
                    f"[stock_repo] seed_watchlist: {inserted} new rows "
                    f"(of {len(symbols)} attempted)"
                )
                return inserted
        except Exception as exc:
            logger.error(f"[stock_repo] seed_watchlist failed: {exc}")
            raise DatabaseError("Failed to seed watchlist") from exc

    def deactivate(self, symbol: str) -> None:
        """
        Mark a stock as inactive (soft delete). Does not remove historical data.

        Args:
            symbol: NSE ticker to deactivate.
        """
        with self._db.session() as s:
            s.execute(
                update(Stock)
                .where(Stock.symbol == symbol)
                .values(is_active=False)
            )
        logger.info(f"[stock_repo] deactivated {symbol!r}")
