"""
services/market_data_service.py
================================
Project Atlas — Market Data Ingestion Service (Sprint 2)

Purpose
-------
Orchestrate the full NIFTY 50 market data ingestion pipeline:

    NIFTY 50 symbols (config)
        ↓
    Seed stocks table (idempotent)
        ↓
    For each symbol — determine fetch window (delta or full history)
        ↓
    YFinanceCollector.fetch_history()   [@retry handled inside]
        ↓
    OHLCVProcessor.validate()
        ↓
    MarketDataRepository.bulk_upsert()  [ON CONFLICT DO NOTHING]
        ↓
    SystemLogRepository.info()
        ↓
    Return IngestReport

Architecture
------------
- ``run()`` is the single public entry point, decorated with @timed.
- Per-symbol failures are caught and recorded in SymbolResult.error;
  the pipeline continues with the next symbol — never aborts.
- Delta fetch: on subsequent runs only the gap between
  ``latest_ts + 1 day`` and ``today`` is downloaded. On first run,
  ``history_days`` (365) of data is fetched.
- Batch delay: ``fetch_delay_seconds`` pause between symbol batches
  to respect yfinance rate limits.

Dependencies
------------
    config.settings.Settings
    data.collectors.yfinance_collector.YFinanceCollector
    data.processors.ohlcv_processor.OHLCVProcessor
    repositories.stock_repository.StockRepository
    repositories.market_data_repository.MarketDataRepository
    repositories.system_log_repository.SystemLogRepository
    core.decorators.timed
    core.types.IngestReport, SymbolResult
    core.logging

Failure Scenarios
-----------------
- yfinance returns empty list    → SymbolResult recorded, pipeline continues
- All records fail validation    → SymbolResult with rows_valid=0, continues
- DB write fails                 → SymbolResult.error set, pipeline continues
- Unhandled exception per symbol → caught, logged, pipeline continues
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from config.settings import Settings
from core.decorators import timed
from core.logging import get_logger
from core.types import IngestReport, SymbolResult
from data.collectors.yfinance_collector import YFinanceCollector
from data.processors.ohlcv_processor import OHLCVProcessor
from database.connection import DatabaseManager
from repositories.market_data_repository import MarketDataRepository
from repositories.stock_repository import StockRepository
from repositories.system_log_repository import SystemLogRepository

logger = get_logger(__name__)

_SERVICE_NAME = "services.market_data_service"


class MarketDataService:
    """
    Orchestrates NIFTY 50 market data ingestion into Supabase PostgreSQL.

    Args:
        db:       Initialised DatabaseManager.
        settings: Application settings singleton.

    Example
    -------
        service = MarketDataService(db=db, settings=settings)
        report = service.run()
        print(report.summary())
    """

    def __init__(self, db: DatabaseManager, settings: Settings) -> None:
        self._db = db
        self._settings = settings
        self._collector = YFinanceCollector()
        self._processor = OHLCVProcessor()
        self._stock_repo = StockRepository(db)
        self._market_repo = MarketDataRepository(db)
        self._log_repo = SystemLogRepository(db)

    @timed
    def run(self) -> IngestReport:
        """
        Execute the full ingestion pipeline for all watchlist symbols.

        Steps:
            1. Seed stocks table with NIFTY 50 symbols (idempotent).
            2. For each symbol, determine fetch window and ingest.
            3. Log aggregate result to system_logs and file logger.

        Returns:
            IngestReport — aggregate counts and per-symbol results.
        """
        symbols = list(self._settings.market.watchlist)
        report = IngestReport(total_symbols=len(symbols))

        logger.info(f"[market_data_svc] Starting ingestion for {len(symbols)} symbols")

        self._log_repo.info(
            _SERVICE_NAME,
            "Ingestion run started",
            {"total_symbols": len(symbols)},
        )

        # Step 1: Seed the stocks table (idempotent — uses ON CONFLICT DO NOTHING)
        self._seed_stocks(symbols)

        # Step 2: Ingest each symbol
        batch_size = self._settings.market.fetch_batch_size
        delay = self._settings.market.fetch_delay_seconds

        for batch_start in range(0, len(symbols), batch_size):
            batch = symbols[batch_start : batch_start + batch_size]

            for symbol in batch:
                result = self._ingest_symbol(symbol)
                report.results.append(result)
                report.total_fetched += result.rows_fetched
                report.total_inserted += result.rows_inserted

                if result.ok:
                    report.succeeded += 1
                else:
                    report.failed += 1

            # Polite delay between batches
            if batch_start + batch_size < len(symbols):
                time.sleep(delay)

        # Step 3: Log final summary
        logger.info(f"[market_data_svc] Ingestion complete - {report.summary()}")
        self._log_repo.info(
            _SERVICE_NAME,
            "Ingestion run complete",
            {
                "total_symbols": report.total_symbols,
                "succeeded": report.succeeded,
                "failed": report.failed,
                "total_fetched": report.total_fetched,
                "total_inserted": report.total_inserted,
            },
        )

        if report.failed > 0:
            failed_symbols = [r.symbol for r in report.results if not r.ok]
            logger.warning(
                f"[market_data_svc] {report.failed} symbols failed: {failed_symbols}"
            )

        return report

    # ── Private helpers ────────────────────────────────────────────────────────

    def _seed_stocks(self, symbols: list[str]) -> None:
        """Seed the stocks table with all watchlist symbols (idempotent)."""
        try:
            inserted = self._stock_repo.seed_watchlist(symbols)
            if inserted > 0:
                logger.info(f"[market_data_svc] Seeded {inserted} new stock symbols")
        except Exception as exc:
            logger.error(f"[market_data_svc] Failed to seed stocks: {exc}")
            # Non-fatal — continue with ingestion

    def _ingest_symbol(self, symbol: str) -> SymbolResult:
        """
        Run the full ingestion pipeline for a single symbol.

        Returns a SymbolResult with counts and error info.
        Never raises — all exceptions are caught and recorded.
        """
        result = SymbolResult(symbol=symbol)

        try:
            # Determine fetch window
            start_dt, end_dt = self._fetch_window(symbol)

            if start_dt >= end_dt:
                logger.debug(
                    f"[market_data_svc] {symbol!r} - already up to date, skipping"
                )
                return result  # 0 rows, no error

            # Fetch from yfinance
            records = self._collector.fetch_history(
                symbol=symbol,
                start=start_dt.date(),
                end=end_dt.date(),
            )
            result.rows_fetched = len(records)

            if not records:
                logger.warning(
                    f"[market_data_svc] {symbol!r} - no data returned for "
                    f"{start_dt.date()} -> {end_dt.date()}"
                )
                return result

            # Validate
            valid, rejected = self._processor.validate(records, symbol=symbol)
            result.rows_valid = len(valid)
            result.rows_rejected = len(rejected)

            # Check minimum valid records threshold (only on first run / large fetch)
            min_records = self._settings.market.min_valid_records
            is_initial_run = result.rows_fetched > 100  # heuristic: >100 rows = initial
            if is_initial_run and result.rows_valid < min_records:
                result.error = (
                    f"too few valid records: {result.rows_valid} < "
                    f"min_valid_records={min_records}"
                )
                logger.warning(f"[market_data_svc] {symbol!r} — {result.error}")
                return result

            # Persist
            inserted = self._market_repo.bulk_upsert(valid, timeframe="1d")
            result.rows_inserted = inserted

            logger.info(
                f"[market_data_svc] {symbol!r} - "
                f"fetched={result.rows_fetched} valid={result.rows_valid} "
                f"rejected={result.rows_rejected} inserted={result.rows_inserted}"
            )

            # Structured DB log
            self._log_repo.info(
                _SERVICE_NAME,
                f"Symbol ingested: {symbol}",
                {
                    "symbol": symbol,
                    "rows_fetched": result.rows_fetched,
                    "rows_valid": result.rows_valid,
                    "rows_rejected": result.rows_rejected,
                    "rows_inserted": result.rows_inserted,
                },
            )

        except Exception as exc:
            result.error = str(exc)
            logger.error(
                f"[market_data_svc] {symbol!r} — unhandled error: {exc}",
                exc_info=True,
            )
            self._log_repo.error(
                _SERVICE_NAME,
                f"Symbol failed: {symbol} — {exc}",
                {"symbol": symbol, "error": str(exc)},
            )

        return result

    def _fetch_window(self, symbol: str) -> tuple[datetime, datetime]:
        """
        Compute the (start, end) UTC datetime range to fetch for ``symbol``.

        - First run (no data): start = today − history_days
        - Subsequent runs:     start = latest_ts + 1 day
        - end is always midnight UTC of today (exclusive per yfinance convention)
        """
        end_dt = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        latest_ts = self._market_repo.get_latest_ts(symbol, timeframe="1d")

        if latest_ts is None:
            # First run — full history
            start_dt = end_dt - timedelta(days=self._settings.market.history_days)
        else:
            # Delta — start the day after the last stored candle
            if latest_ts.tzinfo is None:
                latest_ts = latest_ts.replace(tzinfo=timezone.utc)
            start_dt = latest_ts + timedelta(days=1)

        return start_dt, end_dt
