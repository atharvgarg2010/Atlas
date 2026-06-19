"""
tests/integration/test_market_data_ingest.py
=============================================
End-to-end integration test for the market data ingestion pipeline.

Requires:
    - Live Supabase PostgreSQL connection (DATABASE_URL in .env)
    - Internet access (downloads 30 days of data from yfinance)
    - Tables created (alembic upgrade head)

Uses 3 symbols from test_settings watchlist (not all 50) and 30 days
of history to keep the test fast (~30 rows per symbol).

Run with:
    pytest tests/integration/test_market_data_ingest.py -v -m integration
"""

from __future__ import annotations

import pytest

from core.types import IngestReport


@pytest.mark.integration
class TestFullIngestionPipeline:
    """End-to-end: fetch → validate → store → verify."""

    def test_run_returns_ingest_report(self, db_manager, test_settings):
        from services.market_data_service import MarketDataService

        svc = MarketDataService(db=db_manager, settings=test_settings)
        report = svc.run()
        assert isinstance(report, IngestReport)

    def test_all_symbols_succeed(self, db_manager, test_settings):
        from services.market_data_service import MarketDataService

        svc = MarketDataService(db=db_manager, settings=test_settings)
        report = svc.run()
        assert report.failed == 0, (
            f"Expected 0 failures, got {report.failed}. "
            f"Errors: {[r.error for r in report.results if not r.ok]}"
        )

    def test_rows_are_stored_in_db(self, db_manager, test_settings, market_data_repo):
        """After ingestion, market_data should have rows for each symbol."""
        from services.market_data_service import MarketDataService

        svc = MarketDataService(db=db_manager, settings=test_settings)
        svc.run()

        for symbol in test_settings.market.watchlist:
            count = market_data_repo.count_by_symbol(symbol)
            assert count > 0, f"No rows stored for {symbol!r}"

    def test_second_run_inserts_zero_rows(self, db_manager, test_settings):
        """Re-running ingestion for the same date range should insert 0 new rows."""
        from services.market_data_service import MarketDataService

        # First run (may already have data from previous test in this session)
        svc = MarketDataService(db=db_manager, settings=test_settings)
        svc.run()

        # Second run — all data is already present, delta window collapses
        report2 = svc.run()
        assert report2.total_inserted == 0, (
            f"Expected 0 rows on second run, got {report2.total_inserted}"
        )

    def test_symbols_seeded_in_stocks_table(self, db_manager, test_settings, stock_repo):
        """After ingestion, stocks table should contain all watchlist symbols."""
        from services.market_data_service import MarketDataService

        svc = MarketDataService(db=db_manager, settings=test_settings)
        svc.run()

        for symbol in test_settings.market.watchlist:
            stock = stock_repo.get_by_symbol(symbol)
            assert stock is not None, f"{symbol!r} not found in stocks table"
            assert stock.is_active is True
