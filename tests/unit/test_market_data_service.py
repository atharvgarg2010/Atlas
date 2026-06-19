"""
tests/unit/test_market_data_service.py
========================================
Unit tests for MarketDataService orchestration logic.

All external dependencies (collector, repos, DB) are mocked.
Tests verify correct delegation, error isolation, and report aggregation.
No database, no network required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.types import IngestReport, SymbolResult


def _make_valid_records(n: int = 5) -> list[dict]:
    """Generate n valid OHLCV records for testing."""
    return [
        {
            "symbol": "RELIANCE.NS",
            "ts": datetime(2024, 1, i, tzinfo=timezone.utc),
            "open": 2900.0, "high": 2950.0,
            "low": 2870.0, "close": 2930.0,
            "volume": 1_000_000,
        }
        for i in range(1, n + 1)
    ]


@pytest.fixture
def mock_settings(test_settings):
    """Use the session-scoped test_settings from conftest (3 symbols, 30 days)."""
    return test_settings


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def service(mock_db, mock_settings):
    """Build a MarketDataService with all dependencies mocked."""
    from services.market_data_service import MarketDataService

    svc = MarketDataService(db=mock_db, settings=mock_settings)

    # Replace all dependencies with mocks
    svc._collector = MagicMock()
    svc._processor = MagicMock()
    svc._stock_repo = MagicMock()
    svc._market_repo = MagicMock()
    svc._log_repo = MagicMock()

    return svc


class TestRunReport:
    def test_run_returns_ingest_report(self, service):
        service._collector.fetch_history.return_value = _make_valid_records(5)
        service._processor.validate.return_value = (_make_valid_records(5), [])
        service._market_repo.bulk_upsert.return_value = 5
        service._market_repo.get_latest_ts.return_value = None
        service._stock_repo.seed_watchlist.return_value = 3

        report = service.run()
        assert isinstance(report, IngestReport)

    def test_total_symbols_matches_watchlist(self, service, mock_settings):
        service._collector.fetch_history.return_value = _make_valid_records(5)
        service._processor.validate.return_value = (_make_valid_records(5), [])
        service._market_repo.bulk_upsert.return_value = 5
        service._market_repo.get_latest_ts.return_value = None
        service._stock_repo.seed_watchlist.return_value = 3

        report = service.run()
        assert report.total_symbols == len(mock_settings.market.watchlist)

    def test_all_succeed_when_no_errors(self, service, mock_settings):
        service._collector.fetch_history.return_value = _make_valid_records(5)
        service._processor.validate.return_value = (_make_valid_records(5), [])
        service._market_repo.bulk_upsert.return_value = 5
        service._market_repo.get_latest_ts.return_value = None
        service._stock_repo.seed_watchlist.return_value = 3

        report = service.run()
        assert report.succeeded == report.total_symbols
        assert report.failed == 0

    def test_total_inserted_aggregated_correctly(self, service, mock_settings):
        service._collector.fetch_history.return_value = _make_valid_records(5)
        service._processor.validate.return_value = (_make_valid_records(5), [])
        service._market_repo.bulk_upsert.return_value = 5
        service._market_repo.get_latest_ts.return_value = None
        service._stock_repo.seed_watchlist.return_value = 3

        report = service.run()
        # 5 rows × 3 symbols = 15
        assert report.total_inserted == 5 * len(mock_settings.market.watchlist)


class TestErrorIsolation:
    def test_collector_failure_does_not_abort_pipeline(self, service, mock_settings):
        """If the collector raises for one symbol, others should still complete."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated network failure")
            return _make_valid_records(3)

        service._collector.fetch_history.side_effect = side_effect
        service._processor.validate.return_value = (_make_valid_records(3), [])
        service._market_repo.bulk_upsert.return_value = 3
        service._market_repo.get_latest_ts.return_value = None
        service._stock_repo.seed_watchlist.return_value = 3

        report = service.run()
        assert report.failed == 1
        assert report.succeeded == len(mock_settings.market.watchlist) - 1

    def test_failed_symbol_recorded_in_results(self, service):
        service._collector.fetch_history.side_effect = ValueError("bad data")
        service._processor.validate.return_value = ([], [])
        service._market_repo.bulk_upsert.return_value = 0
        service._market_repo.get_latest_ts.return_value = None
        service._stock_repo.seed_watchlist.return_value = 3

        report = service.run()
        failed = [r for r in report.results if not r.ok]
        assert len(failed) > 0
        assert "bad data" in failed[0].error

    def test_empty_collector_response_not_an_error(self, service, mock_settings):
        """Empty yfinance response is a warning, not a failure."""
        service._collector.fetch_history.return_value = []
        service._market_repo.get_latest_ts.return_value = None
        service._stock_repo.seed_watchlist.return_value = 3

        report = service.run()
        # No error — just 0 rows
        assert report.failed == 0
        assert report.total_inserted == 0


class TestDeltaFetch:
    def test_delta_start_is_latest_ts_plus_one_day(self, service):
        """When latest_ts is set, start should be latest_ts + 1 day."""
        from datetime import timedelta

        latest = datetime(2024, 6, 1, tzinfo=timezone.utc)
        service._market_repo.get_latest_ts.return_value = latest

        start, end = service._fetch_window("RELIANCE.NS")
        assert start == latest + timedelta(days=1)

    def test_first_run_start_uses_history_days(self, service, mock_settings):
        """When no data exists, start should be today − history_days."""
        from datetime import timedelta

        service._market_repo.get_latest_ts.return_value = None

        start, end = service._fetch_window("RELIANCE.NS")
        expected_start = end - timedelta(days=mock_settings.market.history_days)
        # Allow 1-second tolerance for test timing
        assert abs((start - expected_start).total_seconds()) < 2
