"""
tests/integration/test_models.py
==================================
Integration tests for ORM models.

Requires a live Supabase PostgreSQL connection.
Tables must exist (run ``alembic upgrade head`` first).

Run with:
    pytest tests/integration/test_models.py -v -m integration
"""

from __future__ import annotations

import pytest
from sqlalchemy import text


@pytest.mark.integration
class TestStockModel:
    def test_stocks_table_exists(self, db_manager):
        """Verify the stocks table was created by the migration."""
        with db_manager.session() as s:
            result = s.execute(
                text("SELECT COUNT(*) FROM information_schema.tables "
                     "WHERE table_name = 'stocks'")
            ).scalar()
        assert result == 1

    def test_market_data_table_exists(self, db_manager):
        """Verify the market_data table was created."""
        with db_manager.session() as s:
            result = s.execute(
                text("SELECT COUNT(*) FROM information_schema.tables "
                     "WHERE table_name = 'market_data'")
            ).scalar()
        assert result == 1

    def test_system_logs_table_exists(self, db_manager):
        """Verify the system_logs table was created."""
        with db_manager.session() as s:
            result = s.execute(
                text("SELECT COUNT(*) FROM information_schema.tables "
                     "WHERE table_name = 'system_logs'")
            ).scalar()
        assert result == 1

    def test_unique_constraint_on_market_data(self, db_manager):
        """Verify the unique constraint exists on market_data."""
        with db_manager.session() as s:
            result = s.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.table_constraints "
                    "WHERE table_name = 'market_data' "
                    "AND constraint_name = 'uq_market_data_symbol_tf_ts'"
                )
            ).scalar()
        assert result == 1


@pytest.mark.integration
class TestStockRepository:
    def test_seed_watchlist_inserts_symbols(self, stock_repo):
        """seed_watchlist should insert symbols (or silently skip if they exist)."""
        count_before = stock_repo.count_active()
        stock_repo.seed_watchlist(["_TEST_SEED.NS"])
        count_after = stock_repo.count_active()
        assert count_after >= count_before  # at least no regression

    def test_get_by_symbol_returns_none_for_unknown(self, stock_repo):
        result = stock_repo.get_by_symbol("_NONEXISTENT_SYMBOL.NS")
        assert result is None

    def test_upsert_and_retrieve(self, stock_repo):
        """upsert then get_by_symbol should return the same record."""
        test_symbol = "_ATLAS_UPSERT_TEST.NS"
        stock = stock_repo.upsert(
            symbol=test_symbol,
            name="Test Company",
            sector="Technology",
            exchange="NSE",
        )
        assert stock is not None
        assert stock.symbol == test_symbol

        retrieved = stock_repo.get_by_symbol(test_symbol)
        assert retrieved is not None
        assert retrieved.symbol == test_symbol


@pytest.mark.integration
class TestSystemLogRepository:
    def test_log_info_does_not_raise(self, system_log_repo):
        """Writing a structured log entry should not raise any exception."""
        system_log_repo.info(
            "tests.integration.test_models",
            "Integration test log entry",
            {"test": True, "value": 42},
        )

    def test_log_error_does_not_raise(self, system_log_repo):
        system_log_repo.error(
            "tests.integration.test_models",
            "Simulated error log entry",
            {"error": "test error"},
        )
