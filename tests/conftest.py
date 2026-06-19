"""
tests/conftest.py
=================
Project Atlas — Pytest Fixtures

Provides shared fixtures for unit and integration tests.

Test DB strategy:
- Integration tests use a real PostgreSQL connection (Supabase).
- TEST_DATABASE_URL is read from the environment. If not set, falls back
  to the value in .env (via get_settings()).
- Unit tests never require a DB — they use mocked or in-memory data only.
- Integration tests should be marked @pytest.mark.integration and can be
  skipped with: pytest tests/unit/ -v  (or -m "not integration")
"""

from __future__ import annotations

import os

import pytest

from config.settings import (
    MarketConfig,
    PaperTradingConfig,
    SchedulerConfig,
    ScoringConfig,
    ScoringWeights,
    Settings,
)


# ─── pytest marks ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require a live PostgreSQL database",
    )


# ─── Settings fixture ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Return a Settings instance configured for testing.

    Uses a minimal 3-symbol watchlist and 30-day history to keep test
    data small and fast. DATABASE_URL is read from TEST_DATABASE_URL env
    var (CI/CD) or falls back to the production .env value.
    """
    from config.settings import get_settings

    # In CI: set TEST_DATABASE_URL to a separate test database.
    # In local dev: uses the same Supabase DB as production (acceptable for Sprint 2).
    db_url = os.getenv("TEST_DATABASE_URL") or get_settings().database_url

    return Settings(
        database_url=db_url,
        app_env="test",
        log_level="DEBUG",
        app_name="Project Atlas (Test)",
        market=MarketConfig(
            exchange="NSE",
            watchlist=("RELIANCE.NS", "TCS.NS", "INFY.NS"),
            data_timeframes=("1d",),
            history_days=30,
            fetch_batch_size=3,
            fetch_delay_seconds=0.5,
            min_valid_records=5,
        ),
        scheduler=SchedulerConfig(),
        paper_trading=PaperTradingConfig(
            initial_capital=10_000.0,
            position_size_pct=0.10,
        ),
        scoring=ScoringConfig(
            weights=ScoringWeights(
                technical=0.40,
                momentum=0.30,
                news=0.20,
                volume=0.10,
            )
        ),
    )


# ─── Database fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def db_manager(test_settings):
    """
    Return an initialised DatabaseManager connected to the test database.

    Disposes the connection pool after the test session completes.
    """
    from database.connection import DatabaseManager

    manager = DatabaseManager(
        database_url=test_settings.database_url,
        echo=False,
    )
    yield manager
    manager.dispose()


@pytest.fixture
def db_session(db_manager):
    """
    Yield a live SQLAlchemy session that rolls back after each test.

    Use this fixture for integration tests that write to the DB —
    changes are never committed to the real tables.
    """
    sess = db_manager._session_factory()
    try:
        yield sess
    finally:
        sess.rollback()
        sess.close()


# ─── Repository fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def stock_repo(db_manager):
    """Return a StockRepository connected to the test DB."""
    from repositories.stock_repository import StockRepository
    return StockRepository(db_manager)


@pytest.fixture(scope="session")
def market_data_repo(db_manager):
    """Return a MarketDataRepository connected to the test DB."""
    from repositories.market_data_repository import MarketDataRepository
    return MarketDataRepository(db_manager)


@pytest.fixture(scope="session")
def system_log_repo(db_manager):
    """Return a SystemLogRepository connected to the test DB."""
    from repositories.system_log_repository import SystemLogRepository
    return SystemLogRepository(db_manager)
