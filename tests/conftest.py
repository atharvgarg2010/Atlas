"""
tests/conftest.py
=================
Project Atlas — Pytest Fixtures

Provides shared fixtures for unit and integration tests.
All tests that need the settings object or a database session
should use these fixtures rather than calling get_settings() directly.

The test database uses a real PostgreSQL connection (same Docker container)
with a separate test schema, not SQLite. This catches PostgreSQL-specific
behaviours (e.g. NUMERIC precision, TIMESTAMPTZ) that SQLite silently ignores.
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


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Return a Settings instance configured for testing.

    Uses a minimal watchlist to keep test data small.
    DATABASE_URL is read from the environment so tests work both locally
    (via Docker) and in CI (via environment variable injection).
    """
    db_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://atlas:atlas_secret@localhost:5432/atlas",
    )
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
