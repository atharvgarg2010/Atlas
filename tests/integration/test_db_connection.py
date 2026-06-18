"""
tests/integration/test_db_connection.py
========================================
Integration tests for database connectivity.

These tests require the Docker PostgreSQL container to be running:
    docker-compose -f deployment/docker-compose.yml up -d db
"""

from __future__ import annotations

import pytest

from core.exceptions import DatabaseConnectionError


class TestDatabasePing:
    def test_ping_succeeds(self, db_manager):
        """ping() returns True when the database is reachable."""
        result = db_manager.ping()
        assert result is True

    def test_ping_raises_on_bad_url(self):
        """ping() raises DatabaseConnectionError for unreachable databases."""
        from database.connection import DatabaseManager

        bad_db = DatabaseManager(
            database_url="postgresql://invalid:invalid@localhost:9999/noexist",
            echo=False,
        )
        with pytest.raises(DatabaseConnectionError):
            bad_db.ping()


class TestSessionContextManager:
    def test_session_commits_on_success(self, db_manager):
        """Session should commit without error when no exception is raised."""
        with db_manager.session() as sess:
            # Simple query — no writes, just verify session is functional
            from sqlalchemy import text

            result = sess.execute(text("SELECT 1")).scalar()
            assert result == 1

    def test_session_rolls_back_on_exception(self, db_manager):
        """Session should roll back when an exception occurs inside the block."""
        with pytest.raises(ValueError):
            with db_manager.session() as sess:
                from sqlalchemy import text

                sess.execute(text("SELECT 1"))
                raise ValueError("Deliberate test error — should trigger rollback")


class TestGetDb:
    def test_get_db_raises_before_init(self):
        """get_db() raises RuntimeError if init_db() hasn't been called."""
        from database import connection

        original = connection._db_manager
        connection._db_manager = None  # reset singleton
        try:
            from database.connection import get_db

            with pytest.raises(RuntimeError, match="not been initialised"):
                get_db()
        finally:
            connection._db_manager = original  # restore
