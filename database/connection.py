"""
database/connection.py
======================
Project Atlas — Database Connection Layer

Architecture
------------
- SQLAlchemy 2.x engine with connection pooling (pool_size=5, overflow=10)
- ``pool_pre_ping=True`` — silently drops stale connections before use
- ``pool_recycle=3600``  — recycles connections every hour to prevent timeouts
- ``DeclarativeBase`` (``Base``) is defined here so Alembic's env.py and all
  model files can import from a single location

Usage
-----
    from database.connection import get_db, init_db

    # At startup (main.py):
    db = init_db(database_url="postgresql://...")
    db.ping()

    # In a service:
    db = get_db()
    with db.session() as s:
        s.add(some_model_instance)

    # In tests (override with SQLite):
    db = init_db(database_url="sqlite:///:memory:")

Purpose:
    Provide a single, testable database manager used by all repositories.
    No other layer should create engines or sessions directly.

Dependencies:
    sqlalchemy, core.exceptions, core.logging

Failure Scenarios:
    - Wrong DATABASE_URL          → OperationalError on first connect
    - DB container not running    → DatabaseConnectionError raised by ping()
    - Session not committed       → rolled back automatically by context manager
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.exceptions import DatabaseConnectionError
from core.logging import get_logger

logger = get_logger(__name__)


# ─── ORM Base ────────────────────────────────────────────────────────────────
# All model classes must inherit from Base so Alembic can discover them.

class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base for all Project Atlas ORM models.

    Import this Base in every models file and in database/migrations/env.py.
    Never define a separate Base in individual model files.
    """


# ─── Database Manager ─────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Manages the SQLAlchemy engine and session lifecycle.

    Instantiated once at application startup via init_db() and reused
    throughout the application via get_db().

    Args:
        database_url: Full SQLAlchemy database URL.
                      Example: 'postgresql://atlas:secret@localhost:5432/atlas'
        echo:         If True, SQLAlchemy logs all SQL statements. Use only
                      in development (controlled by APP_ENV).
    """

    def __init__(self, database_url: str, echo: bool = False) -> None:
        self._engine = create_engine(
            database_url,
            pool_size=3,           # keep small — Supabase free plan caps connections
            max_overflow=2,
            pool_pre_ping=True,    # verify connections are alive before use
            pool_recycle=300,      # recycle every 5 min — Supabase drops idle after ~5 min
            echo=echo,
            future=True,
            connect_args={"connect_timeout": 10},
        )
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,  # safe to access attributes after commit
        )
        logger.info(
            f"DatabaseManager initialised "
            f"[pool_size=3, max_overflow=2, pool_recycle=300s, echo={echo}]"
        )

    @property
    def engine(self):
        """Return the raw SQLAlchemy engine (use sparingly)."""
        return self._engine

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Provide a transactional database session as a context manager.

        Behaviour:
            - Commits automatically on clean exit.
            - Rolls back automatically on any exception.
            - Always closes the session (returns connection to pool).

        Example:
            with db.session() as s:
                stock = Stock(symbol="RELIANCE.NS", name="Reliance Industries")
                s.add(stock)
            # committed here

        Raises:
            Any exception raised inside the block after rolling back.
        """
        sess: Session = self._session_factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    def ping(self) -> bool:
        """
        Verify database connectivity with a lightweight SELECT 1.

        Returns:
            True if the database responds successfully.

        Raises:
            DatabaseConnectionError: If the database cannot be reached.

        Note:
            Called once at startup in main.py. On failure, check:
            - DATABASE_URL is correct in .env
            - Network connectivity to the database host
            - SSL requirements (Supabase requires ?sslmode=require in the URL)
        """
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database health check: PASSED")
            return True
        except Exception as exc:
            logger.critical(f"Database health check: FAILED - {exc}")
            raise DatabaseConnectionError(
                "Cannot reach the database. "
                "Check DATABASE_URL in .env and verify network/SSL settings."
            ) from exc

    def dispose(self) -> None:
        """
        Dispose of all connection pool connections.

        Call this on application shutdown to release resources cleanly.
        """
        self._engine.dispose()
        logger.info("Database connection pool disposed")


# ─── Module-level Singleton ───────────────────────────────────────────────────

_db_manager: DatabaseManager | None = None


def init_db(database_url: str, echo: bool = False) -> DatabaseManager:
    """
    Initialise the module-level DatabaseManager singleton.

    Must be called exactly once at application startup, before any
    repository or service attempts to use the database.

    Args:
        database_url: SQLAlchemy database URL (typically from settings.database_url).
        echo:         Echo SQL to stdout (development only).

    Returns:
        The initialised DatabaseManager instance.
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url, echo=echo)
    return _db_manager


def get_db() -> DatabaseManager:
    """
    Return the module-level DatabaseManager.

    Raises:
        RuntimeError: If init_db() has not been called first.
    """
    if _db_manager is None:
        raise RuntimeError(
            "DatabaseManager has not been initialised. "
            "Call database.connection.init_db() at application startup."
        )
    return _db_manager
