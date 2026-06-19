"""
repositories/system_log_repository.py
======================================
Project Atlas — Structured Operational Log Data Access Layer (Sprint 2)

Purpose
-------
Write structured operational events to the ``system_logs`` table.
This is separate from the rotating file log — it stores machine-readable,
queryable job-level events (run start/end, per-symbol record counts,
error summaries) that the dashboard can surface.

Rules
-----
- This repository ONLY writes to system_logs — it never reads.
  Reads will be added in Sprint 4 when the dashboard is built.
- Failures here are non-fatal: logged to the file logger and swallowed
  so that a DB log write failure never crashes the ingestion pipeline.

Dependencies
------------
    database.connection.DatabaseManager
    database.models.SystemLog
    core.logging
"""

from __future__ import annotations

from typing import Any

from database.connection import DatabaseManager
from database.models import SystemLog
from core.logging import get_logger

logger = get_logger(__name__)


class SystemLogRepository:
    """
    Write-only data access layer for the ``system_logs`` table.

    Args:
        db: Initialised DatabaseManager singleton.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def log(
        self,
        level: str,
        logger_name: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Insert a structured operational log entry.

        Failures are silently swallowed (logged to file) so that a broken
        DB connection never blocks the data ingestion pipeline.

        Args:
            level:       Log level string — 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
            logger_name: Dotted module path — e.g. 'services.market_data_service'.
            message:     Human-readable event description.
            context:     Optional dict of structured metadata stored as JSONB.
                         Must be JSON-serialisable. Example:
                         {'symbol': 'RELIANCE.NS', 'rows_inserted': 252}
        """
        try:
            entry = SystemLog(
                level=level.upper()[:10],  # cap at column length
                logger=logger_name[:100],
                message=message,
                context=context,
            )
            with self._db.session() as s:
                s.add(entry)
        except Exception as exc:
            # Never let a log write crash the caller
            logger.warning(
                f"[system_log_repo] Failed to write structured log: {exc}"
            )

    # ── Convenience helpers ────────────────────────────────────────────────────

    def info(
        self, logger_name: str, message: str, context: dict[str, Any] | None = None
    ) -> None:
        """Shorthand for log(level='INFO', ...)."""
        self.log("INFO", logger_name, message, context)

    def warning(
        self, logger_name: str, message: str, context: dict[str, Any] | None = None
    ) -> None:
        """Shorthand for log(level='WARNING', ...)."""
        self.log("WARNING", logger_name, message, context)

    def error(
        self, logger_name: str, message: str, context: dict[str, Any] | None = None
    ) -> None:
        """Shorthand for log(level='ERROR', ...)."""
        self.log("ERROR", logger_name, message, context)
