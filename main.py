"""
Project Atlas — Entry Point
===========================
Initialises the configuration, logging, and database connection, then
performs a health check to verify all systems are operational.

Usage
-----
    python main.py

This script is intentionally minimal in Sprint 1. It will grow to
launch the scheduler and dashboard in later phases.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Bootstrap the Atlas platform and verify system health."""
    # ── Step 1: Load config ────────────────────────────────────────────────
    from config import get_settings
    from core.exceptions import AtlasError, DatabaseConnectionError
    from core.logging import get_logger, setup_logging

    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    logger = get_logger(__name__)

    # ── Step 2: Print startup banner ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name}  |  v0.1.0  |  Sprint 1")
    logger.info("=" * 60)
    logger.info(f"  Environment     : {settings.app_env.upper()}")
    logger.info(f"  Log Level       : {settings.log_level}")
    logger.info(f"  Exchange        : {settings.market.exchange}")
    logger.info(f"  Watchlist       : {len(settings.market.watchlist)} symbols (NIFTY 50)")
    logger.info(f"  History         : {settings.market.history_days} days")
    logger.info(f"  Paper Capital   : ₹{settings.paper_trading.initial_capital:,.0f}")
    logger.info(f"  Position Size   : {settings.paper_trading.position_size_pct * 100:.0f}% per trade")
    logger.info("-" * 60)

    # ── Step 3: Database health check ─────────────────────────────────────
    from database.connection import init_db

    try:
        db = init_db(
            database_url=settings.database_url,
            echo=settings.is_development,
        )
        db.ping()
        logger.info("  Database        : OK")
    except DatabaseConnectionError as exc:
        logger.critical(f"  Database        : FAILED — {exc}")
        logger.critical(
            "  Ensure the database container is running:\n"
            "    docker-compose up -d db"
        )
        sys.exit(1)
    except AtlasError as exc:
        logger.critical(f"  Startup error   : {exc}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  System check: PASSED — all systems operational")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
