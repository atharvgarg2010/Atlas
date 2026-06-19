"""
Project Atlas — Entry Point (Sprint 2)
=======================================
Bootstraps config, logging, database, and runs the market data ingestion pipeline.

Usage
-----
    python main.py

Sprint 2 flow:
    1. Load config from .env + settings.yaml
    2. Setup logging (file + console)
    3. Connect to Supabase PostgreSQL — health check (SELECT 1)
    4. Run Alembic migrations to ensure tables exist
    5. Run MarketDataService — ingest NIFTY 50 OHLCV data
    6. Print IngestReport summary
"""

from __future__ import annotations

import sys


def main() -> None:
    """Bootstrap the Atlas platform and run Sprint 2 market data ingestion."""

    # ── Step 1: Load config ────────────────────────────────────────────────
    from config import get_settings
    from core.exceptions import AtlasError, DatabaseConnectionError
    from core.logging import get_logger, setup_logging

    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    logger = get_logger(__name__)

    # ── Step 2: Print startup banner ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name}  |  v0.2.0  |  Sprint 2")
    logger.info("=" * 60)
    logger.info(f"  Environment     : {settings.app_env.upper()}")
    logger.info(f"  Log Level       : {settings.log_level}")
    logger.info(f"  Exchange        : {settings.market.exchange}")
    logger.info(f"  Watchlist       : {len(settings.market.watchlist)} symbols (NIFTY 50)")
    logger.info(f"  History         : {settings.market.history_days} days")
    logger.info(f"  Paper Capital   : INR {settings.paper_trading.initial_capital:,.0f}")
    logger.info(f"  Position Size   : {settings.paper_trading.position_size_pct * 100:.0f}% per trade")
    logger.info("-" * 60)

    # ── Step 3: Database health check ─────────────────────────────────────
    from database.connection import init_db

    try:
        db = init_db(
            database_url=settings.database_url,
            echo=False,  # keep SQL quiet — too noisy for 50-symbol ingestion
        )
        db.ping()
        logger.info("  Database        : OK (Supabase PostgreSQL)")
    except DatabaseConnectionError as exc:
        logger.critical(f"  Database        : FAILED — {exc}")
        logger.critical(
            "  Check DATABASE_URL in .env — ensure it points to your Supabase "
            "project with ?sslmode=require"
        )
        sys.exit(1)
    except AtlasError as exc:
        logger.critical(f"  Startup error   : {exc}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  System check: PASSED — all systems operational")
    logger.info("=" * 60)

    # ── Step 4: Run market data ingestion ─────────────────────────────────
    logger.info("")
    logger.info("  Starting market data ingestion...")
    logger.info("-" * 60)

    try:
        from services.market_data_service import MarketDataService

        service = MarketDataService(db=db, settings=settings)
        report = service.run()

        logger.info("")
        logger.info("=" * 60)
        logger.info("  INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Symbols total   : {report.total_symbols}")
        logger.info(f"  Succeeded       : {report.succeeded}")
        logger.info(f"  Failed          : {report.failed}")
        logger.info(f"  Rows fetched    : {report.total_fetched:,}")
        logger.info(f"  Rows inserted   : {report.total_inserted:,}")
        logger.info("=" * 60)

        if report.failed > 0:
            logger.warning(f"  {report.failed} symbols had errors:")
            for r in report.results:
                if not r.ok:
                    logger.warning(f"    [FAIL] {r.symbol}: {r.error}")

        sys.exit(0 if report.failed == 0 else 1)

    except Exception as exc:
        logger.critical(f"  Ingestion crashed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
