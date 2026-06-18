"""
database/migrations/env.py
==========================
Alembic migration environment — Project Atlas

This file is executed by Alembic for both ``alembic upgrade`` (online mode)
and ``alembic revision --autogenerate`` (offline autogenerate).

Key decisions:
    - DATABASE_URL is loaded from our settings system (config.get_settings),
      NOT from alembic.ini. This ensures .env is the single source of truth.
    - Base.metadata is imported AFTER importing all model modules so
      autogenerate sees every table.
    - render_as_batch=False: we are targeting PostgreSQL only, which supports
      ALTER TABLE natively (unlike SQLite which needs batch mode).
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Load Atlas settings so DATABASE_URL comes from .env ─────────────────────
from config.settings import get_settings

# ── Import Base + all models so autogenerate picks up every table ────────────
from database.connection import Base
import database.models  # noqa: F401 — side-effect import registers all models

# ─── Alembic config object (wraps alembic.ini) ───────────────────────────────
config = context.config

# Set up Python logging from alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ─── Target metadata for autogenerate ─────────────────────────────────────────
target_metadata = Base.metadata

# ─── Inject DATABASE_URL from our settings ────────────────────────────────────
_settings = get_settings()
config.set_main_option("sqlalchemy.url", _settings.database_url)


# ─── Offline mode (generates SQL script without connecting) ──────────────────

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — generates SQL script, no DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ─── Online mode (applies migrations directly to the DB) ──────────────────────

def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connects to the DB and applies changes."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # use NullPool in migrations — no persistent connections
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


# ─── Entry point ──────────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
