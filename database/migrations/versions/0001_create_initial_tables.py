"""Create stocks, market_data, system_logs tables

Revision ID: 0001
Revises:
Create Date: 2026-06-19 00:00:00.000000+00:00

Sprint 2 initial migration — creates the three core tables:
    - stocks         : Symbol master registry
    - market_data    : OHLCV candles (unique per symbol × timeframe × ts)
    - system_logs    : Structured operational event log
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── stocks ────────────────────────────────────────────────────────────────
    op.create_table(
        "stocks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=True),
        sa.Column("sector", sa.String(length=100), nullable=True),
        sa.Column("exchange", sa.String(length=10), nullable=False, server_default="NSE"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol"),
    )
    op.create_index("ix_stocks_symbol", "stocks", ["symbol"])
    op.create_index("ix_stocks_is_active", "stocks", ["is_active"])

    # ── market_data ───────────────────────────────────────────────────────────
    op.create_table(
        "market_data",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("timeframe", sa.String(length=5), nullable=False, server_default="1d"),
        sa.Column("ts", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("high", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("low", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("close", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "symbol", "timeframe", "ts", name="uq_market_data_symbol_tf_ts"
        ),
    )
    op.create_index("ix_market_data_symbol_ts", "market_data", ["symbol", "ts"])
    op.create_index("ix_market_data_ts", "market_data", ["ts"])

    # ── system_logs ───────────────────────────────────────────────────────────
    op.create_table(
        "system_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("level", sa.String(length=10), nullable=False),
        sa.Column("logger", sa.String(length=100), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "ts",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_system_logs_ts", "system_logs", ["ts"])
    op.create_index("ix_system_logs_level", "system_logs", ["level"])


def downgrade() -> None:
    op.drop_index("ix_system_logs_level", table_name="system_logs")
    op.drop_index("ix_system_logs_ts", table_name="system_logs")
    op.drop_table("system_logs")

    op.drop_index("ix_market_data_ts", table_name="market_data")
    op.drop_index("ix_market_data_symbol_ts", table_name="market_data")
    op.drop_table("market_data")

    op.drop_index("ix_stocks_is_active", table_name="stocks")
    op.drop_index("ix_stocks_symbol", table_name="stocks")
    op.drop_table("stocks")
