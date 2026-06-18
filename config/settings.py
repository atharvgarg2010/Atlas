"""
config/settings.py
==================
Project Atlas — Unified Application Settings

Architecture
------------
Settings are split across two sources, merged into a single immutable
``Settings`` dataclass that is cached after the first load:

1. ``.env`` file (secrets only):
   DATABASE_URL, APP_ENV, LOG_LEVEL
   Loaded via pydantic-settings (_EnvSettings) to get strict type validation
   and clear error messages for missing required vars.

2. ``config/settings.yaml`` (all non-secret config):
   Market universe, scheduler intervals, paper trading params, scoring weights.
   Loaded as typed, frozen dataclasses for zero-overhead attribute access.

Usage
-----
    from config import get_settings

    settings = get_settings()
    print(settings.market.watchlist)        # ('RELIANCE.NS', 'TCS.NS', ...)
    print(settings.paper_trading.initial_capital)   # 10000.0

Purpose:
    Centralise all configuration. No other module reads YAML or .env directly.

Dependencies:
    pydantic-settings, pyyaml, python-dotenv

Failure Scenarios:
    - Missing DATABASE_URL in .env → pydantic raises ValidationError with
      a clear message listing the missing variable.
    - Missing settings.yaml → defaults are used (logged as WARNING).
    - Malformed YAML → yaml.YAMLError raised at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = ROOT_DIR / "config"

# Load .env before pydantic-settings reads env vars
load_dotenv(ROOT_DIR / ".env", override=False)


# ─── YAML Loader ──────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents. Returns {} if file missing."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ─── Typed Config Dataclasses (from YAML) ─────────────────────────────────────

@dataclass(frozen=True)
class MarketConfig:
    """Market universe and data collection settings."""

    exchange: str = "NSE"
    watchlist: tuple[str, ...] = ()
    data_timeframes: tuple[str, ...] = ("1d",)
    history_days: int = 365

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketConfig":
        return cls(
            exchange=data.get("exchange", "NSE"),
            watchlist=tuple(data.get("watchlist", [])),
            data_timeframes=tuple(data.get("data_timeframes", ["1d"])),
            history_days=int(data.get("history_days", 365)),
        )


@dataclass(frozen=True)
class SchedulerConfig:
    """Job execution interval settings (all in minutes)."""

    market_data_interval_min: int = 15
    indicator_interval_min: int = 15
    news_interval_min: int = 15
    sentiment_interval_min: int = 60
    scoring_interval_min: int = 60
    signal_interval_min: int = 60

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchedulerConfig":
        return cls(
            market_data_interval_min=int(data.get("market_data_interval_min", 15)),
            indicator_interval_min=int(data.get("indicator_interval_min", 15)),
            news_interval_min=int(data.get("news_interval_min", 15)),
            sentiment_interval_min=int(data.get("sentiment_interval_min", 60)),
            scoring_interval_min=int(data.get("scoring_interval_min", 60)),
            signal_interval_min=int(data.get("signal_interval_min", 60)),
        )


@dataclass(frozen=True)
class PaperTradingConfig:
    """Paper trading simulation settings."""

    initial_capital: float = 10_000.0
    position_size_pct: float = 0.10
    currency: str = "INR"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperTradingConfig":
        return cls(
            initial_capital=float(data.get("initial_capital", 10_000.0)),
            position_size_pct=float(data.get("position_size_pct", 0.10)),
            currency=str(data.get("currency", "INR")),
        )


@dataclass(frozen=True)
class ScoringWeights:
    """Component weights for the composite opportunity score (must sum to 1.0)."""

    technical: float = 0.40
    momentum: float = 0.30
    news: float = 0.20
    volume: float = 0.10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringWeights":
        return cls(
            technical=float(data.get("technical", 0.40)),
            momentum=float(data.get("momentum", 0.30)),
            news=float(data.get("news", 0.20)),
            volume=float(data.get("volume", 0.10)),
        )

    def validate(self) -> None:
        """Raise ValueError if weights do not sum to 1.0 (tolerance: ±0.001)."""
        total = self.technical + self.momentum + self.news + self.volume
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"ScoringWeights must sum to 1.0, got {total:.4f}. "
                f"Check config/settings.yaml → scoring.weights"
            )


@dataclass(frozen=True)
class ScoringConfig:
    """Composite stock scoring settings."""

    weights: ScoringWeights = field(default_factory=ScoringWeights)
    weight_profile: str = "default"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringConfig":
        return cls(
            weights=ScoringWeights.from_dict(data.get("weights", {})),
            weight_profile=str(data.get("weight_profile", "default")),
        )


# ─── Env Settings (from .env) ─────────────────────────────────────────────────

class _EnvSettings(BaseSettings):
    """
    Reads secret values strictly from environment variables / .env file.
    Validated by pydantic — missing required fields raise a clear error.
    """

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = Field(..., validation_alias="DATABASE_URL")
    app_env: str = Field("development", validation_alias="APP_ENV")
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")


# ─── Unified Settings ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """
    Immutable, unified application settings.

    Combines secrets from .env with structured config from settings.yaml.
    Obtain the singleton via get_settings() — never instantiate directly.
    """

    # ── From .env ──────────────────────────────────────────────────────────
    database_url: str
    app_env: str
    log_level: str

    # ── From settings.yaml ─────────────────────────────────────────────────
    app_name: str
    market: MarketConfig
    scheduler: SchedulerConfig
    paper_trading: PaperTradingConfig
    scoring: ScoringConfig

    # ── Convenience properties ─────────────────────────────────────────────
    @property
    def is_development(self) -> bool:
        """True when APP_ENV=development."""
        return self.app_env.lower() == "development"

    @property
    def is_production(self) -> bool:
        """True when APP_ENV=production."""
        return self.app_env.lower() == "production"


# ─── Factory ──────────────────────────────────────────────────────────────────

def _build_settings() -> Settings:
    """Load, merge, and validate all configuration sources."""
    env = _EnvSettings()
    yaml_data = _load_yaml(CONFIG_DIR / "settings.yaml")

    scoring_cfg = ScoringConfig.from_dict(yaml_data.get("scoring", {}))
    scoring_cfg.weights.validate()   # fail fast on bad weight config

    return Settings(
        database_url=env.database_url,
        app_env=env.app_env,
        log_level=env.log_level,
        app_name=yaml_data.get("app", {}).get("name", "Project Atlas"),
        market=MarketConfig.from_dict(yaml_data.get("market", {})),
        scheduler=SchedulerConfig.from_dict(yaml_data.get("scheduler", {})),
        paper_trading=PaperTradingConfig.from_dict(yaml_data.get("paper_trading", {})),
        scoring=scoring_cfg,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached application settings singleton.

    Thread-safe: lru_cache(maxsize=1) is used so the YAML and .env files
    are read exactly once per process lifetime.

    Raises:
        pydantic.ValidationError: If a required .env variable is missing.
        yaml.YAMLError: If settings.yaml is malformed.
        ValueError: If scoring weights do not sum to 1.0.
    """
    return _build_settings()
