"""
core/logging.py
===============
Project Atlas — Structured Logging System

Purpose:
    Provide a single, consistent logging setup for the entire platform.
    All modules must obtain loggers via get_logger(__name__) — never
    instantiate logging.getLogger() directly.

    Loggers are scoped under the 'atlas.' namespace so they can be
    configured centrally in config/logging.yaml without affecting
    third-party library loggers.

Inputs:
    config/logging.yaml — handler, formatter, and level configuration.

Outputs:
    logs/atlas.log        — all log levels, rotating at 10 MB
    logs/atlas_errors.log — ERROR and CRITICAL only, rotating at 10 MB
    stdout                — INFO and above (console)

Dependencies:
    pyyaml — for loading logging.yaml

Failure Scenarios:
    - If logging.yaml is missing, falls back to basicConfig (stdout only).
    - If logs/ directory cannot be created, raises PermissionError.
"""

from __future__ import annotations

import logging
import logging.config
import logging.handlers
from pathlib import Path
from typing import Final

import yaml

# ─── Paths ────────────────────────────────────────────────────────────────────
_ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent
_CONFIG_DIR: Final[Path] = _ROOT_DIR / "config"
_LOGS_DIR: Final[Path] = _ROOT_DIR / "logs"

_FALLBACK_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
)

_logging_configured: bool = False


def setup_logging(log_level: str | None = None) -> None:
    """
    Configure the logging system from config/logging.yaml.

    Must be called once at application startup (in main.py) before any
    logger is used. Subsequent calls are no-ops (idempotent).

    Args:
        log_level: Optional override for the root log level.
                   Accepts: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
                   If None, uses the level defined in logging.yaml.

    Raises:
        PermissionError: If the logs/ directory cannot be created.
    """
    global _logging_configured
    if _logging_configured:
        return

    # Ensure the logs directory exists before handlers try to open files
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    yaml_path = _CONFIG_DIR / "logging.yaml"

    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as fh:
            config: dict = yaml.safe_load(fh)

        # Resolve all filename values to absolute paths under logs/
        for handler_cfg in config.get("handlers", {}).values():
            if "filename" in handler_cfg:
                # Support both bare filenames and relative paths
                handler_cfg["filename"] = str(
                    _LOGS_DIR / Path(handler_cfg["filename"]).name
                )

        # Override root level if explicitly requested
        if log_level:
            config.setdefault("root", {})["level"] = log_level.upper()

        logging.config.dictConfig(config)
    else:
        # Graceful fallback — still usable without the YAML file
        logging.basicConfig(
            level=getattr(logging, (log_level or "INFO").upper()),
            format=_FALLBACK_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger scoped under the 'atlas.' namespace.

    If the provided name does not start with 'atlas.', it will be
    automatically prefixed. This ensures all Atlas loggers are handled
    by the 'atlas' logger configured in logging.yaml.

    Args:
        name: Module name — always pass __name__.

    Returns:
        A logging.Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("MarketDataService started")
        logger.error("Failed to fetch RELIANCE.NS", exc_info=True)
    """
    if not name.startswith("atlas."):
        name = f"atlas.{name}"
    return logging.getLogger(name)
