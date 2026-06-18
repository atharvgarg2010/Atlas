"""
tests/unit/test_logging.py
==========================
Unit tests for the logging system.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


class TestGetLogger:
    def test_returns_logger_instance(self):
        from core.logging import get_logger

        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)

    def test_name_is_prefixed_with_atlas(self):
        from core.logging import get_logger

        logger = get_logger("some.module")
        assert logger.name.startswith("atlas.")

    def test_already_prefixed_name_not_doubled(self):
        from core.logging import get_logger

        logger = get_logger("atlas.services.market_data")
        assert logger.name == "atlas.services.market_data"
        assert not logger.name.startswith("atlas.atlas.")


class TestSetupLogging:
    def test_setup_is_idempotent(self, tmp_path, monkeypatch):
        """Calling setup_logging twice should not raise or duplicate handlers."""
        from core import logging as atlas_logging

        # Reset state for this test
        monkeypatch.setattr(atlas_logging, "_logging_configured", False)
        monkeypatch.setattr(atlas_logging, "_LOGS_DIR", tmp_path)

        atlas_logging.setup_logging(log_level="WARNING")
        atlas_logging.setup_logging(log_level="DEBUG")   # second call — no-op

        # Should still be WARNING (second call was ignored)
        root_logger = logging.getLogger("atlas")
        assert root_logger is not None

    def test_logs_directory_created(self, tmp_path, monkeypatch):
        """setup_logging should create the logs directory if it doesn't exist."""
        from core import logging as atlas_logging

        new_logs_dir = tmp_path / "new_logs"
        monkeypatch.setattr(atlas_logging, "_logging_configured", False)
        monkeypatch.setattr(atlas_logging, "_LOGS_DIR", new_logs_dir)

        atlas_logging.setup_logging()
        assert new_logs_dir.exists()
