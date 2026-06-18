"""
tests/unit/test_config.py
=========================
Unit tests for the configuration system.
"""

from __future__ import annotations

import pytest

from config.settings import (
    MarketConfig,
    PaperTradingConfig,
    ScoringConfig,
    ScoringWeights,
    Settings,
)


class TestMarketConfig:
    def test_watchlist_is_tuple(self, test_settings):
        assert isinstance(test_settings.market.watchlist, tuple)

    def test_watchlist_not_empty(self, test_settings):
        assert len(test_settings.market.watchlist) > 0

    def test_watchlist_symbols_have_ns_suffix(self, test_settings):
        for symbol in test_settings.market.watchlist:
            assert symbol.endswith(".NS"), f"Symbol {symbol} missing .NS suffix"

    def test_history_days_positive(self, test_settings):
        assert test_settings.market.history_days > 0


class TestPaperTradingConfig:
    def test_initial_capital_matches_spec(self, test_settings):
        assert test_settings.paper_trading.initial_capital == 10_000.0

    def test_position_size_pct_matches_spec(self, test_settings):
        assert test_settings.paper_trading.position_size_pct == 0.10

    def test_currency_is_inr(self, test_settings):
        assert test_settings.paper_trading.currency == "INR"


class TestScoringWeights:
    def test_weights_sum_to_one(self, test_settings):
        w = test_settings.scoring.weights
        total = w.technical + w.momentum + w.news + w.volume
        assert abs(total - 1.0) < 0.001

    def test_invalid_weights_raise_error(self):
        bad_weights = ScoringWeights(
            technical=0.50,
            momentum=0.50,
            news=0.50,
            volume=0.50,
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            bad_weights.validate()


class TestSettings:
    def test_is_development_flag(self, test_settings):
        # test env is "test", neither dev nor prod
        assert not test_settings.is_production

    def test_database_url_not_empty(self, test_settings):
        assert test_settings.database_url
        assert "postgresql" in test_settings.database_url
