"""
tests/unit/test_yfinance_collector.py
=======================================
Unit tests for YFinanceCollector using mocked yfinance.

No network calls — yf.download is patched throughout.
Tests cover: happy path, empty response, missing columns, malformed rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.collectors.yfinance_collector import YFinanceCollector


@pytest.fixture
def collector() -> YFinanceCollector:
    return YFinanceCollector()


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal yfinance-style DataFrame from a list of dicts."""
    index = [pd.Timestamp(r["date"], tz="UTC") for r in rows]
    data = {
        "Open":   [r["open"] for r in rows],
        "High":   [r["high"] for r in rows],
        "Low":    [r["low"] for r in rows],
        "Close":  [r["close"] for r in rows],
        "Volume": [r["volume"] for r in rows],
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(index))


_SAMPLE_ROWS = [
    {"date": "2024-01-02", "open": 2800.0, "high": 2850.0, "low": 2780.0, "close": 2830.0, "volume": 1_000_000},
    {"date": "2024-01-03", "open": 2830.0, "high": 2900.0, "low": 2820.0, "close": 2890.0, "volume": 1_200_000},
    {"date": "2024-01-04", "open": 2890.0, "high": 2910.0, "low": 2860.0, "close": 2875.0, "volume": 900_000},
]


class TestFetchHistoryPeriod:
    @patch("data.collectors.yfinance_collector.yf.download")
    def test_returns_list_of_dicts(self, mock_dl, collector):
        mock_dl.return_value = _make_df(_SAMPLE_ROWS)
        result = collector.fetch_history_period("RELIANCE.NS", period="5d")
        assert isinstance(result, list)
        assert len(result) == 3

    @patch("data.collectors.yfinance_collector.yf.download")
    def test_each_record_has_required_keys(self, mock_dl, collector):
        mock_dl.return_value = _make_df(_SAMPLE_ROWS)
        result = collector.fetch_history_period("RELIANCE.NS", period="5d")
        for rec in result:
            assert "ts" in rec
            assert "open" in rec
            assert "high" in rec
            assert "low" in rec
            assert "close" in rec
            assert "volume" in rec
            assert "symbol" in rec

    @patch("data.collectors.yfinance_collector.yf.download")
    def test_ts_is_timezone_aware(self, mock_dl, collector):
        mock_dl.return_value = _make_df(_SAMPLE_ROWS)
        result = collector.fetch_history_period("RELIANCE.NS", period="5d")
        for rec in result:
            assert rec["ts"].tzinfo is not None

    @patch("data.collectors.yfinance_collector.yf.download")
    def test_symbol_is_set_on_every_record(self, mock_dl, collector):
        mock_dl.return_value = _make_df(_SAMPLE_ROWS)
        result = collector.fetch_history_period("TCS.NS", period="5d")
        for rec in result:
            assert rec["symbol"] == "TCS.NS"

    @patch("data.collectors.yfinance_collector.yf.download")
    def test_empty_dataframe_returns_empty_list(self, mock_dl, collector):
        mock_dl.return_value = pd.DataFrame()
        result = collector.fetch_history_period("UNKNOWN.NS", period="5d")
        assert result == []

    @patch("data.collectors.yfinance_collector.yf.download")
    def test_none_response_returns_empty_list(self, mock_dl, collector):
        mock_dl.return_value = None
        result = collector.fetch_history_period("UNKNOWN.NS", period="5d")
        assert result == []

    @patch("data.collectors.yfinance_collector.yf.download")
    def test_missing_columns_returns_empty_list(self, mock_dl, collector):
        """If yfinance returns a DF without expected columns, return []."""
        bad_df = pd.DataFrame({"Wrong": [1, 2, 3]})
        mock_dl.return_value = bad_df
        result = collector.fetch_history_period("RELIANCE.NS", period="5d")
        assert result == []


class TestFetchHistory:
    @patch("data.collectors.yfinance_collector.yf.download")
    def test_date_range_fetch_returns_records(self, mock_dl, collector):
        mock_dl.return_value = _make_df(_SAMPLE_ROWS)
        result = collector.fetch_history(
            "RELIANCE.NS",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        )
        assert len(result) == 3

    @patch("data.collectors.yfinance_collector.yf.download")
    def test_volume_is_integer(self, mock_dl, collector):
        mock_dl.return_value = _make_df(_SAMPLE_ROWS)
        result = collector.fetch_history_period("RELIANCE.NS", period="5d")
        for rec in result:
            assert isinstance(rec["volume"], int)
