"""
tests/unit/test_ohlcv_processor.py
====================================
Unit tests for OHLCVProcessor validation rules.

No database, no network, no external dependencies.
Tests every validation rule independently with valid/invalid inputs.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.processors.ohlcv_processor import OHLCVProcessor


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _valid_record(**overrides) -> dict:
    """Return a valid OHLCV record, with optional field overrides."""
    rec = {
        "symbol": "RELIANCE.NS",
        "ts": datetime(2024, 6, 1, tzinfo=timezone.utc),
        "open": 2900.0,
        "high": 2950.0,
        "low": 2870.0,
        "close": 2930.0,
        "volume": 5_000_000,
    }
    rec.update(overrides)
    return rec


@pytest.fixture
def processor() -> OHLCVProcessor:
    return OHLCVProcessor()


# ─── Rule 1: Required fields ─────────────────────────────────────────────────

class TestRequiredFields:
    def test_valid_record_passes(self, processor):
        valid, rejected = processor.validate([_valid_record()])
        assert len(valid) == 1
        assert len(rejected) == 0

    @pytest.mark.parametrize("missing_key", ["ts", "open", "high", "low", "close", "volume"])
    def test_missing_field_is_rejected(self, processor, missing_key):
        rec = _valid_record()
        del rec[missing_key]
        valid, rejected = processor.validate([rec], symbol="RELIANCE.NS")
        assert len(rejected) == 1
        assert len(valid) == 0

    @pytest.mark.parametrize("null_key", ["open", "high", "low", "close", "volume"])
    def test_none_field_is_rejected(self, processor, null_key):
        rec = _valid_record(**{null_key: None})
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1


# ─── Rule 2: NaN / Inf ───────────────────────────────────────────────────────

class TestNaNInf:
    @pytest.mark.parametrize("field", ["open", "high", "low", "close"])
    def test_nan_price_is_rejected(self, processor, field):
        import math
        rec = _valid_record(**{field: math.nan})
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1

    @pytest.mark.parametrize("field", ["open", "high", "low", "close"])
    def test_inf_price_is_rejected(self, processor, field):
        import math
        rec = _valid_record(**{field: math.inf})
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1


# ─── Rule 3: Price positivity ────────────────────────────────────────────────

class TestPricePositivity:
    @pytest.mark.parametrize("field", ["open", "high", "low", "close"])
    def test_zero_price_is_rejected(self, processor, field):
        rec = _valid_record(**{field: 0.0})
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1

    @pytest.mark.parametrize("field", ["open", "high", "low", "close"])
    def test_negative_price_is_rejected(self, processor, field):
        rec = _valid_record(**{field: -1.0})
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1


# ─── Rule 4: Volume ──────────────────────────────────────────────────────────

class TestVolume:
    def test_zero_volume_is_allowed(self, processor):
        """Zero volume occurs on market holidays — must NOT be rejected."""
        rec = _valid_record(volume=0)
        valid, rejected = processor.validate([rec])
        assert len(valid) == 1
        assert len(rejected) == 0

    def test_negative_volume_is_rejected(self, processor):
        rec = _valid_record(volume=-100)
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1


# ─── Rule 5: OHLC logic ──────────────────────────────────────────────────────

class TestOHLCLogic:
    def test_high_below_close_is_rejected(self, processor):
        # high < close — impossible
        rec = _valid_record(open=2900.0, high=2920.0, low=2870.0, close=2930.0)
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1

    def test_high_below_open_is_rejected(self, processor):
        # high < open — impossible
        rec = _valid_record(open=2950.0, high=2920.0, low=2870.0, close=2930.0)
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1

    def test_low_above_open_is_rejected(self, processor):
        # low > open — impossible
        rec = _valid_record(open=2900.0, high=2950.0, low=2910.0, close=2930.0)
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1

    def test_low_above_close_is_rejected(self, processor):
        # low > close — impossible
        rec = _valid_record(open=2950.0, high=2960.0, low=2940.0, close=2930.0)
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1

    def test_high_equals_low_equals_ohlc_is_valid(self, processor):
        """Flat candle (no movement) — valid on market halt."""
        rec = _valid_record(open=2900.0, high=2900.0, low=2900.0, close=2900.0)
        valid, rejected = processor.validate([rec])
        assert len(valid) == 1


# ─── Rule 6 & 7: Timestamps ──────────────────────────────────────────────────

class TestTimestamps:
    def test_future_timestamp_is_rejected(self, processor):
        from datetime import timedelta
        rec = _valid_record(ts=datetime.now(tz=timezone.utc) + timedelta(days=1))
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1

    def test_pre_1990_timestamp_is_rejected(self, processor):
        rec = _valid_record(ts=datetime(1989, 12, 31, tzinfo=timezone.utc))
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1

    def test_naive_timestamp_is_rejected(self, processor):
        rec = _valid_record(ts=datetime(2024, 1, 1))  # no tzinfo
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1

    def test_non_datetime_timestamp_is_rejected(self, processor):
        rec = _valid_record(ts="2024-01-01")
        valid, rejected = processor.validate([rec])
        assert len(rejected) == 1


# ─── Batch behaviour ─────────────────────────────────────────────────────────

class TestBatchBehaviour:
    def test_empty_input_returns_empty_tuples(self, processor):
        valid, rejected = processor.validate([])
        assert valid == []
        assert rejected == []

    def test_mixed_batch_splits_correctly(self, processor):
        """Only the bad record is rejected — valid ones pass through."""
        records = [
            _valid_record(ts=datetime(2024, 1, i, tzinfo=timezone.utc))
            for i in range(1, 6)
        ]
        # Corrupt the middle record
        records[2]["open"] = -999.0
        valid, rejected = processor.validate(records)
        assert len(valid) == 4
        assert len(rejected) == 1

    def test_all_valid_records_pass(self, processor):
        records = [
            _valid_record(ts=datetime(2024, 1, i, tzinfo=timezone.utc))
            for i in range(1, 11)
        ]
        valid, rejected = processor.validate(records)
        assert len(valid) == 10
        assert len(rejected) == 0
