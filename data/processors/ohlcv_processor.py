"""
data/processors/ohlcv_processor.py
=====================================
Project Atlas — OHLCV Data Validator (Sprint 2)

Purpose
-------
Validate raw OHLCV records returned by YFinanceCollector before they
are persisted to the database. Any record failing a hard rule is dropped
with a WARNING log entry. Processing continues with the remaining records.

Validation Rules (applied in order)
-------------------------------------
1. Required fields    : 'ts', 'open', 'high', 'low', 'close', 'volume' present and non-None.
2. No NaN / Inf       : All price and volume fields are finite numbers.
3. Price positivity   : open > 0, high > 0, low > 0, close > 0.
4. Volume non-negative: volume >= 0  (zero volume on holiday/halt is allowed).
5. OHLC logic         : high >= max(open, close) AND low <= min(open, close).
6. Timestamp sanity   : ts is a valid datetime, not before 1990-01-01, not in the future.
7. Timezone awareness : ts must be timezone-aware (UTC). Naive datetimes are rejected.

Architecture
------------
- ``validate()`` is a pure function — no I/O, no DB access.
- Returns (valid_records, rejected_records) so the caller can log counts.
- Does NOT modify prices or volumes — only filters.

Dependencies
------------
    math, datetime — stdlib only. No pandas dependency here.

Failure Scenarios
-----------------
- Completely empty input → returns ([], []).
- All records invalid   → returns ([], all_records).
- Single bad field      → that record goes to rejected, rest continue.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from core.logging import get_logger

logger = get_logger(__name__)

# Sentinel minimum date — reject anything before Indian equity markets existed in modern form
_MIN_DATE = datetime(1990, 1, 1, tzinfo=timezone.utc)

# Required keys every record must have
_REQUIRED_KEYS = ("ts", "open", "high", "low", "close", "volume")


class OHLCVProcessor:
    """
    Stateless OHLCV validation processor.

    Usage
    -----
        processor = OHLCVProcessor()
        valid, rejected = processor.validate(records, symbol="RELIANCE.NS")

        # Persist valid records only
        repo.bulk_upsert(valid)
    """

    def validate(
        self,
        records: list[dict],
        symbol: str = "",
    ) -> tuple[list[dict], list[dict]]:
        """
        Validate a list of raw OHLCV dicts.

        Args:
            records: Raw dicts from YFinanceCollector.
            symbol:  Symbol name used for log messages (cosmetic only).

        Returns:
            Tuple of (valid_records, rejected_records).
            valid_records   — passed all rules, safe to insert.
            rejected_records — failed at least one rule.
        """
        if not records:
            return [], []

        valid: list[dict] = []
        rejected: list[dict] = []
        now_utc = datetime.now(tz=timezone.utc)

        for rec in records:
            reason = self._check(rec, now_utc)
            if reason is None:
                valid.append(rec)
            else:
                ts_str = rec.get("ts", "?")
                logger.warning(
                    f"[processor] {symbol} {ts_str} — REJECTED: {reason}"
                )
                rejected.append(rec)

        if rejected:
            logger.info(
                f"[processor] {symbol}: {len(valid)} valid, "
                f"{len(rejected)} rejected out of {len(records)} records"
            )

        return valid, rejected

    # ── Private helpers ────────────────────────────────────────────────────────

    def _check(self, rec: dict, now_utc: datetime) -> str | None:
        """
        Run all validation rules against a single record.

        Returns None if all rules pass, or a string description of the
        first failed rule.
        """
        # Rule 1: Required fields present and non-None
        for key in _REQUIRED_KEYS:
            if key not in rec or rec[key] is None:
                return f"missing required field '{key}'"

        ts = rec["ts"]
        o = rec["open"]
        h = rec["high"]
        lo = rec["low"]
        c = rec["close"]
        v = rec["volume"]

        # Rule 2: No NaN / Inf in price fields
        for name, val in (("open", o), ("high", h), ("low", lo), ("close", c)):
            try:
                if not math.isfinite(float(val)):
                    return f"'{name}' is NaN or Inf ({val!r})"
            except (TypeError, ValueError):
                return f"'{name}' is not a valid number ({val!r})"

        # Rule 3: Prices must be positive
        if float(o) <= 0:
            return f"open <= 0 ({o})"
        if float(h) <= 0:
            return f"high <= 0 ({h})"
        if float(lo) <= 0:
            return f"low <= 0 ({lo})"
        if float(c) <= 0:
            return f"close <= 0 ({c})"

        # Rule 4: Volume non-negative
        try:
            if int(v) < 0:
                return f"volume < 0 ({v})"
        except (TypeError, ValueError):
            return f"volume is not a valid integer ({v!r})"

        # Rule 5: OHLC price logic
        o_f, h_f, lo_f, c_f = float(o), float(h), float(lo), float(c)
        if h_f < max(o_f, c_f):
            return (
                f"high ({h_f}) < max(open={o_f}, close={c_f})"
            )
        if lo_f > min(o_f, c_f):
            return (
                f"low ({lo_f}) > min(open={o_f}, close={c_f})"
            )

        # Rule 6: Timestamp sanity
        if not isinstance(ts, datetime):
            return f"ts is not a datetime ({type(ts).__name__})"

        # Rule 7: Timezone-aware
        if ts.tzinfo is None:
            return "ts is timezone-naive (expected UTC-aware)"

        if ts < _MIN_DATE:
            return f"ts {ts.date()} is before minimum date {_MIN_DATE.date()}"

        if ts > now_utc:
            return f"ts {ts.date()} is in the future"

        return None  # all rules passed
