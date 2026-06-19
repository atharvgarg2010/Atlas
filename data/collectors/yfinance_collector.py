"""
data/collectors/yfinance_collector.py
=======================================
Project Atlas — Yahoo Finance Data Collector (Sprint 2)

Purpose
-------
Fetch OHLCV historical data for NSE symbols using yfinance.
This module is the ONLY place in the codebase that imports yfinance.
All other modules receive data as plain Python dicts.

Architecture
------------
- ``fetch_history(symbol, start, end)`` downloads a date range.
  Called with delta range on subsequent runs (not full history every time).
- ``fetch_history_period(symbol, period)`` downloads by period string
  (e.g. '1y') — used for the initial 365-day backfill.
- Both methods are decorated with @retry for network resilience.
- yfinance ``auto_adjust=True`` adjusts OHLCV for splits/dividends.
- ``progress=False`` suppresses yfinance's tqdm progress bars in scheduler.

Output format (list of dicts)
------------------------------
    {
        'ts': datetime (UTC, timezone-aware),
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'volume': int,
    }

Failure Scenarios
-----------------
- Empty DataFrame (unlisted symbol, market holiday)  → return []
- HTTP 429 rate limit                                 → @retry handles
- Network timeout                                     → @retry handles
- Malformed column names from yfinance                → caught, logged, []

Dependencies
------------
    yfinance, pandas, core.decorators.retry, core.logging
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timezone

import pandas as pd
import yfinance as yf

from core.decorators import retry
from core.logging import get_logger

logger = get_logger(__name__)

# Suppress yfinance/pandas FutureWarnings — not actionable at our level
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


class YFinanceCollector:
    """
    Thin adapter around yfinance for OHLCV data fetching.

    All methods return a list of OHLCV dicts with standardised keys.
    Returns an empty list on any unrecoverable failure (never raises).
    """

    # Columns yfinance returns with auto_adjust=True (no Adj Close)
    _REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}

    def fetch_history(
        self,
        symbol: str,
        start: date | datetime | str,
        end: date | datetime | str,
        interval: str = "1d",
    ) -> list[dict]:
        """
        Fetch OHLCV data for a date range.

        Args:
            symbol:   NSE ticker with .NS suffix — e.g. 'RELIANCE.NS'.
            start:    Start date (inclusive). Accepts date, datetime, or 'YYYY-MM-DD'.
            end:      End date (exclusive). Accepts date, datetime, or 'YYYY-MM-DD'.
            interval: yfinance interval string. Default '1d'.

        Returns:
            List of OHLCV dicts. Empty list on failure or no data.
        """
        return self._download(symbol=symbol, interval=interval, start=start, end=end)

    def fetch_history_period(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[dict]:
        """
        Fetch OHLCV data using a yfinance period string.

        Args:
            symbol:   NSE ticker with .NS suffix.
            period:   yfinance period — '1d', '5d', '1mo', '3mo', '6mo',
                      '1y', '2y', '5y', '10y', 'ytd', 'max'.
            interval: yfinance interval string. Default '1d'.

        Returns:
            List of OHLCV dicts. Empty list on failure or no data.
        """
        return self._download(symbol=symbol, interval=interval, period=period)

    @retry(max_attempts=3, backoff_seconds=2.0, exceptions=(Exception,))
    def _download(self, symbol: str, interval: str, **kwargs) -> list[dict]:
        """
        Internal download method, decorated with @retry.

        ``kwargs`` is forwarded to ``yf.download`` and contains either
        ``period`` or ``(start, end)`` — never both.
        """
        logger.debug(f"[yfinance] Downloading {symbol!r} interval={interval!r} {kwargs}")

        # Suppress yfinance's noisy console output
        df: pd.DataFrame = yf.download(
            tickers=symbol,
            interval=interval,
            auto_adjust=True,
            progress=False,
            **kwargs,
        )

        if df is None or df.empty:
            logger.warning(f"[yfinance] Empty response for {symbol!r}")
            return []

        # yfinance may return a MultiIndex if multiple tickers are requested.
        # Flatten to single level for our single-symbol calls.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Validate expected columns exist
        missing = self._REQUIRED_COLS - set(df.columns)
        if missing:
            logger.error(
                f"[yfinance] {symbol!r} missing columns {missing}. "
                f"Got: {list(df.columns)}"
            )
            return []

        records = self._to_records(symbol, df)
        logger.info(f"[yfinance] {symbol!r}: {len(records)} rows downloaded")
        return records

    @staticmethod
    def _to_records(symbol: str, df: pd.DataFrame) -> list[dict]:
        """
        Convert a yfinance DataFrame to a list of OHLCV dicts.

        The DataFrame index is the candle date (DatetimeIndex). We convert
        each date to a timezone-aware datetime at midnight UTC so it can be
        stored in a TIMESTAMPTZ column without ambiguity.
        """
        records: list[dict] = []

        for ts, row in df.iterrows():
            try:
                # Normalise timestamp to midnight UTC
                if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                    dt = ts.to_pydatetime().astimezone(timezone.utc)
                else:
                    dt = datetime(ts.year, ts.month, ts.day, tzinfo=timezone.utc)

                records.append(
                    {
                        "symbol": symbol,
                        "ts": dt,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]),
                    }
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    f"[yfinance] Skipping malformed row for {symbol!r} "
                    f"at {ts}: {exc}"
                )

        return records
