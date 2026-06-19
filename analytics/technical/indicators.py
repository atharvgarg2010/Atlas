"""
analytics/technical/indicators.py
===================================
Project Atlas — Technical Indicator Engine v1

Purpose
-------
Transform a raw list of OHLCV candle dicts into enriched candle dicts that
include all technical indicators required by the Atlas signal engine.

Design Decisions
----------------
- Pure pandas + numpy — no dependency on the ``ta`` library at the core
  computation level. This keeps the module independently testable and
  portable. The IndicatorService (services/indicator_service.py) may use
  ``ta`` for additional indicators in later sprints.
- All functions are *pure* (no I/O, no global state). They accept a
  ``pd.DataFrame`` and return a new ``pd.DataFrame`` with new columns
  appended.
- The public API is ``IndicatorEngine.enrich(candles)`` which accepts a
  list of OHLCV dicts and returns a list of enriched dicts. All internal
  pandas work is hidden from callers.
- NaN values for the warm-up period (e.g. first 50 candles before EMA-50
  is valid) are preserved as ``None`` in the output dicts so callers can
  decide how to handle them.

Indicators Computed (v1)
------------------------
    EMA-20        Exponential Moving Average (period=20)
    EMA-50        Exponential Moving Average (period=50)
    SMA-20        Simple Moving Average (period=20)
    RSI-14        Relative Strength Index (period=14)
    MACD          MACD line (EMA12 - EMA26)
    MACD Signal   9-period EMA of MACD line
    ATR-14        Average True Range (period=14)

Usage
-----
    from analytics.technical.indicators import IndicatorEngine

    engine = IndicatorEngine()
    enriched = engine.enrich(raw_candles)   # list[dict] -> list[dict]

Input schema (each dict in raw_candles)
----------------------------------------
    {
        "timestamp": datetime | str,
        "open":      float,
        "high":      float,
        "low":       float,
        "close":     float,
        "volume":    int,
    }

Output schema (each dict in returned list)
-------------------------------------------
    {
        "timestamp":   ...,
        "open":        float,
        "high":        float,
        "low":         float,
        "close":       float,
        "volume":      int,
        "ema_20":      float | None,
        "ema_50":      float | None,
        "sma_20":      float | None,
        "rsi_14":      float | None,
        "macd":        float | None,
        "macd_signal": float | None,
        "atr_14":      float | None,
    }

Dependencies
------------
    pandas>=1.5, numpy>=1.23
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ─── Indicator Window Constants ───────────────────────────────────────────────
# Defined once here — never duplicated in individual functions.

EMA_FAST = 20
EMA_SLOW = 50
SMA_PERIOD = 20
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14


# ─── Core Indicator Functions (operate on pd.Series / pd.DataFrame) ───────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Uses pandas ``ewm`` with ``adjust=False`` which matches the standard
    EMA recurrence: EMA_t = close_t * k + EMA_{t-1} * (1-k)
    where k = 2 / (period + 1).

    Args:
        series: Close price series.
        period: EMA period.

    Returns:
        pd.Series of EMA values (NaN for warm-up candles).
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        series: Close price series.
        period: Lookback window.

    Returns:
        pd.Series of SMA values (NaN for first ``period - 1`` rows).
    """
    return series.rolling(window=period, min_periods=period).mean()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothed method).

    Implementation follows the original Wilder (1978) calculation:
        - Average gain / loss computed using Wilder's smoothing (EWM with
          ``com = period - 1``, ``adjust=False``), which is equivalent to
          Wilder's method and matches most trading platforms.

    Args:
        series: Close price series.
        period: RSI period (typically 14).

    Returns:
        pd.Series of RSI values in [0, 100] range.
        NaN for the first ``period`` rows (warm-up).
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's EMA: com = period - 1  ↔  alpha = 1 / period
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_macd(
    series: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence).

    MACD Line   = EMA(fast) - EMA(slow)
    Signal Line = EMA(signal) of MACD Line

    Args:
        series: Close price series.
        fast:   Fast EMA period (typically 12).
        slow:   Slow EMA period (typically 26).
        signal: Signal line EMA period (typically 9).

    Returns:
        Tuple of (macd_line, signal_line) as pd.Series.
        NaN during warm-up (first ``slow + signal - 1`` rows).
    """
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """
    Average True Range (Wilder's smoothing).

    True Range = max(
        high - low,
        |high - prev_close|,
        |low  - prev_close|
    )
    ATR = Wilder's EMA of True Range over ``period`` bars.

    Args:
        high:   High price series.
        low:    Low price series.
        close:  Close price series.
        period: ATR period (typically 14).

    Returns:
        pd.Series of ATR values. NaN for first ``period`` rows.
    """
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's EMA: com = period - 1
    atr = tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    return atr


# ─── Public API ───────────────────────────────────────────────────────────────

class IndicatorEngine:
    """
    Stateless indicator computation engine.

    Accepts a list of raw OHLCV candle dicts and returns the same list
    enriched with all v1 technical indicators.

    Example
    -------
        engine = IndicatorEngine()
        enriched = engine.enrich(raw_candles)

        for candle in enriched:
            print(candle["timestamp"], candle["rsi_14"])
    """

    def enrich(self, candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Compute all v1 indicators and return enriched candle list.

        Candles must be sorted chronologically (oldest first).
        Partial NaN values in the first N rows are expected and preserved.

        Args:
            candles: List of OHLCV dicts. Minimum 1 candle required.
                     Accuracy improves with more historical data
                     (recommend >= 60 candles for all indicators to warm up).

        Returns:
            List of dicts with original OHLCV fields + indicator columns.
            Length matches input exactly.

        Raises:
            ValueError: If candles is empty or missing required OHLCV keys.
        """
        if not candles:
            raise ValueError("candles list must not be empty")

        df = self._to_dataframe(candles)
        df = self._compute_all(df)
        return self._to_records(df)

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _to_dataframe(candles: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert input list of dicts to a typed DataFrame."""
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(candles[0].keys())
        if missing:
            raise ValueError(f"Candles are missing required keys: {missing}")

        df = pd.DataFrame(candles)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Ensure numeric dtypes
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def _compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all indicator functions and attach columns to DataFrame."""
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # ── Trend ──────────────────────────────────────────────────────────────
        df["ema_20"] = compute_ema(close, EMA_FAST)
        df["ema_50"] = compute_ema(close, EMA_SLOW)
        df["sma_20"] = compute_sma(close, SMA_PERIOD)

        # ── Momentum ───────────────────────────────────────────────────────────
        df["rsi_14"] = compute_rsi(close, RSI_PERIOD)

        macd_line, signal_line = compute_macd(
            close, MACD_FAST, MACD_SLOW, MACD_SIGNAL
        )
        df["macd"]        = macd_line
        df["macd_signal"] = signal_line

        # ── Volatility ─────────────────────────────────────────────────────────
        df["atr_14"] = compute_atr(high, low, close, ATR_PERIOD)

        return df

    @staticmethod
    def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Convert DataFrame back to list of dicts.
        Replace numpy NaN with None for clean JSON-serialisable output.
        """
        # Round indicator values to 4 decimal places
        indicator_cols = [
            "ema_20", "ema_50", "sma_20",
            "rsi_14", "macd", "macd_signal", "atr_14",
        ]
        for col in indicator_cols:
            if col in df.columns:
                df[col] = df[col].round(4)

        # NaN → None so the output is JSON-serialisable
        records = df.where(pd.notna(df), other=None).to_dict(orient="records")
        return records
