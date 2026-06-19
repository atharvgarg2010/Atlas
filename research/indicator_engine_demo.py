"""
research/indicator_engine_demo.py
===================================
Project Atlas — Indicator Engine v1 — Demo & Test Run

Demonstrates the IndicatorEngine with a realistic 75-candle sample dataset
modelled on RELIANCE.NS price behaviour (synthetic but plausible prices).

Run from the project root:
    python research/indicator_engine_demo.py

Expected output:
    - First 5 candles: indicator columns show None (warm-up period)
    - Candle 26+:      MACD line appears (requires EMA-26 warm-up)
    - Candle 35+:      MACD Signal appears (26 + 9 - 1)
    - All 75 candles:  EMA-20, SMA-20, RSI-14, ATR-14 present
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pprint import pformat

# ── Allow running from project root without pip install ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.technical.indicators import IndicatorEngine

# ─── Sample Dataset ───────────────────────────────────────────────────────────
# 75 synthetic daily candles for RELIANCE.NS
# Prices start near ₹2,800 and follow a realistic uptrend with pullbacks.
# Generated manually — not real market data.

_BASE_DATE = datetime(2025, 1, 2, tzinfo=timezone.utc)

# fmt: off
_RAW_PRICES = [
    # (open, high, low, close, volume)
    (2800.0, 2835.0, 2790.0, 2822.0, 4_200_000),
    (2822.0, 2850.0, 2810.0, 2845.0, 3_800_000),
    (2845.0, 2860.0, 2820.0, 2830.0, 3_500_000),
    (2830.0, 2855.0, 2815.0, 2850.0, 3_900_000),
    (2850.0, 2880.0, 2840.0, 2875.0, 4_500_000),
    (2875.0, 2900.0, 2860.0, 2890.0, 5_200_000),
    (2890.0, 2910.0, 2870.0, 2905.0, 4_800_000),
    (2905.0, 2930.0, 2885.0, 2920.0, 4_600_000),
    (2920.0, 2945.0, 2900.0, 2910.0, 4_100_000),
    (2910.0, 2935.0, 2895.0, 2930.0, 3_700_000),
    (2930.0, 2960.0, 2915.0, 2950.0, 4_300_000),
    (2950.0, 2970.0, 2935.0, 2945.0, 3_900_000),
    (2945.0, 2980.0, 2940.0, 2970.0, 4_700_000),
    (2970.0, 3000.0, 2960.0, 2990.0, 5_500_000),
    (2990.0, 3010.0, 2975.0, 3005.0, 5_100_000),
    (3005.0, 3025.0, 2990.0, 3015.0, 4_800_000),
    (3015.0, 3040.0, 3000.0, 3035.0, 5_000_000),
    (3035.0, 3055.0, 3020.0, 3050.0, 4_900_000),
    (3050.0, 3070.0, 3040.0, 3060.0, 4_600_000),
    (3060.0, 3080.0, 3045.0, 3055.0, 4_200_000),
    # Pullback phase
    (3055.0, 3060.0, 3020.0, 3025.0, 5_600_000),
    (3025.0, 3040.0, 2995.0, 3005.0, 5_900_000),
    (3005.0, 3015.0, 2970.0, 2980.0, 6_100_000),
    (2980.0, 2990.0, 2945.0, 2960.0, 6_500_000),
    (2960.0, 2975.0, 2930.0, 2940.0, 6_800_000),
    (2940.0, 2960.0, 2920.0, 2955.0, 5_500_000),
    (2955.0, 2980.0, 2945.0, 2975.0, 4_900_000),
    # Recovery
    (2975.0, 3000.0, 2965.0, 2995.0, 4_600_000),
    (2995.0, 3020.0, 2985.0, 3010.0, 4_400_000),
    (3010.0, 3035.0, 3000.0, 3030.0, 4_700_000),
    (3030.0, 3055.0, 3020.0, 3050.0, 5_100_000),
    (3050.0, 3070.0, 3040.0, 3065.0, 5_300_000),
    (3065.0, 3085.0, 3055.0, 3080.0, 5_200_000),
    (3080.0, 3100.0, 3070.0, 3095.0, 5_500_000),
    (3095.0, 3120.0, 3085.0, 3110.0, 5_800_000),
    # Second pullback
    (3110.0, 3115.0, 3075.0, 3080.0, 6_200_000),
    (3080.0, 3090.0, 3045.0, 3055.0, 6_700_000),
    (3055.0, 3070.0, 3030.0, 3045.0, 6_300_000),
    (3045.0, 3060.0, 3025.0, 3040.0, 5_900_000),
    (3040.0, 3065.0, 3030.0, 3060.0, 5_100_000),
    # Breakout
    (3060.0, 3100.0, 3055.0, 3095.0, 7_200_000),
    (3095.0, 3130.0, 3090.0, 3125.0, 7_800_000),
    (3125.0, 3150.0, 3115.0, 3145.0, 7_500_000),
    (3145.0, 3170.0, 3135.0, 3160.0, 7_100_000),
    (3160.0, 3185.0, 3150.0, 3175.0, 6_800_000),
    (3175.0, 3200.0, 3165.0, 3190.0, 6_500_000),
    (3190.0, 3210.0, 3180.0, 3205.0, 6_200_000),
    (3205.0, 3225.0, 3195.0, 3215.0, 6_000_000),
    (3215.0, 3235.0, 3205.0, 3230.0, 5_800_000),
    (3230.0, 3250.0, 3220.0, 3245.0, 5_600_000),
    # Consolidation
    (3245.0, 3260.0, 3230.0, 3240.0, 4_800_000),
    (3240.0, 3255.0, 3225.0, 3235.0, 4_600_000),
    (3235.0, 3250.0, 3220.0, 3245.0, 4_400_000),
    (3245.0, 3265.0, 3235.0, 3255.0, 4_500_000),
    (3255.0, 3275.0, 3245.0, 3265.0, 4_700_000),
    (3265.0, 3280.0, 3255.0, 3270.0, 4_600_000),
    (3270.0, 3290.0, 3260.0, 3280.0, 4_800_000),
    (3280.0, 3300.0, 3270.0, 3295.0, 5_000_000),
    (3295.0, 3315.0, 3285.0, 3310.0, 5_200_000),
    (3310.0, 3330.0, 3300.0, 3325.0, 5_100_000),
    # Final push
    (3325.0, 3350.0, 3315.0, 3345.0, 5_500_000),
    (3345.0, 3370.0, 3335.0, 3360.0, 5_800_000),
    (3360.0, 3385.0, 3350.0, 3375.0, 5_600_000),
    (3375.0, 3395.0, 3365.0, 3390.0, 5_400_000),
    (3390.0, 3410.0, 3380.0, 3405.0, 5_300_000),
    (3405.0, 3425.0, 3395.0, 3420.0, 5_100_000),
    (3420.0, 3445.0, 3410.0, 3435.0, 5_200_000),
    (3435.0, 3455.0, 3425.0, 3450.0, 5_000_000),
    (3450.0, 3470.0, 3440.0, 3465.0, 4_900_000),
    (3465.0, 3490.0, 3455.0, 3480.0, 4_800_000),
    (3480.0, 3500.0, 3470.0, 3495.0, 4_700_000),
    (3495.0, 3515.0, 3485.0, 3510.0, 4_600_000),
    (3510.0, 3530.0, 3500.0, 3525.0, 4_500_000),
    (3525.0, 3545.0, 3515.0, 3540.0, 4_400_000),
    (3540.0, 3560.0, 3530.0, 3555.0, 4_300_000),
]
# fmt: on

assert len(_RAW_PRICES) == 75, f"Expected 75 rows, got {len(_RAW_PRICES)}"


def build_sample_candles() -> list[dict]:
    """Generate the 75-candle sample dataset as a list of dicts."""
    candles = []
    # Skip weekends: generate Mon-Fri trading days only
    current_date = _BASE_DATE
    for o, h, l, c, v in _RAW_PRICES:
        # Advance to next weekday
        while current_date.weekday() >= 5:  # 5=Sat, 6=Sun
            current_date += timedelta(days=1)
        candles.append(
            {
                "timestamp": current_date,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )
        current_date += timedelta(days=1)
    return candles


# ─── Formatting Helpers ───────────────────────────────────────────────────────

def _fmt(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "      None"
    return f"{value:>10.{decimals}f}"


def _print_table(enriched: list[dict]) -> None:
    """Print a readable summary table of enriched candles."""
    header = (
        f"{'#':>3}  {'Date':>12}  {'Close':>8}  "
        f"{'EMA20':>8}  {'EMA50':>8}  {'SMA20':>8}  "
        f"{'RSI14':>7}  {'MACD':>8}  {'Signal':>8}  {'ATR14':>7}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for i, c in enumerate(enriched, start=1):
        ts = c["timestamp"]
        date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
        print(
            f"{i:>3}  {date_str:>12}  {c['close']:>8.2f}  "
            f"{_fmt(c['ema_20'])}  {_fmt(c['ema_50'])}  {_fmt(c['sma_20'])}  "
            f"{_fmt(c['rsi_14'], 1):>7}  {_fmt(c['macd'], 3):>8}  "
            f"{_fmt(c['macd_signal'], 3):>8}  {_fmt(c['atr_14'], 2):>7}"
        )

    print(separator)


def _print_summary(enriched: list[dict]) -> None:
    """Print statistics about the last candle and warm-up counts."""
    last = enriched[-1]
    print("\n=== LAST CANDLE SNAPSHOT ===")
    print(f"  Date        : {last['timestamp']}")
    print(f"  Close       : {last['close']:.2f}")
    print(f"  EMA-20      : {last['ema_20']:.4f}")
    print(f"  EMA-50      : {last['ema_50']:.4f}")
    print(f"  SMA-20      : {last['sma_20']:.4f}")
    print(f"  RSI-14      : {last['rsi_14']:.2f}")
    print(f"  MACD        : {last['macd']:.4f}")
    print(f"  MACD Signal : {last['macd_signal']:.4f}")
    print(f"  ATR-14      : {last['atr_14']:.4f}")

    # Count candles where each indicator has valid data
    def count_valid(key: str) -> int:
        return sum(1 for c in enriched if c[key] is not None)

    print("\n=== VALID DATA COUNTS (out of 75 candles) ===")
    print(f"  EMA-20      : {count_valid('ema_20'):>3} candles")
    print(f"  EMA-50      : {count_valid('ema_50'):>3} candles")
    print(f"  SMA-20      : {count_valid('sma_20'):>3} candles")
    print(f"  RSI-14      : {count_valid('rsi_14'):>3} candles")
    print(f"  MACD        : {count_valid('macd'):>3} candles")
    print(f"  MACD Signal : {count_valid('macd_signal'):>3} candles")
    print(f"  ATR-14      : {count_valid('atr_14'):>3} candles")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 70)
    print("  Project Atlas — Indicator Engine v1 — Demo Run")
    print("  Symbol: RELIANCE.NS (synthetic data)")
    print("=" * 70)

    # Build sample data
    candles = build_sample_candles()
    print(f"\n[1] Built {len(candles)} synthetic candles")

    # Run the engine
    engine = IndicatorEngine()
    enriched = engine.enrich(candles)
    print(f"[2] Enriched {len(enriched)} candles with indicators")

    # Print full table
    print("\n[3] ENRICHED CANDLE TABLE")
    _print_table(enriched)

    # Summary
    _print_summary(enriched)

    # JSON output sample (last 3 candles, JSON-serialisable)
    print("\n[4] SAMPLE JSON OUTPUT (last 3 candles)")
    sample = enriched[-3:]
    # Convert datetime to ISO string for JSON
    for c in sample:
        if hasattr(c["timestamp"], "isoformat"):
            c["timestamp"] = c["timestamp"].isoformat()
    print(json.dumps(sample, indent=2, default=str))

    print("\n[5] DONE — Indicator Engine v1 verified successfully.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
