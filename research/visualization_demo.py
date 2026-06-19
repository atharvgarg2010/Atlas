"""
research/visualization_demo.py
=================================
Project Atlas — Visualization Engine v1 — Demo Run

Generates the 75-candle synthetic RELIANCE.NS dataset, runs it through
IndicatorEngine to enrich with indicators, injects a handful of synthetic
BUY/SELL signals, then renders the full 3-panel Atlas chart.

Run from project root:
    python research/visualization_demo.py

Output:
    - Opens interactive chart in your default browser
    - Saves chart to:  research/output/atlas_chart_demo.html

The chart contains:
    Panel 1: Candlestick + EMA-20 (amber) + EMA-50 (blue) + SMA-20 (dotted)
             + BUY markers (green triangles) + SELL markers (red triangles)
             + Volume overlay (semi-transparent)
    Panel 2: RSI-14 with overbought/oversold bands
    Panel 3: MACD + Signal line + MACD histogram
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Allow running from project root without pip install ───────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.technical.indicators import IndicatorEngine
from analytics.technical.charts import plot_atlas_chart
from research.indicator_engine_demo import build_sample_candles


# ─── Inject Synthetic Signals ─────────────────────────────────────────────────
# These are manually placed to demonstrate the signal marker rendering.
# In production, signals come from SignalService based on indicator thresholds.
#
# Signal logic used here (illustrative):
#   BUY  — RSI crossed up from oversold zone (< 35) during a pullback
#   SELL — RSI reached overbought zone (> 85) during a strong uptrend

def _inject_signals(candles: list[dict]) -> list[dict]:
    """
    Attach synthetic BUY / SELL signals to candles based on RSI thresholds.

    Rules (for demo purposes only — not production logic):
        BUY  when RSI-14 first crosses above 35 after being below 35
        SELL when RSI-14 first crosses above 80 (strong momentum sell)
    """
    prev_rsi = None
    for candle in candles:
        rsi = candle.get("rsi_14")
        candle["signal"] = None  # default

        if rsi is None or prev_rsi is None:
            prev_rsi = rsi
            continue

        # BUY: RSI crossing up from oversold
        if prev_rsi < 35 and rsi >= 35:
            candle["signal"] = "BUY"

        # SELL: RSI crossing into extreme overbought
        elif prev_rsi < 80 and rsi >= 80:
            candle["signal"] = "SELL"

        prev_rsi = rsi

    return candles


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  Project Atlas — Visualization Engine v1 Demo")
    print("=" * 60)

    # Step 1: Build sample candles
    raw = build_sample_candles()
    print(f"\n[1] Generated {len(raw)} synthetic candles (RELIANCE.NS)")

    # Step 2: Enrich with indicators
    engine = IndicatorEngine()
    enriched = engine.enrich(raw)
    print(f"[2] Enriched with IndicatorEngine: EMA/SMA/RSI/MACD/ATR")

    # Step 3: Inject synthetic signals
    enriched = _inject_signals(enriched)
    buy_count  = sum(1 for c in enriched if c.get("signal") == "BUY")
    sell_count = sum(1 for c in enriched if c.get("signal") == "SELL")
    print(f"[3] Injected signals: {buy_count} BUY, {sell_count} SELL")

    # Step 4: Build the chart
    fig = plot_atlas_chart(
        enriched,
        title="Atlas v1 Chart",
        symbol="RELIANCE.NS (Synthetic)",
    )
    print(f"[4] Chart built: 3 panels, {len(fig.data)} traces")

    # Step 5: Save HTML
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    html_path = output_dir / "atlas_chart_demo.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"[5] Saved to: {html_path}")

    # Step 6: Open in browser
    print("[6] Opening chart in browser...")
    fig.show()

    print("\n" + "=" * 60)
    print("  Done. Visualization Engine v1 verified.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
