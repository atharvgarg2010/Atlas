"""
analytics/technical/charts.py
================================
Project Atlas — Visualization Engine v1

Purpose
-------
Render professional-grade interactive trading charts from enriched candle
data produced by ``analytics.technical.indicators.IndicatorEngine``.

Output is a Plotly Figure with three synchronized panels:

    Panel 1 (70%): Candlestick + EMA-20 + EMA-50 + SMA-20
                   + optional BUY/SELL signal markers
    Panel 2 (15%): RSI-14 with overbought/oversold bands
    Panel 3 (15%): MACD line + Signal line + MACD histogram

Design Philosophy
-----------------
- Bloomberg-lite dark theme: charcoal background, white text, accent colours
  that are visible but not garish.
- No grid clutter — faint horizontal reference lines only.
- Volume bar chart rendered as a semi-transparent backdrop inside Panel 1.
- All panels share a linked x-axis so zoom/pan is synchronised.
- The chart is self-contained HTML (``fig.show()`` opens in browser) and can
  also be embedded in the Streamlit dashboard via ``st.plotly_chart(fig)``.

Public API
----------
    from analytics.technical.charts import plot_atlas_chart

    fig = plot_atlas_chart(candles)   # list[dict] → plotly.graph_objects.Figure
    fig.show()                        # opens browser
    fig.write_html("chart.html")      # save to file

Input schema (each dict in candles)
-------------------------------------
    Required: timestamp, open, high, low, close, volume
    Indicators: ema_20, ema_50, sma_20, rsi_14, macd, macd_signal, atr_14
    Optional:   signal  ("BUY" | "SELL" | "HOLD" | None)

Dependencies
------------
    plotly>=5.0
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Theme Constants ──────────────────────────────────────────────────────────

_BG = "#0D1117"           # near-black background (GitHub dark)
_PANEL_BG = "#161B22"     # slightly lighter panel fill
_GRID = "#21262D"         # very subtle gridlines
_TEXT = "#C9D1D9"         # primary text
_TEXT_DIM = "#6E7681"     # secondary labels
_BORDER = "#30363D"       # panel borders

# Candle colours
_BULL = "#26A69A"         # teal-green (bullish candle)
_BEAR = "#EF5350"         # warm red (bearish candle)

# Indicator line colours
_EMA20   = "#F0B429"      # amber
_EMA50   = "#7B68EE"      # medium slate blue
_SMA20   = "#4FC3F7"      # light blue
_MACD    = "#26A69A"      # teal (matches bull colour)
_MACD_SIG = "#FF7043"     # deep orange

# Signal marker colours
_BUY_COLOR  = "#00E676"   # bright green
_SELL_COLOR = "#FF1744"   # bright red

# Volume bar fill (transparent overlay inside price panel)
_VOLUME_BULL_FILL = "rgba(38,166,154,0.15)"
_VOLUME_BEAR_FILL = "rgba(239,83,80,0.15)"

# RSI reference levels
_RSI_OVERBOUGHT = 70
_RSI_OVERSOLD   = 30

# Panel height ratios: price / RSI / MACD
_ROW_HEIGHTS = [0.65, 0.175, 0.175]


# ─── Public API ───────────────────────────────────────────────────────────────

def plot_atlas_chart(
    candles: list[dict[str, Any]],
    title: str = "Atlas — Market Chart",
    symbol: str = "",
    show_volume: bool = True,
) -> go.Figure:
    """
    Build an interactive Atlas trading chart from enriched candle data.

    Args:
        candles:      List of OHLCV + indicator dicts (see module docstring).
        title:        Chart title shown in the top-left.
        symbol:       Optional ticker label appended to the title.
        show_volume:  Whether to render the transparent volume overlay.

    Returns:
        A ``plotly.graph_objects.Figure`` with three linked panels.
        Call ``.show()`` to open in browser or pass to ``st.plotly_chart()``.
    """
    if not candles:
        raise ValueError("candles list must not be empty")

    # ── Data extraction ────────────────────────────────────────────────────────
    timestamps  = [c["timestamp"] for c in candles]
    opens       = [c["open"]   for c in candles]
    highs       = [c["high"]   for c in candles]
    lows        = [c["low"]    for c in candles]
    closes      = [c["close"]  for c in candles]
    volumes     = [c["volume"] for c in candles]

    ema_20      = [c.get("ema_20")      for c in candles]
    ema_50      = [c.get("ema_50")      for c in candles]
    sma_20      = [c.get("sma_20")      for c in candles]
    rsi_14      = [c.get("rsi_14")      for c in candles]
    macd        = [c.get("macd")        for c in candles]
    macd_signal = [c.get("macd_signal") for c in candles]

    # Separate BUY and SELL signals
    buy_ts   = [c["timestamp"] for c in candles if c.get("signal") == "BUY"]
    buy_px   = [c["low"]  * 0.985 for c in candles if c.get("signal") == "BUY"]
    sell_ts  = [c["timestamp"] for c in candles if c.get("signal") == "SELL"]
    sell_px  = [c["high"] * 1.015 for c in candles if c.get("signal") == "SELL"]

    # MACD histogram (MACD - Signal)
    macd_hist = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd, macd_signal)
    ]
    hist_colors = [
        _BULL if (v is not None and v >= 0) else _BEAR
        for v in macd_hist
    ]

    # Volume bar colour (bull / bear)
    vol_colors = [
        _VOLUME_BULL_FILL if c >= o else _VOLUME_BEAR_FILL
        for c, o in zip(closes, opens)
    ]

    # ── Figure scaffold ───────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=_ROW_HEIGHTS,
    )

    chart_title = f"{title}  —  {symbol}" if symbol else title

    # ── Panel 1: Candlestick ──────────────────────────────────────────────────

    # Volume (semi-transparent bars behind candles)
    if show_volume:
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=volumes,
                marker_color=vol_colors,
                name="Volume",
                showlegend=False,
                yaxis="y2",
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="OHLC",
            increasing_line_color=_BULL,
            decreasing_line_color=_BEAR,
            increasing_fillcolor=_BULL,
            decreasing_fillcolor=_BEAR,
            line_width=1,
            whiskerwidth=0.5,
        ),
        row=1, col=1,
    )

    # EMA-20
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=ema_20,
            name="EMA 20",
            line=dict(color=_EMA20, width=1.2),
            connectgaps=False,
        ),
        row=1, col=1,
    )

    # EMA-50
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=ema_50,
            name="EMA 50",
            line=dict(color=_EMA50, width=1.2),
            connectgaps=False,
        ),
        row=1, col=1,
    )

    # SMA-20
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=sma_20,
            name="SMA 20",
            line=dict(color=_SMA20, width=1.0, dash="dot"),
            connectgaps=False,
        ),
        row=1, col=1,
    )

    # BUY signal markers
    if buy_ts:
        fig.add_trace(
            go.Scatter(
                x=buy_ts,
                y=buy_px,
                mode="markers",
                name="BUY",
                marker=dict(
                    symbol="triangle-up",
                    color=_BUY_COLOR,
                    size=12,
                    line=dict(color="white", width=0.5),
                ),
                hovertemplate="BUY<br>%{x}<extra></extra>",
            ),
            row=1, col=1,
        )

    # SELL signal markers
    if sell_ts:
        fig.add_trace(
            go.Scatter(
                x=sell_ts,
                y=sell_px,
                mode="markers",
                name="SELL",
                marker=dict(
                    symbol="triangle-down",
                    color=_SELL_COLOR,
                    size=12,
                    line=dict(color="white", width=0.5),
                ),
                hovertemplate="SELL<br>%{x}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Panel 2: RSI ──────────────────────────────────────────────────────────

    # Overbought band fill
    fig.add_hrect(
        y0=_RSI_OVERBOUGHT, y1=100,
        fillcolor="rgba(239,83,80,0.07)",
        line_width=0,
        row=2, col=1,
    )

    # Oversold band fill
    fig.add_hrect(
        y0=0, y1=_RSI_OVERSOLD,
        fillcolor="rgba(38,166,154,0.07)",
        line_width=0,
        row=2, col=1,
    )

    # RSI line
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=rsi_14,
            name="RSI 14",
            line=dict(color="#BA68C8", width=1.5),
            connectgaps=False,
        ),
        row=2, col=1,
    )

    # RSI reference lines
    for level, label in [(_RSI_OVERBOUGHT, "70"), (_RSI_OVERSOLD, "30"), (50, "50")]:
        fig.add_hline(
            y=level,
            line=dict(color=_GRID, width=0.8, dash="dot"),
            row=2, col=1,
        )

    # ── Panel 3: MACD ─────────────────────────────────────────────────────────

    # MACD histogram
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=macd_hist,
            name="MACD Hist",
            marker_color=hist_colors,
            marker_line_width=0,
            opacity=0.6,
            showlegend=False,
        ),
        row=3, col=1,
    )

    # MACD line
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=macd,
            name="MACD",
            line=dict(color=_MACD, width=1.5),
            connectgaps=False,
        ),
        row=3, col=1,
    )

    # Signal line
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=macd_signal,
            name="Signal",
            line=dict(color=_MACD_SIG, width=1.2),
            connectgaps=False,
        ),
        row=3, col=1,
    )

    # MACD zero line
    fig.add_hline(
        y=0,
        line=dict(color=_GRID, width=0.8),
        row=3, col=1,
    )

    # ── Layout & Theme ────────────────────────────────────────────────────────
    _apply_theme(fig, chart_title, timestamps)

    return fig


# ─── Theme Application ────────────────────────────────────────────────────────

def _apply_theme(
    fig: go.Figure,
    title: str,
    timestamps: list,
) -> None:
    """Apply the Bloomberg-lite dark theme to the figure."""

    fig.update_layout(
        # Background
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL_BG,

        # Title
        title=dict(
            text=title,
            font=dict(color=_TEXT, size=14, family="'Roboto Mono', monospace"),
            x=0.01,
            xanchor="left",
        ),

        # Legend
        legend=dict(
            bgcolor="rgba(22,27,34,0.8)",
            bordercolor=_BORDER,
            borderwidth=1,
            font=dict(color=_TEXT, size=11),
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),

        # Margins
        margin=dict(l=60, r=40, t=60, b=40),

        # Remove range slider (we use the linked axis instead)
        xaxis_rangeslider_visible=False,

        # Hover
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=_PANEL_BG,
            bordercolor=_BORDER,
            font=dict(color=_TEXT, size=11),
        ),

        # Figure size
        height=780,
        autosize=True,

        # Volume y-axis (secondary, inside panel 1)
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            showticklabels=False,
            range=[0, max(1, 5)],  # volume compressed to bottom 20%
        ),
    )

    # Shared x-axis styling
    common_xaxis = dict(
        gridcolor=_GRID,
        gridwidth=0.5,
        tickfont=dict(color=_TEXT_DIM, size=10),
        linecolor=_BORDER,
        showline=True,
        zeroline=False,
    )

    # Common y-axis styling
    common_yaxis = dict(
        gridcolor=_GRID,
        gridwidth=0.5,
        tickfont=dict(color=_TEXT_DIM, size=10),
        linecolor=_BORDER,
        showline=True,
        zeroline=False,
        side="right",
    )

    # Panel 1 axes (price)
    fig.update_xaxes(common_xaxis, row=1, col=1, showticklabels=False)
    fig.update_yaxes(
        **common_yaxis,
        title=dict(text="Price", font=dict(color=_TEXT_DIM, size=10)),
        row=1, col=1,
    )

    # Panel 2 axes (RSI)
    fig.update_xaxes(common_xaxis, row=2, col=1, showticklabels=False)
    fig.update_yaxes(
        **common_yaxis,
        title=dict(text="RSI", font=dict(color=_TEXT_DIM, size=10)),
        range=[0, 100],
        dtick=20,
        row=2, col=1,
    )

    # Panel 3 axes (MACD)
    fig.update_xaxes(
        **common_xaxis,
        row=3, col=1,
        showticklabels=True,
        tickformat="%b %d",
        tickangle=-30,
    )
    fig.update_yaxes(
        **common_yaxis,
        title=dict(text="MACD", font=dict(color=_TEXT_DIM, size=10)),
        row=3, col=1,
    )
