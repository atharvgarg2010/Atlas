"""
analytics/technical/charts.py
================================
Project Atlas — Visualization Engine v2

Three-panel chart (no backtest data) OR four-panel chart (with backtest data):

    Panel 1 (Price):        Candlestick + EMA-20 + EMA-50 + SMA-20 + Volume
                            Strategy signal markers (faint, from candle["signal"])
                            Executed trade markers (prominent, from executed_trades)
    Panel 2 (RSI):          RSI-14 with overbought / oversold bands
    Panel 3 (MACD):         MACD line + Signal line + Histogram
    Panel 4 (Equity Curve): Portfolio value over time (only if equity_curve provided)

Public API
----------
    from analytics.technical.charts import plot_atlas_chart

    # Without backtest data (3 panels):
    fig = plot_atlas_chart(enriched_candles, symbol="TITAN.NS")

    # With backtest data (4 panels):
    fig = plot_atlas_chart(
        enriched_candles,
        symbol="TITAN.NS",
        executed_trades=result.trade_log,        # list[PortfolioTradeRecord]
        equity_curve=result.equity_curve_data,   # list[{timestamp, portfolio_value}]
    )

Executed Trade Marker Types
----------------------------
    BUY          → green  ▲  at exact execution price
    SELL_PARTIAL → orange ◀  at exact execution price
    SELL_FULL    → red    ▼  at exact execution price
    FORCE_CLOSE  → gray   ⬧  at exact execution price

Hover text on trade markers shows:
    price, quantity, avg_cost, pnl, cash_after, strategy
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Theme ────────────────────────────────────────────────────────────────────

_BG       = "#0D1117"
_PANEL_BG = "#161B22"
_GRID     = "#21262D"
_TEXT     = "#C9D1D9"
_TEXT_DIM = "#6E7681"
_BORDER   = "#30363D"

_BULL     = "#26A69A"
_BEAR     = "#EF5350"

_EMA20    = "#F0B429"
_EMA50    = "#7B68EE"
_SMA20    = "#4FC3F7"
_MACD_L   = "#26A69A"
_MACD_SIG = "#FF7043"

# Executed trade marker colours
_MRK_BUY     = "#00E676"   # bright green
_MRK_SELL_P  = "#FF9800"   # orange (partial)
_MRK_SELL_F  = "#FF1744"   # red    (full)
_MRK_FORCE   = "#78909C"   # grey   (force close)

# Strategy-level signal colours (subtle background markers)
_SIG_BUY  = "rgba(0,230,118,0.45)"
_SIG_SELL = "rgba(255,23,68,0.45)"

_EQUITY_LINE = "#00BCD4"   # cyan for equity curve

_RSI_OB = 70
_RSI_OS = 30

_VOLUME_BULL = "rgba(38,166,154,0.15)"
_VOLUME_BEAR = "rgba(239,83,80,0.15)"


# ─── Public API ───────────────────────────────────────────────────────────────

def plot_atlas_chart(
    candles: list[dict[str, Any]],
    title: str = "Atlas — Market Chart",
    symbol: str = "",
    show_volume: bool = True,
    executed_trades: list[Any] | None = None,
    equity_curve: list[dict] | None = None,
) -> go.Figure:
    """
    Build an interactive Atlas trading chart.

    Args:
        candles:         Enriched OHLCV dicts (required).
        title:           Chart title prefix.
        symbol:          Ticker label shown in title.
        show_volume:     Render semi-transparent volume bars behind candles.
        executed_trades: List of PortfolioTradeRecord objects (or dicts).
                         When provided, adds precise execution markers to the
                         price panel with rich hover text.
        equity_curve:    List of {timestamp, portfolio_value} dicts.
                         When provided, adds a 4th equity curve panel.

    Returns:
        plotly.graph_objects.Figure  (call .show() or pass to st.plotly_chart())
    """
    if not candles:
        raise ValueError("candles list must not be empty")

    # Normalise executed_trades: accept dataclass instances or plain dicts
    if executed_trades:
        norm_trades = []
        for t in executed_trades:
            if isinstance(t, dict):
                norm_trades.append(t)
            else:
                try:
                    norm_trades.append(asdict(t))
                except Exception:
                    norm_trades.append(t.__dict__)
        executed_trades = norm_trades

    has_equity  = bool(equity_curve)
    n_rows      = 4 if has_equity else 3
    row_heights = [0.52, 0.15, 0.15, 0.18] if has_equity else [0.65, 0.175, 0.175]

    # ── Data extraction ────────────────────────────────────────────────────────
    timestamps = [c["timestamp"] for c in candles]
    opens      = [c["open"]   for c in candles]
    highs      = [c["high"]   for c in candles]
    lows       = [c["low"]    for c in candles]
    closes     = [c["close"]  for c in candles]
    volumes    = [c["volume"] for c in candles]

    ema_20      = [c.get("ema_20")      for c in candles]
    ema_50      = [c.get("ema_50")      for c in candles]
    sma_20      = [c.get("sma_20")      for c in candles]
    rsi_14      = [c.get("rsi_14")      for c in candles]
    macd        = [c.get("macd")        for c in candles]
    macd_signal = [c.get("macd_signal") for c in candles]

    # Simple strategy-level signal markers (from candle["signal"])
    buy_sig_ts  = [c["timestamp"] for c in candles if c.get("signal") == "BUY"]
    buy_sig_px  = [c["low"] * 0.984 for c in candles if c.get("signal") == "BUY"]
    sell_sig_ts = [c["timestamp"] for c in candles if c.get("signal") == "SELL"]
    sell_sig_px = [c["high"] * 1.016 for c in candles if c.get("signal") == "SELL"]

    # MACD histogram colours
    macd_hist = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd, macd_signal)
    ]
    hist_colors = [
        _BULL if (v is not None and v >= 0) else _BEAR for v in macd_hist
    ]
    vol_colors = [
        _VOLUME_BULL if c >= o else _VOLUME_BEAR
        for c, o in zip(closes, opens)
    ]

    # ── Figure scaffold ───────────────────────────────────────────────────────
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )

    chart_title = f"{title}  —  {symbol}" if symbol else title

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 1: Candlestick + MAs + Markers
    # ═════════════════════════════════════════════════════════════════════════

    if show_volume:
        fig.add_trace(
            go.Bar(
                x=timestamps, y=volumes,
                marker_color=vol_colors,
                name="Volume", showlegend=False,
                yaxis="y2", hoverinfo="skip",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Candlestick(
            x=timestamps, open=opens, high=highs, low=lows, close=closes,
            name="OHLC",
            increasing_line_color=_BULL, decreasing_line_color=_BEAR,
            increasing_fillcolor=_BULL,  decreasing_fillcolor=_BEAR,
            line_width=1, whiskerwidth=0.5,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=timestamps, y=ema_20, name="EMA 20",
                   line=dict(color=_EMA20, width=1.2), connectgaps=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=timestamps, y=ema_50, name="EMA 50",
                   line=dict(color=_EMA50, width=1.2), connectgaps=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=timestamps, y=sma_20, name="SMA 20",
                   line=dict(color=_SMA20, width=1.0, dash="dot"), connectgaps=False),
        row=1, col=1,
    )

    # ── Strategy-level signal markers (subtle, background layer) ─────────────
    if buy_sig_ts and not executed_trades:
        fig.add_trace(
            go.Scatter(
                x=buy_sig_ts, y=buy_sig_px,
                mode="markers", name="Signal BUY",
                marker=dict(symbol="triangle-up", color=_SIG_BUY, size=9,
                            line=dict(color=_MRK_BUY, width=1)),
                hovertemplate="Signal BUY<br>%{x}<extra></extra>",
            ),
            row=1, col=1,
        )
    if sell_sig_ts and not executed_trades:
        fig.add_trace(
            go.Scatter(
                x=sell_sig_ts, y=sell_sig_px,
                mode="markers", name="Signal SELL",
                marker=dict(symbol="triangle-down", color=_SIG_SELL, size=9,
                            line=dict(color=_MRK_SELL_F, width=1)),
                hovertemplate="Signal SELL<br>%{x}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Executed trade markers (precise, rich hover) ──────────────────────────
    if executed_trades:
        _add_trade_markers(fig, executed_trades)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 2: RSI
    # ═════════════════════════════════════════════════════════════════════════

    fig.add_hrect(y0=_RSI_OB, y1=100,
                  fillcolor="rgba(239,83,80,0.07)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=_RSI_OS,
                  fillcolor="rgba(38,166,154,0.07)", line_width=0, row=2, col=1)
    fig.add_trace(
        go.Scatter(x=timestamps, y=rsi_14, name="RSI 14",
                   line=dict(color="#BA68C8", width=1.5), connectgaps=False),
        row=2, col=1,
    )
    for level in (_RSI_OB, _RSI_OS, 50):
        fig.add_hline(y=level, line=dict(color=_GRID, width=0.8, dash="dot"),
                      row=2, col=1)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 3: MACD
    # ═════════════════════════════════════════════════════════════════════════

    fig.add_trace(
        go.Bar(x=timestamps, y=macd_hist, name="MACD Hist",
               marker_color=hist_colors, marker_line_width=0,
               opacity=0.6, showlegend=False),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=timestamps, y=macd, name="MACD",
                   line=dict(color=_MACD_L, width=1.5), connectgaps=False),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=timestamps, y=macd_signal, name="Signal",
                   line=dict(color=_MACD_SIG, width=1.2), connectgaps=False),
        row=3, col=1,
    )
    fig.add_hline(y=0, line=dict(color=_GRID, width=0.8), row=3, col=1)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 4: Equity Curve  (only if provided)
    # ═════════════════════════════════════════════════════════════════════════

    if has_equity:
        eq_ts  = [p["timestamp"]     for p in equity_curve]
        eq_val = [p["portfolio_value"] for p in equity_curve]

        fig.add_trace(
            go.Scatter(
                x=eq_ts, y=eq_val,
                name="Portfolio Value",
                line=dict(color=_EQUITY_LINE, width=1.8),
                fill="tozeroy",
                fillcolor="rgba(0,188,212,0.07)",
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    "Portfolio: INR %{y:,.2f}<extra></extra>"
                ),
            ),
            row=4, col=1,
        )

    # ── Layout & Theme ────────────────────────────────────────────────────────
    _apply_theme(fig, chart_title, has_equity)

    return fig


# ─── Trade Marker Helper ──────────────────────────────────────────────────────

def _add_trade_markers(fig: go.Figure, trades: list[dict]) -> None:
    """
    Add one Scatter trace per action type so legend entries are clean.
    Hover text surfaces: action, price, quantity, avg_cost, pnl, cash, strategy.
    """
    groups: dict[str, dict] = {
        "BUY":         {"ts": [], "px": [], "hover": [], "symbol": "triangle-up",
                        "color": _MRK_BUY, "size": 14, "label": "BUY"},
        "SELL_PARTIAL":{"ts": [], "px": [], "hover": [], "symbol": "triangle-left",
                        "color": _MRK_SELL_P, "size": 13, "label": "PARTIAL EXIT"},
        "SELL_FULL":   {"ts": [], "px": [], "hover": [], "symbol": "triangle-down",
                        "color": _MRK_SELL_F, "size": 14, "label": "SELL"},
        "FORCE_CLOSE": {"ts": [], "px": [], "hover": [], "symbol": "diamond",
                        "color": _MRK_FORCE, "size": 10, "label": "FORCE CLOSE"},
    }

    for t in trades:
        action = t.get("action", "")
        key    = action if action in groups else "FORCE_CLOSE"
        grp    = groups[key]

        ts    = t.get("timestamp")
        price = t.get("price", 0.0)
        qty   = t.get("quantity", 0.0)
        avg   = t.get("avg_entry_price", price)
        pnl   = t.get("pnl_realized", 0.0)
        cash  = t.get("cash_balance", 0.0)
        strat = t.get("strategy", "")

        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"

        grp["ts"].append(ts)
        grp["px"].append(price)
        grp["hover"].append(
            f"<b>{action}</b><br>"
            f"Price : {price:.2f}<br>"
            f"Qty   : {qty:.4f}<br>"
            f"Cost  : {avg:.2f}<br>"
            f"P&amp;L  : INR {pnl_str}<br>"
            f"Cash  : INR {cash:,.0f}<br>"
            f"Strat : {strat}"
        )

    for action, grp in groups.items():
        if not grp["ts"]:
            continue
        fig.add_trace(
            go.Scatter(
                x=grp["ts"],
                y=grp["px"],
                mode="markers",
                name=grp["label"],
                marker=dict(
                    symbol=grp["symbol"],
                    color=grp["color"],
                    size=grp["size"],
                    line=dict(color="white", width=0.8),
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=grp["hover"],
            ),
            row=1, col=1,
        )


# ─── Theme Application ────────────────────────────────────────────────────────

def _apply_theme(fig: go.Figure, title: str, has_equity: bool) -> None:
    """Apply Bloomberg-lite dark theme."""

    chart_height = 900 if has_equity else 780

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL_BG,
        title=dict(
            text=title,
            font=dict(color=_TEXT, size=14, family="'Roboto Mono', monospace"),
            x=0.01, xanchor="left",
        ),
        legend=dict(
            bgcolor="rgba(22,27,34,0.8)", bordercolor=_BORDER, borderwidth=1,
            font=dict(color=_TEXT, size=11),
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        margin=dict(l=60, r=40, t=60, b=40),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=_PANEL_BG, bordercolor=_BORDER,
                        font=dict(color=_TEXT, size=11)),
        height=chart_height,
        autosize=True,
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    showticklabels=False, range=[0, 5]),
    )

    _xax = dict(gridcolor=_GRID, gridwidth=0.5,
                tickfont=dict(color=_TEXT_DIM, size=10),
                linecolor=_BORDER, showline=True, zeroline=False)
    _yax = dict(gridcolor=_GRID, gridwidth=0.5,
                tickfont=dict(color=_TEXT_DIM, size=10),
                linecolor=_BORDER, showline=True, zeroline=False, side="right")

    # Panel 1 — Price
    fig.update_xaxes(**_xax, row=1, col=1, showticklabels=False)
    fig.update_yaxes(**_yax, title=dict(text="Price", font=dict(color=_TEXT_DIM, size=10)),
                     row=1, col=1)

    # Panel 2 — RSI
    fig.update_xaxes(**_xax, row=2, col=1, showticklabels=False)
    fig.update_yaxes(**_yax, title=dict(text="RSI", font=dict(color=_TEXT_DIM, size=10)),
                     range=[0, 100], dtick=20, row=2, col=1)

    # Panel 3 — MACD
    last_row_with_labels = 3 if not has_equity else 3
    fig.update_xaxes(**_xax, row=3, col=1,
                     showticklabels=(not has_equity),
                     tickformat="%b %d", tickangle=-30)
    fig.update_yaxes(**_yax, title=dict(text="MACD", font=dict(color=_TEXT_DIM, size=10)),
                     row=3, col=1)

    # Panel 4 — Equity Curve (conditional)
    if has_equity:
        fig.update_xaxes(**_xax, row=4, col=1,
                         showticklabels=True, tickformat="%b %d", tickangle=-30)
        fig.update_yaxes(**_yax,
                         title=dict(text="Portfolio", font=dict(color=_TEXT_DIM, size=10)),
                         row=4, col=1)
