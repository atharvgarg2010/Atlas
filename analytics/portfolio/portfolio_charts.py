"""
analytics/portfolio/portfolio_charts.py
==========================================
Project Atlas — Portfolio Visualization Engine

Generates a professional multi-panel dashboard for a PortfolioReport:

    Panel 1 (full width):   Portfolio equity curve + per-stock equity curves
    Panel 2 (left):         Capital allocation bar chart (% per stock)
    Panel 3 (right):        Per-stock return % comparison bar chart

Public API
----------
    from analytics.portfolio.portfolio_charts import plot_portfolio_chart

    fig = plot_portfolio_chart(report)
    fig.show()
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Theme (Bloomberg-lite dark, consistent with charts.py) ──────────────────

_BG       = "#0D1117"
_PANEL_BG = "#161B22"
_GRID     = "#21262D"
_TEXT     = "#C9D1D9"
_TEXT_DIM = "#6E7681"
_BORDER   = "#30363D"

_PORTFOLIO_LINE = "#00E5FF"   # bright cyan  — dominant portfolio curve
_STOCK_PALETTE  = [           # per-stock equity curves (faded)
    "#F0B429", "#7B68EE", "#4FC3F7", "#EF5350",
    "#26A69A", "#FF7043", "#BA68C8", "#FFCA28",
    "#42A5F5", "#EC407A", "#66BB6A", "#78909C",
]

_BAR_ALLOC  = "#00E5FF"
_BAR_POS    = "#26A69A"   # positive return
_BAR_NEG    = "#EF5350"   # negative return


# ─── Public API ───────────────────────────────────────────────────────────────

def plot_portfolio_chart(report: Any) -> go.Figure:
    """
    Build a 2-row, 2-col dashboard for the given PortfolioReport.

    Layout:
        Row 1 (cols 1–2):  Portfolio equity curve + per-stock curves
        Row 2 (col 1):     Allocation % bar chart
        Row 2 (col 2):     Per-stock total return % bar chart

    Args:
        report:  PortfolioReport instance from PortfolioManager.run().

    Returns:
        plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"colspan": 2, "type": "xy"}, None],
            [{"type": "xy"},               {"type": "xy"}],
        ],
        row_heights=[0.60, 0.40],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        subplot_titles=[
            "Portfolio Equity Curve  (Total Capital + Per-Stock)",
            "Capital Allocation (%)",
            "Per-Stock Return (%)",
        ],
    )

    # ── Panel 1: Equity Curves ────────────────────────────────────────────────
    _add_equity_curves(fig, report)

    # ── Panel 2 (row 2, col 1): Allocation bar chart ──────────────────────────
    _add_allocation_bars(fig, report)

    # ── Panel 3 (row 2, col 2): Per-stock return bars ─────────────────────────
    _add_return_bars(fig, report)

    # ── Layout & Theme ────────────────────────────────────────────────────────
    _apply_theme(fig, report)

    return fig


# ─── Panel Builders ───────────────────────────────────────────────────────────

def _add_equity_curves(fig: go.Figure, report: Any) -> None:
    """Portfolio curve (dominant) + per-stock curves (faded)."""

    # Per-stock curves first (so portfolio overlays them)
    for i, (symbol, result) in enumerate(report.per_stock_results.items()):
        colour = _STOCK_PALETTE[i % len(_STOCK_PALETTE)]
        eq_ts  = [p["timestamp"] for p in result.equity_curve_data]
        eq_val = [p["portfolio_value"] for p in result.equity_curve_data]

        fig.add_trace(
            go.Scatter(
                x=eq_ts, y=eq_val,
                name=symbol,
                line=dict(color=colour, width=1.0),
                opacity=0.55,
                hovertemplate=(
                    f"<b>{symbol}</b><br>"
                    "%{x}<br>"
                    "Value: INR %{y:,.2f}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # Portfolio total curve — prominent
    port_ts  = [p["timestamp"] for p in report.portfolio_equity_curve]
    port_val = [p["portfolio_value"] for p in report.portfolio_equity_curve]

    sign = "+" if report.total_return_pct >= 0 else ""

    fig.add_trace(
        go.Scatter(
            x=port_ts, y=port_val,
            name=f"PORTFOLIO  ({sign}{report.total_return_pct:.2f}%)",
            line=dict(color=_PORTFOLIO_LINE, width=2.8),
            fill="tozeroy",
            fillcolor="rgba(0,229,255,0.06)",
            hovertemplate=(
                "<b>Portfolio</b><br>"
                "%{x}<br>"
                "Total: INR %{y:,.2f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # Initial capital reference line
    fig.add_hline(
        y=report.initial_capital,
        line=dict(color=_TEXT_DIM, width=1.0, dash="dot"),
        row=1, col=1,
    )


def _add_allocation_bars(fig: go.Figure, report: Any) -> None:
    """Horizontal bar chart showing % allocation per stock."""
    symbols = [e.symbol for e in report.allocation_table]
    allocs  = [round(e.allocation_pct * 100, 1) for e in report.allocation_table]
    colours = [_STOCK_PALETTE[i % len(_STOCK_PALETTE)] for i in range(len(symbols))]

    fig.add_trace(
        go.Bar(
            x=allocs,
            y=symbols,
            orientation="h",
            marker_color=colours,
            text=[f"{a:.1f}%" for a in allocs],
            textposition="outside",
            textfont=dict(color=_TEXT, size=10),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Allocation: %{x:.1f}%<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2, col=1,
    )


def _add_return_bars(fig: go.Figure, report: Any) -> None:
    """Vertical bar chart showing total return % per stock."""
    symbols = []
    returns = []
    colours = []

    for entry in report.allocation_table:
        result = report.per_stock_results.get(entry.symbol)
        if result is None:
            continue
        symbols.append(entry.symbol)
        ret = result.total_return_pct
        returns.append(round(ret, 2))
        colours.append(_BAR_POS if ret >= 0 else _BAR_NEG)

    fig.add_trace(
        go.Bar(
            x=symbols,
            y=returns,
            marker_color=colours,
            text=[f"{r:+.2f}%" for r in returns],
            textposition="outside",
            textfont=dict(color=_TEXT, size=10),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Return: %{y:+.2f}%<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2, col=2,
    )

    # Zero baseline
    fig.add_hline(y=0, line=dict(color=_GRID, width=0.8), row=2, col=2)


# ─── Theme ────────────────────────────────────────────────────────────────────

def _apply_theme(fig: go.Figure, report: Any) -> None:
    """Apply Bloomberg-lite dark theme to portfolio dashboard."""
    title = (
        f"Atlas Portfolio Simulation  —  "
        f"{report.num_stocks} stocks  |  "
        f"INR {report.initial_capital:,.0f}  →  "
        f"INR {report.final_portfolio_value:,.0f}  "
        f"({'+'if report.total_return_pct>=0 else ''}{report.total_return_pct:.2f}%)"
    )

    _xax = dict(
        gridcolor=_GRID, gridwidth=0.5,
        tickfont=dict(color=_TEXT_DIM, size=10),
        linecolor=_BORDER, showline=True, zeroline=False,
    )
    _yax = dict(
        gridcolor=_GRID, gridwidth=0.5,
        tickfont=dict(color=_TEXT_DIM, size=10),
        linecolor=_BORDER, showline=True, zeroline=False,
    )

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL_BG,
        title=dict(
            text=title,
            font=dict(color=_TEXT, size=13, family="'Roboto Mono', monospace"),
            x=0.01, xanchor="left",
        ),
        legend=dict(
            bgcolor="rgba(22,27,34,0.85)", bordercolor=_BORDER, borderwidth=1,
            font=dict(color=_TEXT, size=10),
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        margin=dict(l=60, r=40, t=90, b=50),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=_PANEL_BG, bordercolor=_BORDER,
                        font=dict(color=_TEXT, size=11)),
        height=850,
        autosize=True,
    )

    # Row 1 — Equity curves
    fig.update_xaxes(**_xax, tickformat="%b '%y", tickangle=-25, row=1, col=1)
    fig.update_yaxes(**_yax,
                     title=dict(text="Portfolio Value (INR)", font=dict(color=_TEXT_DIM, size=10)),
                     tickformat=",.0f", row=1, col=1)

    # Row 2, col 1 — Allocation bars
    fig.update_xaxes(**_xax, title=dict(text="Allocation (%)", font=dict(color=_TEXT_DIM, size=10)),
                     row=2, col=1)
    fig.update_yaxes(**_yax, row=2, col=1)

    # Row 2, col 2 — Return bars
    fig.update_xaxes(**_xax, tickangle=-30, row=2, col=2)
    fig.update_yaxes(**_yax,
                     title=dict(text="Return (%)", font=dict(color=_TEXT_DIM, size=10)),
                     row=2, col=2)

    # Subplot title styling
    for ann in fig.layout.annotations:
        ann.font = dict(color=_TEXT_DIM, size=11)
