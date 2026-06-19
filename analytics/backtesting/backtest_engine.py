"""
analytics/backtesting/backtest_engine.py
==========================================
Project Atlas — Backtest Engine v1

Purpose
-------
Simulate LONG-only paper trading on historical OHLCV + signal data.
Measures real strategy performance using a full equity-curve simulation.

Financial Correctness Notes
---------------------------
Execution Price Policy:
    We execute at the CLOSE price of the signal candle. This is a common
    simplification for daily bar backtesting. It introduces a mild optimistic
    bias (in production, use next-candle open for more realistic simulation).
    This is clearly noted in the BacktestResult so consumers can adjust.

Position Sizing:
    Fixed fractional — each trade allocates exactly `position_size_pct`
    (default 10%) of the current balance. This compounds gains/losses
    over time (Kelly-adjacent, conservative).

Max Drawdown:
    Computed on the end-of-trade equity curve (not intra-candle).
    Formula: max( (peak - trough) / peak ) across all recorded balance states.

Open Position at End of Data:
    If the backtest ends with an open LONG position, it is force-closed at
    the last available close price. This prevents an artificially high
    final balance by leaving unrealised P&L hidden.

Classes
-------
    TradeRecord     — Immutable dataclass for a single executed trade.
    BacktestResult  — Aggregated simulation output: metrics + trade log.
    BacktestEngine  — Orchestrator that runs the simulation.

Usage
-----
    from analytics.backtesting.backtest_engine import BacktestEngine

    engine = BacktestEngine(initial_balance=100_000.0)
    result = engine.run(candles=enriched_candles, signals=signals)
    result.print_summary()
    result.to_csv("research/output/trades.csv")
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """A single completed trade leg (BUY entry or SELL exit)."""
    timestamp:          Any             # datetime
    action:             str             # "BUY" | "SELL" | "FORCE_CLOSE"
    price:              float
    quantity:           float
    pnl:                float           # 0.0 for BUY legs; realised P&L for SELL
    balance_after:      float
    strategy:           str
    reason:             str             = ""


@dataclass
class BacktestResult:
    """
    Full output of a BacktestEngine.run() call.

    Attributes
    ----------
    initial_balance:    Starting capital.
    final_balance:      Capital after simulation.
    total_return_pct:   Percentage gain / loss vs initial capital.
    num_trades:         Number of completed round-trips (BUY + matching SELL).
    winning_trades:     Trades where P&L > 0.
    win_rate_pct:       winning_trades / num_trades * 100.
    max_drawdown_pct:   Worst peak-to-trough decline on the equity curve (%).
    trade_log:          Chronological list of every executed TradeRecord.
    open_position:      True if simulation ended with an unclosed position.
    execution_note:     Reminder about execution price policy.
    """
    initial_balance:    float
    final_balance:      float
    total_return_pct:   float
    num_trades:         int
    winning_trades:     int
    win_rate_pct:       float
    max_drawdown_pct:   float
    trade_log:          list[TradeRecord]           = field(default_factory=list)
    open_position:      bool                        = False
    execution_note:     str                         = "Executed at signal-candle close price."

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a formatted performance summary to stdout."""
        sep = "=" * 52
        sign = "+" if self.total_return_pct >= 0 else ""
        win_str = (
            f"{self.win_rate_pct:.1f}%"
            if self.num_trades > 0
            else "N/A (no closed trades)"
        )

        print(f"\n{sep}")
        print("  ATLAS BACKTEST RESULTS")
        print(sep)
        print(f"  Initial Capital  : INR {self.initial_balance:>12,.2f}")
        print(f"  Final Balance    : INR {self.final_balance:>12,.2f}")
        print(f"  Total Return     : {sign}{self.total_return_pct:.2f}%")
        print(f"  Max Drawdown     : {self.max_drawdown_pct:.2f}%")
        print(sep)
        print(f"  Round-Trip Trades: {self.num_trades}")
        print(f"  Winning Trades   : {self.winning_trades}")
        print(f"  Win Rate         : {win_str}")
        print(sep)

        if self.trade_log:
            print(f"\n  TRADE LOG ({len(self.trade_log)} entries)")
            print(f"  {'Date':<12} {'Action':<11} {'Price':>9} {'Qty':>8} "
                  f"{'P&L':>10} {'Balance':>13}")
            print(f"  {'-'*12} {'-'*11} {'-'*9} {'-'*8} {'-'*10} {'-'*13}")
            for t in self.trade_log:
                date_str = (
                    t.timestamp.strftime("%Y-%m-%d")
                    if hasattr(t.timestamp, "strftime")
                    else str(t.timestamp)[:10]
                )
                pnl_str = f"{t.pnl:>+10.2f}" if t.pnl != 0 else f"{'':>10}"
                print(
                    f"  {date_str:<12} {t.action:<11} {t.price:>9.2f} "
                    f"{t.quantity:>8.3f} {pnl_str} "
                    f"INR {t.balance_after:>10,.2f}"
                )

        if self.open_position:
            print("\n  [NOTE] Simulation ended with an open position — "
                  "force-closed at last available price.")
        print(f"\n  {self.execution_note}")
        print(f"{sep}\n")

    def to_csv(self, path: str | Path, symbol: str = "") -> Path:
        """
        Export the trade log to a CSV file.

        Args:
            path: Destination file path (parent directories created if needed).
            symbol: Ticker symbol of the stock.

        Returns:
            The resolved Path of the written file.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "timestamp", "symbol", "action", "price", "quantity",
            "pnl", "balance", "strategy", "reason",
        ]

        file_exists = dest.exists()
        
        with open(dest, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for t in self.trade_log:
                ts_str = (
                    t.timestamp.isoformat()
                    if hasattr(t.timestamp, "isoformat")
                    else str(t.timestamp)
                )
                writer.writerow({
                    "timestamp":  ts_str,
                    "symbol":     symbol,
                    "action":     t.action,
                    "price":      round(t.price, 4),
                    "quantity":   round(t.quantity, 6),
                    "pnl":        round(t.pnl, 4),
                    "balance":    round(t.balance_after, 4),
                    "strategy":   t.strategy,
                    "reason":     t.reason,
                })

        return dest

    def to_markdown(self, path: str | Path, symbol: str) -> Path:
        """
        Export the backtest summary and trade log to a Markdown file.

        Args:
            path: Destination file path.
            symbol: The ticker symbol tested.

        Returns:
            The resolved Path of the written file.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        sign = "+" if self.total_return_pct >= 0 else ""
        win_str = f"{self.win_rate_pct:.1f}%" if self.num_trades > 0 else "N/A"
        
        lines = [
            f"\n<br>\n\n# Atlas Backtest Report: {symbol}",
            f"**Run Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Performance Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Initial Capital | INR {self.initial_balance:,.2f} |",
            f"| Final Balance | INR {self.final_balance:,.2f} |",
            f"| Total Return | {sign}{self.total_return_pct:.2f}% |",
            f"| Max Drawdown | {self.max_drawdown_pct:.2f}% |",
            f"| Total Trades | {self.num_trades} |",
            f"| Winning Trades | {self.winning_trades} |",
            f"| Win Rate | {win_str} |",
            "",
            "## Trade Log",
            ""
        ]
        
        if not self.trade_log:
            lines.append("*No trades executed during this period.*")
        else:
            lines.append("| Date | Action | Price | Quantity | P&L | Balance | Reason |")
            lines.append("|---|---|---|---|---|---|---|")
            for t in self.trade_log:
                date_str = t.timestamp.strftime("%Y-%m-%d") if hasattr(t.timestamp, "strftime") else str(t.timestamp)[:10]
                pnl_str = f"{t.pnl:>+10.2f}" if t.pnl != 0 else ""
                reason_short = t.reason.split(':')[0] if ':' in t.reason else t.reason
                lines.append(f"| {date_str} | **{t.action}** | {t.price:.2f} | {t.quantity:.3f} | {pnl_str} | INR {t.balance_after:,.2f} | {reason_short} |")
                
        if self.open_position:
            lines.extend([
                "",
                "> **Note**: Simulation ended with an open position which was force-closed at the last available price."
            ])
            
        lines.extend([
            "",
            "---",
            f"*{self.execution_note}*"
        ])
        
        with open(dest, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            
        return dest


# ─── Engine ───────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Long-only paper trading simulator.

    Processes a paired list of enriched candles and strategy signals,
    executing BUY/SELL orders and tracking the equity curve.

    Args:
        initial_balance:    Starting capital in INR (default: 100,000).
        position_size_pct:  Fraction of current balance to deploy per trade
                            (default: 0.10 = 10%).
    """

    def __init__(
        self,
        initial_balance:    float = 100_000.0,
        position_size_pct:  float = 0.10,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if not (0 < position_size_pct <= 1.0):
            raise ValueError("position_size_pct must be in (0, 1]")

        self.initial_balance    = initial_balance
        self.position_size_pct  = position_size_pct

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        candles: list[dict[str, Any]],
        signals: list[dict[str, Any]],
    ) -> BacktestResult:
        """
        Execute the backtest simulation.

        Args:
            candles:    Enriched OHLCV dicts (from IndicatorEngine).
            signals:    Signal dicts (from StrategyEngine), same length as candles.

        Returns:
            BacktestResult with full metrics and trade log.

        Raises:
            ValueError: If candles and signals lengths differ.
        """
        if len(candles) != len(signals):
            raise ValueError(
                f"candles ({len(candles)}) and signals ({len(signals)}) "
                "must have the same length"
            )

        # ── Simulation State ─────────────────────────────────────────────────
        balance:        float           = self.initial_balance
        position:       str | None      = None      # None or "LONG"
        entry_price:    float           = 0.0
        quantity:       float           = 0.0
        entry_ts:       Any             = None
        trade_log:      list[TradeRecord] = []
        equity_curve:   list[float]     = [self.initial_balance]
        completed_trades: list[float]   = []        # stores P&L of round-trips

        # ── Main Loop ────────────────────────────────────────────────────────
        for candle, sig in zip(candles, signals):
            action  = sig.get("signal", "HOLD")
            price   = sig.get("price") or candle.get("close")
            ts      = sig.get("timestamp") or candle.get("timestamp")
            strategy = sig.get("strategy", "UNKNOWN")
            reason   = sig.get("reason", "")

            if price is None or price <= 0:
                continue    # malformed candle — skip silently

            # ── BUY ──────────────────────────────────────────────────────────
            if action == "BUY" and position is None:
                allocation  = balance * self.position_size_pct
                quantity    = allocation / price
                entry_price = price
                entry_ts    = ts
                position    = "LONG"

                trade_log.append(TradeRecord(
                    timestamp       = ts,
                    action          = "BUY",
                    price           = price,
                    quantity        = quantity,
                    pnl             = 0.0,
                    balance_after   = balance,   # balance unchanged on entry
                    strategy        = strategy,
                    reason          = reason,
                ))

            # ── SELL ─────────────────────────────────────────────────────────
            elif action == "SELL" and position == "LONG":
                pnl         = (price - entry_price) * quantity
                balance    += pnl
                position    = None

                trade_log.append(TradeRecord(
                    timestamp       = ts,
                    action          = "SELL",
                    price           = price,
                    quantity        = quantity,
                    pnl             = pnl,
                    balance_after   = balance,
                    strategy        = strategy,
                    reason          = reason,
                ))

                completed_trades.append(pnl)
                equity_curve.append(balance)

        # ── Force-close open position at last candle ─────────────────────────
        open_at_end = False
        if position == "LONG" and candles:
            last_price  = candles[-1].get("close", entry_price)
            pnl         = (last_price - entry_price) * quantity
            balance    += pnl
            open_at_end = True

            trade_log.append(TradeRecord(
                timestamp       = candles[-1].get("timestamp"),
                action          = "FORCE_CLOSE",
                price           = last_price,
                quantity        = quantity,
                pnl             = pnl,
                balance_after   = balance,
                strategy        = "SYSTEM",
                reason          = "End-of-data force close",
            ))
            completed_trades.append(pnl)
            equity_curve.append(balance)

        # ── Metrics ───────────────────────────────────────────────────────────
        total_return_pct    = ((balance - self.initial_balance) / self.initial_balance) * 100
        num_trades          = len(completed_trades)
        winning_trades      = sum(1 for p in completed_trades if p > 0)
        win_rate_pct        = (winning_trades / num_trades * 100) if num_trades > 0 else 0.0
        max_drawdown_pct    = self._compute_max_drawdown(equity_curve)

        return BacktestResult(
            initial_balance     = self.initial_balance,
            final_balance       = balance,
            total_return_pct    = total_return_pct,
            num_trades          = num_trades,
            winning_trades      = winning_trades,
            win_rate_pct        = win_rate_pct,
            max_drawdown_pct    = max_drawdown_pct,
            trade_log           = trade_log,
            open_position       = open_at_end,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_max_drawdown(equity_curve: list[float]) -> float:
        """
        Compute maximum peak-to-trough drawdown on the equity curve.

        Returns percentage drawdown (e.g. 5.23 for 5.23%).
        Returns 0.0 if fewer than 2 data points exist.
        """
        if len(equity_curve) < 2:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100 if peak > 0 else 0.0
            if drawdown > max_dd:
                max_dd = drawdown

        return round(max_dd, 4)
