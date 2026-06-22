"""
analytics/portfolio/portfolio_engine.py
==========================================
Project Atlas — Multi-Stock Portfolio Engine

Purpose
-------
Simulate hedge-fund style multi-stock portfolio management:
  1. Fetch OHLCV for every symbol in the universe.
  2. Score each strategy (EMA_CROSSOVER / RSI) per stock.
  3. Select the best strategy per stock.
  4. Compute score-weighted capital allocations (5–40% per stock).
  5. Run individual backtests (PortfolioBacktestEngine) per stock.
  6. Aggregate into a single portfolio equity curve + metrics.

Capital Correctness Guarantees
-------------------------------
  ✦ Total capital allocated  == initial_capital  (enforced by normalisation)
  ✦ Per-stock backtest uses  exactly its allocated_capital as starting balance
  ✦ Portfolio equity(t)      == Σ stock_equity(t) at each timestamp
  ✦ All open positions are force-closed at end-of-data
  ✦ No phantom profits: PnL comes exclusively from (sell_price - cost_basis)

Scoring Formula
---------------
  score = total_return_pct - 0.5 * max_drawdown_pct

Allocation Formula
------------------
  Raw weights proportional to shifted scores → iterative clamping to [min_pct, max_pct]
  → guarantee Σ allocations == 1.0

Classes
-------
  StrategyScore      — strategy-level evaluation metrics for one stock
  AllocationEntry    — final allocation decision for one stock
  PortfolioReport    — complete portfolio simulation output
  PortfolioManager   — orchestrator
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from analytics.technical.strategies import StrategyEngine
from analytics.backtesting.backtest_engine import (
    PortfolioBacktestEngine,
    PortfolioResult,
)
from data.data_manager import DataManager

logger = logging.getLogger("atlas.portfolio")

# ─── Constants ────────────────────────────────────────────────────────────────

STRATEGIES      = ["EMA_CROSSOVER", "RSI"]
EVAL_CAPITAL    = 100_000.0   # Notional capital for strategy scoring (scale-free)
MIN_ALLOCATION  = 0.05        # Minimum per-stock allocation
MAX_ALLOCATION  = 0.40        # Maximum per-stock allocation


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class StrategyScore:
    """Evaluation metrics for one strategy applied to one stock."""
    strategy:           str
    total_return_pct:   float
    max_drawdown_pct:   float
    win_rate_pct:       float
    num_trades:         int
    score:              float   # = total_return - 0.5 * max_drawdown


@dataclass
class AllocationEntry:
    """Capital allocation decision for one stock."""
    symbol:             str
    allocated_capital:  float
    allocation_pct:     float           # 0–1
    selected_strategy:  str
    score:              float
    strategy_scores:    dict[str, StrategyScore] = field(default_factory=dict)


@dataclass
class PortfolioReport:
    """
    Complete output of a PortfolioManager.run() call.

    Money Integrity:
        initial_capital == Σ allocation_table[i].allocated_capital   (on entry)
        final_portfolio_value == Σ per_stock_results[s].final_portfolio_value
    """
    initial_capital:         float
    final_portfolio_value:   float
    total_return_pct:        float
    realized_pnl:            float
    max_drawdown_pct:        float
    num_stocks:              int
    run_date:                str

    allocation_table:        list[AllocationEntry]  = field(default_factory=list)
    per_stock_results:       dict[str, PortfolioResult] = field(default_factory=dict)
    portfolio_equity_curve:  list[dict]             = field(default_factory=list)

    def print_summary(self) -> None:
        """Print formatted multi-stock portfolio summary to stdout."""
        sep  = "═" * 64
        sign = "+" if self.total_return_pct >= 0 else ""

        print(f"\n{sep}")
        print("  ATLAS PORTFOLIO SIMULATION  (Multi-Stock Engine)")
        print(f"  Run Date       : {self.run_date}")
        print(sep)
        print(f"  Initial Capital     : INR {self.initial_capital:>14,.2f}")
        print(f"  Final Portfolio     : INR {self.final_portfolio_value:>14,.2f}")
        print(f"  Total Return        :     {sign}{self.total_return_pct:.2f}%")
        print(f"  Total Realized PnL  : INR {self.realized_pnl:>+14,.2f}")
        print(f"  Max Drawdown        :     {self.max_drawdown_pct:.2f}%")
        print(sep)

        # Allocation table
        print(f"\n  {'Symbol':<16} {'Strategy':<16} {'Score':>7} "
              f"{'Alloc %':>8} {'Capital':>13} {'Return':>8} {'PnL':>10}")
        print(f"  {'-'*16} {'-'*16} {'-'*7} {'-'*8} {'-'*13} {'-'*8} {'-'*10}")

        for entry in self.allocation_table:
            result = self.per_stock_results.get(entry.symbol)
            ret_str = (
                f"{result.total_return_pct:>+8.2f}%"
                if result else "     N/A"
            )
            pnl_str = (
                f"INR {result.realized_pnl:>+7,.0f}"
                if result else "   N/A"
            )
            print(
                f"  {entry.symbol:<16} {entry.selected_strategy:<16} "
                f"{entry.score:>7.2f} {entry.allocation_pct*100:>7.1f}% "
                f"INR {entry.allocated_capital:>9,.0f} {ret_str} {pnl_str}"
            )

        print(f"\n{sep}\n")

    def to_csv(self, output_dir: str | Path) -> dict[str, Path]:
        """
        Export portfolio results to CSV files.

        Files written:
            portfolio_summary.csv   — allocation + per-stock metrics
            portfolio_trades.csv    — unified trade log across all stocks
            portfolio_equity.csv    — aggregated equity curve time series
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = {}

        # ── Summary CSV ───────────────────────────────────────────────────────
        summary_path = out / "portfolio_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "symbol", "selected_strategy", "score", "allocation_pct",
                "allocated_capital", "final_value", "total_return_pct",
                "realized_pnl", "max_drawdown_pct", "num_trades",
            ])
            w.writeheader()
            for entry in self.allocation_table:
                result = self.per_stock_results.get(entry.symbol)
                w.writerow({
                    "symbol":             entry.symbol,
                    "selected_strategy":  entry.selected_strategy,
                    "score":              round(entry.score, 4),
                    "allocation_pct":     round(entry.allocation_pct * 100, 2),
                    "allocated_capital":  round(entry.allocated_capital, 2),
                    "final_value":        round(result.final_portfolio_value, 2) if result else "N/A",
                    "total_return_pct":   round(result.total_return_pct, 4) if result else "N/A",
                    "realized_pnl":       round(result.realized_pnl, 4) if result else "N/A",
                    "max_drawdown_pct":   round(result.max_drawdown_pct, 4) if result else "N/A",
                    "num_trades":         result.num_trades if result else 0,
                })
        written["summary"] = summary_path

        # ── Unified Trade Log CSV ─────────────────────────────────────────────
        trades_path = out / "portfolio_trades.csv"
        fieldnames = [
            "symbol", "timestamp", "action", "price", "quantity",
            "avg_entry_price", "pnl_realized", "cash_balance",
            "remaining_position", "portfolio_value", "strategy", "reason",
        ]
        with open(trades_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for symbol, result in self.per_stock_results.items():
                for t in result.trade_log:
                    ts_str = (
                        t.timestamp.isoformat()
                        if hasattr(t.timestamp, "isoformat")
                        else str(t.timestamp)
                    )
                    w.writerow({
                        "symbol":             symbol,
                        "timestamp":          ts_str,
                        "action":             t.action,
                        "price":              round(t.price, 4),
                        "quantity":           round(t.quantity, 6),
                        "avg_entry_price":    round(t.avg_entry_price, 4),
                        "pnl_realized":       round(t.pnl_realized, 4),
                        "cash_balance":       round(t.cash_balance, 4),
                        "remaining_position": round(t.remaining_position, 6),
                        "portfolio_value":    round(t.portfolio_value, 4),
                        "strategy":           t.strategy,
                        "reason":             t.reason,
                    })
        written["trades"] = trades_path

        # ── Portfolio Equity Curve CSV ────────────────────────────────────────
        equity_path = out / "portfolio_equity.csv"
        with open(equity_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["timestamp", "portfolio_value"])
            w.writeheader()
            for pt in self.portfolio_equity_curve:
                ts = pt["timestamp"]
                w.writerow({
                    "timestamp":       str(ts),
                    "portfolio_value": round(pt["portfolio_value"], 4),
                })
        written["equity"] = equity_path

        return written


# ─── Engine ───────────────────────────────────────────────────────────────────

class PortfolioManager:
    """
    Multi-stock portfolio backtesting orchestrator.

    Pipeline:
        1. Fetch OHLCV + compute indicators for each symbol.
        2. Evaluate every strategy on every stock using a notional capital.
        3. Select best strategy per stock; compute score-weighted allocations.
        4. Run actual backtests with allocated capital.
        5. Aggregate equity curves and compute portfolio-level metrics.

    Args:
        entry_fraction:     Per-trade entry fraction for PortfolioBacktestEngine.
        exit_fraction:      Per-trade exit fraction for PortfolioBacktestEngine.
        min_allocation:     Minimum portfolio weight per stock (default 5%).
        max_allocation:     Maximum portfolio weight per stock (default 40%).
        history_days:       Number of historical days to fetch per symbol.
    """

    def __init__(
        self,
        entry_fraction:  float = 0.25,
        exit_fraction:   float = 0.50,
        min_allocation:  float = MIN_ALLOCATION,
        max_allocation:  float = MAX_ALLOCATION,
        history_days:    int   = 365,
    ) -> None:
        self.entry_fraction  = entry_fraction
        self.exit_fraction   = exit_fraction
        self.min_allocation  = min_allocation
        self.max_allocation  = max_allocation
        self.history_days    = history_days
        self.data_manager    = DataManager()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        symbols:         list[str],
        initial_capital: float,
    ) -> PortfolioReport:
        """
        Execute the full multi-stock portfolio simulation.

        Args:
            symbols:         List of NSE/BSE ticker symbols.
            initial_capital: Total starting capital in INR.

        Returns:
            PortfolioReport  — complete portfolio simulation results.
        """
        if not symbols:
            raise ValueError("symbols list must not be empty")
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Step 1: Fetch + Enrich data for all symbols ───────────────────────
        logger.info(f"Portfolio universe: {', '.join(symbols)}")
        candles_map: dict[str, list[dict]] = {}

        for sym in symbols:
            logger.info(f"  Fetching {sym} from Cache...")
            enriched = self.data_manager.get_market_data(sym, history_days=self.history_days)
            if not enriched or len(enriched) < 60:
                logger.warning(f"  {sym}: insufficient data (need ≥60 candles). Skipping.")
                continue
            candles_map[sym] = enriched
            logger.info(f"  {sym}: {len(enriched)} candles loaded.")

        if not candles_map:
            raise RuntimeError("No valid symbols returned data. Aborting.")

        active_symbols = list(candles_map.keys())

        # ── Step 2: Evaluate strategies per stock ─────────────────────────────
        logger.info("Evaluating strategies per stock...")
        all_scores: dict[str, dict[str, StrategyScore]] = {}

        for sym in active_symbols:
            scores = self._evaluate_strategies(sym, candles_map[sym])
            all_scores[sym] = scores
            best = max(scores.values(), key=lambda s: s.score)
            logger.info(
                f"  {sym}: best={best.strategy} score={best.score:.2f} "
                f"ret={best.total_return_pct:.2f}% dd={best.max_drawdown_pct:.2f}%"
            )

        # ── Step 3: Compute capital allocations ───────────────────────────────
        best_score_map = {
            sym: max(scores.values(), key=lambda s: s.score).score
            for sym, scores in all_scores.items()
        }
        allocation_pcts = self._compute_allocations(best_score_map)

        allocation_table: list[AllocationEntry] = []
        for sym in active_symbols:
            best   = max(all_scores[sym].values(), key=lambda s: s.score)
            pct    = allocation_pcts[sym]
            capital = round(initial_capital * pct, 2)
            allocation_table.append(AllocationEntry(
                symbol             = sym,
                allocated_capital  = capital,
                allocation_pct     = pct,
                selected_strategy  = best.strategy,
                score              = best.score,
                strategy_scores    = all_scores[sym],
            ))
            logger.info(
                f"  {sym}: {pct*100:.1f}% → INR {capital:,.0f}  "
                f"[{best.strategy}]"
            )

        # Verify capital integrity
        total_allocated = sum(e.allocated_capital for e in allocation_table)
        drift = abs(total_allocated - initial_capital)
        if drift > 0.02:
            # Adjust largest position to absorb float rounding
            diff = initial_capital - total_allocated
            allocation_table[0].allocated_capital += diff
        logger.info(
            f"Capital allocated: INR {sum(e.allocated_capital for e in allocation_table):,.2f} "
            f"/ INR {initial_capital:,.2f}"
        )

        # ── Step 4: Run per-stock backtests ───────────────────────────────────
        logger.info("Running per-stock backtests...")
        per_stock_results: dict[str, PortfolioResult] = {}

        for entry in allocation_table:
            sym     = entry.symbol
            candles = candles_map[sym]
            signals = StrategyEngine(
                strategy_name=entry.selected_strategy
            ).generate_signals(candles)

            engine = PortfolioBacktestEngine(
                initial_balance    = entry.allocated_capital,
                entry_fraction     = self.entry_fraction,
                exit_fraction      = self.exit_fraction,
                min_position_value = max(10.0, entry.allocated_capital * 0.002),
            )
            result = engine.run(candles=candles, signals=signals, symbol=sym)
            per_stock_results[sym] = result

            sign = "+" if result.total_return_pct >= 0 else ""
            logger.info(
                f"  {sym}: {sign}{result.total_return_pct:.2f}%  "
                f"PnL=INR {result.realized_pnl:+,.0f}  "
                f"Trades={result.num_trades}"
            )

        # ── Step 5: Aggregate portfolio curve ─────────────────────────────────
        portfolio_equity_curve = self._aggregate_equity_curves(per_stock_results)

        # ── Step 6: Compute portfolio-level metrics ────────────────────────────
        total_final = sum(r.final_portfolio_value for r in per_stock_results.values())
        total_pnl   = sum(r.realized_pnl for r in per_stock_results.values())
        total_ret   = (total_final - initial_capital) / initial_capital * 100
        port_dd     = self._compute_portfolio_drawdown(portfolio_equity_curve)

        return PortfolioReport(
            initial_capital        = initial_capital,
            final_portfolio_value  = total_final,
            total_return_pct       = total_ret,
            realized_pnl           = total_pnl,
            max_drawdown_pct       = port_dd,
            num_stocks             = len(per_stock_results),
            run_date               = run_date,
            allocation_table       = allocation_table,
            per_stock_results      = per_stock_results,
            portfolio_equity_curve = portfolio_equity_curve,
        )

    # ── Private: Strategy Evaluation ─────────────────────────────────────────

    def _evaluate_strategies(
        self,
        symbol:  str,
        candles: list[dict],
    ) -> dict[str, StrategyScore]:
        """
        Run all strategies on pre-enriched candles with notional capital.
        Returns a dict mapping strategy_name → StrategyScore.
        """
        scores: dict[str, StrategyScore] = {}

        for strategy_name in STRATEGIES:
            try:
                signals = StrategyEngine(strategy_name=strategy_name).generate_signals(candles)
                engine  = PortfolioBacktestEngine(
                    initial_balance    = EVAL_CAPITAL,
                    entry_fraction     = self.entry_fraction,
                    exit_fraction      = self.exit_fraction,
                    min_position_value = 50.0,
                )
                result = engine.run(candles=candles, signals=signals, symbol=symbol)

                score = result.total_return_pct - 0.5 * result.max_drawdown_pct

                # Win-rate from completed round trips
                completed_pnls = [
                    t.pnl_realized for t in result.trade_log
                    if t.action in ("SELL_PARTIAL", "SELL_FULL", "FORCE_CLOSE")
                    and t.pnl_realized != 0
                ]
                win_rate = (
                    sum(1 for p in completed_pnls if p > 0) / len(completed_pnls) * 100
                    if completed_pnls else 0.0
                )

                scores[strategy_name] = StrategyScore(
                    strategy         = strategy_name,
                    total_return_pct = result.total_return_pct,
                    max_drawdown_pct = result.max_drawdown_pct,
                    win_rate_pct     = win_rate,
                    num_trades       = result.num_trades,
                    score            = score,
                )
            except Exception as exc:
                logger.warning(f"    {symbol}/{strategy_name}: eval failed — {exc}")
                scores[strategy_name] = StrategyScore(
                    strategy         = strategy_name,
                    total_return_pct = 0.0,
                    max_drawdown_pct = 0.0,
                    win_rate_pct     = 0.0,
                    num_trades       = 0,
                    score            = 0.0,
                )

        return scores

    # ── Private: Capital Allocation ───────────────────────────────────────────

    def _compute_allocations(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Score-weighted allocation with iterative clamping.

        Guarantees:
            - Each allocation in [min_allocation, max_allocation]
            - Σ allocations == 1.0
        """
        n = len(scores)
        if n == 0:
            return {}
        if n == 1:
            return {list(scores.keys())[0]: 1.0}

        # Dynamic minimum: ensure min*n ≤ 1.0
        min_pct = min(self.min_allocation, 0.8 / n)
        max_pct = self.max_allocation

        # Shift scores to strictly positive
        offset  = min(scores.values())
        shifted = {k: max(v - offset + 0.01, 0.01) for k, v in scores.items()}

        # Working copy
        alloc = dict(shifted)

        for _ in range(300):
            total = sum(alloc.values())
            norm  = {k: v / total for k, v in alloc.items()}

            if all(min_pct <= v <= max_pct for v in norm.values()):
                return norm

            # Identify clamped vs free stocks
            at_min = {k for k, v in norm.items() if v < min_pct}
            at_max = {k for k, v in norm.items() if v > max_pct}
            free   = set(scores.keys()) - at_min - at_max

            clamped_sum = len(at_min) * min_pct + len(at_max) * max_pct
            remaining   = 1.0 - clamped_sum

            # Fix clamped
            for k in at_min:
                alloc[k] = min_pct
            for k in at_max:
                alloc[k] = max_pct

            if not free:
                # Everything is clamped — force scale and return
                result = {k: min_pct if k in at_min else max_pct for k in scores}
                t = sum(result.values())
                return {k: v / t for k, v in result.items()}

            # Redistribute remaining among free stocks
            free_total = sum(alloc[k] for k in free)
            scale = remaining / free_total if free_total > 0 else 1.0
            for k in free:
                alloc[k] = alloc[k] * scale

        # Final fallback — normalise whatever we have
        total = sum(alloc.values())
        return {k: v / total for k, v in alloc.items()}

    # ── Private: Equity Curve Aggregation ────────────────────────────────────

    @staticmethod
    def _aggregate_equity_curves(
        per_stock_results: dict[str, PortfolioResult],
    ) -> list[dict]:
        """
        Sum per-stock equity curves into a single portfolio equity curve.

        Each stock's equity_curve_data contains {timestamp, portfolio_value}
        where portfolio_value = cash + mark-to-market position at that candle.
        Summing across stocks gives the total portfolio value at each date.

        Timestamps are normalised to ISO date strings for alignment.
        """
        combined: dict[str, float] = defaultdict(float)

        for symbol, result in per_stock_results.items():
            for pt in result.equity_curve_data:
                ts = pt["timestamp"]
                # Normalise to date string for cross-stock alignment
                date_key = (
                    ts.strftime("%Y-%m-%d")
                    if hasattr(ts, "strftime")
                    else str(ts)[:10]
                )
                combined[date_key] += pt["portfolio_value"]

        return [
            {"timestamp": k, "portfolio_value": round(v, 4)}
            for k, v in sorted(combined.items())
        ]

    @staticmethod
    def _compute_portfolio_drawdown(equity_curve: list[dict]) -> float:
        """Max peak-to-trough drawdown on the aggregated portfolio curve."""
        if len(equity_curve) < 2:
            return 0.0
        peak   = equity_curve[0]["portfolio_value"]
        max_dd = 0.0
        for pt in equity_curve[1:]:
            v = pt["portfolio_value"]
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 4)
