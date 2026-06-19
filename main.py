"""
Project Atlas — CLI Entry Point
===============================
Terminal-based quant research tool.

Usage
-----
    python main.py HDFCBANK.NS                  # Fetch + Indicators + Chart
    python main.py HDFCBANK.NS --backtest       # + Backtest simulation
    python main.py -s HDFCBANK.NS --strategy RSI --backtest

Supported Strategies:
    EMA_CROSSOVER   (default)
    RSI
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from analytics.technical.indicators import IndicatorEngine
from analytics.technical.strategies import StrategyEngine
from analytics.technical.charts import plot_atlas_chart
from analytics.backtesting.backtest_engine import BacktestEngine
from core.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ─── CLI Setup ────────────────────────────────────────────────────────────────

def setup_cli() -> argparse.Namespace:
    """Configure and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Atlas — Quantitative Trading Research System",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "symbol_pos",
        nargs="?",
        help="Stock symbol to analyze (e.g. HDFCBANK.NS)",
    )
    parser.add_argument(
        "-s", "--symbol",
        help="Stock symbol to analyze (e.g. HDFCBANK.NS)",
    )
    parser.add_argument(
        "--strategy",
        default="EMA_CROSSOVER",
        choices=["EMA_CROSSOVER", "RSI"],
        help="Trading strategy to use (default: EMA_CROSSOVER)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest simulation after signal generation",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Initial paper capital in INR. If provided, overwrites the saved balance.",
    )
    parser.add_argument(
        "--reset-balance",
        type=float,
        default=None,
        help="Reset the saved paper trading balance to this amount and exit.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of historical days to fetch (default: 365)",
    )

    args = parser.parse_args()

    if args.reset_balance is not None:
        return args

    final_symbol = args.symbol_pos or args.symbol
    if not final_symbol:
        parser.print_help()
        sys.exit(1)

    args.final_symbol = final_symbol.upper()
    return args


# ─── Data Fetch ───────────────────────────────────────────────────────────────

def fetch_data(symbol: str, history_days: int = 365) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance for the given symbol."""
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=history_days)

    logger.info(f"Fetching market data for {symbol}...")

    df = yf.download(
        tickers=symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    return df


def dataframe_to_candles(df: pd.DataFrame) -> list[dict]:
    """Normalise a yfinance DataFrame into the Atlas candle list[dict] format."""
    # Flatten MultiIndex columns (yfinance 0.2+ sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high",
        "Low": "low",  "Close": "close", "Volume": "volume",
    })
    df = df.reset_index()
    df = df.rename(columns={"Date": "timestamp", "Datetime": "timestamp"})

    return df.to_dict(orient="records")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Execute the full Atlas analysis pipeline."""
    setup_logging(log_level="INFO")

    args = setup_cli()
    
    # ── Handle Balance Reset ──────────────────────────────────────────────────
    if args.reset_balance is not None:
        state_file = Path(__file__).parent / "research" / "output" / "portfolio_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump({"balance": args.reset_balance, "last_updated": datetime.now().isoformat()}, f)
        logger.info(f"Portfolio balance successfully reset to: INR {args.reset_balance:,.2f}")
        logger.info("Historical trade logs (Log.csv) have been preserved.")
        sys.exit(0)

    symbol = args.final_symbol

    try:
        # ── 1. Data Fetch ─────────────────────────────────────────────────────
        df = fetch_data(symbol, history_days=args.days)

        if df.empty:
            logger.error(f"Invalid symbol or no data found for: {symbol}")
            sys.exit(1)

        logger.info(f"Fetched {len(df)} candles.")
        raw_candles = dataframe_to_candles(df)

        # ── 2. Indicator Engine ───────────────────────────────────────────────
        logger.info("Computing indicators...")
        enriched_candles = IndicatorEngine().enrich(raw_candles)

        # ── 3. Strategy Engine ────────────────────────────────────────────────
        logger.info(f"Evaluating strategy: {args.strategy}...")
        strategy = StrategyEngine(strategy_name=args.strategy)
        signals  = strategy.generate_signals(enriched_candles)

        # Merge signals into candles (needed for chart markers)
        signal_count = 0
        for candle, sig in zip(enriched_candles, signals):
            candle["signal"] = sig["signal"]
            if sig["signal"] in ("BUY", "SELL"):
                signal_count += 1
                ts_str = (
                    sig["timestamp"].strftime("%Y-%m-%d")
                    if hasattr(sig["timestamp"], "strftime")
                    else str(sig["timestamp"])[:10]
                )
                logger.info(
                    f"  --> {sig['signal']} at {ts_str}: {sig['reason']}"
                )

        logger.info(f"Generated {signal_count} actionable signals.")

        # ── 4. Backtest Engine (optional) ─────────────────────────────────────
        if args.backtest:
            # Manage Portfolio State
            state_file = Path(__file__).parent / "research" / "output" / "portfolio_state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            initial_balance = 100_000.0
            
            if args.capital is not None:
                initial_balance = args.capital
            elif state_file.exists():
                with open(state_file, "r") as f:
                    try:
                        state = json.load(f)
                        initial_balance = state.get("balance", 100_000.0)
                    except json.JSONDecodeError:
                        pass
                        
            logger.info(
                f"Running backtest simulation "
                f"(capital=INR {initial_balance:,.2f}, strategy={args.strategy})..."
            )
            engine = BacktestEngine(
                initial_balance=initial_balance,
                position_size_pct=0.10,
            )
            result = engine.run(candles=enriched_candles, signals=signals)

            # Print metrics
            result.print_summary()
            
            # Save new balance state
            with open(state_file, "w") as f:
                json.dump({"balance": result.final_balance, "last_updated": datetime.now().isoformat()}, f)
            
            logger.info(f"Updated paper trading balance to: INR {result.final_balance:,.2f}")

            # Export CSV
            csv_dir  = Path(__file__).parent / "research" / "output"
            csv_path = csv_dir / "Log.csv"
            md_path  = csv_dir / "Backtest_Report.md"
            
            written_csv  = result.to_csv(csv_path, symbol=symbol)
            written_md   = result.to_markdown(md_path, symbol=symbol)
            logger.info(f"Trade log appended to CSV: {written_csv}")
            logger.info(f"Backtest report appended to MD: {written_md}")

        # ── 5. Visualization ──────────────────────────────────────────────────
        logger.info("Generating visualization...")
        fig = plot_atlas_chart(enriched_candles, symbol=symbol)
        logger.info("Opening chart in browser...")
        fig.show()

    except Exception as exc:
        logger.critical(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
