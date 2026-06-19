"""
Project Atlas — CLI Entry Point
===============================
Terminal-based quant research tool.

Usage
-----
    python main.py [SYMBOL]
    python main.py -s [SYMBOL]

Example
-------
    python main.py HDFCBANK.NS
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from analytics.technical.indicators import IndicatorEngine
from analytics.technical.strategies import StrategyEngine
from analytics.technical.charts import plot_atlas_chart
from core.logging import get_logger, setup_logging

logger = get_logger(__name__)


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
    
    # Future flags can easily be added here
    # parser.add_argument("--backtest", action="store_true")
    # parser.add_argument("--live", action="store_true")

    args = parser.parse_args()
    
    # Allow both positional and flag-based symbol input
    final_symbol = args.symbol_pos or args.symbol
    
    if not final_symbol:
        parser.print_help()
        sys.exit(1)
        
    args.final_symbol = final_symbol.upper()
    return args


def fetch_data(symbol: str, history_days: int = 365) -> pd.DataFrame:
    """Fetch OHLCV data directly from yfinance for the CLI tool."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=history_days)
    
    logger.info(f"Fetching market data for {symbol}...")
    
    # Fetch data (suppress noisy progress bar)
    df = yf.download(
        tickers=symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    
    return df


def main() -> None:
    """Main execution pipeline."""
    # Ensure logs output to console in utf-8 to prevent Windows CP1252 crashes
    setup_logging(log_level="INFO")
    
    args = setup_cli()
    symbol = args.final_symbol
    
    try:
        # 1. Data Fetch
        df = fetch_data(symbol)
        
        if df.empty:
            logger.error(f"Invalid symbol or no data found for: {symbol}")
            sys.exit(1)
            
        logger.info(f"Fetched {len(df)} rows of data.")
        
        # yfinance returns a MultiIndex column dataframe when fetching a single ticker in newer versions,
        # or single level depending on the version. Let's flatten it safely.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Clean up column names to match what IndicatorEngine expects
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        
        # The index is the timestamp
        df = df.reset_index()
        df = df.rename(columns={"Date": "timestamp", "Datetime": "timestamp"})
        
        # Convert to list of dicts
        raw_candles = df.to_dict(orient="records")
        
        # 2. Indicator Engine
        logger.info("Computing indicators...")
        engine = IndicatorEngine()
        enriched_candles = engine.enrich(raw_candles)
        
        # 3. Strategy Engine
        logger.info("Evaluating trading strategies (EMA Crossover)...")
        strategy = StrategyEngine(strategy_name="EMA_CROSSOVER")
        signals = strategy.generate_signals(enriched_candles)
        
        # Merge signals back into candles for visualization and print summary
        signal_count = 0
        for i, candle in enumerate(enriched_candles):
            sig_dict = signals[i]
            candle["signal"] = sig_dict["signal"]
            
            if sig_dict["signal"] in ("BUY", "SELL"):
                signal_count += 1
                logger.info(f"  --> {sig_dict['signal']} SIGNAL at {sig_dict['timestamp'].strftime('%Y-%m-%d')}: {sig_dict['reason']}")
                
        logger.info(f"Generated {signal_count} total signals over the period.")
        
        # 4. Visualization Engine
        logger.info("Generating visualization...")
        fig = plot_atlas_chart(enriched_candles, symbol=symbol)
        
        logger.info("Opening chart in browser...")
        fig.show()
        
    except Exception as exc:
        logger.critical(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
