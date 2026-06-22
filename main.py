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
from analytics.backtesting.backtest_engine import PortfolioBacktestEngine
from analytics.portfolio.portfolio_engine import PortfolioManager
from analytics.portfolio.portfolio_charts import plot_portfolio_chart
from core.logging import get_logger, setup_logging
from config.settings import get_settings
from database.connection import init_db
from data.data_manager import DataManager

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
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check current paper trading balance and log file locations.",
    )
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Run multi-stock portfolio simulation mode.",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default=None,
        help="Comma-separated NSE symbols for portfolio mode (e.g. HDFCBANK.NS,TCS.NS,TITAN.NS).",
    )
    parser.add_argument(
        "--rank-universe",
        action="store_true",
        help="Run the Factor Ranking Engine to rank the entire universe.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top stocks to select for Factor Backtest or Portfolio mode (default: 10).",
    )
    parser.add_argument(
        "--factor-backtest",
        action="store_true",
        help="Run a monthly rebalancing Factor strategy backtest.",
    )
    parser.add_argument(
        "--optimize-portfolio",
        action="store_true",
        help="Run the Portfolio Optimization Engine on the Top N ranked stocks.",
    )
    parser.add_argument(
        "--weighting",
        type=str,
        default="equal",
        choices=["equal", "minvar", "maxsharpe", "riskparity"],
        help="Weighting scheme for portfolio optimization and factor backtest.",
    )
    # ── Cache Management Commands ──
    parser.add_argument(
        "--sync-symbol",
        type=str,
        default=None,
        help="Sync a single symbol to the Supabase cache (e.g. RELIANCE.NS).",
    )
    parser.add_argument(
        "--sync-universe",
        type=str,
        default=None,
        help="Comma-separated symbols to sync concurrently to the cache.",
    )
    parser.add_argument(
        "--sync-all",
        action="store_true",
        help="Sync all symbols currently marked as ACTIVE in the cache metadata.",
    )
    parser.add_argument(
        "--cache-status",
        action="store_true",
        help="Show statistics for the Supabase OHLCV cache.",
    )

    args = parser.parse_args()

    if (args.reset_balance is not None or args.status or args.portfolio 
        or args.sync_symbol or args.sync_universe or args.sync_all or args.cache_status
        or args.rank_universe or args.factor_backtest or args.optimize_portfolio):
        return args

    final_symbol = args.symbol_pos or args.symbol
    if not final_symbol:
        parser.print_help()
        sys.exit(1)

    args.final_symbol = final_symbol.upper()
    return args


# ─── Removed fetch_data in favour of DataManager ──────────────────────────────


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Execute the full Atlas analysis pipeline."""
    setup_logging(log_level="INFO")

    # Initialize Database Connection
    settings = get_settings()
    init_db(settings.database_url, echo=False)

    args = setup_cli()
    
    # ── Handle Cache Management ───────────────────────────────────────────────
    if args.cache_status:
        dm = DataManager()
        stats = dm.get_cache_status()
        logger.info("=== Supabase Market Data Cache Status ===")
        for k, v in stats.items():
            logger.info(f"{k}: {v}")
        sys.exit(0)

    if args.sync_symbol:
        dm = DataManager()
        sym = args.sync_symbol.upper()
        logger.info(f"Syncing {sym} to Supabase cache...")
        success = dm.sync_symbol(sym, force=True)
        if success:
            logger.info(f"Successfully synced {sym}.")
        else:
            logger.error(f"Failed to sync {sym}.")
        sys.exit(0)

    if args.sync_universe:
        dm = DataManager()
        symbols = [s.strip().upper() for s in args.sync_universe.split(",") if s.strip()]
        logger.info(f"Syncing universe to Supabase cache: {symbols}")
        results = dm.sync_universe(symbols, max_workers=5)
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Universe sync complete. Successful: {success_count}/{len(symbols)}")
        sys.exit(0)

    if args.sync_all:
        dm = DataManager()
        with dm.db.session() as s:
            from database.models.market_data import SymbolMetadata
            from sqlalchemy import select
            active_symbols = s.scalars(select(SymbolMetadata.symbol).where(SymbolMetadata.status == "ACTIVE")).all()
        
        if not active_symbols:
            logger.warning("No ACTIVE symbols found in cache to sync.")
            sys.exit(0)
            
        logger.info(f"Syncing all {len(active_symbols)} ACTIVE symbols to Supabase cache...")
        results = dm.sync_universe(active_symbols, max_workers=5)
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Sync-all complete. Successful: {success_count}/{len(active_symbols)}")
        sys.exit(0)

    # ── Handle Balance Reset ──────────────────────────────────────────────────
    state_file = Path(__file__).parent / "research" / "output" / "portfolio_state.json"
    csv_path   = Path(__file__).parent / "research" / "output" / "Log.csv"
    md_path    = Path(__file__).parent / "research" / "output" / "Backtest_Report.md"

    if args.reset_balance is not None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump({"balance": args.reset_balance, "last_updated": datetime.now().isoformat()}, f)
        logger.info(f"Portfolio balance successfully reset to: INR {args.reset_balance:,.2f}")
        logger.info("Historical trade logs (Log.csv) have been preserved.")
        sys.exit(0)

    # ── Handle Status Check ───────────────────────────────────────────────────
    if args.status:
        balance = 100_000.0
        if state_file.exists():
            with open(state_file, "r") as f:
                try:
                    balance = json.load(f).get("balance", 100_000.0)
                except json.JSONDecodeError:
                    pass
        
        logger.info(f"Current Paper Trading Balance: INR {balance:,.2f}")
        
        if csv_path.exists():
            logger.info(f"Trade Log (CSV) path     : {csv_path.resolve()}")
        else:
            logger.info("Trade Log (CSV)          : No logs generated yet.")
            
        if md_path.exists():
            logger.info(f"Backtest Report (MD) path: {md_path.resolve()}")
            
        sys.exit(0)

    # ── Handle Factor Ranking & Backtest ──────────────────────────────────────
    from datetime import date
    
    if args.rank_universe and not args.portfolio and not args.factor_backtest:
        from analytics.factors.factor_engine import FactorEngine
        engine = FactorEngine()
        ranks = engine.rank_universe(date.today())
        if not ranks.empty:
            logger.info("=== ATLAS FACTOR RANKING (Top 10) ===")
            print(ranks.head(args.top)[['rank', 'symbol', 'composite_score']].to_string(index=False))
            logger.info("=== Bottom 10 ===")
            print(ranks.tail(args.top)[['rank', 'symbol', 'composite_score']].to_string(index=False))
            logger.info(f"=== Factor Breakdown for {ranks.iloc[0]['symbol']} ===")
            for k, v in ranks.iloc[0].to_dict().items():
                print(f"  {k}: {v}")
            
            from analytics.factors.factor_report import generate_reasoning_report
            out_dir = Path(__file__).parent / "research" / "output"
            report_path = generate_reasoning_report(ranks, engine.weights, date.today(), args.top, out_dir)
            logger.info(f"Factor Reasoning Report generated at: {report_path}")

        sys.exit(0)

    if args.factor_backtest:
        from analytics.backtesting.factor_backtest import FactorBacktestEngine
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days)
        engine = FactorBacktestEngine(initial_balance=args.capital or 100_000.0, top_n=args.top, weighting_scheme=args.weighting)
        try:
            result = engine.run(start_date=start_date, end_date=end_date)
            logger.info("=== Factor Backtest Complete ===")
            print("\nStrategy Metrics:")
            for k, v in result['strategy'].items():
                print(f"  {k}: {v}")
            print("\nBenchmark Metrics:")
            for k, v in result['benchmark'].items():
                print(f"  {k}: {v}")
            
            if result['turnover_stats']:
                avg_turnover = sum(t['turnover_pct'] for t in result['turnover_stats']) / len(result['turnover_stats'])
                print(f"\nAverage Monthly Turnover: {avg_turnover:.2f}%")
        except Exception as e:
            logger.error(f"Factor backtest failed: {e}", exc_info=True)
        sys.exit(0)

    if args.optimize_portfolio:
        from analytics.factors.factor_engine import FactorEngine
        from analytics.portfolio.optimizer import PortfolioOptimizer
        from analytics.portfolio.portfolio_report import generate_optimization_report
        import pandas as pd
        
        engine = FactorEngine()
        ranks = engine.rank_universe(date.today())
        if ranks.empty:
            logger.error("Ranking failed. Cannot optimize portfolio.")
            sys.exit(1)
            
        top_stocks = ranks.head(args.top)['symbol'].tolist()
        all_data = engine._fetch_data_batch(date.today())
        
        past_prices = {}
        for sym in top_stocks:
            sym_df = all_data.get(sym)
            if sym_df is not None:
                past_data = sym_df[sym_df['date'] <= date.today()].tail(252)
                past_prices[sym] = past_data.set_index('date')['close']
                
        prices_df = pd.DataFrame(past_prices).ffill().dropna()
        factor_scores = ranks.head(args.top).set_index('symbol')['composite_score']
        
        optimizer = PortfolioOptimizer(prices_df, factor_scores, risk_free_rate=0.05)
        if args.weighting == "minvar":
            opt_res = optimizer.minimum_variance()
        elif args.weighting == "maxsharpe":
            opt_res = optimizer.maximum_sharpe()
        elif args.weighting == "riskparity":
            opt_res = optimizer.risk_parity()
        else:
            opt_res = optimizer.equal_weight()
            
        out_dir = Path(__file__).parent / "research" / "output"
        report_path = generate_optimization_report(ranks, opt_res, args.weighting, date.today(), out_dir)
        logger.info(f"Portfolio Optimization Report generated at: {report_path}")
        sys.exit(0)

    # ── Handle Portfolio Mode ──────────────────────────────────────────────
    if args.portfolio:
        if args.rank_universe:
            from analytics.factors.factor_engine import FactorEngine
            engine = FactorEngine()
            ranks = engine.rank_universe(date.today())
            if ranks.empty:
                logger.error("Ranking failed. Cannot run portfolio.")
                sys.exit(1)
            symbols = ranks.head(args.top)['symbol'].tolist()
        else:
            if not args.universe:
                logger.error("--portfolio requires --universe or --rank-universe")
                sys.exit(1)
            symbols = [s.strip().upper() for s in args.universe.split(",") if s.strip()]

        if len(symbols) < 2:
            logger.error("Portfolio mode requires at least 2 symbols.")
            sys.exit(1)

        initial_capital = args.capital if args.capital else 100_000.0
        logger.info(f"Starting Portfolio Simulation: {len(symbols)} stocks, INR {initial_capital:,.0f}")

        try:
            manager = PortfolioManager(
                entry_fraction = 0.25,
                exit_fraction  = 0.50,
                history_days   = args.days,
            )
            report = manager.run(symbols=symbols, initial_capital=initial_capital)
            report.print_summary()

            # Export CSVs
            out_dir   = Path(__file__).parent / "research" / "output"
            written   = report.to_csv(out_dir)
            logger.info(f"Portfolio summary  : {written['summary']}")
            logger.info(f"Unified trade log  : {written['trades']}")
            logger.info(f"Portfolio equity   : {written['equity']}")

            # Visualization
            logger.info("Generating portfolio dashboard...")
            fig = plot_portfolio_chart(report)
            logger.info("Opening dashboard in browser...")
            fig.show()

        except Exception as exc:
            logger.critical(f"Portfolio simulation failed: {exc}", exc_info=True)
            sys.exit(1)

        sys.exit(0)

    symbol = args.final_symbol

    try:
        # 1. Load Data from Supabase Cache
        dm = DataManager()
        enriched_candles = dm.get_market_data(symbol, history_days=args.days)
        if not enriched_candles:
            logger.critical(f"No market data available for {symbol}")
            sys.exit(1)
        
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
        backtest_result = None
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
                f"Running portfolio simulation "
                f"(capital=INR {initial_balance:,.2f}, strategy={args.strategy}, "
                f"entry=25% cash, exit=50% holdings)..."
            )
            engine = PortfolioBacktestEngine(
                initial_balance    = initial_balance,
                entry_fraction     = 0.25,   # invest 25% of cash per BUY
                exit_fraction      = 0.50,   # sell 50% of holdings per SELL
                min_position_value = 500.0,
            )
            result = engine.run(
                candles = enriched_candles,
                signals = signals,
                symbol  = symbol,
            )

            # Print metrics
            result.print_summary()
            backtest_result = result
            
            # Save updated balance (use final_portfolio_value for persistence)
            with open(state_file, "w") as f:
                json.dump({"balance": result.final_portfolio_value, "last_updated": datetime.now().isoformat()}, f)
            logger.info(f"Portfolio value saved: INR {result.final_portfolio_value:,.2f}")

            # Export CSV and MD
            csv_dir  = Path(__file__).parent / "research" / "output"
            csv_path = csv_dir / "Log.csv"
            md_path  = csv_dir / "Backtest_Report.md"

            written_csv = result.to_csv(csv_path)
            written_md  = result.to_markdown(md_path)
            logger.info(f"Trade log appended to CSV : {written_csv}")
            logger.info(f"Backtest report updated   : {written_md}")

        # ── 5. Visualization ──────────────────────────────────────────────────
        logger.info("Generating visualization...")
        
        # Pass backtest data to chart if backtest was run
        chart_trades = backtest_result.trade_log        if backtest_result else None
        chart_equity = backtest_result.equity_curve_data if backtest_result else None
        
        fig = plot_atlas_chart(
            enriched_candles,
            symbol=symbol,
            executed_trades=chart_trades,
            equity_curve=chart_equity,
        )
        logger.info("Opening chart in browser...")
        fig.show()

    except Exception as exc:
        logger.critical(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
