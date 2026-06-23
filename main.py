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
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Build ML training dataset.",
    )
    parser.add_argument(
        "--validate-dataset",
        action="store_true",
        help="Generate dataset analysis report.",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train the XGBoost Regressor model.",
    )
    parser.add_argument(
        "--predict-universe",
        action="store_true",
        help="Run ML predictions for the current universe.",
    )
    parser.add_argument(
        "--ml-backtest",
        action="store_true",
        help="Run portfolio backtest using ML expected returns.",
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
    # ── Phase 4.1 Commands ──
    parser.add_argument(
        "--show-db-stats",
        action="store_true",
        help="Show comprehensive database statistics and save to report.",
    )
    parser.add_argument(
        "--backfill-history",
        action="store_true",
        help="Run incremental backward fetch for historical data coverage.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Years of historical coverage to target during backfill.",
    )
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="Generate Dataset_Coverage_Report.md.",
    )
    parser.add_argument(
        "--reality-check",
        action="store_true",
        help="Run Phase 4.2 Institutional Validation & Reality Check suite.",
    )
    # ── Phase 5.0 Commands ──
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="Run Phase 5.0 Feature Selection Engine.",
    )
    parser.add_argument(
        "--run-tournament",
        action="store_true",
        help="Run Phase 5.0 True Walk-Forward Model Tournament.",
    )

    args = parser.parse_args()

    if (args.reset_balance is not None or args.status or args.portfolio 
        or args.sync_symbol or args.sync_universe or args.sync_all or args.cache_status
        or args.rank_universe or args.factor_backtest or args.optimize_portfolio
        or args.build_dataset or args.validate_dataset or args.train_model
        or args.predict_universe or args.ml_backtest or args.show_db_stats
        or args.backfill_history or args.coverage_report or args.reality_check
        or args.feature_selection or args.run_tournament):
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

    # ── Handle Phase 4.1 Commands ─────────────────────────────────────────────
    if args.show_db_stats:
        from sqlalchemy import select, func
        from database.models.market_data import MarketData, MarketIndicators, SymbolMetadata
        from datetime import date
        out_dir = Path(__file__).parent / "research" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "Database_Statistics_Report.md"
        
        dm = DataManager()
        with dm.db.session() as s:
            min_date = s.scalar(select(func.min(MarketData.date)))
            max_date = s.scalar(select(func.max(MarketData.date)))
            total_md = s.scalar(select(func.count(MarketData.id)))
            total_ind = s.scalar(select(func.count(MarketIndicators.id)))
            
            active_symbols = s.scalars(select(SymbolMetadata.symbol).where(SymbolMetadata.status == "ACTIVE")).all()
            total_symbols = len(active_symbols)
            
            avg_coverage = (total_md / total_symbols) if total_symbols > 0 else 0
            
            with open(report_path, "w") as f:
                f.write("# Atlas Database Statistics Report\n\n")
                f.write(f"**Generated on:** {date.today()}\n\n")
                f.write("## Global Bounds\n")
                f.write(f"- **Earliest Date in DB:** {min_date}\n")
                f.write(f"- **Latest Date in DB:** {max_date}\n\n")
                f.write("## Volume & Coverage\n")
                f.write(f"- **Symbols Covered (ACTIVE):** {total_symbols}\n")
                f.write(f"- **Total `market_data` Rows:** {total_md:,}\n")
                f.write(f"- **Total `market_indicators` Rows:** {total_ind:,}\n")
                f.write(f"- **Average Rows per Symbol:** {avg_coverage:,.0f}\n")
                
        logger.info(f"Database Statistics Report generated at: {report_path}")
        print(f"Earliest Date: {min_date}")
        print(f"Latest Date: {max_date}")
        print(f"Total Rows: {total_md:,}")
        print(f"Symbols Covered: {total_symbols}")
        sys.exit(0)

    if args.backfill_history:
        from data.backfill_manager import BackfillManager
        manager = BackfillManager()
        
        # NIFTY 100 List
        nifty100 = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "INFY.NS", "ITC.NS", "HINDUNILVR.NS",
            "LT.NS", "SBIN.NS", "BAJFINANCE.NS", "MARUTI.NS", "M&M.NS", "HCLTECH.NS", "TATAMOTORS.NS", "SUNPHARMA.NS",
            "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS", "KOTAKBANK.NS", "AXISBANK.NS", "ONGC.NS", "ULTRACEMCO.NS", "TITAN.NS",
            "ASIANPAINT.NS", "BAJAJFINSV.NS", "WIPRO.NS", "NESTLEIND.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS",
            "DRREDDY.NS", "TECHM.NS", "HINDALCO.NS", "TRENT.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "GRASIM.NS", "BAJAJ-AUTO.NS",
            "INDUSINDBK.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TATACONSUM.NS", "CIPLA.NS", "DIVISLAB.NS", "BPCL.NS",
            "SHRIRAMFIN.NS", "SBILIFE.NS", "HDFCLIFE.NS", "LTIM.NS", "BEL.NS", "HAL.NS", "CHOLAFIN.NS", "TVSMOTOR.NS",
            "INDIGO.NS", "ZOMATO.NS", "DLF.NS", "VBL.NS", "JINDALSTEL.NS", "JSWSTEEL.NS", "TORNTPHARM.NS", "PIDILITIND.NS",
            "GODREJCP.NS", "MAXHEALTH.NS", "BOSCHLTD.NS", "CGPOWER.NS", "CUMMINSIND.NS", "LODHA.NS", "TIINDIA.NS", "TRENT.NS",
            "AMBUJACEM.NS", "SHREECEM.NS", "PNB.NS", "BANKBARODA.NS", "IOB.NS", "UNIONBANK.NS", "CANBK.NS", "IDBI.NS",
            "GAIL.NS", "IOC.NS", "IRFC.NS", "PFC.NS", "RECLTD.NS", "ADANIPOWER.NS", "TATAPOWER.NS", "ZENTEC.NS", "UBL.NS",
            "MARICO.NS", "DABUR.NS", "HAVELLS.NS", "VOLTAS.NS", "DIXON.NS", "PIIND.NS", "AUBANK.NS", "TATACOMM.NS", "OFSS.NS",
            "NAUKRI.NS", "ICICIPRULI.NS", "ICICIGI.NS", "HDFCAMC.NS", "^NSEI"
        ]
        
        # Keep unique
        symbols = list(set(nifty100))
        logger.info(f"Initiating backfill for NIFTY100 universe ({len(symbols)} symbols) targeting {args.years} years...")
        manager.backfill_universe(symbols, years=args.years, max_workers=5)
        sys.exit(0)
        
    if args.coverage_report:
        from analytics.ml.coverage_report import CoverageReporter
        reporter = CoverageReporter()
        reporter.generate_report()
        sys.exit(0)

    # ── Handle Phase 4.2 Reality Check ────────────────────────────────────────
    if args.reality_check:
        from analytics.ml.reality_check import ValidationEngine
        datasets_dir = Path(__file__).parent / "research" / "datasets"
        parquet_files = sorted(list(datasets_dir.glob("*.parquet")))
        if not parquet_files:
            logger.error("No dataset found to run reality check.")
            sys.exit(1)
        engine = ValidationEngine(parquet_files[-1])
        engine.run_full_validation()
        sys.exit(0)

    # ── Handle Phase 5.0 ──────────────────────────────────────────────────────
    if args.feature_selection:
        from analytics.ml.feature_selection import FeatureSelectionEngine
        datasets_dir = Path(__file__).parent / "research" / "datasets"
        parquet_files = sorted(list(datasets_dir.glob("*.parquet")))
        if not parquet_files:
            logger.error("No dataset found to run feature selection.")
            sys.exit(1)
        engine = FeatureSelectionEngine(parquet_files[-1])
        engine.run()
        sys.exit(0)

    if args.run_tournament:
        from analytics.ml.tournament import ModelTournament
        datasets_dir = Path(__file__).parent / "research" / "datasets"
        parquet_files = sorted(list(datasets_dir.glob("*.parquet")))
        if not parquet_files:
            logger.error("No dataset found to run tournament.")
            sys.exit(1)
        tournament = ModelTournament(parquet_files[-1])
        tournament.run()
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

    if args.ml_backtest:
        from analytics.backtesting.factor_backtest import FactorBacktestEngine
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days)
        engine = FactorBacktestEngine(initial_balance=args.capital or 100_000.0, top_n=args.top, weighting_scheme=args.weighting, use_ml=True)
        try:
            result = engine.run(start_date=start_date, end_date=end_date)
            logger.info("=== ML Backtest Complete ===")
            print("\nStrategy Metrics (ML Engine):")
            for k, v in result['strategy'].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.2f}")
                else:
                    print(f"  {k}: {v}")
            print("\nBenchmark Metrics (^NSEI Buy & Hold):")
            for k, v in result['benchmark'].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.2f}")
                else:
                    print(f"  {k}: {v}")
            
            if result['turnover_stats']:
                avg_turnover = sum(t['turnover_pct'] for t in result['turnover_stats']) / len(result['turnover_stats'])
                print(f"\nAverage Monthly Turnover: {avg_turnover:.2f}%")
        except Exception as e:
            logger.error(f"ML backtest failed: {e}", exc_info=True)
        sys.exit(0)
        
    if args.build_dataset:
        from analytics.ml.dataset_builder import DatasetBuilder
        builder = DatasetBuilder()
        builder.build_and_save()
        sys.exit(0)
        
    if args.validate_dataset:
        from analytics.ml.data_validation import DataValidator
        datasets_dir = Path(__file__).parent / "research" / "datasets"
        parquet_files = list(datasets_dir.glob("*.parquet"))
        if not parquet_files:
            logger.error("No dataset found to validate.")
            sys.exit(1)
        latest_dataset = sorted(parquet_files)[-1]
        validator = DataValidator(latest_dataset)
        validator.run_validation()
        sys.exit(0)
        
    if args.train_model:
        from analytics.ml.train_model import ModelTrainer
        from analytics.ml.explainability import ModelExplainer
        datasets_dir = Path(__file__).parent / "research" / "datasets"
        parquet_files = list(datasets_dir.glob("*.parquet"))
        if not parquet_files:
            logger.error("No dataset found to train on.")
            sys.exit(1)
        latest_dataset = sorted(parquet_files)[-1]
        
        trainer = ModelTrainer(latest_dataset)
        registry_entry = trainer.run_training()
        
        explainer = ModelExplainer(registry_entry, latest_dataset)
        explainer.generate_importance_report()
        sys.exit(0)
        
    if args.predict_universe:
        from analytics.ml.predictor import AlphaPredictor
        from analytics.ml.dataset_builder import DatasetBuilder
        from datetime import date
        
        builder = DatasetBuilder()
        features_df = builder.build_features_df(end_date=date.today())
        if features_df.empty:
            logger.error("No features available.")
            sys.exit(1)
            
        today_features = features_df[features_df['date'] == features_df['date'].max()]
        if today_features.empty:
            logger.error("No features available for today.")
            sys.exit(1)
            
        predictor = AlphaPredictor()
        preds = predictor.predict(today_features)
        preds = preds.sort_values('predicted_rank')
        
        logger.info("=== ATLAS ML PREDICTIONS (Top 10) ===")
        top_10 = preds.head(args.top)[['predicted_rank', 'symbol', 'predicted_return']]
        print(top_10.to_string(index=False))
        
        from analytics.ml.explainability import ModelExplainer
        with open(Path(__file__).parent / "models" / "model_registry.json", 'r') as f:
            registry = json.load(f)
        datasets_dir = Path(__file__).parent / "research" / "datasets"
        latest_dataset = sorted(datasets_dir.glob("*.parquet"))[-1]
        explainer = ModelExplainer(registry[0], latest_dataset)
        
        top_row = preds.iloc[0]
        drivers, reasons = explainer.explain_prediction(top_row[predictor.features])
        
        logger.info(f"\nWhy Atlas likes {top_row['symbol']} (Predicted Return: {top_row['predicted_return']*100:.2f}%):")
        for reason in reasons:
            logger.info(f" - {reason}")
            
        # Write to Report
        out_dir = Path(__file__).parent / "research" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "Alpha_Prediction_Report.md"
        with open(report_path, "w") as f:
            f.write(f"# Atlas ML Alpha Prediction Report\n")
            f.write(f"Generated on: {date.today()}\n\n")
            f.write(f"## Top {args.top} Predicted Returns\n\n")
            f.write("| Rank | Symbol | Predicted Return | Expected Return % |\n")
            f.write("|------|--------|------------------|-------------------|\n")
            for _, row in top_10.iterrows():
                f.write(f"| {int(row['predicted_rank'])} | {row['symbol']} | {row['predicted_return']:.4f} | {row['predicted_return']*100:.2f}% |\n")
                
            f.write(f"\n## Explainability for Top Pick ({top_row['symbol']})\n\n")
            f.write(f"**Predicted Return:** {top_row['predicted_return']*100:.2f}%\n\n")
            for reason in reasons:
                f.write(f"- {reason}\n")
                
        logger.info(f"Alpha Prediction Report saved to: {report_path}")
        sys.exit(0)

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
