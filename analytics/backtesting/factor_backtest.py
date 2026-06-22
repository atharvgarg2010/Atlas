from typing import Any
import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from analytics.factors.factor_engine import FactorEngine
from analytics.portfolio.optimizer import PortfolioOptimizer
from analytics.ml.predictor import AlphaPredictor
from analytics.ml.dataset_builder import DatasetBuilder
from core.logging import get_logger

logger = get_logger(__name__)

class FactorBacktestEngine:
    def __init__(self, initial_balance: float = 100000.0, top_n: int = 10, weighting_scheme: str = "equal", use_ml: bool = False):
        self.initial_balance = initial_balance
        self.top_n = top_n
        self.weighting_scheme = weighting_scheme
        self.use_ml = use_ml
        self.transaction_cost = 0.001  # 0.1% per trade
        
        self.factor_engine = FactorEngine()
        
        if self.use_ml:
            logger.info("Initializing ML components for Backtest...")
            self.predictor = AlphaPredictor()
            self.dataset_builder = DatasetBuilder()
            # Cache the entire historical feature dataset in memory to avoid rebuilding every month
            self.historical_features = self.dataset_builder.build_features_df(end_date=None)
        
    def run(self, start_date: date, end_date: date) -> dict[str, Any]:
        """
        Run the monthly rebalancing backtest.
        """
        strat_name = "ML Prediction" if self.use_ml else "Factor Composite"
        logger.info(f"Starting {strat_name} Backtest: {start_date} to {end_date}")
        
        cash = self.initial_balance
        positions = {}  # symbol -> shares
        
        equity_curve = []
        turnover_stats = []
        
        total_transaction_costs = 0.0
        
        # Determine all month-end dates within range
        current_date = start_date.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)
        if current_date > end_date:
            current_date = end_date
            
        logger.info("Fetching entire historical data for simulation...")
        all_data = self.factor_engine._fetch_data_batch(end_date)
        benchmark_symbol = "^NSEI"
        if benchmark_symbol not in all_data:
            raise Exception("Benchmark ^NSEI missing from DB.")
        
        benchmark_df = all_data[benchmark_symbol]
        benchmark_df = benchmark_df[(benchmark_df['date'] >= start_date) & (benchmark_df['date'] <= end_date)]
        
        if len(benchmark_df) == 0:
            raise Exception("No benchmark data in the specified date range.")
            
        benchmark_start_price = benchmark_df.iloc[0]['close']
        benchmark_end_price = benchmark_df.iloc[-1]['close']
        benchmark_cagr = (benchmark_end_price / benchmark_start_price) ** (365 / (end_date - start_date).days) - 1
        
        # Align all dates
        trading_dates = benchmark_df['date'].tolist()
        
        rebalance_dates = []
        last_month = -1
        for d in trading_dates:
            if d.month != last_month:
                if last_month != -1:
                    rebalance_dates.append(d)
                last_month = d.month
        rebalance_dates.append(trading_dates[-1])
        
        portfolio_value = cash
        
        # For Directional Accuracy
        predictions = []
        actuals = []
        
        for d in trading_dates:
            # Mark to market before any trading
            current_pv = cash
            for sym, shares in positions.items():
                sym_df = all_data.get(sym)
                if sym_df is not None:
                    # get price for date d or last available before d
                    past_data = sym_df[sym_df['date'] <= d]
                    if not past_data.empty:
                        current_pv += shares * past_data.iloc[-1]['close']
                        
            equity_curve.append({'date': d, 'portfolio_value': current_pv})
            portfolio_value = current_pv
            
            # Rebalance
            if d in rebalance_dates:
                logger.info(f"Rebalancing on {d}")
                
                top_stocks = []
                scores_for_optimizer = None
                
                if self.use_ml:
                    # ML Mode
                    # Get features available on date d
                    # The historical features dataframe has all dates, so we just filter
                    features_at_d = self.historical_features[self.historical_features['date'] == d].copy()
                    
                    if features_at_d.empty:
                        logger.warning(f"No ML features available for {d}. Holding positions.")
                        continue
                        
                    # Drop the future target before predicting to simulate strict inference
                    if 'target_return_30d' in features_at_d.columns:
                        features_at_d = features_at_d.drop(columns=['target_return_30d'])
                        
                    try:
                        preds_df = self.predictor.predict(features_at_d)
                    except Exception as e:
                        logger.warning(f"ML Prediction failed on {d}: {e}")
                        continue
                        
                    # Sort by predicted rank
                    preds_df = preds_df.sort_values('predicted_rank')
                    top_stocks = preds_df.head(self.top_n)['symbol'].tolist()
                    
                    # Convert to Series for optimizer
                    scores_for_optimizer = preds_df.set_index('symbol')['predicted_return'].loc[top_stocks]
                    
                    # Track for directional accuracy (save prediction to compare next month)
                    # For simplicity, we can just save the prediction. It requires comparing against actual return later.
                    # Since this runs over the whole history, we can peek at the target_return_30d from the cached dataset just for METRICS
                    # (this is NOT used by the model/optimizer).
                    targets_at_d = self.historical_features[self.historical_features['date'] == d]
                    for idx, row in preds_df.iterrows():
                        sym = row['symbol']
                        pred_ret = row['predicted_return']
                        actual_ret_row = targets_at_d[targets_at_d['symbol'] == sym]
                        if not actual_ret_row.empty:
                            actual_ret = actual_ret_row.iloc[0]['target_return_30d']
                            if not pd.isna(actual_ret):
                                predictions.append(pred_ret)
                                actuals.append(actual_ret)
                else:
                    # Heuristic Factor Mode
                    ranks_df = self.factor_engine.rank_universe(d, persist=False)
                    if ranks_df.empty:
                        logger.warning(f"No ranking available for {d}. Holding positions.")
                        continue
                        
                    top_stocks = ranks_df.head(self.top_n)['symbol'].tolist()
                    scores_for_optimizer = ranks_df.head(self.top_n).set_index('symbol')['composite_score']
                
                # Identify Turnover
                current_symbols = set(positions.keys())
                new_symbols = set(top_stocks)
                
                added = list(new_symbols - current_symbols)
                removed = list(current_symbols - new_symbols)
                
                turnover_pct = (len(added) + len(removed)) / (2 * self.top_n) * 100 if self.top_n > 0 else 0
                
                turnover_stats.append({
                    'date': d,
                    'added': added,
                    'removed': removed,
                    'turnover_pct': turnover_pct
                })
                
                # Run Portfolio Optimization
                past_prices = {}
                for sym in top_stocks:
                    sym_df = all_data.get(sym)
                    if sym_df is not None:
                        past_data = sym_df[sym_df['date'] <= d].tail(252) # use up to 1 year of data
                        past_prices[sym] = past_data.set_index('date')['close']
                        
                prices_df = pd.DataFrame(past_prices).ffill().dropna()
                
                if prices_df.empty:
                    continue
                    
                optimizer = PortfolioOptimizer(
                    prices_df, 
                    scores_for_optimizer, 
                    risk_free_rate=0.05, 
                    is_ml_expected_returns=self.use_ml
                )
                
                if self.weighting_scheme == "minvar":
                    opt_res = optimizer.minimum_variance()
                elif self.weighting_scheme == "maxsharpe":
                    opt_res = optimizer.maximum_sharpe()
                elif self.weighting_scheme == "riskparity":
                    opt_res = optimizer.risk_parity()
                else:
                    opt_res = optimizer.equal_weight()
                    
                target_weights = opt_res['weights']
                
                # Liquidate all to cash first 
                cash = portfolio_value
                positions.clear()
                
                # Apply transaction costs on liquidation (if PV > 0)
                # Gross traded value = PV (we sold everything)
                cost = portfolio_value * self.transaction_cost
                cash -= cost
                total_transaction_costs += cost
                
                # The remaining cash is used to buy
                buy_power = cash
                
                # Rebuy
                for sym in top_stocks:
                    weight = target_weights.get(sym, 0.0)
                    if weight <= 0:
                        continue
                        
                    target_value = buy_power * weight
                    sym_df = all_data.get(sym)
                    if sym_df is not None:
                        past_data = sym_df[sym_df['date'] <= d]
                        if not past_data.empty:
                            price = past_data.iloc[-1]['close']
                            
                            # Transaction cost on buy
                            buy_cost = target_value * self.transaction_cost
                            actual_invested = target_value - buy_cost
                            total_transaction_costs += buy_cost
                            
                            shares = actual_invested / price
                            positions[sym] = shares
                            cash -= target_value
                            
                            
        # Final Metrics
        days = (end_date - start_date).days
        years = days / 365.0
        
        # Calculate Gross Portfolio Value if no transaction costs existed
        gross_portfolio_value = portfolio_value + total_transaction_costs
        gross_cagr = (gross_portfolio_value / self.initial_balance) ** (1 / years) - 1 if years > 0 else 0
        net_cagr = (portfolio_value / self.initial_balance) ** (1 / years) - 1 if years > 0 else 0
        
        eq_df = pd.DataFrame(equity_curve)
        eq_df['peak'] = eq_df['portfolio_value'].cummax()
        eq_df['drawdown'] = (eq_df['peak'] - eq_df['portfolio_value']) / eq_df['peak']
        max_dd = eq_df['drawdown'].max() * 100
        
        # Volatility & Sharpe
        eq_df['daily_return'] = eq_df['portfolio_value'].pct_change()
        daily_vol = eq_df['daily_return'].std()
        annual_vol = daily_vol * np.sqrt(252)
        sharpe = net_cagr / annual_vol if annual_vol > 0 else 0
        
        # Benchmark metrics
        bench_df = benchmark_df.copy()
        bench_df['peak'] = bench_df['close'].cummax()
        bench_df['drawdown'] = (bench_df['peak'] - bench_df['close']) / bench_df['peak']
        bench_max_dd = bench_df['drawdown'].max() * 100
        bench_daily_ret = bench_df['close'].pct_change()
        bench_annual_vol = bench_daily_ret.std() * np.sqrt(252)
        bench_sharpe = benchmark_cagr / bench_annual_vol if bench_annual_vol > 0 else 0
        
        # Directional Accuracy (Only tracked for ML mode)
        dir_acc = 0.0
        if self.use_ml and len(predictions) > 0:
            correct = np.sign(predictions) == np.sign(actuals)
            dir_acc = np.mean(correct) * 100.0
        
        result = {
            'strategy': {
                'initial_balance': self.initial_balance,
                'final_balance': portfolio_value,
                'gross_cagr': gross_cagr * 100,
                'net_cagr': net_cagr * 100,
                'max_drawdown': max_dd,
                'annual_volatility': annual_vol * 100,
                'sharpe_ratio': sharpe,
                'total_transaction_costs': total_transaction_costs,
                'directional_accuracy': dir_acc
            },
            'benchmark': {
                'cagr': benchmark_cagr * 100,
                'max_drawdown': bench_max_dd,
                'annual_volatility': bench_annual_vol * 100,
                'sharpe_ratio': bench_sharpe
            },
            'turnover_stats': turnover_stats,
            'equity_curve': equity_curve
        }
        return result
