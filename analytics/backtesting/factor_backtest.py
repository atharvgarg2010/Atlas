from typing import Any
import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from analytics.factors.factor_engine import FactorEngine
from analytics.portfolio.optimizer import PortfolioOptimizer
from core.logging import get_logger

logger = get_logger(__name__)

class FactorBacktestEngine:
    def __init__(self, initial_balance: float = 100000.0, top_n: int = 10, weighting_scheme: str = "equal"):
        self.initial_balance = initial_balance
        self.top_n = top_n
        self.weighting_scheme = weighting_scheme
        self.factor_engine = FactorEngine()
        
    def run(self, start_date: date, end_date: date) -> dict[str, Any]:
        """
        Run the monthly rebalancing backtest.
        """
        logger.info(f"Starting Factor Strategy Backtest: {start_date} to {end_date}")
        
        current_date = start_date
        cash = self.initial_balance
        positions = {}  # symbol -> shares
        
        equity_curve = []
        turnover_stats = []
        
        # Determine all month-end dates within range
        # Approximation: we'll run on the last calendar day of the month, 
        # and pull ranking from available data up to that day.
        current_date = start_date.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)
        
        if current_date > end_date:
            current_date = end_date
            
        # We also need daily benchmark and stock prices for mark-to-market.
        # Fetching everything for the entire period into a dict of DataFrames.
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
                ranks_df = self.factor_engine.rank_universe(d, persist=False)
                
                if ranks_df.empty:
                    logger.warning(f"No ranking available for {d}. Holding positions.")
                    continue
                    
                top_stocks = ranks_df.head(self.top_n)['symbol'].tolist()
                
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
                
                factor_scores = ranks_df.head(self.top_n).set_index('symbol')['composite_score']
                
                optimizer = PortfolioOptimizer(prices_df, factor_scores, risk_free_rate=0.05)
                
                if self.weighting_scheme == "minvar":
                    opt_res = optimizer.minimum_variance()
                elif self.weighting_scheme == "maxsharpe":
                    opt_res = optimizer.maximum_sharpe()
                elif self.weighting_scheme == "riskparity":
                    opt_res = optimizer.risk_parity()
                else:
                    opt_res = optimizer.equal_weight()
                    
                target_weights = opt_res['weights']
                
                # Liquidate all to cash first (simplified rebalance assuming 0 slippage/fees for now)
                cash = portfolio_value
                positions.clear()
                
                # Rebuy
                for sym in top_stocks:
                    weight = target_weights.get(sym, 0.0)
                    if weight <= 0:
                        continue
                        
                    target_value = portfolio_value * weight
                    sym_df = all_data.get(sym)
                    if sym_df is not None:
                        past_data = sym_df[sym_df['date'] <= d]
                        if not past_data.empty:
                            price = past_data.iloc[-1]['close']
                            shares = target_value / price
                            positions[sym] = shares
                            cash -= target_value
                            
                            
        # Final Metrics
        days = (end_date - start_date).days
        years = days / 365.0
        cagr = (portfolio_value / self.initial_balance) ** (1 / years) - 1 if years > 0 else 0
        
        eq_df = pd.DataFrame(equity_curve)
        eq_df['peak'] = eq_df['portfolio_value'].cummax()
        eq_df['drawdown'] = (eq_df['peak'] - eq_df['portfolio_value']) / eq_df['peak']
        max_dd = eq_df['drawdown'].max() * 100
        
        # Volatility & Sharpe
        eq_df['daily_return'] = eq_df['portfolio_value'].pct_change()
        daily_vol = eq_df['daily_return'].std()
        annual_vol = daily_vol * np.sqrt(252)
        sharpe = cagr / annual_vol if annual_vol > 0 else 0
        
        # Benchmark metrics
        bench_df = benchmark_df.copy()
        bench_df['peak'] = bench_df['close'].cummax()
        bench_df['drawdown'] = (bench_df['peak'] - bench_df['close']) / bench_df['peak']
        bench_max_dd = bench_df['drawdown'].max() * 100
        bench_daily_ret = bench_df['close'].pct_change()
        bench_annual_vol = bench_daily_ret.std() * np.sqrt(252)
        bench_sharpe = benchmark_cagr / bench_annual_vol if bench_annual_vol > 0 else 0
        
        result = {
            'strategy': {
                'initial_balance': self.initial_balance,
                'final_balance': portfolio_value,
                'cagr': cagr * 100,
                'max_drawdown': max_dd,
                'annual_volatility': annual_vol * 100,
                'sharpe_ratio': sharpe
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
