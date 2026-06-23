import json
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
from core.logging import get_logger
from data.data_manager import DataManager
from pathlib import Path

logger = get_logger(__name__)

class PerformanceTracker:
    def __init__(self):
        self.dm = DataManager()
        self.live_dir = Path(__file__).resolve().parent.parent.parent / "research" / "live"
        self.portfolios_dir = self.live_dir / "portfolios"
        self.history_file = self.live_dir / "performance_history.json"
        
    def _fetch_nifty_close(self):
        try:
            ticker = yf.Ticker("^NSEI")
            hist = ticker.history(period="5d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Failed to fetch NIFTY close: {e}")
        return None
        
    def _calculate_metrics(self, history: dict) -> dict:
        snapshots = history.get('daily_snapshots', [])
        if len(snapshots) < 2:
            return {}
            
        df = pd.DataFrame(snapshots)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate Returns
        df['returns'] = df['portfolio_value'].pct_change()
        
        metrics = {}
        
        # Win Rate
        wins = (df['returns'].dropna() > 0).sum()
        total_days = len(df['returns'].dropna())
        metrics['win_rate'] = float((wins / total_days) * 100) if total_days > 0 else 0.0
        
        # CAGR (Annualized Return)
        days_elapsed = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        if days_elapsed > 0:
            total_return_ratio = df['portfolio_value'].iloc[-1] / history['initial_capital']
            metrics['cagr'] = float((total_return_ratio ** (365.25 / days_elapsed) - 1) * 100)
        else:
            metrics['cagr'] = 0.0
            
        # Rolling Return (252 trading days)
        if len(df) >= 252:
            metrics['rolling_return_1y'] = float((df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[-252] - 1) * 100)
        else:
            metrics['rolling_return_1y'] = float((df['portfolio_value'].iloc[-1] / history['initial_capital'] - 1) * 100)
            
        # Drawdown
        cum_ret = df['portfolio_value'] / history['initial_capital']
        rolling_max = cum_ret.cummax()
        drawdowns = (cum_ret / rolling_max - 1) * 100
        metrics['max_drawdown'] = float(drawdowns.min())
        metrics['current_drawdown'] = float(drawdowns.iloc[-1])
        
        # Sharpe Ratio
        mean_ret = df['returns'].mean()
        std_ret = df['returns'].std()
        if std_ret > 0:
            metrics['sharpe_ratio'] = float((mean_ret / std_ret) * np.sqrt(252))
        else:
            metrics['sharpe_ratio'] = 0.0
            
        # Benchmark Alpha
        if 'nifty_close' in df.columns and history.get('initial_nifty'):
            nifty_ret = (df['nifty_close'].iloc[-1] / history['initial_nifty']) - 1
            port_ret = (df['portfolio_value'].iloc[-1] / history['initial_capital']) - 1
            metrics['benchmark_alpha'] = float((port_ret - nifty_ret) * 100)
        else:
            metrics['benchmark_alpha'] = 0.0
            
        return metrics

    def run_daily_mtm(self):
        logger.info("Starting Daily Portfolio Mark-to-Market (MTM)...")
        active_port_file = self.portfolios_dir / "active_portfolio.json"
        
        if not active_port_file.exists():
            logger.warning("No active_portfolio.json found. Cannot track performance.")
            return False
            
        with open(active_port_file, "r", encoding="utf-8") as f:
            portfolio = json.load(f)
            
        symbols = [p['symbol'] for p in portfolio['positions']]
        if not symbols:
            logger.warning("Active portfolio is 100% Cash.")
            portfolio_value = portfolio['capital']
        else:
            all_data = []
            for sym in symbols:
                data = self.dm.get_market_data(sym, history_days=5)
                for row in data:
                    all_data.append({"date": row["timestamp"], "symbol": sym, "close": row["close"]})
                    
            prices_df = pd.DataFrame(all_data)
            if prices_df.empty:
                logger.error("Failed to fetch latest prices for MTM.")
                return False
                
            latest_prices = prices_df.sort_values('date').groupby('symbol').last()['close'].to_dict()
            
            invested_value = sum(p['shares'] * latest_prices.get(p['symbol'], p['entry_price']) for p in portfolio['positions'])
            portfolio_value = portfolio.get('cash_balance', 0) + invested_value
            
        nifty_close = self._fetch_nifty_close()
        today_str = date.today().isoformat()
        
        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = {
                "initial_capital": portfolio['capital'],
                "start_date": portfolio['date'],
                "initial_nifty": nifty_close,
                "daily_snapshots": []
            }
            
        history['daily_snapshots'] = [s for s in history['daily_snapshots'] if s['date'] != today_str]
        
        snapshot = {
            "date": today_str,
            "portfolio_value": float(portfolio_value),
            "nifty_close": float(nifty_close) if nifty_close else None
        }
        history['daily_snapshots'].append(snapshot)
        
        # Calculate extended metrics
        metrics = self._calculate_metrics(history)
        history['current_metrics'] = metrics
        
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
            
        pnl_pct = (portfolio_value / history['initial_capital'] - 1) * 100
        logger.info(f"MTM Complete. Portfolio Value: {portfolio_value:,.2f} (Total PnL: {pnl_pct:+.2f}%)")
        return True

if __name__ == "__main__":
    from core.logging import setup_logging
    setup_logging(log_level="INFO")
    tracker = PerformanceTracker()
    tracker.run_daily_mtm()
