import json
from datetime import date, datetime
import pandas as pd
import numpy as np
from pathlib import Path
from core.logging import get_logger
from data.data_manager import DataManager
from analytics.factors.factor_engine import FactorEngine
from analytics.portfolio.optimizer import PortfolioOptimizer

logger = get_logger(__name__)

class PaperTrader:
    def __init__(self, initial_capital=1000000.0, top_n=10, weighting_scheme="maxsharpe"):
        self.dm = DataManager()
        self.engine = FactorEngine()
        
        self.capital = initial_capital
        self.top_n = top_n
        self.weighting = weighting_scheme
        
        self.live_dir = Path(__file__).resolve().parent.parent.parent / "research" / "live"
        self.out_dir = self.live_dir / "portfolios"
        self.reports_dir = self.live_dir / "reports"
        self.trades_dir = self.live_dir / "trades"
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_latest_prices_matrix(self, symbols: list) -> pd.DataFrame:
        # Need 252 days of history for robust covariance matrix
        all_data = []
        for sym in symbols:
            data = self.dm.get_market_data(sym, history_days=252)
            for row in data:
                all_data.append({"date": row["timestamp"], "symbol": sym, "close": row["close"]})
                
        df = pd.DataFrame(all_data)
        if df.empty: return pd.DataFrame()
        # Pivot to get close prices
        prices = df.pivot(index='date', columns='symbol', values='close')
        # Eliminate NaNs with forward fill then backward fill
        prices = prices.ffill().bfill()
        return prices
        
    def _run_data_quality_checks(self, ranks: pd.DataFrame, target_date: date) -> bool:
        logger.info("Running pre-generation Data Quality Checks...")
        issues = []
        
        # Verify finite factor scores
        for idx, row in ranks.iterrows():
            if not np.isfinite(row['composite_score']):
                issues.append(f"Non-finite composite_score for {row['symbol']}")
                
        # Check duplicates
        if ranks['symbol'].duplicated().any():
            issues.append("Duplicate symbols found in ranks.")
            
        report_md = f"# Data Quality Report ({target_date})\n\n"
        if not issues:
            report_md += "**Status:** PASS ✅\nAll factor scores are finite, and no duplicates exist."
            is_valid = True
        else:
            report_md += "**Status:** FAIL ❌\n\n### Issues Detected:\n"
            for issue in issues:
                report_md += f"- {issue}\n"
            is_valid = False
            
        with open(self.reports_dir / "Data_Quality_Report.md", "w", encoding="utf-8") as f:
            f.write(report_md)
            
        return is_valid

    def _verify_portfolio_integrity(self, portfolio: dict, target_date: date) -> bool:
        logger.info("Running Portfolio Valuation Integrity Checks...")
        issues = []
        
        # Check NaNs
        if np.isnan(portfolio.get('invested_capital', np.nan)): issues.append("NaN invested_capital")
        if np.isnan(portfolio.get('cash_balance', np.nan)): issues.append("NaN cash_balance")
        
        total_val = portfolio.get('invested_capital', 0) + portfolio.get('cash_balance', 0)
        if not np.isclose(total_val, portfolio.get('capital', 0), atol=1.0):
            issues.append(f"Capital mismatch: {total_val} != {portfolio.get('capital')}")
            
        weight_sum = 0
        for p in portfolio.get('positions', []):
            if np.isnan(p.get('entry_price', np.nan)): issues.append(f"NaN entry_price for {p['symbol']}")
            if np.isnan(p.get('weight', np.nan)): issues.append(f"NaN weight for {p['symbol']}")
            if p.get('shares', -1) < 0: issues.append(f"Negative shares for {p['symbol']}")
            if not isinstance(p.get('shares'), int): issues.append(f"Non-integer shares for {p['symbol']}")
            weight_sum += p.get('weight', 0)
            
        if portfolio.get('positions') and not np.isclose(weight_sum, 1.0, atol=0.01):
            issues.append(f"Weights do not sum to 100% (sum={weight_sum})")
            
        report_md = f"# Portfolio Integrity Report ({target_date})\n\n"
        if not issues:
            report_md += "**Status:** PASS ✅\nAll valuations and checks are consistent."
            is_valid = True
        else:
            report_md += "**Status:** FAIL ❌\n\n### Integrity Violations:\n"
            for issue in issues:
                report_md += f"- {issue}\n"
            is_valid = False
            
        with open(self.reports_dir / "Portfolio_Integrity_Report.md", "w", encoding="utf-8") as f:
            f.write(report_md)
            
        return is_valid
        
    def _update_trade_ledger(self, new_portfolio: dict):
        ledger_path = self.trades_dir / "trade_ledger.json"
        ledger = []
        if ledger_path.exists():
            with open(ledger_path, "r", encoding="utf-8") as f:
                ledger = json.load(f)
                
        active_path = self.out_dir / "active_portfolio.json"
        old_positions = {}
        if active_path.exists():
            with open(active_path, "r", encoding="utf-8") as f:
                old_port = json.load(f)
                old_positions = {p['symbol']: p for p in old_port.get('positions', [])}
                
        new_positions = {p['symbol']: p for p in new_portfolio.get('positions', [])}
        timestamp = datetime.now().isoformat()
        
        # Sells
        for sym, old_p in old_positions.items():
            if sym not in new_positions:
                ledger.append({
                    "timestamp": timestamp,
                    "action": "SELL",
                    "symbol": sym,
                    "shares": old_p['shares'],
                    "execution_price": old_p['entry_price'],  # idealized paper execution
                    "portfolio_weight": 0.0,
                    "factor_score": old_p.get('factor_score', 0.0)
                })
        
        # Buys
        for sym, new_p in new_positions.items():
            if sym not in old_positions:
                ledger.append({
                    "timestamp": timestamp,
                    "action": "BUY",
                    "symbol": sym,
                    "shares": new_p['shares'],
                    "execution_price": new_p['entry_price'],
                    "portfolio_weight": new_p['weight'],
                    "factor_score": new_p.get('factor_score', 0.0)
                })
            else:
                # Rebalance adjustments could be recorded here if desired
                # But for simplicity tracking the net new buys and sells
                pass
                
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=4)

    def run_rebalance(self):
        target_date = date.today()
        logger.info(f"Starting Live Paper Trading Rebalance for {target_date}...")
        
        # 1. Rank Universe
        ranks = self.engine.rank_universe(target_date)
        if ranks.empty:
            logger.error("Factor Engine returned empty ranks. Cannot construct portfolio.")
            return False
            
        # 2. Select Top N & Run Data Quality
        top_stocks = ranks.head(self.top_n)
        if not self._run_data_quality_checks(top_stocks, target_date):
            logger.error("Data Quality checks failed. Aborting rebalance.")
            return False
            
        symbols = top_stocks['symbol'].tolist()
        scores = top_stocks.set_index('symbol')['composite_score']
        
        logger.info(f"Selected Top {self.top_n} candidates. Fetching correlation matrix...")
        
        # 3. Optimize Weights
        prices = self._get_latest_prices_matrix(symbols)
        if prices.empty:
            logger.error("Failed to fetch historical prices for optimization.")
            return False
            
        optimizer = PortfolioOptimizer(prices, scores)
        
        if self.weighting == "maxsharpe":
            res = optimizer.maximum_sharpe()
        elif self.weighting == "minvar":
            res = optimizer.minimum_variance()
        elif self.weighting == "riskparity":
            res = optimizer.risk_parity()
        else:
            res = optimizer.equal_weight()
            
        weights = res['weights']
        
        # 4. Construct Portfolio Object
        latest_closes = prices.iloc[-1].to_dict()
        positions = []
        
        for sym, weight in weights.items():
            alloc = self.capital * weight
            price = latest_closes.get(sym, np.nan)
            
            if np.isnan(price):
                logger.warning(f"NaN price detected for {sym}. Attempting to fetch last valid close...")
                fallback_data = self.dm.get_market_data(sym, history_days=5)
                if fallback_data:
                    price = fallback_data[-1]['close']
                else:
                    logger.error(f"Could not resolve valid price for {sym}. Halting.")
                    return False
            
            shares = int(alloc / price) if price > 0 else 0
            
            positions.append({
                "symbol": sym,
                "weight": float(weight),
                "shares": shares,
                "entry_price": float(price),
                "factor_score": float(scores.loc[sym]),
                "entry_date": target_date.isoformat()
            })
            
        invested = sum(p['shares'] * p['entry_price'] for p in positions)
        cash_balance = self.capital - invested
            
        portfolio = {
            "date": target_date.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "capital": float(self.capital),
            "invested_capital": float(invested),
            "cash_balance": float(cash_balance),
            "weighting_scheme": self.weighting,
            "expected_sharpe": float(res.get('sharpe_ratio', 0)),
            "positions": positions
        }
        
        # 5. Integrity Check
        if not self._verify_portfolio_integrity(portfolio, target_date):
            logger.error("Portfolio Integrity checks failed. Aborting save.")
            return False
            
        # 6. Update Ledger & Save
        self._update_trade_ledger(portfolio)
        
        out_path = self.out_dir / f"portfolio_{target_date.isoformat()}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(portfolio, f, indent=4)
            
        with open(self.out_dir / "active_portfolio.json", "w", encoding="utf-8") as f:
            json.dump(portfolio, f, indent=4)
            
        logger.info(f"Portfolio generated successfully. Expected Sharpe: {portfolio['expected_sharpe']:.2f}")
        return True

if __name__ == "__main__":
    from core.logging import setup_logging
    setup_logging(log_level="INFO")
    trader = PaperTrader()
    trader.run_rebalance()
