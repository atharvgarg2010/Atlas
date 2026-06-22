import numpy as np
import pandas as pd
from scipy.optimize import minimize
from core.logging import get_logger

logger = get_logger(__name__)

class PortfolioOptimizer:
    def __init__(self, historical_prices: pd.DataFrame, factor_scores: pd.Series, risk_free_rate: float = 0.05):
        """
        Initialize the PortfolioOptimizer.
        
        Args:
            historical_prices: DataFrame where index is dates and columns are symbols with closing prices.
            factor_scores: Series where index is symbols and values are Composite Factor Scores (0-100).
            risk_free_rate: Annualized risk-free rate.
        """
        self.symbols = list(factor_scores.index)
        
        # Ensure we only use historical prices for the symbols we care about
        # Forward fill missing values to avoid NaNs disrupting correlation
        self.prices = historical_prices[self.symbols].ffill()
        self.returns = self.prices.pct_change().dropna()
        
        # Compute Covariance matrix (Annualized)
        self.cov_matrix = self.returns.cov() * 252
        self.corr_matrix = self.returns.corr()
        
        # Map factor scores to expected returns (0 to 100 maps to -10% to +25%)
        # R = -0.10 + (score / 100) * 0.35
        self.expected_returns = -0.10 + (factor_scores / 100.0) * 0.35
        
        self.rfr = risk_free_rate
        self.n = len(self.symbols)
        
        # Dynamic bounds: handles small N (e.g. N=3 -> max 33.3%, min 2%)
        # For N=10, max=20%, min=2%
        self.min_weight = min(0.02, 1.0 / self.n) if self.n > 0 else 0.0
        self.max_weight = max(0.20, 1.0 / self.n) if self.n > 0 else 1.0
        self.bounds = tuple((self.min_weight, self.max_weight) for _ in range(self.n))
        self.initial_guess = np.array([1.0 / self.n] * self.n)

    def _constraints(self):
        """Constraint: weights must sum to 1."""
        return ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
    def equal_weight(self):
        """Standard 1/N equal weight portfolio."""
        w = np.array([1.0 / self.n] * self.n)
        return self._format_output(w)

    def minimum_variance(self):
        """Minimize portfolio variance."""
        def objective(w):
            return 0.5 * np.dot(w.T, np.dot(self.cov_matrix, w))
            
        res = minimize(objective, self.initial_guess, method='SLSQP', bounds=self.bounds, constraints=self._constraints())
        if not res.success:
            logger.warning(f"MinVar optimization failed: {res.message}. Falling back to Equal Weight.")
            return self.equal_weight()
        return self._format_output(res.x)

    def maximum_sharpe(self):
        """Maximize Sharpe Ratio (Minimize negative Sharpe)."""
        def objective(w):
            port_ret = np.dot(w.T, self.expected_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            if port_vol == 0:
                return 0
            return -(port_ret - self.rfr) / port_vol
            
        res = minimize(objective, self.initial_guess, method='SLSQP', bounds=self.bounds, constraints=self._constraints())
        if not res.success:
            logger.warning(f"MaxSharpe optimization failed: {res.message}. Falling back to Equal Weight.")
            return self.equal_weight()
        return self._format_output(res.x)

    def risk_parity(self):
        """Equal risk contribution portfolio."""
        def objective(w):
            port_variance = np.dot(w.T, np.dot(self.cov_matrix, w))
            marginal_risk = np.dot(self.cov_matrix, w)
            risk_contribution = w * marginal_risk
            target_risk_contribution = port_variance / self.n
            return np.sum((risk_contribution - target_risk_contribution)**2)
            
        res = minimize(objective, self.initial_guess, method='SLSQP', bounds=self.bounds, constraints=self._constraints())
        if not res.success:
            logger.warning(f"RiskParity optimization failed: {res.message}. Falling back to Equal Weight.")
            return self.equal_weight()
        return self._format_output(res.x)

    def _format_output(self, weights: np.ndarray) -> dict:
        """Format the optimized weights and calculate relevant portfolio metrics."""
        w_series = pd.Series(weights, index=self.symbols)
        # Normalize just in case of tiny floating point issues
        w_series = w_series / w_series.sum()
        
        port_ret = np.dot(w_series, self.expected_returns)
        port_vol = np.sqrt(np.dot(w_series, np.dot(self.cov_matrix, w_series)))
        sharpe = (port_ret - self.rfr) / port_vol if port_vol > 0 else 0.0
        
        # Calculate risk contributions
        marginal_risk = np.dot(self.cov_matrix, w_series)
        risk_contributions = (w_series * marginal_risk) / (port_vol**2) if port_vol > 0 else pd.Series(0, index=self.symbols)
        
        # Correlation metrics
        corr_upper = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool))
        
        if self.n > 1:
            stacked = corr_upper.stack()
            avg_corr = stacked.mean()
            least_corr_pair = stacked.idxmin()
            most_corr_pair = stacked.idxmax()
            least_corr_val = stacked.min()
            most_corr_val = stacked.max()
        else:
            avg_corr = 1.0
            least_corr_pair = (self.symbols[0], self.symbols[0])
            most_corr_pair = (self.symbols[0], self.symbols[0])
            least_corr_val = 1.0
            most_corr_val = 1.0
            
        return {
            "weights": w_series.to_dict(),
            "expected_return": port_ret,
            "expected_volatility": port_vol,
            "sharpe_ratio": sharpe,
            "risk_contributions": risk_contributions.to_dict(),
            "correlation": {
                "average": avg_corr,
                "least_correlated_pair": least_corr_pair,
                "least_correlated_value": least_corr_val,
                "most_correlated_pair": most_corr_pair,
                "most_correlated_value": most_corr_val
            }
        }
