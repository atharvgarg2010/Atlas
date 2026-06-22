import pandas as pd
import numpy as np
from analytics.factors.base_factor import BaseFactor

class VolatilityFactor(BaseFactor):
    def __init__(self):
        super().__init__("volatility")

    def compute(self, symbol_data: pd.DataFrame, benchmark_data: pd.DataFrame | None = None) -> float | None:
        if len(symbol_data) < 21:
            return None
        
        latest = symbol_data.iloc[-1]
        if pd.isna(latest.get('atr_14')):
            return None
            
        pct_atr = latest['atr_14'] / latest['close']
        
        # 20-day return std dev
        returns = symbol_data['close'].pct_change().iloc[-20:]
        std_dev = returns.std()
        
        if pd.isna(std_dev):
            return None
            
        # We want lower volatility to score higher, so we negate the sum
        return -1.0 * (pct_atr + std_dev)
