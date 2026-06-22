import pandas as pd
from analytics.factors.base_factor import BaseFactor

class LiquidityFactor(BaseFactor):
    def __init__(self):
        super().__init__("liquidity")

    def compute(self, symbol_data: pd.DataFrame, benchmark_data: pd.DataFrame | None = None) -> float | None:
        if len(symbol_data) < 30:
            return None
            
        # 30-day average volume and price
        recent_30 = symbol_data.iloc[-30:]
        avg_vol = recent_30['volume'].mean()
        avg_price = recent_30['close'].mean()
        
        if pd.isna(avg_vol) or pd.isna(avg_price):
            return None
            
        return avg_vol * avg_price
