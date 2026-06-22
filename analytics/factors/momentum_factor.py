import pandas as pd
from analytics.factors.base_factor import BaseFactor

class MomentumFactor(BaseFactor):
    def __init__(self):
        super().__init__("momentum")

    def compute(self, symbol_data: pd.DataFrame, benchmark_data: pd.DataFrame | None = None) -> float | None:
        if len(symbol_data) < 126:
            return None
        
        # 3 months ~ 63 trading days, 6 months ~ 126 trading days
        current_close = symbol_data['close'].iloc[-1]
        close_3m = symbol_data['close'].iloc[-64]
        close_6m = symbol_data['close'].iloc[-127]
        
        return_3m = (current_close - close_3m) / close_3m
        return_6m = (current_close - close_6m) / close_6m
        
        return 0.4 * return_3m + 0.6 * return_6m
