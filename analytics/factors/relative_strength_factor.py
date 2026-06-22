import pandas as pd
from analytics.factors.base_factor import BaseFactor

class RSFactor(BaseFactor):
    def __init__(self):
        super().__init__("relative_strength")

    def compute(self, symbol_data: pd.DataFrame, benchmark_data: pd.DataFrame | None = None) -> float | None:
        if benchmark_data is None or len(symbol_data) < 126 or len(benchmark_data) < 126:
            return None
        
        # 6 months ~ 126 trading days
        current_close = symbol_data['close'].iloc[-1]
        close_6m = symbol_data['close'].iloc[-127]
        stock_return = (current_close - close_6m) / close_6m
        
        bench_current = benchmark_data['close'].iloc[-1]
        bench_6m = benchmark_data['close'].iloc[-127]
        bench_return = (bench_current - bench_6m) / bench_6m
        
        return stock_return - bench_return
