import pandas as pd
from analytics.factors.base_factor import BaseFactor

class TrendFactor(BaseFactor):
    def __init__(self):
        super().__init__("trend")

    def compute(self, symbol_data: pd.DataFrame, benchmark_data: pd.DataFrame | None = None) -> float | None:
        if len(symbol_data) < 1:
            return None
        
        latest = symbol_data.iloc[-1]
        
        if pd.isna(latest.get('ema_20')) or pd.isna(latest.get('ema_50')) or pd.isna(latest.get('ema_200')):
            return None

        price = latest['close']
        ema20 = latest['ema_20']
        ema50 = latest['ema_50']
        ema200 = latest['ema_200']

        score = 0
        if price > ema20:
            score += 25
        if ema20 > ema50:
            score += 25
        if ema50 > ema200:
            score += 25
        if price > ema200:
            score += 25

        return float(score)
