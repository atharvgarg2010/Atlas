"""
analytics/technical/strategies.py
===================================
Project Atlas — Strategy Engine v1

Purpose
-------
Convert enriched OHLCV candle data (with technical indicators) into
actionable, structured trading signals.

Strategies Included (v1)
------------------------
    1. RSI Mean Reversion: 
       - BUY when RSI < 30
       - SELL when RSI > 70
    
    2. EMA Crossover:
       - BUY when EMA 20 crosses ABOVE EMA 50
       - SELL when EMA 20 crosses BELOW EMA 50

Design Decisions
----------------
- No Signal Spam: We track the current market position ('LONG', 'SHORT', 'NEUTRAL')
  to ensure we only emit a signal when the trend/state actually changes.
- Modular Design: New strategies can be added as private methods and mapped
  in the `__init__` registry.
- Pure Python: Processes the list of dicts directly. No ML or external calls.

Output Format
-------------
List of dictionaries (one per candle) containing:
    {
        "timestamp": ...,
        "signal": "BUY" | "SELL" | "HOLD",
        "reason": str,
        "strategy": str,
        "price": float
    }
"""

from __future__ import annotations

from typing import Any


class StrategyEngine:
    """
    Evaluates technical indicators to generate structured trading signals.
    """

    def __init__(self, strategy_name: str = "EMA_CROSSOVER"):
        """
        Initialize the strategy engine.
        
        Args:
            strategy_name: Name of the strategy to run. 
                           Supported: "RSI", "EMA_CROSSOVER"
        """
        self.strategy_name = strategy_name.upper()
        
        # Strategy registry
        self._strategies = {
            "RSI": self._rsi_mean_reversion,
            "EMA_CROSSOVER": self._ema_crossover,
        }
        
        if self.strategy_name not in self._strategies:
            valid = ", ".join(self._strategies.keys())
            raise ValueError(f"Unknown strategy '{self.strategy_name}'. Valid options: {valid}")

    def generate_signals(self, candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process enriched candles and generate signals based on the selected strategy.
        
        Args:
            candles: List of enriched OHLCV dicts from IndicatorEngine.
            
        Returns:
            List of signal dicts corresponding 1:1 with the input candles.
        """
        if not candles:
            return []
            
        strategy_func = self._strategies[self.strategy_name]
        return strategy_func(candles)

    # ─── Strategy Implementations ─────────────────────────────────────────────

    def _rsi_mean_reversion(self, candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        RSI Mean Reversion Strategy.
        - BUY when RSI crosses below 30
        - SELL when RSI crosses above 70
        """
        signals = []
        current_position = "NEUTRAL"
        
        for c in candles:
            ts = c["timestamp"]
            price = c["close"]
            rsi = c.get("rsi_14")
            
            signal = "HOLD"
            reason = ""
            
            if rsi is not None:
                # Check for BUY condition
                if rsi < 30 and current_position != "LONG":
                    signal = "BUY"
                    reason = f"RSI ({rsi:.2f}) entered oversold territory (< 30)"
                    current_position = "LONG"
                    
                # Check for SELL condition
                elif rsi > 70 and current_position != "SHORT":
                    signal = "SELL"
                    reason = f"RSI ({rsi:.2f}) entered overbought territory (> 70)"
                    current_position = "SHORT"
            
            signals.append({
                "timestamp": ts,
                "signal": signal,
                "reason": reason,
                "strategy": "RSI_MEAN_REVERSION",
                "price": price,
            })
            
        return signals

    def _ema_crossover(self, candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        EMA Crossover Strategy (20-period vs 50-period).
        - BUY when EMA 20 crosses ABOVE EMA 50
        - SELL when EMA 20 crosses BELOW EMA 50
        """
        signals = []
        current_position = "NEUTRAL"
        prev_ema_20 = None
        prev_ema_50 = None
        
        for c in candles:
            ts = c["timestamp"]
            price = c["close"]
            ema_20 = c.get("ema_20")
            ema_50 = c.get("ema_50")
            
            signal = "HOLD"
            reason = ""
            
            # Need valid EMAs for the current AND previous candle to detect a crossover
            if ema_20 is not None and ema_50 is not None and prev_ema_20 is not None and prev_ema_50 is not None:
                
                # BUY: EMA 20 was below EMA 50, now it's above
                if prev_ema_20 <= prev_ema_50 and ema_20 > ema_50:
                    if current_position != "LONG":
                        signal = "BUY"
                        reason = f"Bullish Crossover: EMA-20 ({ema_20:.2f}) crossed above EMA-50 ({ema_50:.2f})"
                        current_position = "LONG"
                
                # SELL: EMA 20 was above EMA 50, now it's below
                elif prev_ema_20 >= prev_ema_50 and ema_20 < ema_50:
                    if current_position != "SHORT":
                        signal = "SELL"
                        reason = f"Bearish Crossover: EMA-20 ({ema_20:.2f}) crossed below EMA-50 ({ema_50:.2f})"
                        current_position = "SHORT"

            signals.append({
                "timestamp": ts,
                "signal": signal,
                "reason": reason,
                "strategy": "EMA_CROSSOVER",
                "price": price,
            })
            
            prev_ema_20 = ema_20
            prev_ema_50 = ema_50
            
        return signals
