"""
services/signal_service.py
============================
Trade Signal Generation Service — Sprint 3

Purpose:
    Generate explainable BUY / SELL / WATCHLIST signals for stocks that
    pass multi-condition screening (trend + momentum + sentiment + volume).

Output:
    Signals stored in `signals` table with:
    - entry_low, entry_high, stop_loss, target_1, risk_reward, confidence
    - reasoning_json: structured dict of supporting factors
    - reasoning_text: human-readable explanation
"""

from __future__ import annotations

# TODO (Sprint 3): Implement SignalService.
