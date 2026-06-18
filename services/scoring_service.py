"""
services/scoring_service.py
============================
Stock Composite Scoring Service — Sprint 2

Purpose:
    Compute a 0–100 opportunity score for each tracked stock using a
    weighted combination of technical, momentum, news, and volume signals.

Formula (configurable in config/settings.yaml):
    final_score = (technical * 0.40) + (momentum * 0.30) + (news * 0.20) + (volume * 0.10)
"""

from __future__ import annotations

# TODO (Sprint 2): Implement ScoringService.
