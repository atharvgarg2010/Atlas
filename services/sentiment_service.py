"""
services/sentiment_service.py
===============================
News Sentiment Analysis Service — Phase 2 (FinBERT)

Purpose:
    Batch-process unprocessed news articles through FinBERT and update
    sentiment scores in the news table.

NOTE: FinBERT (~440 MB) is loaded lazily and runs HOURLY, not every 15 min,
      to respect VPS memory constraints.
"""

from __future__ import annotations

# TODO (Phase 2): Implement SentimentService.
# Dependencies: transformers, torch (NOT in requirements.txt yet — add in Phase 2)
