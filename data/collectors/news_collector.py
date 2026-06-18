"""
data/collectors/news_collector.py
===================================
Financial News Collector — Sprint 2

Purpose:
    Collect financial news headlines and summaries for NIFTY 50 stocks.
    Sources: NewsAPI.org, GNews, MoneyControl RSS, or yfinance .news attribute.

Output:
    News articles stored in the `news` table with is_processed=False.
    The SentimentService will pick up unprocessed articles on its next run.
"""

from __future__ import annotations

# TODO (Sprint 2): Implement NewsCollector after confirming news source API.
