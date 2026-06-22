"""
data/data_manager.py
====================
Project Atlas — Market Data Cache Layer

Provides a persistent historical data cache using Supabase PostgreSQL.
Handles initial fetch, incremental daily updates, and parallel universe syncing.
Replaces repeated slow yfinance API calls.
"""

from __future__ import annotations

import concurrent.futures
import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from analytics.technical.indicators import IndicatorEngine
from core.logging import get_logger
from database.connection import get_db
from database.models.market_data import MarketData, MarketIndicators, SymbolMetadata

logger = get_logger(__name__)


class DataManager:
    """
    Manages OHLCV market data and technical indicator caching.
    """

    def __init__(self) -> None:
        self.db = get_db()
        self.indicator_engine = IndicatorEngine()

    def get_market_data(self, symbol: str, history_days: int = 365) -> list[dict[str, Any]]:
        """
        Get enriched market data (OHLCV + indicators) for a symbol.
        Automatically syncs if data is missing or stale.

        Args:
            symbol: Ticker symbol (e.g. RELIANCE.NS)
            history_days: Minimum days of history required.

        Returns:
            List of dictionaries containing OHLCV + indicator data.
        """
        start_time = time.perf_counter()

        # Ensure data is fresh
        self.sync_symbol(symbol)

        cutoff_date = date.today() - timedelta(days=history_days)

        with self.db.session() as s:
            # Check if symbol is marked as INVALID to avoid queries
            meta = s.scalar(select(SymbolMetadata).where(SymbolMetadata.symbol == symbol))
            if meta and meta.status == "INVALID":
                logger.warning(f"  {symbol} is marked INVALID. Returning empty list.")
                return []

            # Join MarketData and MarketIndicators
            stmt = (
                select(MarketData, MarketIndicators)
                .join(MarketIndicators, (MarketData.symbol == MarketIndicators.symbol) & (MarketData.date == MarketIndicators.date))
                .where(MarketData.symbol == symbol)
                .where(MarketData.date >= cutoff_date)
                .order_by(MarketData.date.asc())
            )
            results = s.execute(stmt).all()

        if not results:
            return []

        # Convert to list of dicts required by Atlas engines
        enriched_candles = []
        for md, ind in results:
            # Note: Portfolio engine expects 'timestamp' to be datetime or date
            # We map 'date' -> 'timestamp' for compatibility with IndicatorEngine/BacktestEngine
            enriched_candles.append({
                "timestamp":   datetime.combine(md.date, datetime.min.time()),
                "open":        md.open,
                "high":        md.high,
                "low":         md.low,
                "close":       md.close,
                "adj_close":   md.adj_close,
                "volume":      md.volume,
                "ema_20":      ind.ema_20,
                "ema_50":      ind.ema_50,
                "sma_20":      ind.sma_20,
                "rsi_14":      ind.rsi_14,
                "macd":        ind.macd,
                "macd_signal": ind.macd_signal,
                "atr_14":      ind.atr_14,
            })

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"  [{symbol}] get_market_data() completed in {elapsed_ms:.2f}ms")

        return enriched_candles

    def sync_symbol(self, symbol: str, force: bool = False) -> bool:
        """
        Sync OHLCV and Indicators for a single symbol.
        Only fetches missing dates if data exists.

        Args:
            symbol: Ticker symbol.
            force: If True, bypasses INVALID status check.

        Returns:
            True if sync was successful or already up-to-date, False if failed.
        """
        with self.db.session() as s:
            meta = s.scalar(select(SymbolMetadata).where(SymbolMetadata.symbol == symbol))

            if meta and meta.status == "INVALID" and not force:
                return False

            if not meta:
                meta = SymbolMetadata(symbol=symbol, status="ACTIVE", total_candles=0)
                s.add(meta)
                s.commit()

            latest_date = s.scalar(select(func.max(MarketData.date)).where(MarketData.symbol == symbol))

        today = date.today()

        # Determine fetch range
        if not latest_date:
            # Cache miss: Fetch full history (1 year for safety, though backtests typically use 365 days)
            start_date = today - timedelta(days=365)
            is_full_fetch = True
            logger.info(f"[CACHE MISS] {symbol}: No local data found. Fetching from {start_date}")
        else:
            # Cache hit / Stale check
            # Note: Market data might not be available for 'today' (weekends, before market close).
            # But if latest_date is within 1-2 days, it's mostly fresh. We fetch from latest_date onwards
            # to capture any new completed candles.
            if latest_date >= today - timedelta(days=1):
                return True  # Already fresh enough

            start_date = latest_date
            is_full_fetch = False
            logger.info(f"[SYNC] {symbol}: Downloading missing candles from {start_date}")

        # Fetch from Yahoo Finance
        try:
            df = yf.download(
                tickers=symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False  # We want both Close and Adj Close
            )
        except Exception as exc:
            logger.error(f"[ERROR] {symbol}: yfinance fetch failed - {exc}")
            self._mark_invalid(symbol)
            return False

        if df.empty:
            if is_full_fetch:
                logger.warning(f"[ERROR] {symbol}: yfinance returned no data.")
                self._mark_invalid(symbol)
            return False

        # Handle multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        # Rename columns safely
        col_map = {
            "Date": "date", "Datetime": "date",
            "Open": "open", "High": "high", "Low": "low", 
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
        }
        df = df.rename(columns=col_map)
        
        # Ensure we have required columns
        required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
        if not required.issubset(df.columns):
            logger.error(f"[ERROR] {symbol}: missing required columns from yfinance.")
            self._mark_invalid(symbol)
            return False

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.dropna(subset=["close"])
        if df.empty:
            return False

        # We must calculate indicators using the full history, not just the missing chunk!
        # If this is an incremental fetch, we must merge it with the existing DB data, 
        # compute indicators on the full series, and then UPSERT everything.
        
        with self.db.session() as s:
            if not is_full_fetch:
                # Load existing DB OHLCV to compute indicators correctly
                existing_records = s.execute(
                    select(MarketData).where(MarketData.symbol == symbol).order_by(MarketData.date.asc())
                ).scalars().all()
                
                existing_df = pd.DataFrame([{
                    "date": r.date, "open": r.open, "high": r.high,
                    "low": r.low, "close": r.close, "adj_close": r.adj_close, "volume": r.volume
                } for r in existing_records])
                
                # Append new data and drop duplicates
                full_df = pd.concat([existing_df, df]).drop_duplicates(subset=["date"], keep="last")
            else:
                full_df = df
            
            full_df = full_df.sort_values("date").reset_index(drop=True)

        # Map to format required by IndicatorEngine
        candles_for_indicators = full_df.copy()
        candles_for_indicators["timestamp"] = pd.to_datetime(candles_for_indicators["date"])
        candles_list = candles_for_indicators.to_dict(orient="records")

        # Compute Indicators
        try:
            enriched_list = self.indicator_engine.enrich(candles_list)
        except Exception as exc:
            logger.error(f"[ERROR] {symbol}: indicator calculation failed - {exc}")
            return False

        # Convert back to DataFrame for easy bulk UPSERT
        enriched_df = pd.DataFrame(enriched_list)
        
        # Only UPSERT the newly fetched rows to save DB write overhead
        if not is_full_fetch:
            new_dates = df["date"].unique()
            upsert_df = enriched_df[enriched_df["date"].isin(new_dates)]
        else:
            upsert_df = enriched_df

        if upsert_df.empty:
            return True

        md_records = []
        ind_records = []
        
        for _, row in upsert_df.iterrows():
            md_records.append({
                "symbol": symbol,
                "date": row["date"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "adj_close": float(row["adj_close"]),
                "volume": int(row["volume"])
            })
            ind_records.append({
                "symbol": symbol,
                "date": row["date"],
                "ema_20": float(row["ema_20"]) if pd.notnull(row["ema_20"]) else None,
                "ema_50": float(row["ema_50"]) if pd.notnull(row["ema_50"]) else None,
                "sma_20": float(row["sma_20"]) if pd.notnull(row["sma_20"]) else None,
                "rsi_14": float(row["rsi_14"]) if pd.notnull(row["rsi_14"]) else None,
                "macd": float(row["macd"]) if pd.notnull(row["macd"]) else None,
                "macd_signal": float(row["macd_signal"]) if pd.notnull(row["macd_signal"]) else None,
                "atr_14": float(row["atr_14"]) if pd.notnull(row["atr_14"]) else None,
            })

        # Upsert into PostgreSQL
        with self.db.session() as s:
            # 1. UPSERT MarketData
            md_stmt = insert(MarketData).values(md_records)
            md_update_dict = {c.name: c for c in md_stmt.excluded if c.name not in ["id", "symbol", "date", "created_at"]}
            md_stmt = md_stmt.on_conflict_do_update(
                constraint="uq_market_data_symbol_date",
                set_=md_update_dict
            )
            s.execute(md_stmt)

            # 2. UPSERT MarketIndicators
            ind_stmt = insert(MarketIndicators).values(ind_records)
            ind_update_dict = {c.name: c for c in ind_stmt.excluded if c.name not in ["id", "symbol", "date"]}
            ind_stmt = ind_stmt.on_conflict_do_update(
                constraint="uq_market_indicators_symbol_date",
                set_=ind_update_dict
            )
            s.execute(ind_stmt)

            # 3. Update Metadata
            meta = s.scalar(select(SymbolMetadata).where(SymbolMetadata.symbol == symbol))
            meta.status = "ACTIVE"
            meta.first_date = full_df["date"].min()
            meta.last_date = full_df["date"].max()
            meta.total_candles = len(full_df)
            meta.last_synced = datetime.now()

            logger.info(f"[SAVED] {symbol}: Inserted/Updated {len(upsert_df)} candles")

        return True

    def sync_universe(self, symbols: list[str], max_workers: int = 5) -> dict[str, bool]:
        """
        Parallel sync of multiple symbols.
        """
        logger.info(f"Starting parallel sync for {len(symbols)} symbols with {max_workers} workers.")
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.sync_symbol, sym): sym for sym in symbols}
            for future in concurrent.futures.as_completed(futures):
                sym = futures[future]
                try:
                    results[sym] = future.result()
                except Exception as exc:
                    logger.error(f"[ERROR] Parallel sync failed for {sym}: {exc}")
                    results[sym] = False
        return results

    def get_cache_status(self) -> dict[str, Any]:
        """
        Get high-level statistics about the cache from the metadata table.
        """
        with self.db.session() as s:
            active = s.scalar(select(func.count()).where(SymbolMetadata.status == "ACTIVE"))
            invalid = s.scalar(select(func.count()).where(SymbolMetadata.status == "INVALID"))
            total_candles = s.scalar(select(func.sum(SymbolMetadata.total_candles)))
            latest_date = s.scalar(select(func.max(SymbolMetadata.last_date)))

        return {
            "Total Symbols Cached (ACTIVE)": active or 0,
            "Total Symbols (INVALID)": invalid or 0,
            "Total Candles": total_candles or 0,
            "Latest Data Date": latest_date,
        }

    def _mark_invalid(self, symbol: str) -> None:
        """Mark a symbol as INVALID in metadata to skip future fetch attempts."""
        with self.db.session() as s:
            meta = s.scalar(select(SymbolMetadata).where(SymbolMetadata.symbol == symbol))
            if not meta:
                meta = SymbolMetadata(symbol=symbol, status="INVALID", total_candles=0)
                s.add(meta)
            else:
                meta.status = "INVALID"
                meta.last_synced = datetime.now()
            logger.warning(f"  {symbol} marked as INVALID.")
