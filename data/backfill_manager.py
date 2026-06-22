"""
data/backfill_manager.py
========================
Project Atlas — Historical Backfill Engine

Intelligently backfills missing historical data backwards from the earliest
available date in the database to achieve a target historical coverage.
"""

from __future__ import annotations

import concurrent.futures
import time
from datetime import date, datetime, timedelta
from typing import Any
from pathlib import Path

import pandas as pd
import yfinance as yf
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from analytics.technical.indicators import IndicatorEngine
from core.logging import get_logger
from database.connection import get_db
from database.models.market_data import MarketData, MarketIndicators, SymbolMetadata

logger = get_logger(__name__)

class BackfillManager:
    def __init__(self) -> None:
        self.db = get_db()
        self.indicator_engine = IndicatorEngine()
        self.out_dir = Path(__file__).parent.parent / "research" / "output"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def backfill_symbol(self, symbol: str, target_start_date: date) -> dict[str, Any]:
        """
        Backfills historical data for a single symbol.
        """
        result = {
            "symbol": symbol,
            "status": "SKIPPED",
            "earliest_before": None,
            "earliest_after": None,
            "rows_inserted": 0,
            "rows_updated": 0,
            "error": None
        }
        
        with self.db.session() as s:
            meta = s.scalar(select(SymbolMetadata).where(SymbolMetadata.symbol == symbol))
            
            if meta and meta.status == "INVALID":
                result["status"] = "INVALID_SYMBOL"
                return result
                
            earliest_date = s.scalar(select(func.min(MarketData.date)).where(MarketData.symbol == symbol))
            result["earliest_before"] = earliest_date
            
            if earliest_date and earliest_date <= target_start_date:
                result["status"] = "ALREADY_COVERED"
                result["earliest_after"] = earliest_date
                return result
                
            # If no data exists, fetch from target_start_date to today
            fetch_end_date = earliest_date if earliest_date else date.today()
            
            logger.info(f"[BACKFILL] {symbol}: Fetching backwards from {fetch_end_date} to {target_start_date}")
            
        try:
            df = yf.download(
                tickers=symbol,
                start=target_start_date.strftime("%Y-%m-%d"),
                end=fetch_end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False
            )
        except Exception as exc:
            logger.error(f"[ERROR] {symbol}: yfinance fetch failed - {exc}")
            result["status"] = "FETCH_FAILED"
            result["error"] = str(exc)
            return result

        if df.empty:
            logger.warning(f"[WARNING] {symbol}: yfinance returned no data for backfill period.")
            result["status"] = "NO_DATA_RETURNED"
            return result

        # Handle multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        col_map = {
            "Date": "date", "Datetime": "date",
            "Open": "open", "High": "high", "Low": "low", 
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
        }
        df = df.rename(columns=col_map)
        
        required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
        if not required.issubset(df.columns):
            logger.error(f"[ERROR] {symbol}: missing required columns from yfinance.")
            result["status"] = "MISSING_COLUMNS"
            return result

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.dropna(subset=["close"])
        if df.empty:
            result["status"] = "NO_VALID_DATA"
            return result

        # Merge new backfill chunk with existing DB chunk
        with self.db.session() as s:
            existing_records = s.execute(
                select(MarketData).where(MarketData.symbol == symbol).order_by(MarketData.date.asc())
            ).scalars().all()
            
            if existing_records:
                existing_df = pd.DataFrame([{
                    "date": r.date, "open": r.open, "high": r.high,
                    "low": r.low, "close": r.close, "adj_close": r.adj_close, "volume": r.volume
                } for r in existing_records])
                full_df = pd.concat([df, existing_df]).drop_duplicates(subset=["date"], keep="last")
            else:
                full_df = df
            
            full_df = full_df.sort_values("date").reset_index(drop=True)

        candles_for_indicators = full_df.copy()
        candles_for_indicators["timestamp"] = pd.to_datetime(candles_for_indicators["date"])
        candles_list = candles_for_indicators.to_dict(orient="records")

        # Compute Indicators
        try:
            enriched_list = self.indicator_engine.enrich(candles_list)
        except Exception as exc:
            logger.error(f"[ERROR] {symbol}: indicator calculation failed - {exc}")
            result["status"] = "INDICATOR_ERROR"
            result["error"] = str(exc)
            return result

        enriched_df = pd.DataFrame(enriched_list)
        
        # Only UPSERT the newly fetched rows to save DB write overhead
        # Wait, if we prepend data, the EMA/MACD of the EXISTING rows will CHANGE!
        # Because EMA is recursive and starts from the beginning of the series.
        # So we MUST UPSERT the ENTIRE full_df to correct the existing indicators!
        upsert_df = enriched_df

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
                "ema_200": float(row["ema_200"]) if pd.notnull(row["ema_200"]) else None,
            })

        with self.db.session() as s:
            md_stmt = insert(MarketData).values(md_records)
            md_update_dict = {c.name: c for c in md_stmt.excluded if c.name not in ["id", "symbol", "date", "created_at"]}
            md_stmt = md_stmt.on_conflict_do_update(
                constraint="uq_market_data_symbol_date",
                set_=md_update_dict
            )
            s.execute(md_stmt)

            ind_stmt = insert(MarketIndicators).values(ind_records)
            ind_update_dict = {c.name: c for c in ind_stmt.excluded if c.name not in ["id", "symbol", "date"]}
            ind_stmt = ind_stmt.on_conflict_do_update(
                constraint="uq_market_indicators_symbol_date",
                set_=ind_update_dict
            )
            s.execute(ind_stmt)

            meta = s.scalar(select(SymbolMetadata).where(SymbolMetadata.symbol == symbol))
            if not meta:
                meta = SymbolMetadata(symbol=symbol, status="ACTIVE")
                s.add(meta)
            
            meta.status = "ACTIVE"
            meta.first_date = full_df["date"].min()
            meta.last_date = full_df["date"].max()
            meta.total_candles = len(full_df)
            meta.last_synced = datetime.now()

            s.commit()
            
            result["earliest_after"] = meta.first_date
            result["rows_inserted"] = len(df)
            result["rows_updated"] = len(upsert_df) - len(df)
            result["status"] = "SUCCESS"
            
            logger.info(f"[BACKFILL SAVED] {symbol}: Inserted {result['rows_inserted']}, Updated {result['rows_updated']} candles.")

        return result

    def backfill_universe(self, symbols: list[str], years: int = 5, max_workers: int = 5):
        logger.info(f"Starting parallel BACKFILL for {len(symbols)} symbols over {years} years with {max_workers} workers.")
        
        target_start_date = date.today() - timedelta(days=int(years * 365.25))
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.backfill_symbol, sym, target_start_date): sym for sym in symbols}
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    logger.error(f"[ERROR] Parallel backfill failed for {futures[future]}: {exc}")
                    
        self._generate_report(results, years)
        return results
        
    def _generate_report(self, results: list[dict], years: int):
        report_path = self.out_dir / "Backfill_Report.md"
        
        successful = [r for r in results if r['status'] == 'SUCCESS']
        skipped = [r for r in results if r['status'] in ('ALREADY_COVERED', 'INVALID_SYMBOL')]
        failed = [r for r in results if r['status'] not in ('SUCCESS', 'ALREADY_COVERED', 'INVALID_SYMBOL')]
        
        total_inserted = sum(r['rows_inserted'] for r in results)
        total_updated = sum(r['rows_updated'] for r in results)
        
        with open(report_path, "w") as f:
            f.write("# Atlas Historical Backfill Report\n\n")
            f.write(f"**Date:** {date.today()}\n")
            f.write(f"**Target History:** {years} Years\n\n")
            
            f.write("## Summary\n")
            f.write(f"- **Total Symbols Processed:** {len(results)}\n")
            f.write(f"- **Symbols Backfilled:** {len(successful)}\n")
            f.write(f"- **Symbols Skipped (Already Covered):** {len(skipped)}\n")
            f.write(f"- **Symbols Failed:** {len(failed)}\n")
            f.write(f"- **Total Rows Inserted:** {total_inserted:,}\n")
            f.write(f"- **Total Rows Updated (Indicator Recomputation):** {total_updated:,}\n\n")
            
            f.write("## Detailed Breakdown\n\n")
            f.write("| Symbol | Status | Earliest Before | Earliest After | Inserted | Updated |\n")
            f.write("|--------|--------|-----------------|----------------|----------|---------|\n")
            
            for r in sorted(results, key=lambda x: x['symbol']):
                eb = r['earliest_before'] or "None"
                ea = r['earliest_after'] or "None"
                f.write(f"| {r['symbol']} | {r['status']} | {eb} | {ea} | {r['rows_inserted']} | {r['rows_updated']} |\n")
                
        logger.info(f"Backfill Report generated at: {report_path}")
