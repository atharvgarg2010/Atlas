import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.connection import init_db, get_db
from database.models.market_data import MarketData, MarketIndicators, SymbolMetadata
from analytics.technical.indicators import IndicatorEngine
from config.settings import get_settings
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
import pandas as pd

def main():
    settings = get_settings()
    init_db(settings.database_url, echo=False)

    db = get_db()
    engine = IndicatorEngine()

    with db.session() as s:
        symbols = s.scalars(select(SymbolMetadata.symbol).where(SymbolMetadata.status == 'ACTIVE')).all()
        
        for sym in symbols:
            print(f"Recalculating {sym}...")
            records = s.execute(
                select(MarketData).where(MarketData.symbol == sym).order_by(MarketData.date.asc())
            ).scalars().all()
            
            if not records:
                continue
                
            df = pd.DataFrame([{
                "date": r.date, "open": r.open, "high": r.high,
                "low": r.low, "close": r.close, "adj_close": r.adj_close, "volume": r.volume
            } for r in records])
            
            df["timestamp"] = pd.to_datetime(df["date"])
            candles_list = df.to_dict(orient="records")
            
            enriched_list = engine.enrich(candles_list)
            
            ind_records = []
            for row in enriched_list:
                ind_records.append({
                    "symbol": sym,
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
                
            # Batch insert
            for i in range(0, len(ind_records), 1000):
                batch = ind_records[i:i+1000]
                ind_stmt = insert(MarketIndicators).values(batch)
                ind_update_dict = {c.name: c for c in ind_stmt.excluded if c.name not in ["id", "symbol", "date"]}
                ind_stmt = ind_stmt.on_conflict_do_update(
                    constraint="uq_market_indicators_symbol_date",
                    set_=ind_update_dict
                )
                s.execute(ind_stmt)
            s.commit()
            print(f"Done {sym}.")

if __name__ == "__main__":
    main()
