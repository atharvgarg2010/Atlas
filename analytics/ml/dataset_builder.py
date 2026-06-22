import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

from database.connection import get_db
from sqlalchemy import select
from database.models.market_data import MarketData, MarketIndicators, SymbolMetadata
from core.logging import get_logger

logger = get_logger(__name__)

class DatasetBuilder:
    def __init__(self):
        self.db = get_db()
        self.output_dir = Path(__file__).parent.parent.parent / "research" / "datasets"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def build_features_df(self, end_date: date | None = None) -> pd.DataFrame:
        logger.info("Starting ML Dataset generation (feature engineering)...")
        
        with self.db.session() as s:
            active_symbols = s.scalars(select(SymbolMetadata.symbol).where(SymbolMetadata.status == "ACTIVE")).all()
            
            stmt = (
                select(MarketData, MarketIndicators)
                .join(MarketIndicators, (MarketData.symbol == MarketIndicators.symbol) & (MarketData.date == MarketIndicators.date))
                .where(MarketData.symbol.in_(active_symbols + ["^NSEI"]))
            )
            
            if end_date:
                stmt = stmt.where(MarketData.date <= end_date)
                
            stmt = stmt.order_by(MarketData.symbol, MarketData.date.asc())
            results = s.execute(stmt).all()
            
        data_by_symbol = {}
        for md, ind in results:
            if md.symbol not in data_by_symbol:
                data_by_symbol[md.symbol] = []
            data_by_symbol[md.symbol].append({
                "date": md.date,
                "close": md.close,
                "volume": md.volume,
                "ema_20": ind.ema_20,
                "ema_50": ind.ema_50,
                "ema_200": ind.ema_200,
                "sma_20": ind.sma_20,
                "rsi_14": ind.rsi_14,
                "macd": ind.macd,
                "macd_signal": ind.macd_signal,
                "atr_14": ind.atr_14
            })
            
        dfs = {sym: pd.DataFrame(records).set_index("date") for sym, records in data_by_symbol.items()}
        
        benchmark_df = dfs.get("^NSEI")
        if benchmark_df is None:
            raise Exception("Benchmark ^NSEI is missing.")
            
        processed_dfs = []
        
        for sym in active_symbols:
            if sym not in dfs:
                continue
                
            df = dfs[sym].copy()
            df['symbol'] = sym
            
            df['ret_1m'] = df['close'].pct_change(21)
            df['ret_3m'] = df['close'].pct_change(63)
            df['ret_6m'] = df['close'].pct_change(126)
            
            df['momentum_raw'] = 0.4 * df['ret_3m'] + 0.6 * df['ret_6m']
            
            trend_score = (
                25.0 * (df['close'] > df['ema_20']).astype(float) +
                25.0 * (df['ema_20'] > df['ema_50']).astype(float) +
                25.0 * (df['ema_50'] > df['ema_200']).astype(float) +
                25.0 * (df['close'] > df['ema_200']).astype(float)
            )
            df['trend_raw'] = trend_score
            
            bench_aligned = benchmark_df.reindex(df.index)
            bench_ret_6m = bench_aligned['close'].pct_change(126)
            df['rs_raw'] = df['ret_6m'] - bench_ret_6m
            
            pct_atr = df['atr_14'] / df['close']
            std_dev = df['close'].pct_change(1).rolling(20).std()
            df['daily_volatility'] = std_dev
            df['volatility_raw'] = -1.0 * (pct_atr + std_dev)
            
            df['avg_volume_30'] = df['volume'].rolling(30).mean()
            df['liquidity_raw'] = df['avg_volume_30'] * df['close'].rolling(30).mean()
            
            df['ema20_dist'] = (df['close'] - df['ema_20']) / df['ema_20']
            df['ema50_dist'] = (df['close'] - df['ema_50']) / df['ema_50']
            df['ema200_dist'] = (df['close'] - df['ema_200']) / df['ema_200']
            
            df['target_return_30d'] = (df['close'].shift(-21) - df['close']) / df['close']
            
            processed_dfs.append(df.reset_index())
            
        full_df = pd.concat(processed_dfs, ignore_index=True)
        
        def winsorize_group(g):
            q01 = g.quantile(0.01)
            q99 = g.quantile(0.99)
            return g.clip(lower=q01, upper=q99)
            
        for factor in ['momentum', 'trend', 'rs', 'volatility', 'liquidity']:
            raw_col = f"{factor}_raw"
            full_df[raw_col] = full_df.groupby('date')[raw_col].transform(winsorize_group)
            full_df[f"{factor}_score"] = full_df.groupby('date')[raw_col].transform(lambda x: x.rank(pct=True) * 100.0)
            
        full_df['composite_score'] = (
            full_df['momentum_score'] * 0.30 +
            full_df['trend_score'] * 0.25 +
            full_df['rs_score'] * 0.20 +
            full_df['volatility_score'] * 0.15 +
            full_df['liquidity_score'] * 0.10
        )
        
        full_df = full_df.dropna(subset=['composite_score', 'rsi_14', 'ema200_dist', 'macd']).copy()
        
        columns_to_keep = [
            'date', 'symbol', 'close', 'target_return_30d',
            'momentum_score', 'trend_score', 'rs_score', 'volatility_score', 'liquidity_score', 'composite_score',
            'rsi_14', 'macd', 'macd_signal', 'atr_14', 'ema20_dist', 'ema50_dist', 'ema200_dist',
            'daily_volatility', 'avg_volume_30', 'ret_1m', 'ret_3m', 'ret_6m'
        ]
        
        return full_df[columns_to_keep].sort_values(['date', 'symbol'])

    def build_and_save(self) -> Path:
        import json
        from datetime import datetime
        
        final_df = self.build_features_df(end_date=None)
        
        version_date = date.today().strftime("%Y%m%d")
        dataset_version = f"atlas_dataset_v{version_date}"
        parquet_path = self.output_dir / f"{dataset_version}.parquet"
        
        final_df.to_parquet(parquet_path, index=False)
        
        # ── Dataset Snapshot Metadata ──
        valid_rows = int(final_df['target_return_30d'].notna().sum())
        metadata = {
            "dataset_version": dataset_version,
            "created_at": datetime.now().isoformat(),
            "symbols": len(final_df['symbol'].unique()),
            "date_start": final_df['date'].min().isoformat() if not final_df.empty else None,
            "date_end": final_df['date'].max().isoformat() if not final_df.empty else None,
            "rows": len(final_df),
            "valid_training_rows": valid_rows,
            "feature_count": len([c for c in final_df.columns if c not in ['date', 'symbol', 'target_return_30d']]),
            "features": [c for c in final_df.columns if c not in ['date', 'symbol', 'target_return_30d']]
        }
        
        metadata_path = self.output_dir / f"{dataset_version}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        # ── Global Dataset Registry ──
        registry_path = self.output_dir / "dataset_registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)
        else:
            registry = []
            
        registry.insert(0, metadata)
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=4)
        
        logger.info(f"Dataset Built: {len(final_df)} samples, {len(final_df['symbol'].unique())} symbols.")
        logger.info(f"Valid Training Rows: {valid_rows}")
        logger.info(f"Saved dataset and metadata to {self.output_dir}")
        return parquet_path
