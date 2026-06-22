import uuid
import yaml
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np

from sqlalchemy import select, delete
from database.connection import get_db
from database.models.factors import FactorRanking
from database.models.market_data import MarketData, MarketIndicators, SymbolMetadata

from analytics.factors.momentum_factor import MomentumFactor
from analytics.factors.trend_factor import TrendFactor
from analytics.factors.relative_strength_factor import RSFactor
from analytics.factors.volatility_factor import VolatilityFactor
from analytics.factors.liquidity_factor import LiquidityFactor

from core.logging import get_logger
logger = get_logger(__name__)

class FactorEngine:
    def __init__(self):
        self.db = get_db()
        self.weights = self._load_weights()
        self.factors = {
            "momentum": MomentumFactor(),
            "trend": TrendFactor(),
            "relative_strength": RSFactor(),
            "volatility": VolatilityFactor(),
            "liquidity": LiquidityFactor()
        }

    def _load_weights(self) -> dict:
        config_path = Path(__file__).parent.parent.parent / "config" / "factor_weights.yaml"
        if not config_path.exists():
            logger.warning("factor_weights.yaml not found, using default equal weights.")
            return {"momentum": 0.2, "trend": 0.2, "relative_strength": 0.2, "volatility": 0.2, "liquidity": 0.2}
        
        with open(config_path, "r") as f:
            weights = yaml.safe_load(f)
            
        # Normalize weights to sum to 1.0 just in case
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _fetch_data_batch(self, ranking_date: date) -> dict[str, pd.DataFrame]:
        """Fetch all active symbol data up to ranking_date."""
        # Using the same mapping approach as DataManager
        with self.db.session() as s:
            active_symbols = s.scalars(select(SymbolMetadata.symbol).where(SymbolMetadata.status == "ACTIVE")).all()
            
            stmt = (
                select(MarketData, MarketIndicators)
                .join(MarketIndicators, (MarketData.symbol == MarketIndicators.symbol) & (MarketData.date == MarketIndicators.date))
                .where(MarketData.date <= ranking_date)
                .order_by(MarketData.symbol, MarketData.date.asc())
            )
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
                "atr_14": ind.atr_14
            })
            
        return {sym: pd.DataFrame(records) for sym, records in data_by_symbol.items()}

    def _winsorize_series(self, series: pd.Series, limits=(0.01, 0.09)) -> pd.Series:
        """Winsorize extreme outliers at 1st and 99th percentiles."""
        lower = series.quantile(limits[0])
        upper = series.quantile(1.0 - limits[1])  # actually limits[1] is upper tail probability if specified as such, but standard is upper bound
        # Let's fix to strict 1% and 99%
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        return series.clip(lower=lower, upper=upper)

    def _percentile_rank(self, series: pd.Series) -> pd.Series:
        """Rank to 0-100 scale."""
        return series.rank(pct=True) * 100

    def rank_universe(self, ranking_date: date, persist: bool = True) -> pd.DataFrame:
        logger.info(f"Starting factor ranking for date: {ranking_date}")
        
        all_data = self._fetch_data_batch(ranking_date)
        
        benchmark_symbol = "^NSEI"
        if benchmark_symbol not in all_data:
            raise Exception(f"Benchmark {benchmark_symbol} missing from cache up to {ranking_date}. Please sync it first.")
            
        benchmark_df = all_data[benchmark_symbol]
        
        # Data Quality Layer
        valid_symbols = []
        for sym, df in all_data.items():
            if sym == benchmark_symbol:
                continue
                
            if len(df) < 200:
                logger.warning(f"Data Quality: {sym} has < 200 candles ({len(df)}). Excluded.")
                continue
                
            if pd.isna(df['ema_200'].iloc[-1]):
                logger.warning(f"Data Quality: {sym} has missing EMA_200. Excluded.")
                continue
                
            valid_symbols.append(sym)
            
        logger.info(f"Universe size after Data Quality layer: {len(valid_symbols)}")
        if not valid_symbols:
            logger.warning("No valid symbols remaining to rank.")
            return pd.DataFrame()
            
        raw_scores = []
        
        # Compute raw scores
        for sym in valid_symbols:
            sym_df = all_data[sym]
            row = {"symbol": sym}
            valid = True
            
            for factor_name, factor_obj in self.factors.items():
                raw_val = factor_obj.compute(sym_df, benchmark_df)
                if raw_val is None:
                    logger.warning(f"Factor compute returned None for {sym} ({factor_name}). Excluded.")
                    valid = False
                    break
                row[f"{factor_name}_raw"] = raw_val
                
            if valid:
                raw_scores.append(row)
                
        if not raw_scores:
            logger.warning("No valid symbols remaining after factor computation.")
            return pd.DataFrame()
            
        df_scores = pd.DataFrame(raw_scores).set_index("symbol")
        
        # Normalize
        for factor_name in self.factors.keys():
            raw_col = f"{factor_name}_raw"
            score_col = f"{factor_name}_score"
            
            # Winsorize
            df_scores[raw_col] = self._winsorize_series(df_scores[raw_col])
            
            # Percentile rank to 0-100
            df_scores[score_col] = self._percentile_rank(df_scores[raw_col])
            
        # Composite score
        df_scores['composite_score'] = 0.0
        for factor_name, weight in self.weights.items():
            df_scores['composite_score'] += df_scores[f"{factor_name}_score"] * weight
            
        # Sort and rank
        df_scores = df_scores.sort_values(by='composite_score', ascending=False)
        df_scores['rank'] = range(1, len(df_scores) + 1)
        
        df_scores = df_scores.reset_index()
        
        if persist:
            # Persist to DB
            run_id = str(uuid.uuid4())
            records = []
            for _, row in df_scores.iterrows():
                record = FactorRanking(
                    run_id=run_id,
                    ranking_date=ranking_date,
                    symbol=row['symbol'],
                    universe_size=len(df_scores),
                    top_n=None,  # This might be populated later by portfolio engine if needed
                    momentum_raw=row.get('momentum_raw'),
                    trend_raw=row.get('trend_raw'),
                    rs_raw=row.get('relative_strength_raw'),
                    volatility_raw=row.get('volatility_raw'),
                    liquidity_raw=row.get('liquidity_raw'),
                    momentum_score=row.get('momentum_score'),
                    trend_score=row.get('trend_score'),
                    rs_score=row.get('relative_strength_score'),
                    volatility_score=row.get('volatility_score'),
                    liquidity_score=row.get('liquidity_score'),
                    composite_score=row['composite_score'],
                    rank=row['rank']
                )
                records.append(record)
                
            with self.db.session() as s:
                # Remove previous runs for the same date to prevent UniqueConstraint errors
                s.execute(delete(FactorRanking).where(FactorRanking.ranking_date == ranking_date))
                s.add_all(records)
                
            logger.info(f"Ranking complete. Run ID: {run_id}. Top symbol: {df_scores.iloc[0]['symbol']}")
        else:
            logger.info(f"Ranking complete for {ranking_date} (persist=False). Top symbol: {df_scores.iloc[0]['symbol']}")
            
        return df_scores
