import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
from core.logging import get_logger
from analytics.ml.models.cross_sectional_ranker import CrossSectionalRankerModel
from scipy.stats import spearmanr

logger = get_logger(__name__)

class FeatureSelectionEngine:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.out_dir = Path(__file__).parent.parent.parent / "research" / "output"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.families = {
            "Momentum": ['ret_1m', 'ret_3m', 'ret_6m', 'momentum_score'],
            "Volatility": ['atr_14', 'daily_volatility', 'volatility_score'],
            "Trend": ['ema20_dist', 'ema50_dist', 'ema200_dist', 'trend_score'],
            "Relative Strength": ['rs_score'],
            "Liquidity": ['avg_volume_30', 'liquidity_score']
        }
        
        self.combinations = {
            "Momentum Only": ["Momentum"],
            "Volatility Only": ["Volatility"],
            "Mom + Vol": ["Momentum", "Volatility"],
            "Mom + Vol + Liq": ["Momentum", "Volatility", "Liquidity"],
            "All Features": ["Momentum", "Volatility", "Trend", "Relative Strength", "Liquidity"]
        }
        
    def run(self):
        logger.info("Starting Feature Selection Engine...")
        df = pd.read_parquet(self.dataset_path)
        df = df.sort_values('date').dropna(subset=['target_return_30d'] + [f for sublist in self.families.values() for f in sublist])
        
        X = df[[f for sublist in self.families.values() for f in sublist]]
        y = df['target_return_30d']
        
        results = []
        
        for name, fam_list in self.combinations.items():
            logger.info(f"Testing combination: {name}")
            active_feats = []
            for fam in fam_list:
                active_feats.extend(self.families[fam])
                
            model = CrossSectionalRankerModel(features=active_feats)
            
            # Fast 5-fold walk-forward
            tscv = TimeSeriesSplit(n_splits=5)
            oof_dfs = []
            
            for train_idx, test_idx in tscv.split(df):
                df_train = df.iloc[train_idx].copy()
                df_test = df.iloc[test_idx].copy()
                
                model.fit(df_train)
                preds = model.predict(df_test)
                oof_dfs.append(preds)
                
            oof_df = pd.concat(oof_dfs)
            
            # Metrics
            spearman = oof_df.groupby('date').apply(lambda x: spearmanr(x['predicted_rank'], x['target_return_30d'])[0]).mean()
            if pd.isna(spearman): spearman = 0
            
            # Simple portfolio simulation: Top 10 equal weight
            cagr, sharpe, max_dd = self._simulate_top_10(oof_df)
            
            results.append({
                "Combination": name,
                "CAGR": cagr,
                "Sharpe": sharpe,
                "Max_DD": max_dd,
                "Spearman": spearman
            })
            
        res_df = pd.DataFrame(results)
        
        with open(self.out_dir / "Feature_Selection_Report.md", "w") as f:
            f.write("# Phase 5.0 Feature Selection Report\n\n")
            f.write("| Combination | CAGR | Sharpe | Max Drawdown | Spearman Rank Corr |\n")
            f.write("|-------------|------|--------|--------------|--------------------|\n")
            for r in results:
                f.write(f"| {r['Combination']} | {r['CAGR']:.2f}% | {r['Sharpe']:.2f} | {r['Max_DD']:.2f}% | {r['Spearman']:.4f} |\n")
                
        logger.info(f"Feature Selection Report generated: {self.out_dir / 'Feature_Selection_Report.md'}")
        return res_df

    def _simulate_top_10(self, oof_df):
        dates = sorted(oof_df['date'].unique())
        equity = 1.0
        equity_curve = []
        for d in dates:
            day_data = oof_df[oof_df['date'] == d]
            top10 = day_data[day_data['predicted_rank'] <= 10]
            if top10.empty: continue
            ret = top10['target_return_30d'].mean() - 0.002 # 0.20% slip/cost
            equity *= (1 + ret)
            equity_curve.append(equity)
            
        eq_df = pd.Series(equity_curve)
        if eq_df.empty: return 0, 0, 0
        
        years = len(dates) / 252.0
        cagr = (eq_df.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else 0
        
        daily_ret = eq_df.pct_change()
        vol = daily_ret.std() * np.sqrt(252)
        sharpe = (cagr / 100) / vol if vol > 0 else 0
        
        peak = eq_df.cummax()
        dd = (peak - eq_df) / peak
        max_dd = dd.max() * 100
        
        return cagr, sharpe, max_dd
