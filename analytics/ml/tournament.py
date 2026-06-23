import pandas as pd
import numpy as np
from pathlib import Path
from core.logging import get_logger
from analytics.backtesting.walk_forward_backtest import WalkForwardBacktester
from analytics.ml.models.binary_classifier import BinaryClassifierModel
from analytics.ml.models.cross_sectional_ranker import CrossSectionalRankerModel
from analytics.ml.models.hybrid_factor_ranker import HybridFactorRankerModel
import xgboost as xgb
from scipy.stats import spearmanr
from analytics.portfolio.validation_charts import plot_calibration_curve, plot_rank_correlation, plot_cumulative_hit_rate

logger = get_logger(__name__)

class LegacyRegressorModel:
    def __init__(self, features):
        self.features = features
        self.target = 'target_return_30d'
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        
    def fit(self, df_train):
        df_train = df_train.dropna(subset=[self.target] + self.features)
        self.model.fit(df_train[self.features], df_train[self.target])
        
    def predict(self, df_test):
        preds = self.model.predict(df_test[self.features])
        res = df_test.copy()
        res['y_pred'] = preds
        res['predicted_rank'] = res.groupby('date')['y_pred'].rank(ascending=False, method='first')
        return res

class RandomBaselineModel:
    def fit(self, df_train): pass
    def predict(self, df_test):
        np.random.seed(len(df_test))
        res = df_test.copy()
        res['y_pred'] = np.random.uniform(-0.1, 0.1, len(res))
        res['predicted_rank'] = res.groupby('date')['y_pred'].rank(ascending=False, method='first')
        return res

class MeanBaselineModel:
    def __init__(self): self.mean_val = 0
    def fit(self, df_train):
        df_train = df_train.dropna(subset=['target_return_30d'])
        self.mean_val = df_train['target_return_30d'].mean()
    def predict(self, df_test):
        res = df_test.copy()
        res['y_pred'] = self.mean_val
        res['predicted_rank'] = res.groupby('date')['y_pred'].rank(ascending=False, method='first')
        return res

class ModelTournament:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.out_dir = Path(__file__).parent.parent.parent / "research" / "output"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir = self.out_dir / "validation"
        self.val_dir.mkdir(exist_ok=True)
        
        self.all_features = [
            'momentum_score', 'trend_score', 'rs_score', 'volatility_score', 'liquidity_score', 'composite_score',
            'rsi_14', 'macd', 'macd_signal', 'atr_14', 'ema20_dist', 'ema50_dist', 'ema200_dist',
            'daily_volatility', 'avg_volume_30', 'ret_1m', 'ret_3m', 'ret_6m'
        ]
        
    def run(self):
        logger.info("Initializing Phase 5.0 Model Tournament...")
        df = pd.read_parquet(self.dataset_path)
        df = df.dropna(subset=['target_return_30d', 'nifty_return_30d', 'target_outperform_30d'] + self.all_features).copy()
        
        models_to_test = {
            "Random Baseline": RandomBaselineModel(),
            "Mean Baseline": MeanBaselineModel(),
            "Momentum Baseline": CrossSectionalRankerModel(['momentum_score', 'ret_3m', 'ret_6m']),
            "Legacy Atlas (Regression)": LegacyRegressorModel(self.all_features),
            "New Ranker (All Features)": CrossSectionalRankerModel(self.all_features),
            "New Classifier (All Features)": BinaryClassifierModel(self.all_features),
            "Hybrid Factor Ranker": HybridFactorRankerModel()
        }
        
        results = []
        best_spearman = -999
        best_oof = None
        best_name = None
        all_deciles = {}
        
        for name, model in models_to_test.items():
            logger.info(f"Running True Walk-Forward Backtest for: {name}")
            backtester = WalkForwardBacktester(model=model, top_n=10)
            oof_df, metrics = backtester.run(df)
            
            if oof_df.empty:
                logger.warning(f"Model {name} returned empty OOF dataframe.")
                continue
                
            spearman = oof_df.groupby('date').apply(lambda x: spearmanr(x['predicted_rank'], x['target_return_30d'])[0]).mean()
            if pd.isna(spearman): spearman = 0
            
            metrics['spearman'] = spearman
            metrics['model'] = name
            results.append(metrics)
            all_deciles[name] = metrics.pop('deciles', {})
            
            if spearman > best_spearman and "Baseline" not in name:
                best_spearman = spearman
                best_oof = oof_df
                best_name = name
                
        # Generate Report
        res_df = pd.DataFrame(results).sort_values('spearman', ascending=False)
        
        with open(self.out_dir / "Model_Tournament_Report.md", "w") as f:
            f.write("# Phase 5.0 Model Tournament Report\n\n")
            f.write("## Consolidated Model Evaluation\n")
            f.write("| Model | Spearman Rank Corr | Decile Spread | CAGR | Sharpe | Max Drawdown | Monthly Turnover |\n")
            f.write("|-------|--------------------|---------------|------|--------|--------------|------------------|\n")
            for _, r in res_df.iterrows():
                f.write(f"| {r['model']} | {r['spearman']:.4f} | {r.get('decile_spread',0):.2f}% | {r['cagr']:.2f}% | {r['sharpe']:.2f} | {r['max_dd']:.2f}% | {r.get('turnover',0):.1f}% |\n")
                
            f.write("\n## Decile Monotonicity Test\n")
            f.write("Avg Realized Return per Prediction Decile (10 = Highest Conviction, 1 = Lowest Conviction)\n\n")
            for name, deciles in all_deciles.items():
                f.write(f"### {name}\n")
                f.write("| Decile | Avg Realized Return (%) |\n")
                f.write("|--------|-------------------------|\n")
                for d in range(10, 0, -1):
                    val = deciles.get(d, 0)
                    f.write(f"| {d} | {val:.2f}% |\n")
                f.write("\n")
                
            f.write("## Hard Promotion Check\n")
            best = res_df.iloc[0]
            if best['spearman'] > 0.05 and best['decile_spread'] > 0:
                f.write(f"**Verdict:** {best['model']} PASSED promotion criteria.\n")
                f.write("Conclusion: Atlas possesses a statistically valid, generalizable ML edge.\n")
            else:
                f.write(f"**Verdict:** ALL ML MODELS REJECTED.\n")
                f.write("Conclusion: None of the models achieved the >0.05 Spearman threshold with a positive decile spread. Momentum/Volatility heuristics outperformed complex ML mapping. Atlas should operate strictly as a Factor-Driven ranking engine.\n")
                
        # Generate Diagnostics for Best Model
        if best_oof is not None:
            self._generate_diagnostics(best_oof, best_name)
            
        logger.info(f"Tournament Complete. Results saved to {self.out_dir / 'Model_Tournament_Report.md'}")
        
    def _generate_diagnostics(self, oof_df, model_name):
        spearman_df = oof_df.groupby('date').apply(lambda x: spearmanr(x['predicted_rank'], x['target_return_30d'])[0]).reset_index(name='spearman')
        spearman_df['spearman'] = spearman_df['spearman'].rolling(3).mean()
        plot_rank_correlation(spearman_df.dropna(), self.val_dir / "rank_correlation.png")
        
        oof_df['correct_direction'] = (np.sign(oof_df['y_pred'] - 0.5) == np.sign(oof_df['target_outperform_30d'] - 0.5)).astype(float) if 'Classifier' in model_name else (np.sign(oof_df['y_pred']) == np.sign(oof_df['target_return_30d'])).astype(float)
        hit_df = oof_df.groupby('date')['correct_direction'].mean().reset_index()
        hit_df['cumulative_hit_rate'] = hit_df['correct_direction'].expanding().mean()
        plot_cumulative_hit_rate(hit_df, self.val_dir / "cumulative_hit_rate.png")
        
        if 'Classifier' in model_name:
            plot_calibration_curve(oof_df['target_outperform_30d'], oof_df['y_pred'], self.val_dir / "calibration_curve.png")
            
        with open(self.out_dir / "Prediction_Quality_Report.md", "w") as f:
            f.write(f"# Prediction Quality Diagnostics: {model_name}\n\n")
            f.write("![Rank Correlation](validation/rank_correlation.png)\n")
            f.write("![Cumulative Hit Rate](validation/cumulative_hit_rate.png)\n")
            if 'Classifier' in model_name:
                f.write("![Calibration Curve](validation/calibration_curve.png)\n")
