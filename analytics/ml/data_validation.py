import pandas as pd
from pathlib import Path
from core.logging import get_logger
from datetime import date

logger = get_logger(__name__)

class DataValidator:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.output_dir = Path(__file__).parent.parent.parent / "research" / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_validation(self):
        logger.info(f"Running data validation on {self.dataset_path}...")
        df = pd.read_parquet(self.dataset_path)
        
        # ── 1. Basic Stats ──
        total_samples = len(df)
        total_symbols = df['symbol'].nunique()
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # ── 2. Missing Values ──
        missing = df.isnull().sum()
        missing_pct = (missing / total_samples) * 100
        
        # ── 3. Target Distribution ──
        target_col = 'target_return_30d'
        target_valid = df[target_col].dropna()
        target_stats = target_valid.describe()
        
        # ── 4. Correlation with Target ──
        features = [
            'momentum_score', 'trend_score', 'rs_score', 'volatility_score', 'liquidity_score', 'composite_score',
            'rsi_14', 'macd', 'macd_signal', 'atr_14', 'ema20_dist', 'ema50_dist', 'ema200_dist',
            'daily_volatility', 'avg_volume_30', 'ret_1m', 'ret_3m', 'ret_6m'
        ]
        
        # Use Spearman correlation for robustness against non-linear/monotonic relationships
        corr_matrix = df[features + [target_col]].corr(method='spearman')
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        
        # ── 5. Generate Report ──
        report_path = self.output_dir / "ML_Data_Report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Atlas ML Dataset Validation Report\n\n")
            f.write(f"**Dataset:** `{self.dataset_path.name}`\n")
            f.write(f"**Validation Date:** {date.today()}\n\n")
            
            f.write("## Overview\n")
            f.write(f"- **Total Samples:** {total_samples:,}\n")
            f.write(f"- **Unique Symbols:** {total_symbols}\n")
            f.write(f"- **Date Range:** {min_date} to {max_date}\n\n")
            
            f.write("## Target Distribution (`target_return_30d`)\n")
            f.write(f"- **Count:** {int(target_stats['count']):,} ({(target_stats['count']/total_samples)*100:.1f}% valid)\n")
            f.write(f"- **Mean:** {target_stats['mean']*100:.2f}%\n")
            f.write(f"- **Std Dev:** {target_stats['std']*100:.2f}%\n")
            f.write(f"- **Min:** {target_stats['min']*100:.2f}%\n")
            f.write(f"- **Max:** {target_stats['max']*100:.2f}%\n\n")
            
            f.write("## Missing Values Analysis\n")
            f.write("| Feature | Missing Count | Missing % |\n")
            f.write("|---------|---------------|-----------|\n")
            for col in missing.index:
                if missing[col] > 0:
                    f.write(f"| {col} | {missing[col]:,} | {missing_pct[col]:.2f}% |\n")
            f.write("\n*(Note: Missing targets at the end of the dataset are expected due to the 30-day look-ahead window.)*\n\n")
            
            f.write("## Feature Correlation with Target (Spearman)\n")
            f.write("Highlights linear and monotonic relationships between features and future 30-day returns.\n\n")
            f.write("| Feature | Correlation |\n")
            f.write("|---------|-------------|\n")
            for feat, val in target_corr.items():
                f.write(f"| {feat} | {val:.4f} |\n")
                
        logger.info(f"Validation Report generated at: {report_path}")
        return report_path
