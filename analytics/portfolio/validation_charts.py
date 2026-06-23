import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def setup_style():
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#121212", 
        "figure.facecolor": "#121212", 
        "text.color": "white", 
        "axes.labelcolor": "white", 
        "xtick.color": "white", 
        "ytick.color": "white", 
        "grid.color": "#2c2c2c"
    })

def plot_equity_curve(ml_equity: pd.Series, factor_equity: pd.Series, benchmark_equity: pd.Series, out_path: Path):
    setup_style()
    plt.figure(figsize=(12, 6))
    
    # Calculate log cumulative returns
    ml_log = np.log1p(ml_equity.pct_change().fillna(0)).cumsum()
    factor_log = np.log1p(factor_equity.pct_change().fillna(0)).cumsum() if factor_equity is not None else None
    bench_log = np.log1p(benchmark_equity.pct_change().fillna(0)).cumsum()
    
    plt.plot(ml_log.index, ml_log, label='Atlas ML', color='#00ffcc', linewidth=2)
    if factor_log is not None:
        plt.plot(factor_log.index, factor_log, label='Heuristic Factor', color='#ff9900', linewidth=2)
    plt.plot(bench_log.index, bench_log, label='Benchmark (^NSEI)', color='#888888', linestyle='--', linewidth=1.5)
    
    plt.title('Log Cumulative Returns (Out-of-Fold / Backtest)')
    plt.ylabel('Log Return')
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_monthly_heatmap(equity_curve: pd.Series, out_path: Path):
    setup_style()
    monthly = equity_curve.resample('ME').last().pct_change()
    
    df = pd.DataFrame({'return': monthly})
    df['Year'] = df.index.year
    df['Month'] = df.index.month_name().str[:3]
    
    pivot = df.pivot(index='Year', columns='Month', values='return')
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns])
    
    plt.figure(figsize=(12, max(4, len(pivot) * 0.8)))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Return %'}, linewidths=.5)
    plt.title('Monthly Returns Heatmap (%)')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_rolling_sharpe(equity_curve: pd.Series, window: int = 126, out_path: Path = None):
    setup_style()
    daily_ret = equity_curve.pct_change().dropna()
    rolling_ret = daily_ret.rolling(window).mean() * 252
    rolling_vol = daily_ret.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_ret - 0.05) / rolling_vol
    
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_sharpe.index, rolling_sharpe, color='#00aaff', linewidth=2)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f'Rolling Sharpe Ratio ({window} Days)')
    plt.ylabel('Sharpe Ratio')
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_drawdown_curve(ml_equity: pd.Series, bench_equity: pd.Series, out_path: Path):
    setup_style()
    
    def calc_dd(equity):
        roll_max = equity.cummax()
        return (equity / roll_max - 1.0) * 100

    ml_dd = calc_dd(ml_equity)
    bench_dd = calc_dd(bench_equity)
    
    plt.figure(figsize=(12, 6))
    plt.fill_between(ml_dd.index, ml_dd, 0, color='#ff3333', alpha=0.5, label='Atlas Drawdown')
    plt.plot(bench_dd.index, bench_dd, color='#888888', linestyle='--', label='Benchmark Drawdown')
    plt.title('Drawdown Analysis')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_scatter(y_true, y_pred, out_path: Path):
    setup_style()
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_pred, y=y_true, alpha=0.5, color='#00ffcc')
    
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
    
    plt.title('Predicted vs Actual Returns (Out-of-Fold)')
    plt.xlabel('Predicted Return')
    plt.ylabel('Actual Future Return')
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_decile_returns(decile_df: pd.DataFrame, out_path: Path):
    setup_style()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Decile', y='Actual_Return', data=decile_df, palette='coolwarm')
    plt.axhline(0, color='white', linestyle='-', linewidth=0.8)
    plt.title('Average Actual Return by Prediction Decile')
    plt.xlabel('Prediction Decile (10 = Highest Predicted)')
    plt.ylabel('Average Actual Return (%)')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_buckets(buckets_df: pd.DataFrame, out_path: Path):
    setup_style()
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Bucket', y='Actual_Return', data=buckets_df, palette='viridis')
    plt.axhline(0, color='white', linestyle='-', linewidth=0.8)
    plt.title('Actual Returns by Prediction Confidence')
    plt.xlabel('Confidence Bucket')
    plt.ylabel('Average Actual Return (%)')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_stability(stability_df: pd.DataFrame, out_path: Path):
    setup_style()
    plt.figure(figsize=(12, 8))
    sns.heatmap(stability_df, cmap='YlGnBu_r', annot=True, fmt=".1f", cbar_kws={'label': 'Rank (1=Best)'})
    plt.title('Feature Rank Stability Across Folds')
    plt.ylabel('Feature')
    plt.xlabel('Fold & Method')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_monte_carlo_distribution(mc_cagrs: list, baseline_cagr: float, out_path: Path):
    setup_style()
    plt.figure(figsize=(10, 6))
    sns.histplot(mc_cagrs, bins=50, kde=True, color='#00ffcc')
    plt.axvline(baseline_cagr, color='white', linestyle='--', linewidth=2, label=f'Baseline CAGR ({baseline_cagr:.2f}%)')
    plt.axvline(np.mean(mc_cagrs), color='#ff9900', linestyle='-', linewidth=2, label=f'Mean MC CAGR ({np.mean(mc_cagrs):.2f}%)')
    plt.axvline(0, color='red', linestyle='-', linewidth=1)
    
    plt.title('Monte Carlo Robustness Test - CAGR Distribution')
    plt.xlabel('Simulated Annualized Return (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_ablation(ablation_df: pd.DataFrame, out_path: Path):
    setup_style()
    plt.figure(figsize=(10, 6))
    
    # Sort so worst impact is at top
    ablation_df = ablation_df.sort_values('CAGR_Change_Pct', ascending=True)
    
    sns.barplot(x='CAGR_Change_Pct', y='Removed_Family', data=ablation_df, palette='Reds_r')
    plt.axvline(0, color='white', linestyle='-', linewidth=1)
    plt.title('Feature Ablation Study - Performance Impact')
    plt.xlabel('Change in CAGR vs Baseline (%)')
    plt.ylabel('Removed Factor Family')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration_curve(y_true, y_prob, out_path: Path):
    setup_style()
    plt.figure(figsize=(8, 8))
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', color='#00ffcc', label='Classifier')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Hit Rate)')
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_rank_correlation(df_spearman: pd.DataFrame, out_path: Path):
    setup_style()
    plt.figure(figsize=(12, 6))
    plt.plot(df_spearman['date'], df_spearman['spearman'], color='#ff9900', linewidth=2)
    plt.axhline(0, color='white', linestyle='--', linewidth=1)
    plt.title('Rolling Spearman Rank Correlation')
    plt.xlabel('Date')
    plt.ylabel('Spearman Correlation')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cumulative_hit_rate(hit_rates: pd.DataFrame, out_path: Path):
    setup_style()
    plt.figure(figsize=(12, 6))
    plt.plot(hit_rates['date'], hit_rates['cumulative_hit_rate'] * 100, color='#00aaff', linewidth=2)
    plt.axhline(50, color='white', linestyle='--', linewidth=1)
    plt.title('Cumulative Hit Rate (Directional Accuracy)')
    plt.xlabel('Date')
    plt.ylabel('Hit Rate (%)')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
