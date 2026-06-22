from pathlib import Path
from datetime import date
import pandas as pd

def generate_optimization_report(ranks_df: pd.DataFrame, optimized_result: dict, weighting_scheme: str, ranking_date: date, output_dir: Path) -> Path:
    """
    Generate a Markdown report explaining the optimization allocations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "Portfolio_Optimization_Report.md"
    
    weights = optimized_result['weights']
    exp_ret = optimized_result['expected_return']
    exp_vol = optimized_result['expected_volatility']
    sharpe = optimized_result['sharpe_ratio']
    risk_contribs = optimized_result['risk_contributions']
    corr = optimized_result['correlation']
    
    universe_size = len(weights)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Atlas Portfolio Optimization Report\n\n")
        f.write(f"**Optimization Date:** {ranking_date}\n")
        f.write(f"**Strategy:** {weighting_scheme.upper()}\n")
        f.write(f"**Portfolio Size:** {universe_size} stocks\n\n")
        
        f.write("## Portfolio Summary\n")
        f.write(f"- **Expected Annual Return:** {exp_ret * 100:.2f}%\n")
        f.write(f"- **Expected Annual Volatility:** {exp_vol * 100:.2f}%\n")
        f.write(f"- **Sharpe Ratio (assumed RFR=0.05):** {sharpe:.2f}\n")
        f.write(f"- **Average Correlation:** {corr['average']:.4f}\n\n")
        
        f.write("### Correlation Extremes\n")
        if universe_size > 1:
            f.write(f"- **Most Correlated Pair:** {corr['most_correlated_pair'][0]} & {corr['most_correlated_pair'][1]} ({corr['most_correlated_value']:.4f})\n")
            f.write(f"- **Least Correlated Pair:** {corr['least_correlated_pair'][0]} & {corr['least_correlated_pair'][1]} ({corr['least_correlated_value']:.4f})\n\n")
        
        f.write("---\n\n")
        f.write("## Allocations\n\n")
        
        f.write("| Symbol | Factor Score | Target Weight | Risk Contribution |\n")
        f.write("|--------|--------------|---------------|-------------------|\n")
        
        # Sort weights descending
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        for sym, w in sorted_weights:
            # find composite score
            score_row = ranks_df[ranks_df['symbol'] == sym]
            score = score_row.iloc[0]['composite_score'] if not score_row.empty else 0.0
            rank = score_row.iloc[0]['rank'] if not score_row.empty else "N/A"
            rc = risk_contribs.get(sym, 0.0)
            
            f.write(f"| {sym} | {score:.2f} (Rank #{rank}) | {w * 100:.2f}% | {rc * 100:.2f}% |\n")
            
        f.write("\n---\n\n")
        
        f.write("## Allocation Reasoning\n\n")
        
        for sym, w in sorted_weights:
            score_row = ranks_df[ranks_df['symbol'] == sym]
            rank = score_row.iloc[0]['rank'] if not score_row.empty else "N/A"
            
            f.write(f"### {sym} (Weight = {w * 100:.2f}%)\n")
            
            reasons = []
            reasons.append(f"Rank #{rank} providing factor momentum.")
            
            if w >= 0.15:
                reasons.append("Assigned a highly concentrated weight due to exceptionally favorable risk-adjusted metrics or strong diversification benefits.")
            elif w <= 0.03:
                reasons.append("Constrained to a minimal weight, likely due to high volatility or high correlation with other heavily weighted assets.")
                
            rc = risk_contribs.get(sym, 0.0)
            avg_rc = 1.0 / universe_size
            if rc > avg_rc * 1.5:
                reasons.append("Note: Contributes disproportionately high risk to the overall portfolio.")
            elif rc < avg_rc * 0.5:
                reasons.append("Acts as a strong volatility dampener for the portfolio.")
                
            for r in reasons:
                f.write(f"- {r}\n")
            f.write("\n")
            
    return report_path
