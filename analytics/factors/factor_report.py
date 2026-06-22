from pathlib import Path
from datetime import date
import pandas as pd

def generate_reasoning_report(ranks_df: pd.DataFrame, weights: dict[str, float], ranking_date: date, top_n: int, output_dir: Path) -> Path:
    """
    Generate a Markdown report explaining the reasoning behind the scoring for the Top N stocks.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "Factor_Reasoning_Report.md"
    
    universe_size = len(ranks_df)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Atlas Factor Reasoning Report\n\n")
        f.write(f"**Ranking Date:** {ranking_date}\n")
        f.write(f"**Universe Size:** {universe_size}\n")
        f.write(f"**Top N Evaluated:** {top_n}\n\n")
        
        f.write("## Factor Methodology\n")
        f.write("Atlas uses a cross-sectional ranking methodology. Each factor computes a raw value for each stock, which is then winsorized (to remove extreme outliers) and converted into a percentile rank (0-100). The final composite score is the weighted sum of these percentiles.\n\n")
        f.write("### Factor Weights\n")
        for factor, weight in weights.items():
            f.write(f"- **{factor.capitalize()}:** {weight * 100:.1f}%\n")
        f.write("\n---\n\n")
        
        f.write("## Top N Stock Breakdowns\n\n")
        
        top_stocks = ranks_df.head(top_n)
        
        for _, row in top_stocks.iterrows():
            symbol = row['symbol']
            rank = row['rank']
            composite_score = row['composite_score']
            
            f.write(f"### {rank}. {symbol} (Score: {composite_score:.2f})\n\n")
            
            # Show raw factor values and normalized scores
            f.write("#### Factor Breakdown\n")
            f.write("| Factor | Raw Value | Normalized Score (0-100) | Weighted Contribution |\n")
            f.write("|--------|-----------|--------------------------|-----------------------|\n")
            
            contributions = []
            strongest_factor = None
            strongest_score = -1
            weakest_factor = None
            weakest_score = 101
            
            for factor, weight in weights.items():
                raw_col = f"{factor}_raw"
                if raw_col not in row and factor == 'relative_strength':
                    raw_col = "relative_strength_raw" if "relative_strength_raw" in row else "rs_raw"
                score_col = f"{factor}_score"
                if score_col not in row and factor == 'relative_strength':
                    score_col = "relative_strength_score" if "relative_strength_score" in row else "rs_score"
                
                raw_val = row.get(raw_col, 0.0)
                score_val = row.get(score_col, 0.0)
                contribution = score_val * weight
                contributions.append(f"({score_val:.2f} * {weight:.2f})")
                
                f.write(f"| {factor.capitalize()} | {raw_val:,.4f} | {score_val:.2f} | {contribution:.2f} |\n")
                
                if score_val > strongest_score:
                    strongest_score = score_val
                    strongest_factor = factor
                if score_val < weakest_score:
                    weakest_score = score_val
                    weakest_factor = factor

            f.write("\n#### Weighted Calculation\n")
            f.write(f"`Composite Score = " + " + ".join(contributions) + f" = {composite_score:.2f}`\n\n")
            
            # Plain English Verdict
            f.write("#### Atlas Verdict\n")
            f.write(f"Atlas ranked **{symbol}** at position **#{rank}** out of {universe_size} stocks. ")
            top_percentile = max(0.1, 100.0 - strongest_score)
            f.write(f"This high placement is primarily driven by its **{strongest_factor}** (scoring in the top {top_percentile:.1f}% of the universe). ")
            if weakest_score < 50:
                f.write(f"However, its relative weakness lies in **{weakest_factor}**, where it scored {weakest_score:.1f}. ")
            f.write(f"Overall, the combination of these factors yields a strong composite score of {composite_score:.2f}, indicating robust multi-factor alignment for the current ranking date.\n\n")
            f.write("---\n\n")
            
    return report_path
