import json
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)

class CoverageReporter:
    def __init__(self):
        self.datasets_dir = Path(__file__).parent.parent.parent / "research" / "datasets"
        self.output_dir = Path(__file__).parent.parent.parent / "research" / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self):
        parquet_files = sorted(list(self.datasets_dir.glob("*.parquet")))
        if not parquet_files:
            logger.error("No parquet dataset found to generate coverage report.")
            return
            
        latest_dataset = parquet_files[-1]
        logger.info(f"Generating Dataset Coverage Report for {latest_dataset.name}")
        
        df = pd.read_parquet(latest_dataset)
        
        # Calculate stats
        total_rows = len(df)
        valid_df = df.dropna(subset=['target_return_30d'])
        valid_targets = len(valid_df)
        
        symbols = df['symbol'].unique()
        num_symbols = len(symbols)
        dates = df['date'].unique()
        date_start = dates.min()
        date_end = dates.max()
        num_trading_days = len(dates)
        
        pos_ret_pct = (valid_df['target_return_30d'] > 0).mean() * 100
        neg_ret_pct = (valid_df['target_return_30d'] < 0).mean() * 100
        avg_ret = valid_df['target_return_30d'].mean() * 100
        
        # Folds
        n_splits = 5
        test_size = valid_targets // (n_splits + 1)
        
        # Missing values (just general missing, primarily targets)
        missing_count = total_rows - valid_targets
        
        # Coverage by symbol
        symbol_counts = df['symbol'].value_counts()
        
        report_path = self.output_dir / "Dataset_Coverage_Report.md"
        
        with open(report_path, "w") as f:
            f.write("# Atlas Dataset Coverage Report\n\n")
            f.write(f"**Generated:** {date.today()}\n")
            f.write(f"**Dataset Version:** {latest_dataset.name}\n\n")
            
            f.write("## Overview\n")
            f.write(f"- **Date Range:** {date_start} to {date_end}\n")
            f.write(f"- **Number of Symbols:** {num_symbols}\n")
            f.write(f"- **Number of Trading Days:** {num_trading_days}\n")
            f.write(f"- **Total Rows:** {total_rows:,}\n")
            f.write(f"- **Valid Target Rows (Training/Validation Samples):** {valid_targets:,}\n")
            f.write(f"- **Missing Values (Current Window / Invalid):** {missing_count:,}\n\n")
            
            f.write("## Target Distribution (`target_return_30d`)\n")
            f.write(f"- **Positive Returns:** {pos_ret_pct:.2f}%\n")
            f.write(f"- **Negative Returns:** {neg_ret_pct:.2f}%\n")
            f.write(f"- **Average Future Return:** {avg_ret:.2f}%\n\n")
            
            f.write("## Walk-Forward Validation Setup\n")
            f.write(f"- **Cross-Validation Scheme:** TimeSeriesSplit (n_splits={n_splits})\n")
            f.write(f"- **Samples per Fold (Validation size):** ~{test_size:,}\n\n")
            
            f.write("## Coverage by Symbol\n")
            f.write("| Symbol | Total Rows | Approx. Years |\n")
            f.write("|--------|------------|---------------|\n")
            for sym, count in symbol_counts.items():
                years = count / 252.0
                f.write(f"| {sym} | {count:,} | {years:.1f}y |\n")
                
        logger.info(f"Dataset Coverage Report generated at: {report_path}")
