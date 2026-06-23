import pandas as pd
import numpy as np
from pathlib import Path
from analytics.ml.reality_check import ValidationEngine
from sklearn.metrics import confusion_matrix

def main():
    datasets_dir = Path(__file__).parent / "research" / "datasets"
    parquet_files = sorted(list(datasets_dir.glob("*.parquet")))
    engine = ValidationEngine(parquet_files[-1])
    
    df_valid = engine.df.dropna(subset=[engine.target] + engine.features).copy()
    X = df_valid[engine.features]
    y = df_valid[engine.target]
    
    # Rerun walk-forward to get exactly 100% identical OOF DF
    print("Generating OOF predictions via Walk-Forward Validation (will take ~15 secs)...")
    t2_res, t11_res, oof_df, models = engine._test2_11_walkforward(X, y, df_valid)
    
    df = oof_df.copy()
    df['Decile'] = pd.qcut(df['y_pred'], 10, labels=False, duplicates='drop') + 1
    
    print("\n========================\nDECILE TABLE\n========================")
    print("Decile | Mean Prediction | Mean Actual Return | Samples")
    print("-" * 65)
    for d in range(1, 11):
        dec_data = df[df['Decile'] == d]
        print(f"{d:6d} | {dec_data['y_pred'].mean()*100:14.2f}% | {dec_data['y_true'].mean()*100:17.2f}% | {len(dec_data):7d}")
        
    print("\n========================\nTOP 20 HIGHEST PREDICTIONS\n========================")
    top20 = df.sort_values('y_pred', ascending=False).head(20).copy()
    top20['y_pred'] = (top20['y_pred'] * 100).apply(lambda x: f"{x:.2f}%")
    top20['y_true'] = (top20['y_true'] * 100).apply(lambda x: f"{x:.2f}%")
    print(top20[['date', 'symbol', 'y_pred', 'y_true']].to_string(index=False))
    
    print("\n========================\nBOTTOM 20 LOWEST PREDICTIONS\n========================")
    bottom20 = df.sort_values('y_pred', ascending=True).head(20).copy()
    bottom20['y_pred'] = (bottom20['y_pred'] * 100).apply(lambda x: f"{x:.2f}%")
    bottom20['y_true'] = (bottom20['y_true'] * 100).apply(lambda x: f"{x:.2f}%")
    print(bottom20[['date', 'symbol', 'y_pred', 'y_true']].to_string(index=False))
    
    print("\n========================\nCONFUSION MATRIX\n========================")
    y_true_bin = (df['y_true'] > 0).astype(int)
    y_pred_bin = (df['y_pred'] > 0).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    print(f"Total Predictions Evaluated: {len(df)}")
    print("True \\ Pred | Predicted Negative | Predicted Positive")
    print("-" * 60)
    print(f"Actual Neg   | {cm[0][0]:18d} | {cm[0][1]:18d}")
    print(f"Actual Pos   | {cm[1][0]:18d} | {cm[1][1]:18d}")

if __name__ == "__main__":
    main()
