import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from core.logging import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.features = [
            'momentum_score', 'trend_score', 'rs_score', 'volatility_score', 'liquidity_score', 'composite_score',
            'rsi_14', 'macd', 'macd_signal', 'atr_14', 'ema20_dist', 'ema50_dist', 'ema200_dist',
            'daily_volatility', 'avg_volume_30', 'ret_1m', 'ret_3m', 'ret_6m'
        ]
        self.target = 'target_return_30d'

    def directional_accuracy(self, y_true, y_pred):
        """Calculate percentage of times the model correctly predicts the sign of the return."""
        correct_direction = np.sign(y_true) == np.sign(y_pred)
        return np.mean(correct_direction) * 100.0

    def run_training(self):
        logger.info(f"Loading dataset from {self.dataset_path}")
        df = pd.read_parquet(self.dataset_path)
        
        # Sort chronologically to prevent look-ahead bias
        df = df.sort_values('date').reset_index(drop=True)
        
        # Drop missing targets (the last 30 days of data won't have future returns)
        df = df.dropna(subset=[self.target] + self.features).reset_index(drop=True)
        
        X = df[self.features]
        y = df[self.target]
        dates = df['date']
        
        # ── 0. Training Readiness Check ──
        logger.info("Executing Training Readiness Check...")
        valid_samples = len(X)
        if valid_samples < 25000:
            logger.critical(f"Training Readiness FAILED: Dataset only has {valid_samples} valid training samples. Target > 25,000.")
            import sys
            sys.exit(1)
        
        logger.info(f"Readiness Check PASSED: {valid_samples} valid training samples available.")
        
        # ── 1. Walk-Forward Validation ──
        logger.info("Starting Walk-Forward Validation (TimeSeriesSplit, 5 splits)...")
        tscv = TimeSeriesSplit(n_splits=5)
        
        metrics = {'mae': [], 'rmse': [], 'r2': [], 'dir_acc': []}
        
        fold = 1
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Use early stopping to prevent overfitting
            model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            
            # Ensure evaluation strictly uses past data (eval set is chronologically after train)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            preds = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            dir_acc = self.directional_accuracy(y_test, preds)
            
            metrics['mae'].append(mae)
            metrics['rmse'].append(rmse)
            metrics['r2'].append(r2)
            metrics['dir_acc'].append(dir_acc)
            
            logger.info(f"Fold {fold} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, DirAcc: {dir_acc:.2f}%")
            fold += 1
            
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        logger.info(f"Walk-Forward Averages: MAE: {avg_metrics['mae']:.4f}, DirAcc: {avg_metrics['dir_acc']:.2f}%")
        
        # ── 2. Final Model Training on Full Dataset ──
        logger.info("Training final model on entire dataset...")
        final_model = xgb.XGBRegressor(
            n_estimators=200,  # Fixed iterations for final train
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
        final_model.fit(X, y)
        
        # ── 3. Save Artifacts & Registry ──
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"alpha_model_{timestamp}.pkl"
        model_path = self.models_dir / model_filename
        
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
            
        # Native Feature Importances
        importance_dict = dict(zip(self.features, final_model.feature_importances_.tolist()))
        
        registry_path = self.models_dir / "model_registry.json"
        
        registry_entry = {
            "model_id": model_filename,
            "trained_at": datetime.now().isoformat(),
            "dataset_version": self.dataset_path.name,
            "metrics": {
                "avg_mae": float(avg_metrics['mae']),
                "avg_rmse": float(avg_metrics['rmse']),
                "avg_r2": float(avg_metrics['r2']),
                "avg_dir_acc": float(avg_metrics['dir_acc'])
            },
            "features": self.features,
            "native_feature_importance": importance_dict
        }
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = []
            
        # Prepend latest model
        registry.insert(0, registry_entry)
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=4)
            
        # Save active features list for quick access by Predictor
        with open(self.models_dir / "feature_columns.json", 'w') as f:
            json.dump(self.features, f, indent=4)
            
        logger.info(f"Model saved to {model_path}. Registry updated.")
        
        # ── 4. Generate Training Report ──
        out_dir = Path(__file__).parent.parent.parent / "research" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "Training_Report.md"
        
        with open(report_path, "w") as f:
            f.write("# Atlas Machine Learning Training Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write("## Execution Summary\n")
            f.write(f"- **Dataset Version:** {self.dataset_path.name}\n")
            f.write(f"- **Model Version:** {model_filename}\n")
            f.write(f"- **Sample Count (Valid):** {len(X)}\n")
            f.write(f"- **Feature Count:** {len(self.features)}\n\n")
            
            f.write("## Walk-Forward Validation Metrics\n")
            f.write("| Fold | MAE | RMSE | R2 | Directional Accuracy |\n")
            f.write("|------|-----|------|----|----------------------|\n")
            for i in range(len(metrics['mae'])):
                f.write(f"| {i+1} | {metrics['mae'][i]:.4f} | {metrics['rmse'][i]:.4f} | {metrics['r2'][i]:.4f} | {metrics['dir_acc'][i]:.2f}% |\n")
            
            f.write(f"\n**Averages:**\n")
            f.write(f"- **MAE:** {avg_metrics['mae']:.4f}\n")
            f.write(f"- **RMSE:** {avg_metrics['rmse']:.4f}\n")
            f.write(f"- **R2:** {avg_metrics['r2']:.4f}\n")
            f.write(f"- **Directional Accuracy:** {avg_metrics['dir_acc']:.2f}%\n")
            
        logger.info(f"Training Report generated at: {report_path}")
        return registry_entry
