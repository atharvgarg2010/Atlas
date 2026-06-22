import pandas as pd
import numpy as np
import shap
import json
import pickle
from pathlib import Path
from datetime import date
from core.logging import get_logger

logger = get_logger(__name__)

class ModelExplainer:
    def __init__(self, registry_entry: dict, dataset_path: Path):
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.output_dir = Path(__file__).parent.parent.parent / "research" / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_id = registry_entry["model_id"]
        self.native_importance = registry_entry["native_feature_importance"]
        self.features = registry_entry["features"]
        
        model_path = self.models_dir / self.model_id
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        logger.info(f"Loading dataset for SHAP explainability: {dataset_path.name}")
        df = pd.read_parquet(dataset_path)
        df = df.dropna(subset=self.features).reset_index(drop=True)
        
        # Take a random subsample of 5000 rows to speed up SHAP (TreeExplainer is fast but good to cap)
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)
            
        self.X = df[self.features]
        self.explainer = shap.TreeExplainer(self.model)
        
    def generate_importance_report(self):
        logger.info("Calculating SHAP values...")
        shap_values = self.explainer.shap_values(self.X)
        
        # Mean absolute SHAP values per feature
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        shap_importance = dict(zip(self.features, shap_abs_mean))
        
        # Sort both lists
        sorted_native = sorted(self.native_importance.items(), key=lambda x: x[1], reverse=True)
        sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
        
        report_path = self.output_dir / "Feature_Importance_Report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Atlas Feature Importance Report\n\n")
            f.write(f"**Model ID:** `{self.model_id}`\n")
            f.write(f"**Report Date:** {date.today()}\n\n")
            
            f.write("## Top 10 Predictive Features (SHAP)\n")
            f.write("SHAP values indicate the average absolute impact each feature has on the model's prediction of future 30-day returns.\n\n")
            f.write("| Rank | Feature | Mean |SHAP| |\n")
            f.write("|------|---------|-------------|\n")
            for i, (feat, val) in enumerate(sorted_shap[:10]):
                f.write(f"| {i+1} | {feat} | {val:.6f} |\n")
                
            f.write("\n## Bottom 10 Predictive Features (Noise Candidates)\n")
            f.write("These features had the lowest SHAP impact and may be candidates for removal in future iterations.\n\n")
            f.write("| Rank | Feature | Mean |SHAP| |\n")
            f.write("|------|---------|-------------|\n")
            for i, (feat, val) in enumerate(reversed(sorted_shap[-10:])):
                f.write(f"| {len(self.features)-i} | {feat} | {val:.6f} |\n")
                
            f.write("\n---\n\n")
            f.write("## Native XGBoost Importances\n")
            f.write("Feature importance based on gain (contribution to the tree splits).\n\n")
            f.write("| Rank | Feature | Gain Importance |\n")
            f.write("|------|---------|-----------------|\n")
            for i, (feat, val) in enumerate(sorted_native):
                f.write(f"| {i+1} | {feat} | {val:.6f} |\n")
                
        logger.info(f"Feature Importance Report generated at: {report_path}")
        return report_path
        
    def explain_prediction(self, feature_series: pd.Series):
        """Explain a single prediction by extracting top contributing factors."""
        # Need a 2D array for SHAP
        X_single = pd.DataFrame([feature_series], columns=self.features)
        shap_vals = self.explainer.shap_values(X_single)[0]
        
        shap_dict = dict(zip(self.features, shap_vals))
        
        # Sort by absolute magnitude to find strongest drivers (positive or negative)
        top_drivers = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        reasons = []
        for feat, val in top_drivers:
            direction = "positive" if val > 0 else "negative"
            reasons.append(f"{feat} pushed the prediction in a {direction} direction.")
            
        return top_drivers, reasons
