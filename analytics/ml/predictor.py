import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from core.logging import get_logger

logger = get_logger(__name__)

class AlphaPredictor:
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.model = None
        self.features = []
        self.model_id = None
        
        self.load_model()
        
    def load_model(self):
        registry_path = self.models_dir / "model_registry.json"
        if not registry_path.exists():
            raise FileNotFoundError("model_registry.json not found. Please train a model first.")
            
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            
        if not registry:
            raise ValueError("model_registry.json is empty.")
            
        latest_entry = registry[0]
        self.model_id = latest_entry["model_id"]
        self.features = latest_entry["features"]
        
        model_path = self.models_dir / self.model_id
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {self.model_id} not found.")
            
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        logger.info(f"Loaded AlphaPredictor model: {self.model_id}")
        
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects a DataFrame containing at least the required features.
        Returns the original DataFrame with 'predicted_return' and 'predicted_rank' columns.
        """
        missing_feats = [f for f in self.features if f not in features_df.columns]
        if missing_feats:
            raise ValueError(f"Missing required features for prediction: {missing_feats}")
            
        X = features_df[self.features]
        
        # Predict
        preds = self.model.predict(X)
        
        result_df = features_df.copy()
        result_df['predicted_return'] = preds
        
        # Rank the predictions cross-sectionally (1 = best, N = worst)
        # Note: higher return means better rank, so we sort descending or rank ascending with method
        result_df['predicted_rank'] = result_df['predicted_return'].rank(ascending=False, method='min')
        
        return result_df
