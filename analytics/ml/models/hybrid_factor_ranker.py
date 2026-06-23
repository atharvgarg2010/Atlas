import xgboost as xgb
import pandas as pd
import numpy as np

class HybridFactorRankerModel:
    def __init__(self):
        # Hardcoded to ONLY the 4 core families
        self.features = ['momentum_score', 'volatility_score', 'rs_score', 'liquidity_score']
        self.target = 'target_return_30d'
        
        # Very shallow trees (max_depth=2) to prevent overfitting and force linear-like weight combinations
        self.model = xgb.XGBRanker(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            objective='rank:pairwise',
            random_state=42
        )
        
    def fit(self, df_train):
        df_train = df_train.dropna(subset=[self.target] + self.features).copy()
        df_train = df_train.sort_values(by=['date', self.target], ascending=[True, False])
        
        groups = df_train.groupby('date').size().values
        
        X = df_train[self.features]
        y = df_train[self.target]
        
        self.model.fit(X, y, group=groups)
        
    def predict(self, df_test):
        X = df_test[self.features]
        scores = self.model.predict(X)
        
        res = df_test.copy()
        res['y_pred'] = scores
        res['predicted_rank'] = res.groupby('date')['y_pred'].rank(ascending=False, method='first')
        return res
