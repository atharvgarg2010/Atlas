import xgboost as xgb
import pandas as pd
import numpy as np

class CrossSectionalRankerModel:
    def __init__(self, features):
        self.features = features
        self.target = 'target_return_30d'
        self.model = xgb.XGBRanker(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='rank:pairwise',
            random_state=42
        )
        
    def fit(self, df_train):
        df_train = df_train.dropna(subset=[self.target] + self.features).copy()
        
        # XGBRanker requires queries to be grouped and sorted by the group ID
        df_train = df_train.sort_values(by=['date', self.target], ascending=[True, False])
        
        # Calculate group sizes
        groups = df_train.groupby('date').size().values
        
        X = df_train[self.features]
        # For rank:pairwise, the target can be continuous returns or discrete ranks.
        # It attempts to maximize the ranking of instances with higher target values.
        y = df_train[self.target]
        
        self.model.fit(X, y, group=groups)
        
    def predict(self, df_test):
        # The raw output is a ranking score (higher is better)
        X = df_test[self.features]
        scores = self.model.predict(X)
        
        res = df_test.copy()
        res['y_pred'] = scores
        res['predicted_rank'] = res.groupby('date')['y_pred'].rank(ascending=False, method='first')
        return res
