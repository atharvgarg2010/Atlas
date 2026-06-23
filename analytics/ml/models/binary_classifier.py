import xgboost as xgb
import pandas as pd
import numpy as np

class BinaryClassifierModel:
    def __init__(self, features):
        self.features = features
        self.target = 'target_outperform_30d'
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
        )
        
    def fit(self, df_train):
        # Drop NaNs specific to this target
        df_train = df_train.dropna(subset=[self.target] + self.features)
        X = df_train[self.features]
        y = df_train[self.target]
        self.model.fit(X, y)
        
    def predict(self, df_test):
        # Returns probability of outperforming (Class 1)
        X = df_test[self.features]
        probs = self.model.predict_proba(X)[:, 1]
        
        res = df_test.copy()
        res['y_pred'] = probs
        # We rank by probability of outperformance (highest probability -> rank 1)
        res['predicted_rank'] = res.groupby('date')['y_pred'].rank(ascending=False, method='first')
        return res
