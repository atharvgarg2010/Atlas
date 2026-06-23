import pandas as pd
import numpy as np
from core.logging import get_logger

logger = get_logger(__name__)

class WalkForwardBacktester:
    def __init__(self, model, top_n=10, transaction_cost=0.001):
        self.model = model
        self.top_n = top_n
        self.transaction_cost = transaction_cost
        
    def run(self, df_features: pd.DataFrame, retrain_every='month'):
        df = df_features.sort_values('date').copy()
        rebalance_dates = df['date'].unique()
        warmup_cutoff = rebalance_dates[min(250, len(rebalance_dates) // 5)]
        test_dates = [d for d in rebalance_dates if d >= warmup_cutoff]
        
        # Enforce strict monthly rebalancing (approx 21 trading days)
        monthly_dates = test_dates[::21]
        
        oof_predictions = []
        
        for i, current_date in enumerate(monthly_dates):
            if i % 5 == 0: logger.info(f"Walk-forward month {i}/{len(monthly_dates)}: {current_date}")
            
            # STRICT LEAKAGE PREVENTION: Train only on data fully resolved
            # Find the actual index of current_date in the global rebalance_dates array
            global_idx = np.where(rebalance_dates == current_date)[0][0]
            cutoff_train = rebalance_dates[max(0, global_idx - 21)] if global_idx >= 21 else rebalance_dates[0]
            
            df_train = df[df['date'] <= cutoff_train].copy()
            df_test = df[df['date'] == current_date].copy()
            
            if len(df_train) < 1000 or df_test.empty: continue
                
            self.model.fit(df_train)
            df_pred = self.model.predict(df_test)
            oof_predictions.append(df_pred)
            
        if not oof_predictions:
            return pd.DataFrame(), {'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'turnover': 0, 'decile_spread': 0}
            
        full_oof = pd.concat(oof_predictions)
        port_metrics = self._simulate_portfolio(full_oof)
        decile_dict, decile_spread = self._calc_deciles(full_oof)
        port_metrics['decile_spread'] = decile_spread
        port_metrics['deciles'] = decile_dict
        
        return full_oof, port_metrics
        
    def _calc_deciles(self, oof_df):
        # Rank by y_pred cross-sectionally. Decile 10 = Highest y_pred, Decile 1 = Lowest y_pred
        def get_decile(g):
            try:
                return pd.qcut(g.rank(method='first'), 10, labels=False) + 1
            except:
                return pd.Series(5, index=g.index) # fallback if too small
                
        oof_df['decile'] = oof_df.groupby('date')['y_pred'].transform(get_decile)
        decile_ret = oof_df.groupby('decile')['target_return_30d'].mean() * 100
        
        if 10 in decile_ret.index and 1 in decile_ret.index:
            spread = decile_ret[10] - decile_ret[1]
        else:
            spread = 0.0
            
        return decile_ret.to_dict(), spread
        
    def _simulate_portfolio(self, oof_df):
        dates = sorted(oof_df['date'].unique())
        equity = 1.0
        equity_curve = []
        turnovers = []
        prev_portfolio = set()
        
        for d in dates:
            day_data = oof_df[oof_df['date'] == d]
            top_stocks = day_data[day_data['predicted_rank'] <= self.top_n]
            current_portfolio = set(top_stocks['symbol'])
            
            if prev_portfolio:
                # Turnover = % of capital changed
                turnover = len(current_portfolio - prev_portfolio) / max(1, len(prev_portfolio)) * 100
                turnovers.append(turnover)
            prev_portfolio = current_portfolio
            
            if top_stocks.empty:
                equity_curve.append({'date': d, 'portfolio_value': equity})
                continue
                
            basket_ret = top_stocks['target_return_30d'].mean()
            if pd.isna(basket_ret): basket_ret = 0.0
            
            net_ret = basket_ret - (self.transaction_cost * 2)
            equity *= (1 + net_ret)
            equity_curve.append({'date': d, 'portfolio_value': equity})
            
        eq_df = pd.DataFrame(equity_curve)
        if eq_df.empty: return {'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'turnover': 0}
        
        # We rebalance monthly, so we have 12 periods a year
        years = len(dates) / 12.0
        cagr = (eq_df['portfolio_value'].iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
        
        monthly_vol = eq_df['portfolio_value'].pct_change().std()
        annual_vol = monthly_vol * np.sqrt(12)
        sharpe = cagr / annual_vol if annual_vol > 0 else 0
        
        eq_df['peak'] = eq_df['portfolio_value'].cummax()
        eq_df['drawdown'] = (eq_df['peak'] - eq_df['portfolio_value']) / eq_df['peak']
        max_dd = eq_df['drawdown'].max() * 100
        
        avg_turnover = np.mean(turnovers) if turnovers else 0.0
        
        return {
            'cagr': cagr * 100,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'turnover': avg_turnover
        }
