# Atlas Feature Importance Report

**Model ID:** `alpha_model_20260623_121449.pkl`
**Report Date:** 2026-06-23

## Top 10 Predictive Features (SHAP)
SHAP values indicate the average absolute impact each feature has on the model's prediction of future 30-day returns.

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | atr_14 | 0.006639 |
| 2 | ema200_dist | 0.005299 |
| 3 | avg_volume_30 | 0.005242 |
| 4 | ret_6m | 0.004016 |
| 5 | trend_score | 0.002803 |
| 6 | ret_3m | 0.002792 |
| 7 | daily_volatility | 0.002728 |
| 8 | rs_score | 0.002681 |
| 9 | composite_score | 0.002261 |
| 10 | liquidity_score | 0.002003 |

## Bottom 10 Predictive Features (Noise Candidates)
These features had the lowest SHAP impact and may be candidates for removal in future iterations.

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 18 | rsi_14 | 0.000477 |
| 17 | ema50_dist | 0.000563 |
| 16 | ret_1m | 0.000573 |
| 15 | ema20_dist | 0.000658 |
| 14 | macd | 0.000726 |
| 13 | volatility_score | 0.000944 |
| 12 | momentum_score | 0.001223 |
| 11 | macd_signal | 0.001466 |
| 10 | liquidity_score | 0.002003 |
| 9 | composite_score | 0.002261 |

---

## Native XGBoost Importances
Feature importance based on gain (contribution to the tree splits).

| Rank | Feature | Gain Importance |
|------|---------|-----------------|
| 1 | ema200_dist | 0.087517 |
| 2 | ret_3m | 0.077289 |
| 3 | ret_6m | 0.076785 |
| 4 | atr_14 | 0.066370 |
| 5 | avg_volume_30 | 0.061092 |
| 6 | rs_score | 0.059758 |
| 7 | daily_volatility | 0.059426 |
| 8 | liquidity_score | 0.057120 |
| 9 | momentum_score | 0.052287 |
| 10 | trend_score | 0.051863 |
| 11 | composite_score | 0.050374 |
| 12 | macd_signal | 0.049505 |
| 13 | ret_1m | 0.049214 |
| 14 | volatility_score | 0.047432 |
| 15 | macd | 0.046379 |
| 16 | ema50_dist | 0.041042 |
| 17 | ema20_dist | 0.034383 |
| 18 | rsi_14 | 0.032165 |
