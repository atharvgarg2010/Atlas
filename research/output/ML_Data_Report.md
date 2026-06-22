# Atlas ML Dataset Validation Report

**Dataset:** `atlas_dataset_v20260622.parquet`
**Validation Date:** 2026-06-22

## Overview
- **Total Samples:** 100,446
- **Unique Symbols:** 97
- **Date Range:** 2022-04-08 to 2026-06-22

## Target Distribution (`target_return_30d`)
- **Count:** 98,505 (98.1% valid)
- **Mean:** 1.64%
- **Std Dev:** 8.68%
- **Min:** -61.18%
- **Max:** 81.44%

## Missing Values Analysis
| Feature | Missing Count | Missing % |
|---------|---------------|-----------|
| target_return_30d | 1,941 | 1.93% |

*(Note: Missing targets at the end of the dataset are expected due to the 30-day look-ahead window.)*

## Feature Correlation with Target (Spearman)
Highlights linear and monotonic relationships between features and future 30-day returns.

| Feature | Correlation |
|---------|-------------|
| daily_volatility | 0.0515 |
| ret_3m | 0.0097 |
| ret_1m | 0.0082 |
| ema50_dist | 0.0066 |
| ema200_dist | 0.0063 |
| rsi_14 | 0.0054 |
| ema20_dist | 0.0038 |
| avg_volume_30 | 0.0023 |
| ret_6m | 0.0000 |
| rs_score | -0.0005 |
| momentum_score | -0.0049 |
| macd | -0.0117 |
| macd_signal | -0.0150 |
| trend_score | -0.0178 |
| composite_score | -0.0228 |
| atr_14 | -0.0287 |
| liquidity_score | -0.0321 |
| volatility_score | -0.0442 |
