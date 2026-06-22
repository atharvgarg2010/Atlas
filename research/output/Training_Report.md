# Atlas Machine Learning Training Report

**Generated:** 2026-06-22T19:27:01.372417

## Execution Summary
- **Dataset Version:** atlas_dataset_v20260622.parquet
- **Model Version:** alpha_model_20260622_192701.pkl
- **Sample Count (Valid):** 98505
- **Feature Count:** 18

## Walk-Forward Validation Metrics
| Fold | MAE | RMSE | R2 | Directional Accuracy |
|------|-----|------|----|----------------------|
| 1 | 0.0697 | 0.0987 | -0.2527 | 54.28% |
| 2 | 0.0790 | 0.1100 | -0.4601 | 47.85% |
| 3 | 0.0713 | 0.0921 | -0.2457 | 48.21% |
| 4 | 0.0594 | 0.0798 | -0.0802 | 54.94% |
| 5 | 0.0638 | 0.0851 | -0.1031 | 49.29% |

**Averages:**
- **MAE:** 0.0686
- **RMSE:** 0.0932
- **R2:** -0.2283
- **Directional Accuracy:** 50.91%
