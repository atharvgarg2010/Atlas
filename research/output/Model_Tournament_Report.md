# Phase 5.0 Model Tournament Report

## Consolidated Model Evaluation
| Model | Spearman Rank Corr | Decile Spread | CAGR | Sharpe | Max Drawdown | Monthly Turnover |
|-------|--------------------|---------------|------|--------|--------------|------------------|
| Mean Baseline | 0.0250 | 0.11% | 20.85% | 1.11 | 25.10% | 15.8% |
| Momentum Baseline | 0.0213 | -0.13% | 21.82% | 1.06 | 26.68% | 74.6% |
| Random Baseline | -0.0079 | -0.38% | 16.81% | 0.99 | 15.89% | 56.8% |
| New Classifier (All Features) | -0.0273 | 0.92% | 32.53% | 1.66 | 13.84% | 70.9% |
| Legacy Atlas (Regression) | -0.0347 | 1.34% | 38.44% | 1.47 | 20.39% | 61.6% |
| Hybrid Factor Ranker | -0.0449 | 1.53% | 25.20% | 1.23 | 14.56% | 64.9% |
| New Ranker (All Features) | -0.0598 | 1.40% | 24.05% | 1.12 | 19.63% | 73.6% |

## Decile Monotonicity Test
Avg Realized Return per Prediction Decile (10 = Highest Conviction, 1 = Lowest Conviction)

### Random Baseline
| Decile | Avg Realized Return (%) |
|--------|-------------------------|
| 10 | 1.34% |
| 9 | 1.53% |
| 8 | 2.41% |
| 7 | 1.53% |
| 6 | 2.15% |
| 5 | 1.01% |
| 4 | 1.12% |
| 3 | 1.00% |
| 2 | 1.57% |
| 1 | 1.72% |

### Mean Baseline
| Decile | Avg Realized Return (%) |
|--------|-------------------------|
| 10 | 1.76% |
| 9 | 0.75% |
| 8 | 1.88% |
| 7 | 1.42% |
| 6 | 1.65% |
| 5 | 1.34% |
| 4 | 1.37% |
| 3 | 2.42% |
| 2 | 1.17% |
| 1 | 1.65% |

### Momentum Baseline
| Decile | Avg Realized Return (%) |
|--------|-------------------------|
| 10 | 1.75% |
| 9 | 1.64% |
| 8 | 1.04% |
| 7 | 1.33% |
| 6 | 1.01% |
| 5 | 1.74% |
| 4 | 2.19% |
| 3 | 1.13% |
| 2 | 1.42% |
| 1 | 1.89% |

### Legacy Atlas (Regression)
| Decile | Avg Realized Return (%) |
|--------|-------------------------|
| 10 | 2.98% |
| 9 | 1.79% |
| 8 | 2.06% |
| 7 | 2.22% |
| 6 | 0.48% |
| 5 | 0.80% |
| 4 | 1.24% |
| 3 | 1.08% |
| 2 | 0.89% |
| 1 | 1.64% |

### New Ranker (All Features)
| Decile | Avg Realized Return (%) |
|--------|-------------------------|
| 10 | 1.92% |
| 9 | 2.20% |
| 8 | 2.04% |
| 7 | 1.81% |
| 6 | 2.85% |
| 5 | 1.43% |
| 4 | 0.98% |
| 3 | 1.08% |
| 2 | 0.57% |
| 1 | 0.53% |

### New Classifier (All Features)
| Decile | Avg Realized Return (%) |
|--------|-------------------------|
| 10 | 2.47% |
| 9 | 1.86% |
| 8 | 0.95% |
| 7 | 1.84% |
| 6 | 1.67% |
| 5 | 1.34% |
| 4 | 0.94% |
| 3 | 1.29% |
| 2 | 1.30% |
| 1 | 1.55% |

### Hybrid Factor Ranker
| Decile | Avg Realized Return (%) |
|--------|-------------------------|
| 10 | 2.02% |
| 9 | 1.65% |
| 8 | 2.36% |
| 7 | 2.06% |
| 6 | 2.30% |
| 5 | 1.20% |
| 4 | 1.12% |
| 3 | 0.96% |
| 2 | 1.22% |
| 1 | 0.49% |

## Hard Promotion Check
**Verdict:** ALL ML MODELS REJECTED.
Conclusion: None of the models achieved the >0.05 Spearman threshold with a positive decile spread. Momentum/Volatility heuristics outperformed complex ML mapping. Atlas should operate strictly as a Factor-Driven ranking engine.
