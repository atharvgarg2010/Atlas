# Indicator Engine

## Overview
The Indicator Engine is responsible for transforming raw historical and live market data (OHLCV) into actionable technical analysis features. These features (indicators) will subsequently feed into the Ranking Engine and AI Layer for trade signal generation.

## Responsibilities
1. **Fetch Raw Data**: Retrieve clean OHLCV data from the `market_data` table via the `MarketDataRepository`.
2. **Compute Indicators**: Apply standardized mathematical formulas to calculate indicators (e.g., RSI, EMA, MACD, Bollinger Bands, ATR).
3. **Persist Results**: Store the computed indicators into an `indicators` table via the `IndicatorRepository`.
4. **Idempotent Updates**: Ensure that re-running the engine on the same dataset updates existing records gracefully or skips them without causing duplication or errors.

## Supported Indicators (Sprint 3 Scope)
- **RSI (Relative Strength Index)**: 14-period standard
- **EMA (Exponential Moving Average)**: 9, 21, and 50 periods
- **MACD (Moving Average Convergence Divergence)**: 12, 26, 9 periods
- **ATR (Average True Range)**: 14-period standard for volatility measurement
- **Bollinger Bands**: 20-period, 2 standard deviations

## Architecture
- **Service Layer**: `IndicatorService` coordinates fetching data, running calculations, and delegating persistence.
- **Repository Layer**: `IndicatorRepository` handles database abstractions (SQLAlchemy) for the `indicators` table.
- **Calculation Library**: Uses `pandas` and `ta` (or `pandas-ta`) for vectorized, highly performant indicator computation across large DataFrames.

## Database Schema (`indicators`)
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | Integer | Primary Key | Auto-increment identifier |
| symbol | String(20) | Foreign Key (`stocks.symbol`) | Ticker symbol |
| timeframe | String(10) | Not Null | Data resolution (e.g., '1d', '15m') |
| ts | DateTime | Not Null | Timestamp of the indicator values |
| rsi_14 | Numeric(12,4) | Nullable | 14-period RSI |
| ema_9 | Numeric(12,4) | Nullable | 9-period EMA |
| ema_21 | Numeric(12,4) | Nullable | 21-period EMA |
| ema_50 | Numeric(12,4) | Nullable | 50-period EMA |
| macd | Numeric(12,4) | Nullable | MACD line |
| macd_signal | Numeric(12,4) | Nullable | MACD signal line |
| macd_hist | Numeric(12,4) | Nullable | MACD histogram |
| atr_14 | Numeric(12,4) | Nullable | 14-period Average True Range |
| bb_upper | Numeric(12,4) | Nullable | Bollinger Bands Upper |
| bb_middle | Numeric(12,4) | Nullable | Bollinger Bands Middle |
| bb_lower | Numeric(12,4) | Nullable | Bollinger Bands Lower |
| created_at | DateTime | Default `now()` | Record creation timestamp |
| updated_at | DateTime | Default `now()` | Record update timestamp |

**Unique Constraint**: `UNIQUE(symbol, timeframe, ts)` to support `ON CONFLICT DO UPDATE/NOTHING` behaviors.

## Implementation Steps
1. **Migration**: Create Alembic migration for the `indicators` table.
2. **Models**: Define the `Indicator` SQLAlchemy ORM model.
3. **Repository**: Implement `IndicatorRepository` for bulk upserts.
4. **Service**: Implement `IndicatorService` containing the pandas/ta logic.
5. **Testing**: Write unit tests comparing output against known baselines, and integration tests for the repository.
