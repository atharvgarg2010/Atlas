# Project Atlas — Architecture

> **Version:** 1.0 | **Status:** Sprint 1 — Core Infrastructure  
> **Stack:** Python 3.12 · PostgreSQL 15 · SQLAlchemy 2 · Streamlit · Docker

---

## System Overview

Project Atlas is a 24/7 **autonomous quantitative research platform** for Indian equity markets (NSE / NIFTY 50). It is a *decision-support system*, not a trading bot. No real-money execution in V1.

The platform continuously:
1. Ingests OHLCV market data from Yahoo Finance (yfinance)
2. Computes technical indicators (RSI, MACD, EMA, Bollinger Bands, ATR)
3. Scores every tracked stock on a 0–100 composite opportunity scale
4. Generates structured trade setups (entry / stop-loss / target / confidence)
5. Simulates paper trades to measure recommendation quality
6. Surfaces all insights through a Streamlit research dashboard

---

## Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                     │
│  dashboard/ — Streamlit (7 pages)                       │
│  Read-only. Queries repositories. Never calls services. │
└───────────────────────────┬─────────────────────────────┘
                            │ reads
┌───────────────────────────▼─────────────────────────────┐
│  RECOMMENDATION LAYER                                   │
│  services/signal_service.py                             │
│  services/paper_trading_service.py                      │
└───────────────────────────┬─────────────────────────────┘
                            │ reads / writes
┌───────────────────────────▼─────────────────────────────┐
│  SCORING LAYER                                          │
│  services/scoring_service.py                            │
│  analytics/portfolio/metrics.py                         │
└───────────────────────────┬─────────────────────────────┘
                            │ reads
┌───────────────────────────▼─────────────────────────────┐
│  ANALYSIS LAYER                                         │
│  services/indicator_service.py                          │
│  services/sentiment_service.py  (Phase 2+)              │
│  analytics/technical/indicators.py                      │
└───────────────────────────┬─────────────────────────────┘
                            │ reads / writes
┌───────────────────────────▼─────────────────────────────┐
│  FEATURE LAYER                                          │
│  data/processors/ohlcv_processor.py                     │
│  ml/ranking/feature_builder.py                          │
└───────────────────────────┬─────────────────────────────┘
                            │ reads / writes
┌───────────────────────────▼─────────────────────────────┐
│  DATA LAYER                                             │
│  database/connection.py  (engine + session)             │
│  repositories/*          (all DB queries)               │
│  database/models.py      (ORM models)                   │
│  PostgreSQL 15                                          │
└───────────────────────────┬─────────────────────────────┘
                            │ fetches
┌───────────────────────────▼─────────────────────────────┐
│  COLLECTION LAYER                                       │
│  data/collectors/yfinance_collector.py                  │
│  data/collectors/news_collector.py                      │
│  services/market_data_service.py                        │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│  EXTERNAL SOURCES                                       │
│  Yahoo Finance (yfinance)   NewsAPI / RSS Feeds         │
└─────────────────────────────────────────────────────────┘
```

**Rules:**
- Each layer may only call the layer directly below it
- No layer skips another layer
- Dashboard never calls services — it reads from repositories only
- No raw SQL from any layer above the repository

---

## Data Flow

```
[Yahoo Finance API]
      │  yfinance_collector.py (@retry)
      ▼
[market_data table]  ← MarketDataService (15-min scheduler job)
      │
      ▼
[indicators table]   ← IndicatorService (15-min scheduler job)
      │
      ▼
[news table]         ← NewsCollector (15-min scheduler job)
      │
      ▼
[stock_scores table] ← ScoringService (60-min scheduler job)
      │  40% technical + 30% momentum + 20% news + 10% volume
      ▼
[signals table]      ← SignalService (60-min scheduler job)
      │  entry / stop-loss / target / confidence / reasoning_json
      ▼
[paper_trades table] ← PaperTradingService (event-driven on signal)
      │
      ▼
[Streamlit Dashboard] ← repositories (read-only, @st.cache_data TTL=15min)
```

---

## Database Schema (Sprint 1 — 3 tables)

> Full 10-table schema will be created incrementally as each module is built.

### `stocks` — Master instrument registry

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL PK | |
| `symbol` | VARCHAR(20) UNIQUE | e.g. `RELIANCE.NS` |
| `name` | VARCHAR(100) | Full company name |
| `sector` | VARCHAR(50) | e.g. `Energy` |
| `market_cap` | VARCHAR(20) | `large` / `mid` / `small` |
| `exchange` | VARCHAR(10) | Default: `NSE` |
| `is_active` | BOOLEAN | Soft-disable without deleting |
| `created_at` | TIMESTAMPTZ | UTC |
| `updated_at` | TIMESTAMPTZ | UTC |

### `market_data` — OHLCV candles (core time-series)

| Column | Type | Notes |
|---|---|---|
| `id` | BIGSERIAL PK | |
| `symbol` | VARCHAR(20) FK→stocks | |
| `timeframe` | VARCHAR(10) | `1d` / `1h` / `15m` |
| `ts` | TIMESTAMPTZ | Candle open timestamp |
| `open` | NUMERIC(12,4) | |
| `high` | NUMERIC(12,4) | |
| `low` | NUMERIC(12,4) | |
| `close` | NUMERIC(12,4) | |
| `volume` | BIGINT | |
| UNIQUE | `(symbol, timeframe, ts)` | Prevents duplicate candles |

### `system_logs` — Operational health log

| Column | Type | Notes |
|---|---|---|
| `id` | BIGSERIAL PK | |
| `job_name` | VARCHAR(100) | Scheduler job name |
| `level` | VARCHAR(10) | `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `message` | TEXT | Human-readable message |
| `detail_json` | JSONB | Structured context (optional) |
| `logged_at` | TIMESTAMPTZ | UTC |

---

## Configuration System

Two-source design — secrets are never co-located with non-secret config:

```
.env                          config/settings.yaml
  DATABASE_URL                  market.watchlist (NIFTY 50)
  APP_ENV                       scheduler.intervals
  LOG_LEVEL                     paper_trading.initial_capital
        │                       scoring.weights
        └──────────┬────────────────────────┘
                   ▼
            config/settings.py
            get_settings() → Settings (frozen dataclass, cached)
```

- `get_settings()` is `lru_cache(maxsize=1)` — files read exactly once per process
- Settings are frozen dataclasses — immutable after load, no accidental mutation
- Scoring weights are validated to sum to 1.0 at startup

---

## Scheduler Design

`apscheduler` with `ThreadPoolExecutor`. Each job is independent:

| Job | Interval | Module |
|---|---|---|
| Market Data Sync | 15 min | `MarketDataService` |
| Indicator Update | 15 min | `IndicatorService` |
| News Sync | 15 min | `NewsCollector` |
| Sentiment Analysis | 60 min | `SentimentService` (Phase 2+) |
| Stock Scoring | 60 min | `ScoringService` |
| Signal Generation | 60 min | `SignalService` |

**Key rules:**
- All jobs are idempotent (safe to re-run)
- All jobs have an individual timeout
- All jobs use `@retry` on external API calls
- All jobs log start / end / duration via `@timed`

---

## Deployment

```
Ubuntu VPS
└── Docker
    ├── atlas_db   (postgres:15-alpine)  — persistent volume
    └── atlas_app  (python:3.12-slim)    — depends_on: db healthy
```

Secrets are injected via `.env` file on the VPS. Never baked into images.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Repository pattern | Isolates all DB queries; services are DB-agnostic |
| Frozen dataclasses for config | Immutable — no risk of mutation mid-run |
| `pool_pre_ping=True` | Silently handles stale connections on 24/7 VPS |
| FinBERT runs hourly not 15-min | 440 MB model + VPS RAM constraint |
| VectorBT for backtesting | Vectorised — orders of magnitude faster than event-driven |
| `@retry` on all collectors | yfinance has no SLA; must tolerate transient failures |
| NIFTY 50 only (V1) | Manageable data volume; representative of Indian large-cap market |
