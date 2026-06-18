# Project Atlas

**Autonomous AI Trading Research Platform**

> Sprint 1 — Core Infrastructure

---

## What Is This?

Project Atlas is a 24/7 autonomous quantitative research platform for Indian equity markets (NIFTY 50). It is a **decision-support system**, not a trading bot. No real-money execution in V1.

The platform analyzes market data, technical indicators, and financial news to identify high-probability trading opportunities and generate explainable recommendations.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12 |
| Database | PostgreSQL 15 (Docker) |
| ORM | SQLAlchemy 2.0 + Alembic |
| Data | yfinance, pandas, numpy, ta |
| ML | scikit-learn, XGBoost, VectorBT |
| Dashboard | Streamlit + Plotly |
| Deployment | Docker + Ubuntu VPS |

---

## Project Structure

```
project-atlas/
├── config/             # YAML settings + logging config
├── core/               # Shared utilities (logging, exceptions, decorators)
├── database/           # SQLAlchemy connection + ORM base
├── repositories/       # All database queries (data access layer)
├── services/           # Business logic (one service per domain)
├── data/
│   ├── collectors/     # External API adapters (yfinance, news)
│   └── processors/     # Data cleaning and validation
├── ml/                 # Model wrappers (FinBERT, XGBoost)
├── analytics/          # Technical indicators, portfolio metrics
├── scheduler/          # APScheduler jobs
├── dashboard/          # Streamlit pages and components
├── tests/              # Unit + integration tests
├── deployment/         # Dockerfile, docker-compose, scripts
├── docs/               # Architecture documentation
├── research/           # Notebooks and experiments
└── logs/               # Runtime logs (git-ignored)
```

---

## Setup (Sprint 1)

### Prerequisites

- Python 3.12+
- Docker Desktop (running)
- Git

### 1. Clone and enter the project

```bash
cd project-atlas
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — the defaults work with Docker Compose as-is:

```
DATABASE_URL=postgresql://atlas:atlas_secret@localhost:5432/atlas
APP_ENV=development
LOG_LEVEL=INFO
```

### 4. Start the database

```bash
docker-compose -f deployment/docker-compose.yml up -d db
```

Wait ~10 seconds for PostgreSQL to initialize, then verify:

```bash
docker-compose -f deployment/docker-compose.yml ps
# atlas_db should show "healthy"
```

### 5. Run the health check

```bash
python main.py
```

Expected output:
```
============================================================
  Project Atlas  |  v0.1.0  |  Sprint 1
============================================================
  Environment     : DEVELOPMENT
  Log Level       : INFO
  Exchange        : NSE
  Watchlist       : 50 symbols (NIFTY 50)
  History         : 365 days
  Paper Capital   : ₹10,000
  Position Size   : 10% per trade
------------------------------------------------------------
  Database        : OK
============================================================
  System check: PASSED — all systems operational
============================================================
```

### 6. Install pre-commit hooks

```bash
pre-commit install
```

---

## Running Tests

```bash
# All tests (requires DB container running)
pytest tests/ -v

# Unit tests only (no DB required)
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Configuration

| File | Purpose |
|---|---|
| `.env` | Secrets — `DATABASE_URL`, `APP_ENV`, `LOG_LEVEL` |
| `config/settings.yaml` | App config — watchlist, intervals, paper trading params, scoring weights |
| `config/logging.yaml` | Log handlers, levels, rotation settings |

All settings are accessed via:

```python
from config import get_settings
settings = get_settings()
print(settings.market.watchlist)        # NIFTY 50 symbols
print(settings.paper_trading.initial_capital)  # 10000.0
```

---

## Development Workflow

```bash
# Start DB
docker-compose -f deployment/docker-compose.yml up -d db

# Activate venv
.venv\Scripts\activate   # Windows

# Make changes
# ...

# Run tests
pytest tests/ -v

# Format and lint (runs automatically on git commit)
black .
isort .
```

---

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for:
- Full system layer diagram
- Data flow diagram
- Database schema
- Configuration system design
- Key design decisions

---

## Development Phases

| Phase | Scope | Status |
|---|---|---|
| **Sprint 1** | Scaffold, config, DB connection, logging | ✅ Complete |
| Sprint 2 | Market data, technical indicators, news | 🔲 Next |
| Sprint 3 | Signal generation, paper trading | 🔲 Planned |
| Sprint 4 | Streamlit dashboard (7 pages) | 🔲 Planned |
| Sprint 5 | VectorBT backtesting, VPS deployment | 🔲 Planned |

---

## Rules

- Never expose `.env` or secrets in code or commits
- All DB queries live in `repositories/` only
- Dashboard reads from repositories only — never calls services
- All external API calls use `@retry` decorator
- Every module has docstring, type hints, and error handling
