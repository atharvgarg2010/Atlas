# research/

This directory is for research, experimentation, and one-off analysis.

## What Goes Here

- Jupyter notebooks for exploratory data analysis
- Strategy experiments and prototypes
- Feature engineering experiments
- Backtesting explorations before formalisation
- Ad-hoc data quality checks
- Model evaluation notebooks

## Rules

- Code here is **not production code** — no linting enforcement
- Do not import from `research/` into any other module
- Notebooks are git-ignored to avoid merge conflicts on cell outputs
- Validated ideas should be migrated to the relevant `services/` or `analytics/` module

## Structure (suggested)

```
research/
├── notebooks/
│   ├── 01_market_data_eda.ipynb
│   ├── 02_indicator_analysis.ipynb
│   └── 03_scoring_experiments.ipynb
├── scripts/
│   └── one_off_analysis.py
└── README.md
```
