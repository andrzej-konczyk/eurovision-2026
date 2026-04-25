# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eurovision 2026 Outcome Prediction System — a data science platform that produces probabilistic contest predictions for broadcast planning and editorial decision-making. All infrastructure runs **locally** (no paid cloud services).

- **MVP target:** June 30, 2026
- **Final predictions:** September 30, 2026
- **Primary KPI:** ≥ 70% top-10 finalist accuracy (target 80%)

## Key Commands

All commands assume the project root is `eurovision-2026/` with `.venv` activated.

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt

# Run dashboard (MVP)
streamlit run src/dashboard/app.py
# → http://localhost:8501  (must load in < 5s)

# Run scenario API
uvicorn src.scenario.main:app --reload
# → http://localhost:8000  (must respond in < 10s)

# MLflow experiment tracking
mlflow ui --backend-store-uri ./models/mlruns
# → http://127.0.0.1:5000

# DVC — data versioning
dvc add data/raw/eurovision_2016_26_kaggle.csv
dvc push / dvc pull

# Tests
pytest tests/
```

## Directory Structure

```
eurovision-2026/
├── data/
│   ├── raw/          # Source files — never modified directly; tracked by DVC
│   ├── processed/    # Cleaned & validated datasets
│   ├── features/     # Engineered feature matrices
│   └── dvc-cache/    # DVC local cache (git-ignored)
├── models/
│   ├── artefacts/    # Trained model files (.pkl, .pt, .json) — tracked by DVC
│   └── mlruns/       # MLflow runs (git-ignored)
├── src/
│   ├── data/         # C-01: ingestion, cleaning, validation
│   ├── features/     # C-02: feature engineering pipeline
│   ├── models/       # C-03/C-04/C-05: ensemble, backtest, leakage audit
│   ├── explainability/ # C-07/C-08: SHAP narratives, voting bloc model
│   ├── dashboard/    # C-09: Streamlit app (MVP) or React+Plotly (final)
│   └── scenario/     # C-06: FastAPI scenario engine
├── tests/
├── notebooks/        # EDA only — not production code
├── reports/          # Backtest reports, PDF executive summary
└── .env              # API keys — never commit
```

## Five-Layer Architecture

| Layer | Components | Technology |
|-------|-----------|------------|
| **L1 Data Sources** | Primary dataset, betting odds, social media, song metadata, EBU rules | CSV (client), Oddsportal, YouTube API, Spotify, MusicBrainz |
| **L2 Ingestion & Store** | C-01 Data Ingestion, C-11 Artefact Registry, C-12 Experiment Tracker | pandas + pydantic, DVC, MLflow |
| **L3 ML Pipeline** | C-02 Feature Engineering, C-03 Ensemble Model, C-04 Backtest Engine, C-05 Leakage Audit, C-07 SHAP Narratives, C-08 Voting Bloc Model | scikit-learn, XGBoost, LightGBM, PyTorch, SHAP, networkx |
| **L4 Inference API** | C-06 Scenario Engine | FastAPI + pre-computed surrogate model |
| **L5 Presentation** | C-09 Interactive Dashboard, C-10 PDF Export | Streamlit (MVP) → React+Plotly+D3 (final), WeasyPrint/Puppeteer |

## Core Architecture Constraints

- **Temporal isolation (PR-07):** No future-year data may appear in any training fold at any stage. The leakage audit (C-05) produces a signed report and is mandatory before any model release.
- **Backtesting:** Leave-last-year-out on 2022, 2023, 2024 holdouts (C-04). Model artefacts used in backtest must match the production inference interface exactly.
- **Reproducibility:** Every training run must log to MLflow: all hyperparameters, per-fold CV scores, model artefact, random seed (`RANDOM_SEED=42`), DVC dataset hash, and leakage check result.
- **48h re-run capability (PR-04):** Pipeline can be fully re-triggered with new participant/song data within 48 hours.
- **Performance SLAs:** Dashboard initial load < 5s; scenario recalculation < 10s. Validate with browser Network tab before each milestone.

## ML Model Details

- **Ensemble (C-03):** XGBoost + LightGBM + Neural Network, with 80% and 95% confidence intervals via time-series cross-validation.
- **Features (C-02):** Historical rank aggregates, temporal rule-change flags, voting bloc indicators, Spotify audio features, social/betting signals.
- **Voting bloc model (C-08):** Correlation-based clustering of bilateral voting history (scipy + networkx + sklearn).
- **SHAP narratives (C-07):** 2–4 sentence plain-language explanation per country, generated from SHAP feature importances.

## Output Tiers

| Tier | Content |
|------|---------|
| T1 | Probable winners (top-3 with confidence intervals) |
| T2 | Contenders (top-10 predictions) |
| T3 | Wildcards |
| T4 | Semi-final elimination risk |

## Data

- **Primary dataset:** `data/raw/eurovision_2016_26_kaggle.csv` — 2016–2026 contest data with 35+ features (placement, points, running order, genre, language, country metadata, community scores, betting odds, qualification record).
- **`data/raw/` is never modified** — always work with `data/processed/` or `data/features/`.
- **Secret keys** (`YOUTUBE_API_KEY`, `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`) live in `.env` only.

## Git Branching

| Branch | Purpose |
|--------|---------|
| `main` | Protected — release-ready only, merge via PR |
| `develop` | Sprint integration branch |
| `feature/US-SXXX-xxx` | Per user story, branch from `develop` |
| `data/xxx` | Data pipeline changes isolated from model code |

## Environment Variables (`.env`)

```
YOUTUBE_API_KEY=
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=
MLFLOW_TRACKING_URI=./models/mlruns
RANDOM_SEED=42
```
