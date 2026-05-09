# Eurovision 2026 Forecast Suite

Local machine-learning pipeline and Streamlit dashboard for Eurovision Song Contest 2026 outcome forecasts.

The project estimates Grand Final top-10 probability, semi-final qualification probability, derived podium/winner signals, model confidence intervals, SHAP-based country narratives, and historical voting-bloc structure. It is built as a local-first workflow with committed dashboard artifacts under `reports/`.

## What Is Included

- Streamlit dashboard in `app.py`
- Grand Final ranking and country detail cards
- Semi-final qualifier predictions for SF1 and SF2
- Podium probability heatmap and winner probability gauges
- SHAP-based narrative explanations
- Voting bloc matrix and D3 voting network
- Backtest and data-health views
- Local training, validation, feature engineering, and refresh scripts

## Quick Start

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the dashboard:

```powershell
python -m streamlit run app.py
```

The dashboard reads prebuilt artifacts from `reports/` and CSV feature data from `Dataset/` and `data/features/`.

## Main Commands

Rebuild Grand Final prediction payload:

```powershell
python -B scripts\build_predictions_json.py
```

Rebuild semi-final prediction payload:

```powershell
python -B scripts\build_semi_predictions_json.py --n-bootstrap 1000
```

Validate the guarded final-refresh pipeline:

```powershell
python scripts\refresh_final_predictions.py
```

Run the guarded final-refresh pipeline:

```powershell
python scripts\refresh_final_predictions.py --run
```

Run tests:

```powershell
python -m pytest
```

For a faster dashboard smoke check:

```powershell
python -m pytest tests\test_dashboard_navigation.py tests\test_dashboard_tiers.py
```

## Project Structure

```text
app.py                         Streamlit dashboard entrypoint
requirements.txt               Python dependencies
.streamlit/config.toml         Streamlit local config
Dataset/                       Enriched Eurovision datasets and DVC metadata
data/features/                 Feature-engineering outputs
docs/                          Operational notes and runbooks
models/artefacts/              Trained model artifacts
reports/                       Dashboard JSON/MD/PNG outputs
scripts/                       Artifact build and refresh scripts
src/data/                      Data ingestion, validation, and processing
src/features/                  Feature engineering
src/models/                    Training, backtest, confidence, SHAP, ensemble
src/tracking/                  Local MLflow configuration
tests/                         Pytest suite
```

## Dashboard Artifacts

The dashboard expects these files to exist:

| File | Purpose |
| --- | --- |
| `reports/predictions_2026.json` | Grand Final country ranking, model probabilities, confidence intervals |
| `reports/semi_predictions_2026.json` | Semi-final qualification probabilities |
| `reports/narratives_2026.json` | Country-level SHAP narrative cards |
| `reports/voting_network_2026.json` | D3 voting network payload |
| `reports/backtest_2022_2025.json` | Grand Final backtest metrics |
| `reports/backtest_semi_2022_2025.json` | Semi-final qualification backtest metrics |
| `Dataset/eurovision_2016_26_enriched.csv` | Main enriched dataset |
| `data/features/bloc_cooccurrence.csv` | Voting bloc matrix |

## Model Summary

The main prediction target is Grand Final top-10 placement. The system combines:

- XGBoost classifier
- LightGBM classifier
- PyTorch MLP classifier
- Weighted ensemble over model outputs
- Bootstrap confidence intervals
- SHAP explainability artifacts

Semi-final qualification is modelled separately because it is a threshold problem: 10 acts qualify from each semi-final.

The dashboard also shows top-3 and winner views, but these are derived interpretation layers on top of the top-10 model. They should be read as relative signals, not direct model outputs.

## Data And Tracking

- DVC is used for selected datasets.
- MLflow is configured for local experiment tracking.
- `.env.example` documents expected local environment variables.
- The project is designed to run locally without paid cloud services.

Useful local MLflow command:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Refresh Workflow

Use `scripts/refresh_final_predictions.py` before final prediction updates. It validates required inputs and can run the refresh pipeline only when `--run` is provided.

Typical dry run:

```powershell
python scripts\refresh_final_predictions.py
```

Typical execution after running-order and odds updates:

```powershell
python scripts\refresh_final_predictions.py --run
```

See `docs/sprint11_running_order_refresh.md` for operational details.

## Current Notes

- Primary KPI: at least 70% top-10 finalist accuracy in backtests.
- Random seed: `42`.
- Python 3.11+ is the intended runtime; the current local environment has also been exercised with Python 3.13.
- Some browser autoplay policies can block dashboard audio until the user interacts with the page.
