# Eurovision 2026 — Project Status

> **Last updated:** 2026-04-30 (Sprint 6 — US-S6-01 Backtest Engine complete)
> **Author:** Andrzej  
> **Active branch:** `main` / `develop` (synchronized)  
> **Session purpose:** Quick-start reference — read this before any other file.

---

## 1. Project at a Glance

**Goal:** Build a local ML platform that produces probabilistic outcome predictions for the Eurovision Song Contest 2026. Output drives broadcast planning and editorial decisions.

| Item | Value |
|------|-------|
| MVP deadline | 2026-06-30 (Streamlit dashboard) |
| Final predictions deadline | 2026-09-30 |
| Primary KPI | ≥ 70% top-10 finalist accuracy (stretch: 80%) |
| Infrastructure | 100% local — no paid cloud services |
| Random seed | 42 (all experiments) |
| Primary language | Python 3.11+ |

---

## 2. Sprint History & Completion Status

### Sprint 1 — Environment & Foundations ✅ (tagged `sprint-1`)
| Story | Description | Status |
|-------|-------------|--------|
| US-S1-01 | Local environment setup (venv, requirements) | Done |
| US-S1-02 | Feasibility analysis + PRD documents | Done |
| US-S1-03 | Data gap report script + initial gap report | Done |
| US-S1-04 | MLflow local setup + smoke-test experiment | Done |
| US-S1-05 | DVC init; Kaggle CSV tracked via `dvc add` | Done |

### Sprint 2 — Data Ingestion & Cleaning ✅ (tagged `sprint-2`)
| Story | Description | Status |
|-------|-------------|--------|
| US-S2-01 | Jury/televote split fetch + merge pipeline (`src/data/fetch_jury_tele.py`, `merge_jury_tele.py`) | Done |
| US-S2-02 | Validation script (`src/data/validate.py`) + validation report per DD-01 | Done |
| US-S2-03 | Betting odds processor + clean CSV (`src/data/process_odds.py`) | Done |
| US-S2-04 | 2026 participant validation + withdrawal flags; hotfix for SF assignments, running orders, artist/song names | Done |

### Sprint 3 — Feature Engineering ✅ (tagged `sprint-3`, merged to `main`)
| Story | Description | Status |
|-------|-------------|--------|
| US-S3-01 | Country fixed effects: `avg_final_rank_3yr`, `avg_jury_3yr`, `avg_tele_3yr` | Done |
| US-S3-02 | Rule-change binary flags: `rule_2019_semifinal_reform`, `rule_2023_jury_weight_reform` | Done |
| US-S3-03 | Voting bloc co-occurrence matrix → `avg_bloc_jury_3yr`, `avg_bloc_tele_3yr` | Done |
| US-S3-04 | Genre enrichment pipeline (Spotify + MusicBrainz) → `broad_genre` + binary flags | Done |
| US-S3-05 | Social proxy z-score normalisation: `zscore_myesb_community`, `zscore_myesb_personal`, `zscore_ogae_points` | Done |
| US-S3-06 | EDA report — 10 interactive Plotly charts (`reports/charts/eda_20260425.html`) | Done |

**⚠ Open item (KL-05):** Genre coverage 74.3% < 90% threshold — genre flags not yet in FEATURE_COLS. Revisit if SHAP shows fan/odds features dominant.

### Sprint 4 — Model Training ✅ (tagged `sprint-4`, merged to `main` 2026-04-28)
| Story | Description | Status |
|-------|-------------|--------|
| US-S4-01 | `LeaveLastYearOut` CV splitter (`src/models/cv.py`) | Done |
| US-S4-02 | XGBoost + LightGBM grid search; DVC artefacts; 2024 holdout evaluation; betting odds feature | Done |
| US-S4-03 | Bootstrap CI n=1000; 80%/50% CI per country (`src/models/confidence.py`) | Done |
| US-S4-04 | SHAP TreeExplainer; top-5 features per country; beeswarm plot (`src/models/shap_pipeline.py`) | Done |
| US-S4-05 | Leakage audit — 8 programmatic checks + peer-review doc; MLflow tag `leakage_check_passed=true` | Done |

### Sprint 5 — Ensemble & Inference ✅ (tagged `sprint-5`, merged to `main` 2026-04-29)
| Story | Description | Status |
|-------|-------------|--------|
| US-S5-01 | PyTorch MLP — third ensemble member; CV grid search; DVC artefacts (`src/models/nn.py`) | Done — CV ROC-AUC 0.6148 ± 0.1760 (8 folds) |
| US-S5-02 | Weighted ensemble XGB+LGBM+MLP; weight grid search (step=0.1); holdout evaluation (`src/models/ensemble.py`) | Done — best blend lgbm=1.0; 70% (7/10); KPI ✅ PASS; KL-06 open risk logged |
| US-S5-03 | SHAP → plain-language prediction cards per country (`src/models/narratives.py`) | Done — 35-country narrative report; `reports/narratives_2026.{json,md}` |
| US-S5-04 | Bilateral jury co-occurrence network → D3 JSON (`src/features/voting_network.py`) | Done — 35 nodes, 72 edges (min_weight=2); top pair Italy–Ukraine(6); `reports/voting_network_2026.json` |
| US-S5-05 | Linear surrogate model — Ridge poly-2 distilled (`src/models/surrogate.py`) | Done — Inference 1.3 ms ✅ PASS; Rank delta 6.6 ❌ KL-07 (structural: no 2026 odds yet) |

### Sprint 6 — Validation & Scenario Engine 🔄 (in progress)
| Story | Description | Status |
|-------|-------------|--------|
| US-S6-01 | Backtest 2022/23/24 — train < year, Top-10 acc + CI calibration per year (`src/models/backtest.py`) | Done — XGB 80/70/70% top-10; CI-80 coverage 92/88/88%; all KPIs PASS ✅ |
| US-S6-01b | Semi-final qualification backtest 2022/23/24 — separate binary classifier, target Grand_Final_Ind (`src/models/backtest_semi.py`) | Done — XGB avg 97% qual acc; LGBM avg 98%; CI-80 coverage 96/99%; all KPIs PASS ✅ |

---

## 3. Model Status

### Architecture
Three binary classifiers predict **Top-10 Grand Final placement**:
- **XGBoost** (`models/artefacts/xgb_model.pkl`)
- **LightGBM** (`models/artefacts/lgbm_model.pkl`)
- **MLP** (`models/artefacts/nn_model.pkl`) — PyTorch, `NNPipeline`

All use `SimpleImputer(median, keep_empty_features=True)` preprocessing.  
Cross-validation: `LeaveLastYearOut` (temporal, no future leakage).  
Grid search tracked in MLflow experiment `eurovision-2026-ensemble`.  
Artefacts tracked by DVC (`models/artefacts/`).

### 2024 Holdout Results (PRD KPI check)

| Snapshot | XGB | LGBM | MLP | Ensemble (best blend) | KPI (≥ 70%) |
|----------|-----|------|-----|-----------------------|-------------|
| Without betting odds | 50% (5/10) | 50% (5/10) | — | — | ❌ FAIL |
| With `implied_prob_close` | **70% (7/10)** | **70% (7/10)** | 50% (5/10) | **70% (7/10)** lgbm=1.0 | ✅ PASS |

Holdout: 2024 Grand Final (25 countries; Netherlands excluded — disqualified).  
Train window: 2016–2023 Grand Final entries (Final_Place known).  
Ensemble hits (7/10): Croatia, France, Ireland, Israel, Italy, Switzerland, Ukraine.  
**Open risk KL-06:** ensemble = best single model, no lift — see `docs/known_limitations.md`.

### Feature Set (23 features)

**Raw (from enriched CSV — 12):**  
`Big6_Ind`, `National_Final`, `Solo_Artist`, `Returning_Artist_Ind`, `Number of Members`,  
`Multiple_Language`, `EU`, `NATO`, `Qualification_Record`,  
`Semi_Final_Num`, `Running_Order_Semi`, `Running_Order_Final`

**Engineered (11):**  
`avg_final_rank_3yr`, `avg_jury_3yr`, `avg_tele_3yr` (country fixed effects)  
`avg_bloc_jury_3yr`, `avg_bloc_tele_3yr` (voting blocs)  
`rule_2019_semifinal_reform`, `rule_2023_jury_weight_reform` (rule flags)  
`zscore_myesb_community`, `zscore_myesb_personal`, `zscore_ogae_points` (social proxy)  
`implied_prob_close` (betting odds closing price — overround-normalised)

**Not yet in feature set (planned for future stories):**  
Genre flags (`genre_pop`, `genre_dance`, …) — pending coverage ≥ 90%  
Spotify audio features (energy, danceability, acousticness) — not yet fetched

---

## 4. Dataset Overview

### Primary: `Dataset/eurovision_2016_26_enriched.csv`
- 393 rows × 41 columns
- Years: 2016–2026 (2020 absent — Eurovision cancelled, COVID-19)
- 10 contests; 35 countries entered in 2026
- All mandatory fields pass DD-01 validation (zero unexplained nulls)
- Jury + televote split added via `fetch_jury_tele.py` (originally absent from Kaggle source)
- Tracked by DVC

### Betting Odds: `Dataset/betting_odds_clean.csv`
- 221 rows × 31 columns; 2018–2025 (no 2016/2017 data)
- `implied_prob` derived from harmonic mean across core bookmakers, overround-normalised to sum to 1.0 per year
- Core bookmakers (≥80% coverage): BETSSON, UNIBET, LAD BROKES, SKY BET, WILLIAM HILL, BET FRED, BFX, COOL BET, BET365
- `odds_open` always NULL — no opening odds source found (KL-01)

### Jury/Tele Split: `Dataset/jury_tele_raw.csv` (tracked by DVC)

### Genre Features: `data/features/genre.csv`
- Coverage: 292/393 = **74.3%** (as of 2026-04-26)
- Source: Spotify (primary) + MusicBrainz (fallback)
- Below 90% threshold → treated as optional/experimental (KL-05)

---

## 5. 2026 Participants (35 countries)

**Auto-qualifiers (Big 5 + host Austria):** Austria, France, Germany, Italy, United Kingdom  
**SF1 (15):** Belgium, Croatia, Estonia, Finland, Georgia, Greece, Israel, Lithuania, Moldova, Montenegro, Poland, Portugal, San Marino, Serbia, Sweden  
**SF2 (15):** Albania, Armenia, Australia, Azerbaijan, Bulgaria, Cyprus, Czech Republic, Denmark, Latvia, Luxembourg, Malta, Norway, Romania, Switzerland, Ukraine

**Boycotts/withdrawals (6):** Spain (notable — first Big 5 absence since Italy rejoined 2011), Ireland, Iceland, Netherlands, Slovenia, North Macedonia

---

## 6. Known Limitations (KL-XX)

| ID | Field | Status | Severity |
|----|-------|--------|----------|
| KL-01 | `odds_open` always NULL | Documented — does not block modeling | Low |
| KL-02 | `Genre` 100% NULL in raw Kaggle source | Resolved via S3 genre enrichment pipeline | N/A |
| KL-03 | `Qualification_Record` NULL for Big 5 + debutants | Structural; exclude from qualification-history features | Low |
| KL-04 | `OGAE_Points` sparse (early years / small nations) | Imputed with per-year median | Low |
| KL-05 | Genre coverage 74.3% (below 90% threshold) | **Open** — blocks genre features in modeling | Medium |
| KL-06 | Ensemble = best single model (lgbm=1.0), no lift | **Open** — mitigation: Spotify audio features (Sprint 12), S6 backtest | Low |
| KL-07 | Surrogate rank delta 6.6 (KPI < 2.0 not met) | **Open** — fix: re-train once 2026 betting odds loaded | Medium |

---

## 7. Repository & Infrastructure

### Git Branches
| Branch | State |
|--------|-------|
| `main` | Sprint 5 release (tagged `sprint-5`, 2026-04-29) |
| `develop` | = main (fully synchronized) |
| `feature/*` | All cleaned up — no local or remote feature branches remain |

### Key Directories
```
Dataset/           — Raw DVC-tracked CSVs (primary + odds + jury/tele + enriched)
src/data/          — Ingestion, cleaning, validation, odds processing
src/features/      — Feature engineering (country effects, rule flags, blocs, genre, social)
src/models/        — CV splitter, train, evaluate
tests/             — 9 test files covering all major modules
docs/              — Gap report, validation report, participant report, known_limitations.md
reports/           — Data profile, EDA charts
models/artefacts/  — Trained model .pkl files (DVC-tracked, git-ignored)
models/mlruns/     — MLflow experiment runs (git-ignored)
```

### Untracked Files (not yet committed to git)
- `Feasibility Analysis/` — feasibility & requirements analysis docx
- `PRD/` — product requirements document
- `Project_Execution_documents/` — ARCH-01, DD-01, DS-GAP-01, ENV-01, MK-01, OQ-01, PP-01, RR-01, SB-01, TC-01, system architecture SVG

### MLflow
```bash
mlflow ui --backend-store-uri ./models/mlruns
# → http://127.0.0.1:5000
```
Experiment: `eurovision-2026-ensemble`

### DVC
```bash
dvc pull   # fetch dataset artefacts
dvc push   # push after new dvc add
```

---

## 8. Components Not Yet Built

Per the five-layer architecture (CLAUDE.md), these components remain to be implemented:

| Component | Description | Priority |
|-----------|-------------|----------|
| C-03 ✅ | Neural Network third ensemble member (`src/models/nn.py`) | Done (US-S5-01) |
| C-04 ✅ | Backtest Engine — XGB/LGBM 2022/23/24 holdout; top-10 acc + CI calibration (`src/models/backtest.py`) | Done (US-S6-01) |
| C-05 | Leakage Audit (mandatory before any model release) | High |
| C-06 | Scenario Engine — FastAPI (`src/scenario/main.py`) | High (MVP) |
| C-07 ✅ | SHAP Narratives — 2-4 sentence per country (`src/models/narratives.py`) | Done (US-S5-03) |
| C-08 | Voting Bloc Model (networkx clustering) | Medium |
| C-09 | Interactive Dashboard — Streamlit MVP (`src/dashboard/app.py`) | High (MVP) |
| C-10 | PDF Export (WeasyPrint/Puppeteer — OQ-10 pending on Windows GTK) | Low |

**Sprint 5 completed components:**

| Component | Status |
|-----------|--------|
| C-03 ✅ | Neural Network MLP (`src/models/nn.py`) |
| C-07 ✅ | SHAP Narratives (`src/models/narratives.py`) |
| Ensemble blending ✅ | `src/models/ensemble.py` (lgbm=1.0, KPI 70%) |
| Jury network ✅ | `src/features/voting_network.py` (D3 JSON) |
| Linear surrogate ✅* | `src/models/surrogate.py` (inference KPI PASS; rank delta KL-07) |

---

## 9. Immediate Next Actions

1. **Sprint 6 — continuing**: US-S6-01 (Backtest Engine) complete. Next candidates: C-06 (Scenario Engine FastAPI), C-09 (Streamlit Dashboard MVP).
2. **Load 2026 betting odds** — once closing odds available (~2–4 weeks pre-contest), run `src/data/process_odds.py` and re-train surrogate. Expected to fix KL-07 rank delta.
3. **Verify genre coverage** — run `python src/features/genre.py --fetch` with Spotify credentials; if ≥ 90%, add genre flags to `FEATURE_COLS`.
4. **DVC push** — `py -m dvc push` when DVC available in environment to sync model artefacts.

---

## 10. Key Technical Rules

- **Temporal isolation (PR-07):** No future-year data in any training fold at any stage. Leakage audit (C-05) mandatory before model release.
- **Merge & push:** Always ask for explicit approval before merge; ask again before push.
- **Reproducibility:** Every training run logs to MLflow: hyperparams, fold CV scores, artefact, random seed, DVC dataset hash, leakage check result.
- **48h re-run (PR-04):** Pipeline must be fully re-triggerable with new participant/song data within 48 hours.
- **Performance SLAs:** Dashboard load < 5s; scenario recalculation < 10s.
- **Feature whitelist:** `FEATURE_COLS` in `train.py` is the authoritative list — no outcome columns may appear there.
- **`data/raw/` immutable:** Never modify raw files; work in `data/processed/` or `data/features/`.
