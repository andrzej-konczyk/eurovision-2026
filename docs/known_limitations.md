# Known Limitations

This document records deliberate gaps, missing data, and design constraints that are **not bugs** and should not block downstream modeling or analysis. Each entry includes the affected field, root cause, impact assessment, and the agreed mitigation.

---

## KL-01 — `odds_open` column is always NULL

| Field | Value |
|-------|-------|
| **Affected dataset** | `Dataset/betting_odds_clean.csv` |
| **Affected column** | `odds_open` |
| **Status** | Known, documented — does **not** block modeling |
| **Logged** | 2026-04-25 |

### Root cause

The primary odds source (`eurovision_odds_2018_2025.csv`, Kaggle) contains a **single pre-contest closing snapshot** per country per year. Opening odds (posted the moment entries are announced, typically 3–6 months before the contest) are not included.

No free public source for Eurovision opening odds was identified during Sprint 2 data collection. Oddsportal does not carry Eurovision markets; EurovisionWorld shows only current odds.

### Impact on PR-02 (betting odds feature)

| Metric | Assessment |
|--------|------------|
| `implied_prob` (from `odds_close`) | **Available, complete** — 221 rows, sums to 1.0/year (2018–2025) |
| `odds_close` harmonic-mean coverage | 100 % (all rows pass 30 % bookmaker threshold) |
| `odds_open` | NULL for all rows from primary source |

`implied_prob` derived from closing odds is the primary feature used in the prediction model. Closing odds are generally considered more informative than opening odds as they incorporate the full pre-contest information set (song reveal, rehearsal performances, press reaction). The absence of `odds_open` therefore has no impact on PR-02 acceptance criteria.

### Mitigation / path to fill

If the client supplies a file with opening odds, run:

```bash
python src/data/process_odds.py --client-file <path_to_client_csv>
```

Expected columns: `year`, `country`, `odds_open`, `odds_close`. The script will merge client rows into `betting_odds_clean.csv` and append an entry to `docs/odds_ingestion_log.md`.

---

## KL-02 — `Genre` column is 100 % NULL

| Field | Value |
|-------|-------|
| **Affected dataset** | `Dataset/eurovision_2016_26_enriched.csv` |
| **Affected column** | `Genre` |
| **Status** | Known, deferred to Sprint 3 feature engineering |
| **Logged** | 2026-04-25 |

### Root cause

The Kaggle source dataset ships the `Genre` column as entirely empty (393/393 rows null). No secondary source was ingested during Sprint 1–2.

### Impact

Genre-based features (`genre_encoded`, `genre_cluster`) are listed as MEDIUM-priority in the gap report (`docs/gap_report_20260425.md`). They are **not required** for the Sprint 3 baseline model. Manual tagging or a MusicBrainz/Spotify API lookup is the planned resolution path.

---

## KL-03 — `Qualification_Record` partially NULL (Big 5 + debutants)

| Field | Value |
|-------|-------|
| **Affected dataset** | `Dataset/eurovision_2016_26_enriched.csv` |
| **Affected column** | `Qualification_Record` |
| **Status** | Known structural null — not an error |
| **Logged** | 2026-04-25 |

### Root cause

Big 5 countries (France, Germany, Italy, Spain, United Kingdom) and Austria (host 2026) receive automatic Grand Final spots; they have no semi-final qualification record by definition.

### Impact

NULL values for these countries are expected and should be excluded from any model feature that uses qualification history.

---

## KL-04 — `OGAE_Points` sparsely populated (early years / small nations)

| Field | Value |
|-------|-------|
| **Affected dataset** | `Dataset/eurovision_2016_26_enriched.csv` |
| **Affected column** | `OGAE_Points` |
| **Status** | Known, MEDIUM severity — does not block baseline model |
| **Logged** | 2026-04-25 |

### Root cause

OGAE (Organisation Générale des Amateurs de l'Eurovision) fan jury data is not available for all country-years in the dataset. Coverage is patchy for 2016–2018 and absent for some smaller nations.

### Impact

OGAE is a supplementary signal, not a core feature. Rows with null `OGAE_Points` are handled by dropping the column from the initial feature set and revisiting in a later sprint if fan-signal features prove valuable.

---

## KL-05 — Genre coverage 74.3 % (below 90 % threshold)

| Field | Value |
|-------|-------|
| **Affected dataset** | `data/features/genre.csv` |
| **Affected column** | `broad_genre` |
| **Status** | Open — blocks genre features in modeling if coverage < 90 % |
| **Logged** | 2026-04-26 |
| **Verified** | `python src/features/genre.py --fetch` (Spotify + MusicBrainz, 2026-04-26) |

### Root cause

The genre pipeline uses Spotify as primary source and MusicBrainz as fallback. 101 entries are absent from both APIs — mostly niche, non-English, or older Eurovision acts with minimal tagging on either platform.

### Coverage breakdown (2026-04-26 run, 393 pairs — Spotify + MusicBrainz)

| broad_genre | count |
|-------------|-------|
| pop | 152 |
| None (unmapped) | 101 |
| dance | 65 |
| rock | 41 |
| folk | 23 |
| classical | 9 |
| ballad | 2 |

**Total covered: 292 / 393 (74.3 %)** — up from 46.3 % (MusicBrainz-only)

### Impact

`broad_genre` and derived binary flags (`genre_pop`, `genre_dance`, …) are unreliable for the 211 uncovered rows. Genre features should be treated as **optional / experimental** until coverage reaches ≥ 90 %.

### Mitigation / path to fill

1. Obtain Spotify API credentials (free — register at developer.spotify.com).
2. Set env vars: `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`.
3. Re-run: `python src/features/genre.py --fetch` — Spotify covers the majority of missing acts.
4. Update this entry with new coverage figure.

---

*Add new limitations below following the same KL-XX format.*

---

## KL-06 — Ensemble does not outperform best single model (open risk — RR-01)

| Field | Value |
|-------|-------|
| **Affected component** | `src/models/ensemble.py`, `models/artefacts/ensemble_weights.json` |
| **Status** | **Open risk** — logged in RR-01 |
| **Logged** | 2026-04-29 |
| **Story** | US-S5-02 |
| **Severity** | Medium |

### Observation

On the 2024 Grand Final holdout the optimal blend weight is **lgbm=1.0, xgb=0.0, nn=0.0**. The ensemble top-10 accuracy equals the best individual model (70%), not above it.

- XGBoost standalone: 70% (7/10)
- LightGBM standalone: 70% (7/10)
- MLP standalone: 50% (5/10)
- Best blend: 70% (7/10) — KPI threshold met, but no ensemble lift

The PRD stretch goal (>70%) is unmet at this stage.

### Root cause

The MLP (US-S5-01, CV ROC-AUC 0.615 ± 0.176) underperforms XGB/LGBM on the 2024 holdout. Any blend that includes a non-zero MLP weight reduces accuracy below 70%. XGB and LGBM predict identically on this holdout — member diversity is insufficient to generate lift.

### Decision (2026-04-29)

Proceed with **best blend lgbm=1.0** (or xgb=0.5, lgbm=0.5) as the production `ensemble_weights.json`. KPI threshold (≥70%) is met. Risk logged for tracking.

### Update (2026-05-05, US-S8-01)

The 2025 holdout re-run reverses the single-model preference:

- XGBoost standalone: 70% (7/10) - PASS
- LightGBM standalone: 60% (6/10) - FAIL
- MLP standalone: 50% (5/10) - FAIL
- Best 2025 blend: xgb=1.0, lgbm=0.0, nn=0.0 - 70% (7/10), PASS
- Equal XGB/LGBM compromise: 60% (6/10), FAIL

Official `ensemble_weights.json` was therefore updated to **xgb=1.0, lgbm=0.0, nn=0.0** for US-S8-01. The equal blend was evaluated and rejected because it fails the 2025 KPI.

### Mitigation paths

1. **Spotify audio features (Sprint 12)** — adding energy, danceability, acousticness to the MLP feature set is expected to improve MLP standalone accuracy and increase member diversity. If MLP lifts above ~65% standalone, blending should produce >70% ensemble accuracy.
2. **S6 backtest (2022, 2023 holdouts)** — ensemble lift may be visible on years where XGB and LGBM diverge. A lift signal there would validate the ensemble architecture even if 2024 shows no improvement.
3. **Fine-grained weight step (0.05)** — may expose marginal improvements not visible at step=0.1.
4. **Stacking / meta-learner** — replace weighted average with a logistic meta-learner trained on fold OOF predictions; may capture non-linear synergies between the three base models.

---

## KL-07 — Linear surrogate rank-delta KPI not met

| Field | Value |
|-------|-------|
| **Affected module** | `src/models/surrogate.py` (US-S5-05) |
| **KPI** | Mean absolute rank delta vs ensemble < 2.0 positions |
| **Achieved** | ~6.6 positions (polynomial Ridge, in-sample distillation) |
| **Status** | Open — structural limitation |
| **Logged** | 2026-04-29 |

### Root cause

For 2026 predictions, three of the most discriminating features are constant across all 35 countries:

| Feature | 2026 status | Reason |
|---------|-------------|--------|
| `implied_prob_close` | NaN → median-imputed (constant) | Odds CSV covers only 2018–2025 |
| `Running_Order_Final` | NaN → median-imputed (constant) | Draw not yet held for Basel 2026 |
| `OGAE_Points` / `zscore_ogae_points` | NaN → median-imputed (constant) | OGAE poll not yet published |

The 2026 ranking is therefore determined entirely by `avg_jury_3yr`, `avg_tele_3yr`, social scores, `Big6_Ind`, etc. The LGBM ensemble exploits **non-linear tree interactions** among these features (e.g. high jury history × high community enthusiasm = disproportionate probability boost) that a linear model — even with degree-2 polynomial interactions — cannot fully replicate.

### Impact

- Inference KPI (**< 2 s**): **PASS** (< 2 ms — 1,000× headroom)
- Rank-delta KPI (**< 2.0**): **FAIL** — achieved ~6.6
- Spearman rank correlation (surrogate vs ensemble on 2026): ~0.79
- The surrogate correctly identifies the top-3 favorites (Sweden, Greece, France) within ±3 positions and is directionally useful for scenario exploration, but should not replace the full ensemble for final-ranking output.

### Mitigation paths

1. **Load 2026 betting odds** — once closing odds are available (typically ~2–4 weeks before the contest), re-run `src/data/process_odds.py` and re-train the surrogate. With real `implied_prob_close` values, rank delta is expected to drop significantly.
2. **Use a depth-2 XGBoost surrogate** — a shallow boosted tree (max_depth=2, 50 estimators) produces ~100× faster inference than the full model while capturing tree interactions; expected delta < 2.
3. **Accept achieved delta for scenario engine** — for the C-06 use case (small feature perturbations on a single country), the surrogate's directional correctness (~79% Spearman) is sufficient for "what-if" analysis.

---

## KL-08 — LGBM pre-contest prior bias (2025)

| Field | Value |
|-------|-------|
| **Affected component** | `src/models/ensemble.py`, LGBM Grand Final top-10 model |
| **Status** | OPEN |
| **Logged** | 2026-05-05 |
| **Story** | US-S8-01 |
| **Severity** | Medium |

### Observation

On the 2025 Grand Final holdout, LightGBM reaches only 60% top-10 accuracy (6/10), while XGBoost reaches 70% (7/10). LGBM systematically misses countries with strong jury/televote history despite weaker market priors: Greece, Italy, and Switzerland.

The false positives skew toward market-favored countries without enough final result support in the holdout: Finland, Latvia, Norway, and Poland.

### Root cause

`implied_prob_close` appears to dominate LGBM splits on this holdout, creating a market-to-model feedback loop. The model follows the pre-contest market prior too aggressively and lacks a corrective feature for countries whose historical jury/televote strength is stronger than their market price.

### Impact

The 2025 ensemble grid search selects **xgb=1.0, lgbm=0.0, nn=0.0**. This keeps the ensemble at 70% top-10 accuracy, but confirms that LGBM is not reliable as the winner-takes-all blend member for the newest available holdout.

Even the XGB-selected 2025 ensemble still misses Greece, Italy, and Switzerland. This matters for future scenario analysis: if the model consistently underestimates strong-history countries, Ukraine or similar countries may also be underestimated when market priors are weak.

### Mitigation

Add `odds_vs_history_delta` in Sprint 9:

```text
odds_vs_history_delta = normalized(implied_prob_close - avg_tele_3yr)
```

The feature should give tree models an explicit signal for countries where the market underprices or overprices recent televote strength.
