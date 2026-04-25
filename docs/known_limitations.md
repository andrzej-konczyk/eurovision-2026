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

*Add new limitations below following the same KL-XX format.*
