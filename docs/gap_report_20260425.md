# Data Gap Report
Generated: 2026-04-25 19:33  
Primary dataset: 393 rows × 36 columns  
Odds dataset: 221 rows × 31 columns

---

## Dataset coverage

**Year range:** 2016–2026 (10 contests present)  
**Years present:** 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026  
**Missing years:** 2020 — **2020 cancelled due to COVID-19** (no contest held; structurally absent, not a data gap).

---

## Summary

| ID | Severity | Field(s) | Status |
|---|---|---|---|
| GAP-01 | 🔴 CRITICAL | jury_points, tele_points | ABSENT — columns do not exist |
| GAP-02 | 🔴 CRITICAL | Genre | 0 / 393 filled (100% null) |
| GAP-03 | 🟠 HIGH | Country, Song, Artist (header) | Trailing whitespace in column names |
| GAP-04 | 🟡 MEDIUM | Qualification_Record | 50 / 393 null (12.7%) |
| GAP-05 | 🟡 MEDIUM | OGAE_Points | 35 / 393 null (8.9%) |
| GAP-06 | 🟢 LOW | BETANO, EPIC BET, 7BET, OPTIBET | 195 / 221 null (88.2%) — 2025 only |
| GAP-07 | 🟢 LOW | All result columns | 161 / 393 null (41%) — expected |

---

## Known gaps — detail

### GAP-01 — jury_points, tele_points 🔴

**Severity:** CRITICAL  
**Dataset:** Primary (kaggle CSV)  
**Status:** `ABSENT — columns do not exist`  
**Planned fix sprint:** S2  
**Owner:** Data Team

**Description:** The dataset contains only the combined `Final_Points` score. Separate jury and televote scores are absent for all years. Since 2016 Eurovision uses a 50/50 jury/televote split; the two signals frequently diverge (e.g. Ukraine 2022 won on televote but ranked lower with juries). Both signals are strong independent predictors.

**ML impact:** Cannot model jury vs. televote preference separately. Bloc-voting patterns differ between jury and public — losing this signal reduces prediction accuracy for countries with polarised reception (e.g. political favourites).

**Resolution:** Scrape per-year jury + televote breakdowns from the EBU results page or ESC-data GitHub dataset. Reference: DS-GAP-01.

### GAP-02 — Genre 🔴

**Severity:** CRITICAL  
**Dataset:** Primary (kaggle CSV)  
**Status:** `0 / 393 filled (100% null)`  
**Planned fix sprint:** S2  
**Owner:** Data Team / Client

**Description:** The `Genre` column is entirely empty across all 393 rows and all years (2016–2026). No genre label has been provided by the client.

**ML impact:** Genre is a potentially strong categorical feature — uptempo/dance entries have historically outperformed ballads in public vote; genre affects running-order placement decisions. Losing this feature degrades the model's ability to encode song-style effects.

**Resolution:** Option A: client supplies genre labels per entry. Option B: derive genre from Spotify `track_genre` / audio features (energy, danceability, acousticness) — already planned via Spotify API (PR-03). Reference: DS-GAP-01.

### GAP-03 — Country, Song, Artist (header) 🟠

**Severity:** HIGH  
**Dataset:** Primary (kaggle CSV)  
**Status:** `Trailing whitespace in column names`  
**Planned fix sprint:** S1  
**Owner:** Dev Team

**Description:** Column names `Country `, `Song `, `Artist ` contain a trailing space. Any code referencing these columns by exact string will fail silently or require defensive `.strip()` handling everywhere.

**ML impact:** Low direct impact but high maintenance / bug risk.

**Resolution:** Strip column names in C-01 Data Ingestion (load step). Apply `df.columns = df.columns.str.strip()` immediately after `pd.read_csv()`.

### GAP-04 — Qualification_Record 🟡

**Severity:** MEDIUM  
**Dataset:** Primary (kaggle CSV)  
**Status:** `50 / 393 null (12.7%)`  
**Planned fix sprint:** S2  
**Owner:** Dev Team

**Description:** 50 entries lack a `Qualification_Record` value. These are mostly debutant countries or Big6 members with no semi-final history.

**ML impact:** `Qualification_Record` is a key semi-final risk feature. Missing values require imputation; naive mean-fill will underestimate risk for debutants.

**Resolution:** Impute 0.0 for Big6 (exempt from semis) and 0.5 for debutants (unknown track record). Flag imputed rows with `qual_record_imputed` boolean column.

### GAP-05 — OGAE_Points 🟡

**Severity:** MEDIUM  
**Dataset:** Primary (kaggle CSV)  
**Status:** `35 / 393 null (8.9%)`  
**Planned fix sprint:** S2  
**Owner:** Dev Team

**Description:** 35 entries have no OGAE fan-club score. Affects mostly early years and smaller fan-club nations.

**ML impact:** OGAE is a fan-sentiment proxy correlated with televote outcome. Missing values require imputation.

**Resolution:** Impute with per-year median. Flag imputed rows.

### GAP-06 — BETANO, EPIC BET, 7BET, OPTIBET 🟢

**Severity:** LOW  
**Dataset:** Odds CSV  
**Status:** `195 / 221 null (88.2%) — 2025 only`  
**Planned fix sprint:** S2  
**Owner:** Dev Team

**Description:** These four bookmakers only appear in the 2025 dataset. They have no historical odds for 2018–2024.

**ML impact:** Cannot be used as historical predictors. Use only the core bookmakers present across all years: BETSSON, UNIBET, LAD BROKES, SKY BET, WILLIAM HILL, BET FRED, BFX.

**Resolution:** When engineering betting-odds features, restrict to bookmakers with >= 5 years of coverage. New bookmakers treated as supplementary only for 2025/2026 inference.

### GAP-07 — All result columns 🟢

**Severity:** LOW  
**Dataset:** Primary (kaggle CSV)  
**Status:** `161 / 393 null (41%) — expected`  
**Planned fix sprint:** —  
**Owner:** —

**Description:** Final_Place, Final_Points, Running_Order_Final are null for: (a) 2026 entries — contest not yet held; (b) countries eliminated in the semi-finals. This is structurally expected, not a data quality issue.

**ML impact:** No impact — these are target/leakage columns, not features.

**Resolution:** No action required. Document in Data Dictionary (DD-01).

---

## Auto-detected unexpected nulls ≥ 5.0% (primary CSV)

| Column | Null count | Null % |
|---|---|---|
| Genre | 393 | 100.0% |
| Qualification_Record | 50 | 12.7% |
| OGAE_Points | 35 | 8.9% |

---

## Bookmaker odds coverage matrix

| Bookmaker | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| BETSSON | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 100% |
| BOYLE SPORTS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 100% |
| BET365 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 100% |
| COOL BET | ✓ | ✓ | ✓ | ✓ | ✓ | 25/26 | ✓ | ✓ | 100% |
| BWIN | ✓ | ✓ | ✓ | ✓ | 19/25 | 0/26 | ✓ | ✓ | 86% |
| UNIBET | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 100% |
| BET STARS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0/26 | 88% |
| LAD BROKES | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 100% |
| 888 SPORT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 24/25 | ✓ | 100% |
| CORAL | ✓ | ✓ | ✓ | 0/26 | 0/25 | 0/26 | 0/25 | 0/26 | 42% |
| 10BET | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0/25 | 0/26 | 77% |
| BETWAY | ✓ | ✓ | ✓ | ✓ | ✓ | 0/26 | ✓ | ✓ | 88% |
| SKY BET | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 100% |
| WILLIAM HILL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 100% |
| BET FRED | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 100% |
| BETFAIR SPORT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0/25 | 0/26 | 77% |
| BFX | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 100% |
| OLYBET | 0/26 | ✓ | 0/41 | 0/26 | 0/25 | 0/26 | 0/25 | 0/26 | 12% |
| 1XBET | 0/26 | 0/26 | ✓ | ✓ | 0/25 | 0/26 | 0/25 | 0/26 | 30% |
| COMEON | 0/26 | 0/26 | ✓ | ✓ | ✓ | ✓ | 0/25 | 0/26 | 53% |
| SMARKETS | 0/26 | 0/26 | 0/41 | ✓ | ✓ | ✓ | ✓ | ✓ | 58% |
| BETANO | 0/26 | 0/26 | 0/41 | 0/26 | 0/25 | 0/26 | 0/25 | ✓ | 12% |
| EPIC BET | 0/26 | 0/26 | 0/41 | 0/26 | 0/25 | 0/26 | 0/25 | ✓ | 12% |
| 7BET | 0/26 | 0/26 | 0/41 | 0/26 | 0/25 | 0/26 | 0/25 | ✓ | 12% |
| OPTIBET | 0/26 | 0/26 | 0/41 | 0/26 | 0/25 | 0/26 | 0/25 | ✓ | 12% |

> **Core bookmakers** (≥ 80% coverage across 2018–2025):  
> BETSSON · UNIBET · LAD BROKES · SKY BET · WILLIAM HILL · BET FRED · BFX · COOL BET · BET365
