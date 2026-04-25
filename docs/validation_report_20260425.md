# Data Validation Report
Generated: 2026-04-25 20:58  
Dataset: `eurovision_2016_26_enriched.csv` — 393 rows × 41 columns  
Standard: DD-01 Data Dictionary  

---

## Result

> **PASS** — zero unexplained nulls in mandatory fields; all AC met.

| Severity | Check | Affected rows |
|---|---|---|
| 🟢 INFO | country_code — ISO alpha-3 mapping | — |
| 🟢 INFO | country_code — valid ISO 3166-1 alpha-3 | — |
| 🟢 INFO | 2020 absence | — |
| 🟢 INFO | year — NOT NULL | — |
| 🟢 INFO | country — NOT NULL | — |
| 🟢 INFO | artist — NOT NULL | — |
| 🟢 INFO | song_title — NOT NULL | — |
| 🟢 INFO | running_order_final — NOT NULL for Final | — |
| 🟢 INFO | jury_points — NOT NULL for Final | — |
| 🟢 INFO | tele_points — NOT NULL for Final | — |
| 🟢 INFO | final_rank — NOT NULL for Final | — |
| 🟢 INFO | Expected nulls — Final_Points / jury_points / tele_points | — |
| 🟢 INFO | jury + tele == Final_Points | — |

---

## Findings — detail

### 🟢 country_code — ISO alpha-3 mapping

**Severity:** INFO  
**Affected rows:** none  

All 46 countries mapped to ISO 3166-1 alpha-3.

### 🟢 country_code — valid ISO 3166-1 alpha-3

**Severity:** INFO  
**Affected rows:** none  

All 45 codes are valid ISO alpha-3 values.

### 🟢 2020 absence

**Severity:** INFO  
**Affected rows:** none  

Year 2020 absent from dataset — Eurovision 2020 cancelled due to COVID-19 pandemic. Documented absence; no imputation required or appropriate.

### 🟢 year — NOT NULL

**Severity:** INFO  
**Affected rows:** none  

0 nulls.

### 🟢 country — NOT NULL

**Severity:** INFO  
**Affected rows:** none  

0 nulls.

### 🟢 artist — NOT NULL

**Severity:** INFO  
**Affected rows:** none  

0 nulls.

### 🟢 song_title — NOT NULL

**Severity:** INFO  
**Affected rows:** none  

0 nulls.

### 🟢 running_order_final — NOT NULL for Final

**Severity:** INFO  
**Affected rows:** none  

0 unexplained nulls in 232 Final rows with known results.

### 🟢 jury_points — NOT NULL for Final

**Severity:** INFO  
**Affected rows:** none  

0 unexplained nulls in 232 Final rows.

### 🟢 tele_points — NOT NULL for Final

**Severity:** INFO  
**Affected rows:** none  

0 unexplained nulls in 232 Final rows.

### 🟢 final_rank — NOT NULL for Final

**Severity:** INFO  
**Affected rows:** none  

0 unexplained nulls in 232 Final rows.

### 🟢 Expected nulls — Final_Points / jury_points / tele_points

**Severity:** INFO  
**Affected rows:** none  

6 Final rows with null results = 2026 entries (contest not yet held). 155 rows eliminated in semi-finals (no Final result expected). Neither counts as an unexplained null.

### 🟢 jury + tele == Final_Points

**Severity:** INFO  
**Affected rows:** none  

Passed for all 232 checked rows (tolerance ±1pt).

---

## Coverage statistics

| Year | Total rows | Finalists | Known results | jury_points coverage |
|---|---|---|---|---|
| 2016 | 42 | 26 | 26 | 26/26 |
| 2017 | 42 | 26 | 26 | 26/26 |
| 2018 | 43 | 26 | 26 | 26/26 |
| 2019 | 41 | 26 | 26 | 26/26 |
| 2021 | 39 | 26 | 26 | 26/26 |
| 2022 | 40 | 25 | 25 | 25/25 |
| 2023 | 37 | 26 | 26 | 26/26 |
| 2024 | 37 | 26 | 25 | 25/25 |
| 2025 | 37 | 26 | 26 | 26/26 |
| 2026 | 35 | 5 | 0 | N/A (not yet held) |
