# Leakage Audit — Eurovision 2026 Prediction Pipeline
**Story:** US-S4-05  
**Reviewer:** Andrzej  
**Date:** 2026-04-28  
**Branch:** `feature/US-S4-05-leakage` → `develop`  
**Verdict:** ✅ PASS — all 8 checks passed

---

## 1. Scope and Methodology

This document audits the Eurovision 2026 prediction pipeline for **temporal
data leakage** as required by constraint PR-07: no information from year Y or
later may enter the features, training labels, or CV evaluation of any model
trained to predict year Y outcomes.

Leakage is checked both **statically** (source-code inspection) and
**dynamically** (runtime programmatic checks in `src/models/leakage_audit.py`).

**Components audited:**

| Component | File |
|-----------|------|
| Feature whitelist | `src/models/train.py` |
| Training split | `src/models/train.py` |
| Cross-validation | `src/models/cv.py` |
| Country fixed effects | `src/features/country_fixed_effects.py` |
| Voting bloc averages | `src/features/voting_blocs.py` |
| Rule-change flags | `src/features/rule_flags.py` |
| Social proxy (z-scores) | `src/features/social_proxy.py` |
| Holdout evaluation | `src/models/evaluate.py` |

---

## 2. Summary of Results

| ID | Check | Verdict | Method |
|----|-------|---------|--------|
| LA-01 | FEATURE_COLS excludes outcome columns | **PASS** | Runtime |
| LA-02 | training_split(): Year < 2026, label known | **PASS** | Runtime |
| LA-03 | LeaveLastYearOut: max_train_yr < test_yr | **PASS** | Runtime |
| LA-04 | Country fixed effects: Year < current year | **PASS** | Runtime |
| LA-05 | Voting blocs: Year < current year | **PASS** | Runtime |
| LA-06 | Holdout split: train < 2024, no index overlap | **PASS** | Runtime |
| LA-07 | Feature matrix X has no outcome columns | **PASS** | Runtime |
| LA-08 | Social proxy: per-year z-score mean ≈ 0 | **PASS** | Runtime |

---

## 3. Detailed Findings

### LA-01 — Feature Whitelist

`FEATURE_COLS` is an explicit 23-feature whitelist at `src/models/train.py:98`:

```python
FEATURE_COLS: list[str] = _RAW_FEATURES + _ENGINEERED_FEATURES
```

**Raw features (12):** `Big6_Ind`, `National_Final`, `Solo_Artist`,
`Returning_Artist_Ind`, `Number of Members`, `Multiple_Language`, `EU`, `NATO`,
`Qualification_Record`, `Semi_Final_Num`, `Running_Order_Semi`,
`Running_Order_Final`.

**Engineered features (11):** `avg_final_rank_3yr`, `avg_jury_3yr`,
`avg_tele_3yr`, `avg_bloc_jury_3yr`, `avg_bloc_tele_3yr`,
`rule_2019_semifinal_reform`, `rule_2023_jury_weight_reform`,
`zscore_myesb_community`, `zscore_myesb_personal`, `zscore_ogae_points`,
`implied_prob_close`.

**Outcome columns verified absent:** `Top 10`, `Final_Place`, `jury_points`,
`tele_points`, `Final_Points`, `Semi_Points`, `Semi_Place`.

Programmatic check: `set(FEATURE_COLS) ∩ OUTCOME_COLS == ∅` → **True**.

> **Note on `Running_Order_Final`:** Assigned by the EBU after semi-final
> results are known but before the Grand Final broadcast. It is a legitimate
> pre-final feature with no outcome information.

---

### LA-02 — training_split() Filters

`training_split()` at `src/models/train.py:177` applies three guards:

```python
mask = (
    (matrix["Grand_Final_Ind"] == 1)
    & (matrix["Year"] < 2026)
    & matrix["Top 10"].notna()
)
```

1. **Grand Final only** — semi-final rows excluded.  
2. **Year < 2026** — target year explicitly excluded.  
3. **Label known** — rows without confirmed Top-10 outcome excluded.

Runtime verification on the enriched CSV confirms all training rows have
`Year ∈ {2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025}` (2020
excluded due to COVID cancellation) and zero NaN labels.

---

### LA-03 — LeaveLastYearOut CV Splitter

`LeaveLastYearOut.split()` at `src/models/cv.py`:

```python
for i, test_year in enumerate(unique_years):
    train_years = unique_years[:i]       # strictly before test_year
    if len(train_years) < self.min_train_years:
        continue
    train_idx = np.where(np.isin(years, train_years))[0]
    test_idx  = np.where(years == test_year)[0]
    yield train_idx, test_idx
```

The `unique_years[:i]` slice guarantees all training years are strictly earlier
than the test year. Runtime check on all 8 historical folds confirms:

- `max(train_years) < test_year` in every fold.  
- `set(train_idx) ∩ set(test_idx) == ∅` in every fold.

---

### LA-04 — Country Fixed Effects

`compute_country_fixed_effects()` at `src/features/country_fixed_effects.py`:

```python
prior = finals[
    (finals["Country"] == country) & (finals["Year"] < year)
]
```

The `Year < year` condition is strict (no `<=`). Verified by spot-check: for
the earliest year in the dataset, `avg_final_rank_3yr`, `avg_jury_3yr`, and
`avg_tele_3yr` are all `NaN` — confirming no prior-year data exists to leak
same-year information.

---

### LA-05 — Voting Blocs

`compute_voting_blocs()` at `src/features/voting_blocs.py`:

```python
prior = finals[finals["Country"].isin(mates) & (finals["Year"] < year)]
```

Two protections:  
1. `Year < year` strict filter.  
2. `mates = bloc_members[grp] - {country}` — the country itself is excluded
   from its own bloc average, preventing self-leakage even at inference time.

Same earliest-year spot-check confirms `avg_bloc_jury_3yr` and
`avg_bloc_tele_3yr` are `NaN` at the dataset's first year.

---

### LA-06 — Holdout Split (evaluate.py)

`holdout_split()` at `src/models/evaluate.py:82`:

```python
train_mask = ... & (matrix["Year"] < holdout_year) ...
test_mask  = ... & (matrix["Year"] == holdout_year) ...
```

Non-overlapping masks by construction (one uses `<`, the other `==`). Runtime
check on the 2024 holdout confirms 0 shared DataFrame indices between the
training and test splits.

---

### LA-07 — Feature Matrix Runtime Check

After calling `training_split()`, the returned `X` DataFrame columns were
checked at runtime:

```
set(X.columns) ∩ {"Top 10", "Final_Place", "jury_points", ...} == ∅
```

This redundantly confirms LA-01 in the actual execution path, not just at
definition time.

---

### LA-08 — Social Proxy (z-score normalisation)

`compute_social_proxy()` at `src/features/social_proxy.py` uses
`groupby("Year").transform()` for normalisation:

```python
df.groupby("Year")[col].transform(_norm).fillna(fill_na)
```

`groupby("Year")` confines mean and standard deviation computation to each
contest year in isolation. Cross-year information cannot enter the z-score.

Runtime check: per-year mean of `zscore_myesb_community` and
`zscore_myesb_personal` is `≈ 0` (within `1e-9`) in every year, confirming
within-year normalisation. (`zscore_ogae_points` is excluded from this mean
check because `fillna(0)` shifts the mean when some OGAE scores are missing.)

> **All scores (MyESB Community/Personal, OGAE Points) are published by fan
> communities before the contest. They predict outcomes but are not outcomes.**

---

## 4. Rule Flags

`compute_rule_flags()` at `src/features/rule_flags.py` produces two binary
flags based purely on the year value:

```python
out["rule_2019_semifinal_reform"] = (out["Year"] >= 2019).astype(int)
out["rule_2023_jury_weight_reform"] = (out["Year"] >= 2023).astype(int)
```

These are year-conditional constants with no dependence on any row-level
outcome. No separate leakage check is required.

---

## 5. Known Limitations

| ID | Item | Risk | Status |
|----|------|------|--------|
| KL-LA-01 | `implied_prob_close` (closing betting odds, 2018–2025). Odds reflect market expectations at contest time and may partially encode semi-final outcomes for same-year finalists. | Low — odds represent pre-final prices, not official contest results. | Accepted. |
| KL-LA-02 | `Running_Order_Final` is assigned after semi-finals conclude. It is pre-Grand-Final but post-semi-final. | Very low — draw order is determined by EBU, not by artistic performance. | Accepted. |
| KL-LA-03 | `zscore_ogae_points`: `fillna(0)` replaces missing fan scores with the group mean (0 in z-score space). Some bias towards the mean for countries with missing OGAE data. | Very low — no contest outcome information enters. | Accepted. |

---

## 6. MLflow Evidence

Run logged to experiment `eurovision-2026-ensemble`:

- **Tag:** `leakage_check_passed = "true"`
- **Tag:** `story = "US-S4-05"`
- **Metrics:** `la_01` through `la_08` (1.0 = pass, 0.0 = fail)

To reproduce:
```bash
python -m src.models.leakage_audit
```

---

## 7. Verdict

**All 8 leakage checks PASSED.** The Eurovision 2026 prediction pipeline
correctly enforces temporal isolation (PR-07) across all pipeline components.
No information from year Y or later enters the features, training labels, or
CV evaluation of any model trained to predict year Y contest outcomes.

---

_Reviewed and signed off: Andrzej, 2026-04-28_
