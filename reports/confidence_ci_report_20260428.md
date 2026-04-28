# US-S4-03 Bootstrap CI — Test & Prediction Report
Generated: 2026-04-28  
Branch: `develop` (merged from `feature/US-S4-03-confidence`)  
Commit: refactor(models): change CI levels from 80%/95% to 80%/50%

---

## 1. Test Suite

```
platform win32 — Python 3.13.5, pytest 9.0.3
collected 139 items

tests/test_confidence.py::TestBootstrapProba::test_output_shape                      PASSED
tests/test_confidence.py::TestBootstrapProba::test_probabilities_in_unit_interval    PASSED
tests/test_confidence.py::TestBootstrapProba::test_deterministic_with_same_seed      PASSED
tests/test_confidence.py::TestBootstrapProba::test_different_seeds_differ            PASSED
tests/test_confidence.py::TestBootstrapProba::test_single_class_bootstrap_skipped    PASSED
tests/test_confidence.py::TestComputeCI::test_columns_present                        PASSED
tests/test_confidence.py::TestComputeCI::test_row_count                              PASSED
tests/test_confidence.py::TestComputeCI::test_sorted_by_prob_mean_descending         PASSED
tests/test_confidence.py::TestComputeCI::test_ci_ordering                            PASSED
tests/test_confidence.py::TestComputeCI::test_degenerate_constant_proba              PASSED
tests/test_confidence.py::TestComputeCI::test_single_country                         PASSED
tests/test_confidence.py::TestComputeCI::test_probabilities_in_unit_interval         PASSED
tests/test_confidence.py::TestConfidenceIntegration::test_returns_both_models        PASSED
tests/test_confidence.py::TestConfidenceIntegration::test_output_csvs_written        PASSED
tests/test_confidence.py::TestConfidenceIntegration::test_meta_json_written          PASSED
tests/test_confidence.py::TestConfidenceIntegration::test_csv_has_expected_columns   PASSED
tests/test_confidence.py::TestConfidenceIntegration::test_target_year_not_in_training PASSED
tests/test_confidence.py::TestConfidenceIntegration::test_invalid_target_year_raises PASSED
... (121 further tests across test_cv, test_train, test_evaluate, test_genre,
     test_rule_flags, test_social_proxy, test_voting_blocs,
     test_country_fixed_effects — all PASSED)

==================== 139 passed, 74 warnings in 24.40s ====================
```

---

## 2. Real Pipeline Run — 2026 Predictions

```
python -m src.models.confidence --n-bootstrap 1000

Bootstrap CI  n=1000  seed=42
Train years   : [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
Target year   : 2026  (35 entries)
Features      : 23
```

### 2a. XGBoost — Top-10 probability with 80 % and 50 % CI

Best params: `n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`

| # | Country | Mean prob | 80% CI lo | 80% CI hi | 50% CI lo | 50% CI hi |
|---|---------|----------:|----------:|----------:|----------:|----------:|
| 1 | Sweden | 0.708 | 0.482 | 0.878 | 0.621 | 0.830 |
| 2 | Greece | 0.658 | 0.402 | 0.867 | 0.558 | 0.793 |
| 3 | Bulgaria | 0.584 | 0.335 | 0.799 | 0.467 | 0.716 |
| 4 | Ukraine | 0.579 | 0.321 | 0.807 | 0.457 | 0.718 |
| 5 | Croatia | 0.517 | 0.266 | 0.757 | 0.379 | 0.664 |
| 6 | Israel | 0.496 | 0.253 | 0.734 | 0.363 | 0.636 |
| 7 | Azerbaijan | 0.468 | 0.205 | 0.717 | 0.324 | 0.610 |
| 8 | France | 0.460 | 0.206 | 0.720 | 0.314 | 0.598 |
| 9 | Finland | 0.453 | 0.214 | 0.703 | 0.308 | 0.592 |
| 10 | Italy | 0.428 | 0.177 | 0.697 | 0.265 | 0.573 |
| 11 | Australia | 0.394 | 0.150 | 0.656 | 0.244 | 0.528 |
| 12 | Switzerland | 0.393 | 0.171 | 0.643 | 0.255 | 0.523 |
| 13 | Austria | 0.378 | 0.132 | 0.643 | 0.225 | 0.523 |
| 14 | Cyprus | 0.367 | 0.158 | 0.600 | 0.234 | 0.493 |
| 15 | Estonia | 0.348 | 0.134 | 0.618 | 0.212 | 0.462 |
| 16 | Portugal | 0.345 | 0.134 | 0.590 | 0.208 | 0.460 |
| 17 | Georgia | 0.345 | 0.157 | 0.562 | 0.224 | 0.449 |
| 18 | Romania | 0.338 | 0.139 | 0.578 | 0.203 | 0.445 |
| 19 | Malta | 0.333 | 0.117 | 0.578 | 0.195 | 0.459 |
| 20 | United Kingdom | 0.322 | 0.112 | 0.567 | 0.183 | 0.430 |
| 21 | Montenegro | 0.301 | 0.101 | 0.536 | 0.167 | 0.409 |
| 22 | Moldova | 0.300 | 0.125 | 0.520 | 0.179 | 0.394 |
| 23 | Serbia | 0.292 | 0.111 | 0.515 | 0.169 | 0.390 |
| 24 | Belgium | 0.284 | 0.101 | 0.510 | 0.159 | 0.381 |
| 25 | Armenia | 0.272 | 0.087 | 0.505 | 0.146 | 0.373 |
| 26 | Czech Republic | 0.262 | 0.082 | 0.494 | 0.140 | 0.369 |
| 27 | Luxembourg | 0.253 | 0.082 | 0.461 | 0.139 | 0.342 |
| 28 | Latvia | 0.209 | 0.063 | 0.401 | 0.102 | 0.282 |
| 29 | Poland | 0.195 | 0.072 | 0.367 | 0.102 | 0.257 |
| 30 | San Marino | 0.193 | 0.069 | 0.358 | 0.105 | 0.254 |
| 31 | Denmark | 0.186 | 0.069 | 0.344 | 0.100 | 0.245 |
| 32 | Albania | 0.184 | 0.067 | 0.344 | 0.104 | 0.239 |
| 33 | Germany | 0.177 | 0.058 | 0.330 | 0.095 | 0.232 |
| 34 | Lithuania | 0.139 | 0.052 | 0.258 | 0.075 | 0.175 |
| 35 | Norway | 0.124 | 0.041 | 0.235 | 0.059 | 0.164 |

### 2b. LightGBM — Top-10 probability with 80 % and 50 % CI

Best params: `n_estimators=100, num_leaves=31, learning_rate=0.05, subsample=0.8, min_child_samples=5`

| # | Country | Mean prob | 80% CI lo | 80% CI hi | 50% CI lo | 50% CI hi |
|---|---------|----------:|----------:|----------:|----------:|----------:|
| 1 | Sweden | 0.760 | 0.227 | 0.987 | 0.653 | 0.971 |
| 2 | Greece | 0.691 | 0.104 | 0.982 | 0.453 | 0.960 |
| 3 | Ukraine | 0.570 | 0.057 | 0.976 | 0.235 | 0.912 |
| 4 | Bulgaria | 0.479 | 0.047 | 0.940 | 0.140 | 0.824 |
| 5 | Croatia | 0.451 | 0.030 | 0.932 | 0.101 | 0.786 |
| 6 | Israel | 0.417 | 0.018 | 0.940 | 0.078 | 0.775 |
| 7 | France | 0.407 | 0.016 | 0.919 | 0.076 | 0.718 |
| 8 | Italy | 0.406 | 0.011 | 0.930 | 0.052 | 0.741 |
| 9 | Switzerland | 0.357 | 0.017 | 0.874 | 0.062 | 0.643 |
| 10 | Azerbaijan | 0.339 | 0.016 | 0.851 | 0.053 | 0.609 |
| 11 | Finland | 0.316 | 0.014 | 0.852 | 0.044 | 0.541 |
| 12 | Australia | 0.310 | 0.011 | 0.861 | 0.039 | 0.539 |
| 13 | Georgia | 0.282 | 0.014 | 0.771 | 0.043 | 0.469 |
| 14 | Cyprus | 0.278 | 0.012 | 0.780 | 0.037 | 0.449 |
| 15 | Serbia | 0.268 | 0.012 | 0.765 | 0.033 | 0.452 |
| 16 | Romania | 0.262 | 0.010 | 0.779 | 0.030 | 0.434 |
| 17 | Austria | 0.243 | 0.006 | 0.743 | 0.020 | 0.386 |
| 18 | Portugal | 0.242 | 0.008 | 0.738 | 0.024 | 0.404 |
| 19 | Malta | 0.238 | 0.007 | 0.734 | 0.022 | 0.359 |
| 20 | Estonia | 0.231 | 0.006 | 0.753 | 0.022 | 0.367 |
| 21 | Armenia | 0.219 | 0.007 | 0.718 | 0.018 | 0.355 |
| 22 | Luxembourg | 0.212 | 0.005 | 0.705 | 0.014 | 0.345 |
| 23 | Czech Republic | 0.201 | 0.005 | 0.667 | 0.014 | 0.301 |
| 24 | Moldova | 0.200 | 0.006 | 0.674 | 0.018 | 0.272 |
| 25 | Belgium | 0.194 | 0.006 | 0.662 | 0.015 | 0.285 |
| 26 | United Kingdom | 0.192 | 0.006 | 0.635 | 0.018 | 0.268 |
| 27 | Montenegro | 0.191 | 0.004 | 0.675 | 0.012 | 0.257 |
| 28 | Poland | 0.128 | 0.004 | 0.437 | 0.009 | 0.123 |
| 29 | San Marino | 0.108 | 0.003 | 0.349 | 0.007 | 0.099 |
| 30 | Latvia | 0.095 | 0.002 | 0.304 | 0.005 | 0.074 |
| 31 | Albania | 0.091 | 0.004 | 0.275 | 0.008 | 0.070 |
| 32 | Denmark | 0.081 | 0.003 | 0.240 | 0.006 | 0.062 |
| 33 | Germany | 0.074 | 0.003 | 0.200 | 0.006 | 0.060 |
| 34 | Norway | 0.065 | 0.003 | 0.165 | 0.005 | 0.051 |
| 35 | Lithuania | 0.048 | 0.002 | 0.118 | 0.004 | 0.027 |

---

## 3. Observations

**Model agreement (top-5):** Both models agree on Sweden #1 and Greece #2 with high
confidence. XGBoost and LightGBM agree on 4/5 top entries (Sweden, Greece, Bulgaria,
Ukraine, Croatia).

**CI width:** LightGBM produces much wider 80 % intervals (avg ~0.76) vs XGBoost
(avg ~0.40), reflecting higher variance in the LGBM bootstrap fits across the
small training set (9 years). XGBoost intervals are tighter and more stable.

**Caution — semi-final filter not yet applied:** These probabilities cover all 35
participants. Once semi-finals are held, entries eliminated before the Grand Final
should be removed. Confidence module accepts `--target-year` and processes whatever
rows exist in the enriched CSV for that year.

**Next step:** Ensemble blending (C-03) to produce a single merged probability
from XGBoost + LightGBM before final predictions are published.
