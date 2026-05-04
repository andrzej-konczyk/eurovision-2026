# Semi-Final Backtest Report - 2022 / 2023 / 2024 / 2025

*Generated: 2026-05-04  |  n_bootstrap = 1000  |  K = 10 per semi*

Target: `Grand_Final_Ind` (qualified from semi = 1).  Features `Running_Order_Final` and `implied_prob_close` excluded. `implied_prob_semi` is used for semi-final qualification market odds.

## Qualification Accuracy

| Year | Model | SF1 | SF2 | Overall | KPI >=70% SF1 | KPI >=70% SF2 |
|------|-------|-----|-----|---------|--------------|--------------|
| 2022 | XGB | 90% (9/10) | 100% (10/10) | 95% | PASS | PASS |
| 2022 | LGBM | 90% (9/10) | 100% (10/10) | 95% | PASS | PASS |
| 2023 | XGB | 100% (10/10) | 100% (10/10) | 100% | PASS | PASS |
| 2023 | LGBM | 100% (10/10) | 100% (10/10) | 100% | PASS | PASS |
| 2024 | XGB | 100% (10/10) | 90% (9/10) | 95% | PASS | PASS |
| 2024 | LGBM | 100% (10/10) | 90% (9/10) | 95% | PASS | PASS |
| 2025 | XGB | 100% (10/10) | 100% (10/10) | 100% | PASS | PASS |
| 2025 | LGBM | 90% (9/10) | 100% (10/10) | 95% | PASS | PASS |

## CI Calibration — 80% CI Empirical Coverage

| Year | XGB | LGBM | KPI >=80% |
|------|-----|------|----------|
| 2022 | 91% | 97% | PASS |
| 2023 | 97% | 100% | PASS |
| 2024 | 97% | 97% | PASS |
| 2025 | 100% | 100% | PASS |

## Per-Year Country Detail

### 2022 (train: [2016, 2017, 2018, 2019, 2021])

**XGB**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Netherlands | SF1 | Q | 0.980 | 0.972 | 0.987 | v |
| Sweden | SF2 | Q | 0.975 | 0.963 | 0.985 | v |
| Ukraine | SF1 | Q | 0.966 | 0.942 | 0.982 | v |
| Greece | SF1 | Q | 0.966 | 0.944 | 0.982 | v |
| Serbia | SF2 | Q | 0.965 | 0.940 | 0.983 | v |
| Czech Republic | SF2 | Q | 0.957 | 0.927 | 0.981 | v |
| Norway | SF1 | Q | 0.953 | 0.925 | 0.977 | v |
| Portugal | SF1 | Q | 0.952 | 0.918 | 0.978 | v |
| Australia | SF2 | Q | 0.947 | 0.905 | 0.979 | v |
| Poland | SF2 | Q | 0.936 | 0.874 | 0.976 | v |
| Armenia | SF1 | Q | 0.901 | 0.822 | 0.961 | v |
| Belgium | SF2 | Q | 0.886 | 0.798 | 0.950 | v |
| Moldova | SF1 | Q | 0.872 | 0.767 | 0.948 | v |
| Estonia | SF2 | Q | 0.861 | 0.747 | 0.946 | v |
| Finland | SF2 | Q | 0.859 | 0.748 | 0.942 | v |
| Lithuania | SF1 | Q | 0.858 | 0.741 | 0.944 | v |
| Romania | SF2 | Q | 0.851 | 0.719 | 0.944 | v |
| Austria | SF1 |  | 0.770 | 0.581 | 0.920 | x |
| Albania | SF1 |  | 0.724 | 0.515 | 0.899 | x |
| Iceland | SF1 | Q | 0.724 | 0.506 | 0.897 | v |
| Cyprus | SF2 |  | 0.707 | 0.474 | 0.911 | v |
| Azerbaijan | SF2 | Q | 0.680 | 0.432 | 0.888 | v |
| Ireland | SF2 |  | 0.533 | 0.317 | 0.735 | v |
| Switzerland | SF1 | Q | 0.177 | 0.090 | 0.279 | x |
| San Marino | SF2 |  | 0.108 | 0.026 | 0.215 | v |
| Slovenia | SF1 |  | 0.080 | 0.028 | 0.145 | v |
| Israel | SF2 |  | 0.077 | 0.031 | 0.140 | v |
| Croatia | SF1 |  | 0.060 | 0.021 | 0.108 | v |
| Georgia | SF2 |  | 0.039 | 0.022 | 0.059 | v |
| Montenegro | SF2 |  | 0.039 | 0.022 | 0.059 | v |
| Latvia | SF1 |  | 0.038 | 0.018 | 0.062 | v |
| Bulgaria | SF1 |  | 0.037 | 0.021 | 0.058 | v |
| North Macedonia | SF2 |  | 0.037 | 0.023 | 0.054 | v |
| Malta | SF2 |  | 0.033 | 0.021 | 0.048 | v |
| Denmark | SF1 |  | 0.028 | 0.017 | 0.042 | v |

**LGBM**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Netherlands | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Sweden | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Ukraine | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Norway | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Greece | SF1 | Q | 0.999 | 1.000 | 1.000 | v |
| Serbia | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Australia | SF2 | Q | 0.998 | 0.999 | 1.000 | v |
| Portugal | SF1 | Q | 0.998 | 1.000 | 1.000 | v |
| Czech Republic | SF2 | Q | 0.997 | 0.999 | 1.000 | v |
| Armenia | SF1 | Q | 0.996 | 0.999 | 1.000 | v |
| Belgium | SF2 | Q | 0.995 | 0.999 | 1.000 | v |
| Moldova | SF1 | Q | 0.993 | 0.997 | 1.000 | v |
| Romania | SF2 | Q | 0.989 | 0.995 | 1.000 | v |
| Lithuania | SF1 | Q | 0.989 | 0.996 | 1.000 | v |
| Estonia | SF2 | Q | 0.989 | 0.994 | 1.000 | v |
| Finland | SF2 | Q | 0.987 | 0.995 | 1.000 | v |
| Iceland | SF1 | Q | 0.942 | 0.919 | 1.000 | v |
| Poland | SF2 | Q | 0.940 | 0.990 | 1.000 | v |
| Azerbaijan | SF2 | Q | 0.932 | 0.843 | 1.000 | v |
| Austria | SF1 |  | 0.582 | 0.005 | 0.999 | v |
| Cyprus | SF2 |  | 0.391 | 0.002 | 0.995 | v |
| Albania | SF1 |  | 0.369 | 0.002 | 0.994 | v |
| Ireland | SF2 |  | 0.312 | 0.001 | 0.978 | v |
| San Marino | SF2 |  | 0.069 | 0.000 | 0.176 | v |
| Slovenia | SF1 |  | 0.035 | 0.000 | 0.014 | v |
| Switzerland | SF1 | Q | 0.027 | 0.000 | 0.005 | x |
| Latvia | SF1 |  | 0.016 | 0.000 | 0.001 | v |
| Croatia | SF1 |  | 0.011 | 0.000 | 0.001 | v |
| Israel | SF2 |  | 0.006 | 0.000 | 0.001 | v |
| Bulgaria | SF1 |  | 0.005 | 0.000 | 0.000 | v |
| Montenegro | SF2 |  | 0.004 | 0.000 | 0.000 | v |
| Georgia | SF2 |  | 0.001 | 0.000 | 0.000 | v |
| Malta | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| North Macedonia | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Denmark | SF1 |  | 0.000 | 0.000 | 0.000 | v |

### 2023 (train: [2016, 2017, 2018, 2019, 2021, 2022])

**XGB**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Finland | SF1 | Q | 0.980 | 0.971 | 0.987 | v |
| Sweden | SF1 | Q | 0.977 | 0.966 | 0.986 | v |
| Austria | SF2 | Q | 0.977 | 0.963 | 0.987 | v |
| Slovenia | SF2 | Q | 0.976 | 0.963 | 0.987 | v |
| Israel | SF1 | Q | 0.975 | 0.965 | 0.986 | v |
| Czech Republic | SF1 | Q | 0.974 | 0.963 | 0.984 | v |
| Norway | SF1 | Q | 0.972 | 0.960 | 0.986 | v |
| Armenia | SF2 | Q | 0.969 | 0.949 | 0.983 | v |
| Serbia | SF1 | Q | 0.958 | 0.928 | 0.980 | v |
| Portugal | SF1 | Q | 0.934 | 0.890 | 0.968 | v |
| Australia | SF2 | Q | 0.930 | 0.882 | 0.966 | v |
| Cyprus | SF2 | Q | 0.924 | 0.873 | 0.965 | v |
| Moldova | SF1 | Q | 0.923 | 0.868 | 0.966 | v |
| Estonia | SF2 | Q | 0.903 | 0.829 | 0.958 | v |
| Switzerland | SF1 | Q | 0.900 | 0.822 | 0.957 | v |
| Belgium | SF2 | Q | 0.896 | 0.823 | 0.954 | v |
| Lithuania | SF2 | Q | 0.827 | 0.681 | 0.930 | v |
| Croatia | SF1 | Q | 0.761 | 0.540 | 0.912 | v |
| Georgia | SF2 |  | 0.372 | 0.190 | 0.584 | v |
| Poland | SF2 | Q | 0.372 | 0.176 | 0.580 | v |
| Latvia | SF1 |  | 0.293 | 0.145 | 0.466 | v |
| Albania | SF2 | Q | 0.290 | 0.127 | 0.471 | x |
| Netherlands | SF1 |  | 0.080 | 0.031 | 0.145 | v |
| Romania | SF2 |  | 0.039 | 0.021 | 0.063 | v |
| Azerbaijan | SF1 |  | 0.035 | 0.019 | 0.056 | v |
| Iceland | SF2 |  | 0.028 | 0.016 | 0.044 | v |
| San Marino | SF2 |  | 0.028 | 0.017 | 0.042 | v |
| Malta | SF1 |  | 0.027 | 0.017 | 0.040 | v |
| Denmark | SF2 |  | 0.023 | 0.014 | 0.035 | v |
| Greece | SF2 |  | 0.023 | 0.015 | 0.032 | v |
| Ireland | SF1 |  | 0.018 | 0.013 | 0.024 | v |

**LGBM**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Finland | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Czech Republic | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Austria | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Slovenia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Armenia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Sweden | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Israel | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Serbia | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Portugal | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Norway | SF1 | Q | 0.999 | 1.000 | 1.000 | v |
| Switzerland | SF1 | Q | 0.999 | 0.999 | 1.000 | v |
| Belgium | SF2 | Q | 0.998 | 0.999 | 1.000 | v |
| Moldova | SF1 | Q | 0.998 | 0.999 | 1.000 | v |
| Australia | SF2 | Q | 0.997 | 0.999 | 1.000 | v |
| Cyprus | SF2 | Q | 0.994 | 0.998 | 1.000 | v |
| Estonia | SF2 | Q | 0.992 | 0.998 | 1.000 | v |
| Lithuania | SF2 | Q | 0.979 | 0.991 | 1.000 | v |
| Croatia | SF1 | Q | 0.885 | 0.441 | 1.000 | v |
| Poland | SF2 | Q | 0.625 | 0.001 | 0.998 | v |
| Albania | SF2 | Q | 0.531 | 0.000 | 0.998 | v |
| Georgia | SF2 |  | 0.314 | 0.000 | 0.983 | v |
| Latvia | SF1 |  | 0.118 | 0.000 | 0.588 | v |
| Netherlands | SF1 |  | 0.007 | 0.000 | 0.002 | v |
| Romania | SF2 |  | 0.006 | 0.000 | 0.000 | v |
| Azerbaijan | SF1 |  | 0.001 | 0.000 | 0.000 | v |
| Iceland | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Denmark | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| San Marino | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Malta | SF1 |  | 0.000 | 0.000 | 0.000 | v |
| Greece | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Ireland | SF1 |  | 0.000 | 0.000 | 0.000 | v |

### 2024 (train: [2016, 2017, 2018, 2019, 2021, 2022, 2023])

**XGB**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Lithuania | SF1 | Q | 0.984 | 0.978 | 0.989 | v |
| Croatia | SF1 | Q | 0.982 | 0.973 | 0.989 | v |
| Norway | SF2 | Q | 0.982 | 0.974 | 0.988 | v |
| Austria | SF2 | Q | 0.981 | 0.972 | 0.988 | v |
| Ukraine | SF1 | Q | 0.979 | 0.971 | 0.987 | v |
| Switzerland | SF2 | Q | 0.966 | 0.941 | 0.985 | v |
| Greece | SF2 | Q | 0.964 | 0.940 | 0.982 | v |
| Serbia | SF1 | Q | 0.961 | 0.938 | 0.980 | v |
| Georgia | SF2 | Q | 0.953 | 0.926 | 0.974 | v |
| Armenia | SF2 | Q | 0.945 | 0.914 | 0.972 | v |
| Cyprus | SF1 | Q | 0.945 | 0.913 | 0.969 | v |
| Slovenia | SF1 | Q | 0.942 | 0.907 | 0.970 | v |
| Estonia | SF2 | Q | 0.936 | 0.898 | 0.967 | v |
| Israel | SF2 | Q | 0.930 | 0.880 | 0.968 | v |
| Luxembourg | SF1 | Q | 0.926 | 0.871 | 0.968 | v |
| Belgium | SF2 |  | 0.878 | 0.794 | 0.947 | x |
| Netherlands | SF2 | Q | 0.874 | 0.790 | 0.945 | v |
| Ireland | SF1 | Q | 0.855 | 0.740 | 0.951 | v |
| Finland | SF1 | Q | 0.614 | 0.385 | 0.810 | v |
| Portugal | SF1 | Q | 0.557 | 0.341 | 0.756 | v |
| Latvia | SF2 | Q | 0.347 | 0.192 | 0.516 | v |
| Denmark | SF2 |  | 0.221 | 0.098 | 0.373 | v |
| Czech Republic | SF2 |  | 0.188 | 0.086 | 0.325 | v |
| Poland | SF1 |  | 0.135 | 0.059 | 0.230 | v |
| Azerbaijan | SF1 |  | 0.043 | 0.020 | 0.071 | v |
| Moldova | SF1 |  | 0.036 | 0.018 | 0.058 | v |
| Australia | SF1 |  | 0.033 | 0.019 | 0.050 | v |
| San Marino | SF2 |  | 0.033 | 0.019 | 0.048 | v |
| Malta | SF2 |  | 0.031 | 0.018 | 0.045 | v |
| Albania | SF2 |  | 0.026 | 0.016 | 0.039 | v |
| Iceland | SF1 |  | 0.024 | 0.015 | 0.036 | v |

**LGBM**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Lithuania | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Norway | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Georgia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Serbia | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Armenia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Estonia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Croatia | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Cyprus | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Slovenia | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Greece | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Ukraine | SF1 | Q | 0.999 | 1.000 | 1.000 | v |
| Austria | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Switzerland | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Luxembourg | SF1 | Q | 0.998 | 1.000 | 1.000 | v |
| Israel | SF2 | Q | 0.997 | 0.999 | 1.000 | v |
| Belgium | SF2 |  | 0.980 | 0.991 | 1.000 | x |
| Netherlands | SF2 | Q | 0.979 | 0.992 | 1.000 | v |
| Ireland | SF1 | Q | 0.903 | 0.724 | 1.000 | v |
| Finland | SF1 | Q | 0.809 | 0.018 | 1.000 | v |
| Portugal | SF1 | Q | 0.801 | 0.013 | 1.000 | v |
| Latvia | SF2 | Q | 0.386 | 0.000 | 0.997 | v |
| Czech Republic | SF2 |  | 0.045 | 0.000 | 0.060 | v |
| Denmark | SF2 |  | 0.035 | 0.000 | 0.024 | v |
| Poland | SF1 |  | 0.019 | 0.000 | 0.006 | v |
| Albania | SF2 |  | 0.003 | 0.000 | 0.000 | v |
| Azerbaijan | SF1 |  | 0.003 | 0.000 | 0.000 | v |
| Malta | SF2 |  | 0.002 | 0.000 | 0.000 | v |
| Moldova | SF1 |  | 0.002 | 0.000 | 0.000 | v |
| Australia | SF1 |  | 0.000 | 0.000 | 0.000 | v |
| San Marino | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Iceland | SF1 |  | 0.000 | 0.000 | 0.000 | v |

### 2025 (train: [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024])

**XGB**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Finland | SF2 | Q | 0.988 | 0.984 | 0.992 | v |
| Greece | SF2 | Q | 0.987 | 0.982 | 0.991 | v |
| Sweden | SF1 | Q | 0.986 | 0.980 | 0.990 | v |
| Norway | SF1 | Q | 0.986 | 0.980 | 0.990 | v |
| Austria | SF2 | Q | 0.983 | 0.974 | 0.990 | v |
| Albania | SF1 | Q | 0.979 | 0.969 | 0.987 | v |
| Poland | SF1 | Q | 0.973 | 0.959 | 0.985 | v |
| Denmark | SF2 | Q | 0.973 | 0.961 | 0.986 | v |
| Lithuania | SF2 | Q | 0.972 | 0.959 | 0.983 | v |
| Ukraine | SF1 | Q | 0.965 | 0.947 | 0.978 | v |
| Malta | SF2 | Q | 0.958 | 0.950 | 0.989 | v |
| Netherlands | SF1 | Q | 0.951 | 0.917 | 0.989 | v |
| Latvia | SF2 | Q | 0.947 | 0.915 | 0.972 | v |
| Luxembourg | SF2 | Q | 0.933 | 0.885 | 0.968 | v |
| San Marino | SF1 | Q | 0.918 | 0.866 | 0.960 | v |
| Israel | SF2 | Q | 0.899 | 0.858 | 0.973 | v |
| Iceland | SF1 | Q | 0.886 | 0.768 | 0.960 | v |
| Estonia | SF1 | Q | 0.871 | 0.761 | 0.967 | v |
| Portugal | SF1 | Q | 0.810 | 0.637 | 0.931 | v |
| Armenia | SF2 | Q | 0.635 | 0.394 | 0.842 | v |
| Australia | SF2 |  | 0.454 | 0.261 | 0.652 | v |
| Czech Republic | SF2 |  | 0.381 | 0.207 | 0.602 | v |
| Belgium | SF1 |  | 0.281 | 0.130 | 0.470 | v |
| Cyprus | SF1 |  | 0.055 | 0.025 | 0.096 | v |
| Ireland | SF2 |  | 0.043 | 0.021 | 0.070 | v |
| Croatia | SF1 |  | 0.033 | 0.019 | 0.052 | v |
| Georgia | SF2 |  | 0.028 | 0.016 | 0.045 | v |
| Serbia | SF2 |  | 0.028 | 0.016 | 0.043 | v |
| Slovenia | SF1 |  | 0.025 | 0.016 | 0.036 | v |
| Azerbaijan | SF1 |  | 0.021 | 0.014 | 0.030 | v |
| Montenegro | SF2 |  | 0.020 | 0.013 | 0.029 | v |

**LGBM**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Finland | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Norway | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Greece | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Sweden | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Poland | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Albania | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Austria | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Ukraine | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Luxembourg | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| San Marino | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Lithuania | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Latvia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Iceland | SF1 | Q | 0.992 | 1.000 | 1.000 | v |
| Denmark | SF2 | Q | 0.990 | 1.000 | 1.000 | v |
| Portugal | SF1 | Q | 0.981 | 0.998 | 1.000 | v |
| Netherlands | SF1 | Q | 0.951 | 0.996 | 1.000 | v |
| Malta | SF2 | Q | 0.942 | 0.996 | 1.000 | v |
| Armenia | SF2 | Q | 0.902 | 0.595 | 1.000 | v |
| Israel | SF2 | Q | 0.868 | 0.057 | 1.000 | v |
| Estonia | SF1 | Q | 0.847 | 0.001 | 1.000 | v |
| Australia | SF2 |  | 0.197 | 0.000 | 0.900 | v |
| Belgium | SF1 |  | 0.098 | 0.000 | 0.393 | v |
| Czech Republic | SF2 |  | 0.094 | 0.000 | 0.344 | v |
| Cyprus | SF1 |  | 0.004 | 0.000 | 0.001 | v |
| Ireland | SF2 |  | 0.002 | 0.000 | 0.000 | v |
| Georgia | SF2 |  | 0.002 | 0.000 | 0.000 | v |
| Croatia | SF1 |  | 0.001 | 0.000 | 0.000 | v |
| Serbia | SF2 |  | 0.001 | 0.000 | 0.000 | v |
| Azerbaijan | SF1 |  | 0.001 | 0.000 | 0.000 | v |
| Montenegro | SF2 |  | 0.001 | 0.000 | 0.000 | v |
| Slovenia | SF1 |  | 0.001 | 0.000 | 0.000 | v |
