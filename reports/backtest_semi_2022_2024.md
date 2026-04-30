# Semi-Final Backtest Report — 2022 / 2023 / 2024

*Generated: 2026-04-30  |  n_bootstrap = 1000  |  K = 10 per semi*

Target: `Grand_Final_Ind` (qualified from semi = 1).  Feature `Running_Order_Final` excluded (not known during semi).

## Qualification Accuracy

| Year | Model | SF1 | SF2 | Overall | KPI >=70% SF1 | KPI >=70% SF2 |
|------|-------|-----|-----|---------|--------------|--------------|
| 2022 | XGB | 90% (9/10) | 100% (10/10) | 95% | PASS | PASS |
| 2022 | LGBM | 100% (10/10) | 100% (10/10) | 100% | PASS | PASS |
| 2023 | XGB | 100% (10/10) | 100% (10/10) | 100% | PASS | PASS |
| 2023 | LGBM | 100% (10/10) | 100% (10/10) | 100% | PASS | PASS |
| 2024 | XGB | 100% (10/10) | 90% (9/10) | 95% | PASS | PASS |
| 2024 | LGBM | 100% (10/10) | 90% (9/10) | 95% | PASS | PASS |

## CI Calibration — 80% CI Empirical Coverage

| Year | XGB | LGBM | KPI >=80% |
|------|-----|------|----------|
| 2022 | 91% | 100% | PASS |
| 2023 | 100% | 100% | PASS |
| 2024 | 97% | 97% | PASS |

## Per-Year Country Detail

### 2022 (train: [2016, 2017, 2018, 2019, 2021])

**XGB**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Netherlands | SF1 | Q | 0.979 | 0.969 | 0.988 | v |
| Sweden | SF2 | Q | 0.972 | 0.957 | 0.984 | v |
| Serbia | SF2 | Q | 0.964 | 0.935 | 0.983 | v |
| Czech Republic | SF2 | Q | 0.964 | 0.940 | 0.983 | v |
| Portugal | SF1 | Q | 0.962 | 0.938 | 0.981 | v |
| Ukraine | SF1 | Q | 0.961 | 0.936 | 0.981 | v |
| Australia | SF2 | Q | 0.960 | 0.931 | 0.983 | v |
| Greece | SF1 | Q | 0.952 | 0.913 | 0.982 | v |
| Norway | SF1 | Q | 0.943 | 0.905 | 0.972 | v |
| Poland | SF2 | Q | 0.936 | 0.882 | 0.974 | v |
| Belgium | SF2 | Q | 0.929 | 0.882 | 0.966 | v |
| Armenia | SF1 | Q | 0.928 | 0.873 | 0.969 | v |
| Lithuania | SF1 | Q | 0.902 | 0.823 | 0.959 | v |
| Finland | SF2 | Q | 0.899 | 0.819 | 0.958 | v |
| Estonia | SF2 | Q | 0.895 | 0.812 | 0.959 | v |
| Romania | SF2 | Q | 0.894 | 0.806 | 0.958 | v |
| Moldova | SF1 | Q | 0.824 | 0.588 | 0.951 | v |
| Iceland | SF1 | Q | 0.814 | 0.648 | 0.933 | v |
| Azerbaijan | SF2 | Q | 0.786 | 0.589 | 0.930 | v |
| Austria | SF1 |  | 0.780 | 0.597 | 0.925 | x |
| Cyprus | SF2 |  | 0.714 | 0.491 | 0.915 | v |
| Albania | SF1 |  | 0.710 | 0.504 | 0.895 | x |
| Ireland | SF2 |  | 0.512 | 0.294 | 0.719 | v |
| Switzerland | SF1 | Q | 0.263 | 0.130 | 0.432 | x |
| San Marino | SF2 |  | 0.097 | 0.025 | 0.197 | v |
| Slovenia | SF1 |  | 0.077 | 0.028 | 0.139 | v |
| Israel | SF2 |  | 0.071 | 0.029 | 0.123 | v |
| Croatia | SF1 |  | 0.053 | 0.020 | 0.100 | v |
| Georgia | SF2 |  | 0.040 | 0.022 | 0.062 | v |
| Bulgaria | SF1 |  | 0.039 | 0.021 | 0.062 | v |
| North Macedonia | SF2 |  | 0.038 | 0.022 | 0.057 | v |
| Latvia | SF1 |  | 0.036 | 0.019 | 0.059 | v |
| Montenegro | SF2 |  | 0.035 | 0.021 | 0.053 | v |
| Malta | SF2 |  | 0.035 | 0.021 | 0.052 | v |
| Denmark | SF1 |  | 0.028 | 0.017 | 0.041 | v |

**LGBM**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Ukraine | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Sweden | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Czech Republic | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Portugal | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Netherlands | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Belgium | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Australia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Norway | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Armenia | SF1 | Q | 0.999 | 1.000 | 1.000 | v |
| Serbia | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Estonia | SF2 | Q | 0.998 | 0.999 | 1.000 | v |
| Finland | SF2 | Q | 0.997 | 0.999 | 1.000 | v |
| Lithuania | SF1 | Q | 0.997 | 0.999 | 1.000 | v |
| Romania | SF2 | Q | 0.997 | 0.999 | 1.000 | v |
| Greece | SF1 | Q | 0.997 | 0.999 | 1.000 | v |
| Iceland | SF1 | Q | 0.990 | 0.998 | 1.000 | v |
| Azerbaijan | SF2 | Q | 0.988 | 0.994 | 1.000 | v |
| Poland | SF2 | Q | 0.929 | 0.916 | 1.000 | v |
| Moldova | SF1 | Q | 0.827 | 0.010 | 1.000 | v |
| Austria | SF1 |  | 0.573 | 0.007 | 0.999 | v |
| Cyprus | SF2 |  | 0.397 | 0.002 | 0.995 | v |
| Ireland | SF2 |  | 0.272 | 0.001 | 0.965 | v |
| Albania | SF1 |  | 0.266 | 0.001 | 0.977 | v |
| Switzerland | SF1 | Q | 0.253 | 0.000 | 0.995 | v |
| San Marino | SF2 |  | 0.046 | 0.000 | 0.057 | v |
| Slovenia | SF1 |  | 0.027 | 0.000 | 0.012 | v |
| Croatia | SF1 |  | 0.008 | 0.000 | 0.001 | v |
| Latvia | SF1 |  | 0.008 | 0.000 | 0.000 | v |
| Israel | SF2 |  | 0.006 | 0.000 | 0.001 | v |
| Bulgaria | SF1 |  | 0.003 | 0.000 | 0.000 | v |
| Montenegro | SF2 |  | 0.001 | 0.000 | 0.000 | v |
| Georgia | SF2 |  | 0.001 | 0.000 | 0.000 | v |
| North Macedonia | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Malta | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Denmark | SF1 |  | 0.000 | 0.000 | 0.000 | v |

### 2023 (train: [2016, 2017, 2018, 2019, 2021, 2022])

**XGB**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Austria | SF2 | Q | 0.979 | 0.968 | 0.988 | v |
| Slovenia | SF2 | Q | 0.978 | 0.966 | 0.987 | v |
| Finland | SF1 | Q | 0.977 | 0.967 | 0.985 | v |
| Sweden | SF1 | Q | 0.975 | 0.964 | 0.985 | v |
| Israel | SF1 | Q | 0.975 | 0.962 | 0.986 | v |
| Czech Republic | SF1 | Q | 0.974 | 0.961 | 0.984 | v |
| Norway | SF1 | Q | 0.973 | 0.960 | 0.983 | v |
| Armenia | SF2 | Q | 0.972 | 0.957 | 0.985 | v |
| Cyprus | SF2 | Q | 0.964 | 0.944 | 0.980 | v |
| Australia | SF2 | Q | 0.964 | 0.945 | 0.981 | v |
| Serbia | SF1 | Q | 0.963 | 0.937 | 0.981 | v |
| Estonia | SF2 | Q | 0.953 | 0.922 | 0.975 | v |
| Portugal | SF1 | Q | 0.943 | 0.906 | 0.972 | v |
| Moldova | SF1 | Q | 0.935 | 0.889 | 0.971 | v |
| Lithuania | SF2 | Q | 0.922 | 0.858 | 0.967 | v |
| Switzerland | SF1 | Q | 0.916 | 0.854 | 0.964 | v |
| Belgium | SF2 | Q | 0.827 | 0.671 | 0.944 | v |
| Croatia | SF1 | Q | 0.694 | 0.361 | 0.909 | v |
| Albania | SF2 | Q | 0.418 | 0.209 | 0.643 | v |
| Poland | SF2 | Q | 0.376 | 0.151 | 0.614 | v |
| Georgia | SF2 |  | 0.360 | 0.183 | 0.559 | v |
| Latvia | SF1 |  | 0.277 | 0.137 | 0.440 | v |
| Netherlands | SF1 |  | 0.080 | 0.032 | 0.142 | v |
| Romania | SF2 |  | 0.038 | 0.020 | 0.060 | v |
| Azerbaijan | SF1 |  | 0.034 | 0.020 | 0.056 | v |
| San Marino | SF2 |  | 0.028 | 0.017 | 0.043 | v |
| Malta | SF1 |  | 0.027 | 0.017 | 0.041 | v |
| Iceland | SF2 |  | 0.026 | 0.016 | 0.040 | v |
| Greece | SF2 |  | 0.023 | 0.015 | 0.033 | v |
| Denmark | SF2 |  | 0.022 | 0.014 | 0.033 | v |
| Ireland | SF1 |  | 0.018 | 0.013 | 0.024 | v |

**LGBM**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Finland | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Israel | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Norway | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Czech Republic | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Cyprus | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Serbia | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Portugal | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Estonia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Sweden | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Moldova | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Austria | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Switzerland | SF1 | Q | 0.999 | 0.999 | 1.000 | v |
| Slovenia | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Armenia | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Lithuania | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Australia | SF2 | Q | 0.998 | 1.000 | 1.000 | v |
| Belgium | SF2 | Q | 0.934 | 0.928 | 1.000 | v |
| Croatia | SF1 | Q | 0.824 | 0.001 | 1.000 | v |
| Albania | SF2 | Q | 0.780 | 0.000 | 1.000 | v |
| Poland | SF2 | Q | 0.699 | 0.000 | 1.000 | v |
| Georgia | SF2 |  | 0.153 | 0.000 | 0.749 | v |
| Latvia | SF1 |  | 0.070 | 0.000 | 0.173 | v |
| Netherlands | SF1 |  | 0.009 | 0.000 | 0.003 | v |
| Romania | SF2 |  | 0.003 | 0.000 | 0.000 | v |
| Azerbaijan | SF1 |  | 0.000 | 0.000 | 0.000 | v |
| San Marino | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Iceland | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Denmark | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Malta | SF1 |  | 0.000 | 0.000 | 0.000 | v |
| Greece | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Ireland | SF1 |  | 0.000 | 0.000 | 0.000 | v |

### 2024 (train: [2016, 2017, 2018, 2019, 2021, 2022, 2023])

**XGB**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Lithuania | SF1 | Q | 0.985 | 0.979 | 0.989 | v |
| Norway | SF2 | Q | 0.982 | 0.975 | 0.988 | v |
| Austria | SF2 | Q | 0.980 | 0.974 | 0.988 | v |
| Croatia | SF1 | Q | 0.979 | 0.969 | 0.988 | v |
| Ukraine | SF1 | Q | 0.978 | 0.968 | 0.986 | v |
| Greece | SF2 | Q | 0.967 | 0.945 | 0.983 | v |
| Serbia | SF1 | Q | 0.965 | 0.946 | 0.981 | v |
| Switzerland | SF2 | Q | 0.960 | 0.931 | 0.984 | v |
| Georgia | SF2 | Q | 0.958 | 0.936 | 0.975 | v |
| Armenia | SF2 | Q | 0.954 | 0.928 | 0.975 | v |
| Cyprus | SF1 | Q | 0.952 | 0.925 | 0.973 | v |
| Estonia | SF2 | Q | 0.950 | 0.924 | 0.973 | v |
| Luxembourg | SF1 | Q | 0.950 | 0.921 | 0.975 | v |
| Slovenia | SF1 | Q | 0.949 | 0.920 | 0.972 | v |
| Israel | SF2 | Q | 0.895 | 0.814 | 0.955 | v |
| Belgium | SF2 |  | 0.888 | 0.812 | 0.950 | x |
| Netherlands | SF2 | Q | 0.885 | 0.804 | 0.948 | v |
| Ireland | SF1 | Q | 0.769 | 0.587 | 0.898 | v |
| Portugal | SF1 | Q | 0.676 | 0.427 | 0.869 | v |
| Latvia | SF2 | Q | 0.482 | 0.252 | 0.726 | v |
| Finland | SF1 | Q | 0.445 | 0.217 | 0.677 | v |
| Denmark | SF2 |  | 0.231 | 0.101 | 0.382 | v |
| Czech Republic | SF2 |  | 0.185 | 0.081 | 0.320 | v |
| Poland | SF1 |  | 0.138 | 0.056 | 0.250 | v |
| Moldova | SF1 |  | 0.039 | 0.018 | 0.068 | v |
| Azerbaijan | SF1 |  | 0.039 | 0.018 | 0.066 | v |
| San Marino | SF2 |  | 0.033 | 0.018 | 0.049 | v |
| Australia | SF1 |  | 0.032 | 0.019 | 0.050 | v |
| Malta | SF2 |  | 0.028 | 0.017 | 0.042 | v |
| Albania | SF2 |  | 0.026 | 0.015 | 0.041 | v |
| Iceland | SF1 |  | 0.024 | 0.014 | 0.037 | v |

**LGBM**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Lithuania | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Norway | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Croatia | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Ukraine | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Luxembourg | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Georgia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Serbia | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Armenia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Slovenia | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Estonia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Cyprus | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Switzerland | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Greece | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Austria | SF2 | Q | 0.998 | 1.000 | 1.000 | v |
| Israel | SF2 | Q | 0.980 | 0.989 | 1.000 | v |
| Belgium | SF2 |  | 0.976 | 0.986 | 1.000 | x |
| Netherlands | SF2 | Q | 0.976 | 0.987 | 1.000 | v |
| Portugal | SF1 | Q | 0.960 | 0.986 | 1.000 | v |
| Ireland | SF1 | Q | 0.956 | 0.921 | 1.000 | v |
| Latvia | SF2 | Q | 0.736 | 0.002 | 1.000 | v |
| Finland | SF1 | Q | 0.500 | 0.000 | 0.998 | v |
| Czech Republic | SF2 |  | 0.061 | 0.000 | 0.120 | v |
| Denmark | SF2 |  | 0.048 | 0.000 | 0.073 | v |
| Poland | SF1 |  | 0.041 | 0.000 | 0.023 | v |
| Azerbaijan | SF1 |  | 0.005 | 0.000 | 0.000 | v |
| Albania | SF2 |  | 0.005 | 0.000 | 0.000 | v |
| Moldova | SF1 |  | 0.005 | 0.000 | 0.000 | v |
| San Marino | SF2 |  | 0.002 | 0.000 | 0.000 | v |
| Malta | SF2 |  | 0.001 | 0.000 | 0.000 | v |
| Iceland | SF1 |  | 0.000 | 0.000 | 0.000 | v |
| Australia | SF1 |  | 0.000 | 0.000 | 0.000 | v |
