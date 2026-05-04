# Semi-Final Backtest Report — 2022 / 2023 / 2024

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

## CI Calibration — 80% CI Empirical Coverage

| Year | XGB | LGBM | KPI >=80% |
|------|-----|------|----------|
| 2022 | 91% | 97% | PASS |
| 2023 | 97% | 100% | PASS |
| 2024 | 97% | 97% | PASS |

## Per-Year Country Detail

### 2022 (train: [2016, 2017, 2018, 2019, 2021])

**XGB**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Netherlands | SF1 | Q | 0.980 | 0.971 | 0.987 | v |
| Sweden | SF2 | Q | 0.975 | 0.963 | 0.985 | v |
| Ukraine | SF1 | Q | 0.966 | 0.944 | 0.982 | v |
| Greece | SF1 | Q | 0.966 | 0.943 | 0.983 | v |
| Serbia | SF2 | Q | 0.964 | 0.939 | 0.983 | v |
| Czech Republic | SF2 | Q | 0.957 | 0.925 | 0.981 | v |
| Norway | SF1 | Q | 0.954 | 0.924 | 0.977 | v |
| Portugal | SF1 | Q | 0.953 | 0.920 | 0.977 | v |
| Australia | SF2 | Q | 0.947 | 0.906 | 0.979 | v |
| Poland | SF2 | Q | 0.936 | 0.874 | 0.976 | v |
| Armenia | SF1 | Q | 0.901 | 0.819 | 0.960 | v |
| Belgium | SF2 | Q | 0.887 | 0.804 | 0.950 | v |
| Moldova | SF1 | Q | 0.871 | 0.764 | 0.947 | v |
| Estonia | SF2 | Q | 0.861 | 0.744 | 0.947 | v |
| Lithuania | SF1 | Q | 0.859 | 0.739 | 0.944 | v |
| Finland | SF2 | Q | 0.859 | 0.744 | 0.943 | v |
| Romania | SF2 | Q | 0.852 | 0.728 | 0.945 | v |
| Austria | SF1 |  | 0.769 | 0.583 | 0.920 | x |
| Iceland | SF1 | Q | 0.723 | 0.509 | 0.899 | v |
| Albania | SF1 |  | 0.721 | 0.514 | 0.896 | x |
| Cyprus | SF2 |  | 0.705 | 0.471 | 0.912 | v |
| Azerbaijan | SF2 | Q | 0.680 | 0.441 | 0.891 | v |
| Ireland | SF2 |  | 0.532 | 0.316 | 0.738 | v |
| Switzerland | SF1 | Q | 0.177 | 0.090 | 0.279 | x |
| San Marino | SF2 |  | 0.107 | 0.026 | 0.211 | v |
| Slovenia | SF1 |  | 0.080 | 0.029 | 0.141 | v |
| Israel | SF2 |  | 0.077 | 0.031 | 0.141 | v |
| Croatia | SF1 |  | 0.059 | 0.022 | 0.111 | v |
| Georgia | SF2 |  | 0.039 | 0.022 | 0.060 | v |
| Montenegro | SF2 |  | 0.039 | 0.022 | 0.060 | v |
| Latvia | SF1 |  | 0.037 | 0.018 | 0.063 | v |
| Bulgaria | SF1 |  | 0.037 | 0.021 | 0.057 | v |
| North Macedonia | SF2 |  | 0.037 | 0.023 | 0.054 | v |
| Malta | SF2 |  | 0.033 | 0.021 | 0.049 | v |
| Denmark | SF1 |  | 0.028 | 0.018 | 0.041 | v |

**LGBM**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Sweden | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Netherlands | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Ukraine | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Greece | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Norway | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Czech Republic | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Serbia | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Australia | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Portugal | SF1 | Q | 0.998 | 1.000 | 1.000 | v |
| Belgium | SF2 | Q | 0.997 | 0.999 | 1.000 | v |
| Armenia | SF1 | Q | 0.997 | 0.999 | 1.000 | v |
| Moldova | SF1 | Q | 0.995 | 0.998 | 1.000 | v |
| Estonia | SF2 | Q | 0.992 | 0.995 | 1.000 | v |
| Romania | SF2 | Q | 0.991 | 0.996 | 1.000 | v |
| Lithuania | SF1 | Q | 0.990 | 0.996 | 1.000 | v |
| Finland | SF2 | Q | 0.989 | 0.998 | 1.000 | v |
| Poland | SF2 | Q | 0.944 | 0.987 | 1.000 | v |
| Iceland | SF1 | Q | 0.940 | 0.902 | 1.000 | v |
| Azerbaijan | SF2 | Q | 0.930 | 0.855 | 1.000 | v |
| Austria | SF1 |  | 0.631 | 0.007 | 1.000 | v |
| Cyprus | SF2 |  | 0.448 | 0.002 | 0.999 | v |
| Albania | SF1 |  | 0.401 | 0.002 | 0.998 | v |
| Ireland | SF2 |  | 0.355 | 0.001 | 0.996 | v |
| San Marino | SF2 |  | 0.063 | 0.000 | 0.092 | v |
| Slovenia | SF1 |  | 0.029 | 0.000 | 0.007 | v |
| Switzerland | SF1 | Q | 0.018 | 0.000 | 0.001 | x |
| Latvia | SF1 |  | 0.009 | 0.000 | 0.000 | v |
| Croatia | SF1 |  | 0.007 | 0.000 | 0.000 | v |
| Israel | SF2 |  | 0.006 | 0.000 | 0.001 | v |
| Bulgaria | SF1 |  | 0.003 | 0.000 | 0.000 | v |
| Montenegro | SF2 |  | 0.002 | 0.000 | 0.000 | v |
| Georgia | SF2 |  | 0.001 | 0.000 | 0.000 | v |
| Malta | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| North Macedonia | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Denmark | SF1 |  | 0.000 | 0.000 | 0.000 | v |

### 2023 (train: [2016, 2017, 2018, 2019, 2021, 2022])

**XGB**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Finland | SF1 | Q | 0.979 | 0.971 | 0.987 | v |
| Sweden | SF1 | Q | 0.977 | 0.966 | 0.986 | v |
| Austria | SF2 | Q | 0.977 | 0.963 | 0.987 | v |
| Slovenia | SF2 | Q | 0.976 | 0.962 | 0.987 | v |
| Israel | SF1 | Q | 0.975 | 0.964 | 0.986 | v |
| Czech Republic | SF1 | Q | 0.974 | 0.963 | 0.984 | v |
| Norway | SF1 | Q | 0.972 | 0.960 | 0.986 | v |
| Armenia | SF2 | Q | 0.969 | 0.949 | 0.983 | v |
| Serbia | SF1 | Q | 0.958 | 0.930 | 0.980 | v |
| Portugal | SF1 | Q | 0.934 | 0.891 | 0.968 | v |
| Australia | SF2 | Q | 0.930 | 0.879 | 0.967 | v |
| Cyprus | SF2 | Q | 0.924 | 0.873 | 0.965 | v |
| Moldova | SF1 | Q | 0.923 | 0.869 | 0.965 | v |
| Estonia | SF2 | Q | 0.902 | 0.830 | 0.957 | v |
| Switzerland | SF1 | Q | 0.900 | 0.823 | 0.958 | v |
| Belgium | SF2 | Q | 0.896 | 0.827 | 0.954 | v |
| Lithuania | SF2 | Q | 0.826 | 0.686 | 0.931 | v |
| Croatia | SF1 | Q | 0.761 | 0.547 | 0.913 | v |
| Georgia | SF2 |  | 0.371 | 0.200 | 0.587 | v |
| Poland | SF2 | Q | 0.370 | 0.179 | 0.583 | v |
| Latvia | SF1 |  | 0.292 | 0.144 | 0.468 | v |
| Albania | SF2 | Q | 0.289 | 0.126 | 0.469 | x |
| Netherlands | SF1 |  | 0.080 | 0.031 | 0.138 | v |
| Romania | SF2 |  | 0.039 | 0.021 | 0.064 | v |
| Azerbaijan | SF1 |  | 0.035 | 0.019 | 0.056 | v |
| San Marino | SF2 |  | 0.028 | 0.017 | 0.043 | v |
| Iceland | SF2 |  | 0.028 | 0.016 | 0.043 | v |
| Malta | SF1 |  | 0.027 | 0.016 | 0.041 | v |
| Denmark | SF2 |  | 0.024 | 0.014 | 0.036 | v |
| Greece | SF2 |  | 0.023 | 0.015 | 0.032 | v |
| Ireland | SF1 |  | 0.018 | 0.013 | 0.024 | v |

**LGBM**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Finland | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Austria | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Israel | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Czech Republic | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Sweden | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Armenia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Slovenia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Serbia | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Portugal | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Norway | SF1 | Q | 0.999 | 1.000 | 1.000 | v |
| Moldova | SF1 | Q | 0.998 | 0.999 | 1.000 | v |
| Switzerland | SF1 | Q | 0.998 | 0.998 | 1.000 | v |
| Belgium | SF2 | Q | 0.997 | 1.000 | 1.000 | v |
| Australia | SF2 | Q | 0.997 | 1.000 | 1.000 | v |
| Cyprus | SF2 | Q | 0.995 | 1.000 | 1.000 | v |
| Estonia | SF2 | Q | 0.995 | 0.999 | 1.000 | v |
| Lithuania | SF2 | Q | 0.979 | 0.993 | 1.000 | v |
| Croatia | SF1 | Q | 0.861 | 0.201 | 1.000 | v |
| Poland | SF2 | Q | 0.588 | 0.000 | 0.998 | v |
| Albania | SF2 | Q | 0.497 | 0.000 | 0.997 | v |
| Georgia | SF2 |  | 0.322 | 0.000 | 0.993 | v |
| Latvia | SF1 |  | 0.118 | 0.000 | 0.603 | v |
| Netherlands | SF1 |  | 0.006 | 0.000 | 0.001 | v |
| Romania | SF2 |  | 0.005 | 0.000 | 0.000 | v |
| Malta | SF1 |  | 0.001 | 0.000 | 0.000 | v |
| San Marino | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Azerbaijan | SF1 |  | 0.000 | 0.000 | 0.000 | v |
| Iceland | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Denmark | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Greece | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Ireland | SF1 |  | 0.000 | 0.000 | 0.000 | v |

### 2024 (train: [2016, 2017, 2018, 2019, 2021, 2022, 2023])

**XGB**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Lithuania | SF1 | Q | 0.984 | 0.978 | 0.989 | v |
| Croatia | SF1 | Q | 0.982 | 0.973 | 0.989 | v |
| Norway | SF2 | Q | 0.982 | 0.974 | 0.988 | v |
| Austria | SF2 | Q | 0.981 | 0.973 | 0.988 | v |
| Ukraine | SF1 | Q | 0.979 | 0.971 | 0.987 | v |
| Switzerland | SF2 | Q | 0.966 | 0.941 | 0.985 | v |
| Greece | SF2 | Q | 0.964 | 0.938 | 0.983 | v |
| Serbia | SF1 | Q | 0.961 | 0.939 | 0.980 | v |
| Georgia | SF2 | Q | 0.952 | 0.925 | 0.973 | v |
| Armenia | SF2 | Q | 0.945 | 0.912 | 0.972 | v |
| Cyprus | SF1 | Q | 0.944 | 0.913 | 0.970 | v |
| Slovenia | SF1 | Q | 0.942 | 0.909 | 0.970 | v |
| Estonia | SF2 | Q | 0.936 | 0.898 | 0.967 | v |
| Israel | SF2 | Q | 0.930 | 0.881 | 0.969 | v |
| Luxembourg | SF1 | Q | 0.926 | 0.876 | 0.967 | v |
| Belgium | SF2 |  | 0.878 | 0.791 | 0.944 | x |
| Netherlands | SF2 | Q | 0.873 | 0.788 | 0.945 | v |
| Ireland | SF1 | Q | 0.855 | 0.745 | 0.950 | v |
| Finland | SF1 | Q | 0.614 | 0.385 | 0.808 | v |
| Portugal | SF1 | Q | 0.557 | 0.346 | 0.752 | v |
| Latvia | SF2 | Q | 0.347 | 0.195 | 0.514 | v |
| Denmark | SF2 |  | 0.221 | 0.095 | 0.366 | v |
| Czech Republic | SF2 |  | 0.188 | 0.085 | 0.318 | v |
| Poland | SF1 |  | 0.135 | 0.058 | 0.228 | v |
| Azerbaijan | SF1 |  | 0.042 | 0.020 | 0.070 | v |
| Moldova | SF1 |  | 0.036 | 0.018 | 0.059 | v |
| Australia | SF1 |  | 0.033 | 0.019 | 0.050 | v |
| San Marino | SF2 |  | 0.033 | 0.019 | 0.048 | v |
| Malta | SF2 |  | 0.031 | 0.018 | 0.046 | v |
| Albania | SF2 |  | 0.026 | 0.016 | 0.038 | v |
| Iceland | SF1 |  | 0.024 | 0.015 | 0.036 | v |

**LGBM**

| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |
|---------|----|----|------|---------|---------|------------|
| Lithuania | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Norway | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Croatia | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Armenia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Serbia | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Georgia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Estonia | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Cyprus | SF1 | Q | 1.000 | 1.000 | 1.000 | v |
| Switzerland | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Austria | SF2 | Q | 1.000 | 1.000 | 1.000 | v |
| Greece | SF2 | Q | 0.999 | 1.000 | 1.000 | v |
| Slovenia | SF1 | Q | 0.999 | 1.000 | 1.000 | v |
| Ukraine | SF1 | Q | 0.999 | 1.000 | 1.000 | v |
| Luxembourg | SF1 | Q | 0.998 | 1.000 | 1.000 | v |
| Israel | SF2 | Q | 0.997 | 0.999 | 1.000 | v |
| Netherlands | SF2 | Q | 0.982 | 0.997 | 1.000 | v |
| Belgium | SF2 |  | 0.982 | 0.998 | 1.000 | x |
| Ireland | SF1 | Q | 0.900 | 0.495 | 1.000 | v |
| Finland | SF1 | Q | 0.789 | 0.003 | 1.000 | v |
| Portugal | SF1 | Q | 0.772 | 0.004 | 0.999 | v |
| Latvia | SF2 | Q | 0.352 | 0.000 | 0.992 | v |
| Czech Republic | SF2 |  | 0.049 | 0.000 | 0.057 | v |
| Denmark | SF2 |  | 0.037 | 0.000 | 0.032 | v |
| Poland | SF1 |  | 0.021 | 0.000 | 0.004 | v |
| Azerbaijan | SF1 |  | 0.003 | 0.000 | 0.000 | v |
| Albania | SF2 |  | 0.003 | 0.000 | 0.000 | v |
| Moldova | SF1 |  | 0.001 | 0.000 | 0.000 | v |
| Malta | SF2 |  | 0.001 | 0.000 | 0.000 | v |
| Australia | SF1 |  | 0.000 | 0.000 | 0.000 | v |
| San Marino | SF2 |  | 0.000 | 0.000 | 0.000 | v |
| Iceland | SF1 |  | 0.000 | 0.000 | 0.000 | v |
