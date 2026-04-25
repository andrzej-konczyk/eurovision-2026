# Data Profile Report
Generated: 2026-04-25 16:21

---

## Primary dataset — eurovision_2016_26_kaggle.csv

**Shape:** 393 rows × 36 columns
**Duplicates:** 0 rows

### Year distribution

| Year | Entries |
|------|---------|
| 2016 | 42 |
| 2017 | 42 |
| 2018 | 43 |
| 2019 | 41 |
| 2021 | 39 |
| 2022 | 40 |
| 2023 | 37 |
| 2024 | 37 |
| 2025 | 37 |
| 2026 | 35 |

### Missing values (null + empty string)

|                      |   missing |   pct |
|:---------------------|----------:|------:|
| Genre                |       393 | 100   |
| Language5            |       391 |  99.5 |
| Language6            |       391 |  99.5 |
| Language4            |       389 |  99   |
| Language3            |       383 |  97.5 |
| Language2            |       317 |  80.7 |
| Final_Place          |       161 |  41   |
| Final_Points         |       161 |  41   |
| Running_Order_Final  |       160 |  40.7 |
| Top 5                |        89 |  22.6 |
| Semi_Place           |        88 |  22.4 |
| Semi_Points          |        88 |  22.4 |
| Top 10               |        84 |  21.4 |
| Running_Order_Semi   |        58 |  14.8 |
| Semi_Final_Num       |        58 |  14.8 |
| Qualification_Record |        50 |  12.7 |
| OGAE_Points          |        35 |   8.9 |
| Grand_Final_Ind      |        30 |   7.6 |
| Big6_Ind             |        30 |   7.6 |

### Numeric columns — descriptive stats

|                      |   min |     mean |   max |     std |   nulls |
|:---------------------|------:|---------:|------:|--------:|--------:|
| Year                 |  2016 | 2020.9   |  2026 |   3.295 |       0 |
| Final_Place          |     1 |   13.392 |    26 |   7.458 |     161 |
| Final_Points         |     0 |  179.75  |   758 | 148.551 |     161 |
| Top 5                |     0 |    0.086 |     1 |   0.28  |      89 |
| Top 10               |     0 |    0.168 |     1 |   0.375 |      84 |
| Running_Order_Final  |     1 |   13.446 |    26 |   7.487 |     160 |
| Grand_Final_Ind      |     0 |    0.656 |     1 |   0.476 |      30 |
| Big6_Ind             |     0 |    0.16  |     1 |   0.367 |      30 |
| Semi_Final_Num       |     1 |    1.507 |     2 |   0.501 |      58 |
| Semi_Place           |     1 |    9.016 |    19 |   4.966 |      88 |
| Semi_Points          |     0 |  116.19  |   403 |  81.929 |      88 |
| Running_Order_Semi   |     1 |    8.925 |    19 |   4.92  |      58 |
| National_Final       |     0 |    0.634 |     1 |   0.482 |       0 |
| Genre                |   nan |  nan     |   nan | nan     |     393 |
| Solo_Artist          |     0 |    0.735 |     1 |   0.442 |       0 |
| Returning_Artist_Ind |     0 |    0.089 |     1 |   0.285 |       0 |
| Number of Members    |     1 |    1.664 |     6 |   1.34  |       0 |
| Multiple_Language    |     0 |    0.193 |     1 |   0.395 |       0 |
| EU                   |     0 |    0.598 |     1 |   0.491 |       0 |
| NATO                 |     0 |    0.58  |     1 |   0.494 |       0 |
| MyESB_Community      |     1 |   20.234 |    43 |  11.5   |       0 |
| MyESB_Personal       |     1 |   20.234 |    43 |  11.5   |       0 |
| OGAE_Points          |     0 |   62.863 |   497 | 105.524 |      35 |
| Qualification_Record |     0 |    0.585 |     1 |   0.231 |      50 |

### Categorical / text columns (first 20)

| column                 |   unique | top_value   |   nulls+empty |
|:-----------------------|---------:|:------------|--------------:|
| Country                |       46 | Albania     |             0 |
| Song                   |      390 | Stay        |             0 |
| Artist                 |      389 | Stefania    |             0 |
| Sex                    |        4 | M           |             0 |
| Language1              |       32 | English     |             0 |
| Language2              |       30 | English     |           317 |
| Language3              |        8 | Spanish     |           383 |
| Language4              |        3 | French      |           389 |
| Language5              |        2 | German      |           391 |
| Language6              |        1 | Italian     |           391 |
| National_Language_Used |        2 | False       |             0 |
| Country_Group          |        6 | Western     |             0 |

### All columns & dtypes

| Column | dtype |
|--------|-------|
| `Year` | int64 |
| `Country ` | object |
| `Song ` | object |
| `Artist ` | object |
| `Final_Place` | float64 |
| `Final_Points` | float64 |
| `Top 5` | float64 |
| `Top 10` | float64 |
| `Running_Order_Final` | float64 |
| `Grand_Final_Ind` | float64 |
| `Big6_Ind` | float64 |
| `Semi_Final_Num` | float64 |
| `Semi_Place` | float64 |
| `Semi_Points` | float64 |
| `Running_Order_Semi` | float64 |
| `National_Final` | int64 |
| `Genre` | float64 |
| `Solo_Artist` | int64 |
| `Sex` | object |
| `Returning_Artist_Ind` | int64 |
| `Number of Members` | int64 |
| `Language1` | object |
| `Language2` | object |
| `Language3` | object |
| `Language4` | object |
| `Language5` | object |
| `Language6` | object |
| `Multiple_Language` | int64 |
| `National_Language_Used` | bool |
| `EU` | int64 |
| `NATO` | int64 |
| `Country_Group` | object |
| `MyESB_Community` | int64 |
| `MyESB_Personal` | int64 |
| `OGAE_Points` | float64 |
| `Qualification_Record` | float64 |

---

## Betting odds — eurovision_odds_2018_2025.csv

**Shape:** 221 rows × 31 columns
**Duplicates:** 0 rows

### Year distribution

| Year | Entries |
|------|---------|
| 2018 | 26 |
| 2019 | 26 |
| 2020 | 41 |
| 2021 | 26 |
| 2022 | 25 |
| 2023 | 26 |
| 2024 | 25 |
| 2025 | 26 |

### Missing values (null + empty string)

|               |   missing |   pct |
|:--------------|----------:|------:|
| BETANO        |       195 |  88.2 |
| EPIC BET      |       195 |  88.2 |
| 7BET          |       195 |  88.2 |
| OPTIBET       |       195 |  88.2 |
| OLYBET        |       195 |  88.2 |
| 1XBET         |       154 |  69.7 |
| CORAL         |       128 |  57.9 |
| COMEON        |       103 |  46.6 |
| SMARKETS      |        93 |  42.1 |
| 10BET         |        51 |  23.1 |
| BETFAIR SPORT |        51 |  23.1 |
| BWIN          |        32 |  14.5 |
| BETWAY        |        26 |  11.8 |
| BET STARS     |        26 |  11.8 |
| COOL BET      |         1 |   0.5 |
| 888 SPORT     |         1 |   0.5 |

### Numeric columns — descriptive stats

|               |     min |     mean |   max |     std |   nulls |
|:--------------|--------:|---------:|------:|--------:|--------:|
| year          | 2018    | 2021.38  |  2025 |   2.253 |       0 |
| rank          |    1    |   14.778 |    41 |   8.947 |       0 |
| BETSSON       |    1.22 |  144.226 |   500 | 142.395 |       0 |
| BOYLE SPORTS  |    1.25 |  168.549 |  1001 | 181.165 |       0 |
| BET365        |    1.25 |  227.864 |  1001 | 264.5   |       0 |
| COOL BET      |    1.34 |  159.258 |   501 | 135.23  |       1 |
| BWIN          |    1.3  |  121.293 |   501 | 113.445 |      32 |
| UNIBET        |    1.22 |  146.394 |   501 | 141.773 |       0 |
| BET STARS     |    1.5  |  170.739 |   751 | 173.538 |      26 |
| LAD BROKES    |    1.3  |  123.694 |   501 | 109.777 |       0 |
| 888 SPORT     |    1.29 |  128.29  |   601 | 126.427 |       1 |
| CORAL         |    1.57 |   84.334 |   301 |  67.236 |     128 |
| 10BET         |    1.3  |  120.755 |   501 | 130.533 |      51 |
| BETWAY        |    1.29 |  116.173 |  1001 | 123.656 |      26 |
| SKY BET       |    1.29 |  137.508 |  1001 | 153.152 |       0 |
| WILLIAM HILL  |    1.33 |  143.119 |   501 | 146.608 |       0 |
| BET FRED      |    1.25 |  101.57  |   501 | 107.864 |       0 |
| BETFAIR SPORT |    1.25 |  110.238 |   501 | 112.726 |      51 |
| BFX           |    1.34 |  388.623 |  1000 | 368.636 |       0 |
| OLYBET        |    1.65 |  141.64  |   301 | 114.523 |     195 |
| 1XBET         |    3.24 |  102.11  |   501 | 118.751 |     154 |
| COMEON        |    1.3  |  127.356 |   501 | 141.661 |     103 |
| SMARKETS      |    1.34 |  264.976 |   500 | 186.759 |      93 |
| BETANO        |    1.7  |  174.675 |   500 | 136.209 |     195 |
| EPIC BET      |    1.62 |  158.466 |   300 | 108.812 |     195 |
| 7BET          |    1.87 |  152.701 |   300 | 112.101 |     195 |
| OPTIBET       |    1.7  |  134.412 |   301 |  95.412 |     195 |

### Categorical / text columns (first 20)

| column   |   unique | top_value    |   nulls+empty |
|:---------|---------:|:-------------|--------------:|
| country  |       43 | Germany      |             0 |
| artist   |      206 | James Newman |             0 |
| song     |      220 | Storm        |             0 |
| win_pct  |       25 | <1           |             0 |

### All columns & dtypes

| Column | dtype |
|--------|-------|
| `year` | int64 |
| `rank` | int64 |
| `country` | object |
| `artist` | object |
| `song` | object |
| `win_pct` | object |
| `BETSSON` | float64 |
| `BOYLE SPORTS` | float64 |
| `BET365` | float64 |
| `COOL BET` | float64 |
| `BWIN` | float64 |
| `UNIBET` | float64 |
| `BET STARS` | float64 |
| `LAD BROKES` | float64 |
| `888 SPORT` | float64 |
| `CORAL` | float64 |
| `10BET` | float64 |
| `BETWAY` | float64 |
| `SKY BET` | float64 |
| `WILLIAM HILL` | float64 |
| `BET FRED` | float64 |
| `BETFAIR SPORT` | float64 |
| `BFX` | float64 |
| `OLYBET` | float64 |
| `1XBET` | float64 |
| `COMEON` | float64 |
| `SMARKETS` | float64 |
| `BETANO` | float64 |
| `EPIC BET` | float64 |
| `7BET` | float64 |
| `OPTIBET` | float64 |

---
