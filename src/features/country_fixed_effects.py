"""
Country fixed-effects features — US-S3-01
Usage:  python src/features/country_fixed_effects.py
Output: data/features/country_fixed_effects.csv

Features computed per (Country, Year):
  avg_final_rank_3yr  — mean Final_Place over ≤3 most recent grand finals before Year
  avg_jury_3yr        — mean jury_points  over the same window
  avg_tele_3yr        — mean tele_points  over the same window

Leakage guarantee (PR-07):
  Only rows with Year < current row's Year are used as history.
  Source rows are restricted to Grand_Final_Ind == 1 (actual finalists).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_CSV = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
OUT_CSV = ROOT / "data" / "features" / "country_fixed_effects.csv"

WINDOW = 3  # number of prior grand-final editions to average over


def _load() -> pd.DataFrame:
    df = pd.read_csv(ENRICHED_CSV, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def compute_country_fixed_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (Country, Year) pair in *df*, compute rolling-3yr country averages
    using only grand-final history strictly before that year.

    Returns a DataFrame with columns:
        Year, Country, avg_final_rank_3yr, avg_jury_3yr, avg_tele_3yr
    """
    finals = (
        df[df["Grand_Final_Ind"] == 1]
        [["Year", "Country", "Final_Place", "jury_points", "tele_points"]]
        .copy()
    )

    pairs = df[["Year", "Country"]].drop_duplicates()

    rows: list[dict] = []
    for country, year in zip(pairs["Country"], pairs["Year"]):
        # Strictly prior grand-final appearances — the key leakage guard
        prior = finals[(finals["Country"] == country) & (finals["Year"] < year)]
        recent = prior.nlargest(WINDOW, "Year")

        rows.append(
            {
                "Year": year,
                "Country": country,
                # pandas mean() skips NaN by default; returns NaN on empty slice
                "avg_final_rank_3yr": recent["Final_Place"].mean(),
                "avg_jury_3yr": recent["jury_points"].mean(),
                "avg_tele_3yr": recent["tele_points"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(["Country", "Year"]).reset_index(drop=True)


def _validate(fe: pd.DataFrame, df: pd.DataFrame) -> None:
    """Print a coverage and sanity report."""
    total = len(fe)
    has_rank = fe["avg_final_rank_3yr"].notna().sum()
    has_jury = fe["avg_jury_3yr"].notna().sum()
    has_tele = fe["avg_tele_3yr"].notna().sum()

    print(f"\nCountry Fixed Effects — coverage ({total} rows)")
    print(f"  avg_final_rank_3yr : {has_rank:>3} / {total} ({has_rank/total*100:.0f}%)")
    print(f"  avg_jury_3yr       : {has_jury:>3} / {total} ({has_jury/total*100:.0f}%)")
    print(f"  avg_tele_3yr       : {has_tele:>3} / {total} ({has_tele/total*100:.0f}%)")

    # Leakage check: for each computed row, verify no same-year final data was used
    # (spot-check via manual recalculation of a known country/year)
    finals = df[df["Grand_Final_Ind"] == 1][["Year", "Country", "Final_Place", "jury_points", "tele_points"]]

    violations = 0
    for _, row in fe.iterrows():
        country, year = row["Country"], row["Year"]
        same_year_finals = finals[(finals["Country"] == country) & (finals["Year"] == year)]
        if not same_year_finals.empty:
            # This row itself is a finalist — confirm its feature was NOT computed from same year
            prior = finals[(finals["Country"] == country) & (finals["Year"] < year)]
            recent = prior.nlargest(WINDOW, "Year")
            expected_rank = recent["Final_Place"].mean()
            actual_rank = row["avg_final_rank_3yr"]
            if not (
                (pd.isna(expected_rank) and pd.isna(actual_rank))
                or np.isclose(expected_rank, actual_rank, equal_nan=True)
            ):
                print(f"  LEAKAGE VIOLATION: {country} {year} — expected {expected_rank:.2f}, got {actual_rank:.2f}")
                violations += 1

    if violations == 0:
        print("  Leakage check     : PASS (no same-year data used)")
    else:
        print(f"  Leakage check     : FAIL ({violations} violations)")

    # Sample: Albania across years
    sample = fe[fe["Country"] == "Albania"][
        ["Year", "Country", "avg_final_rank_3yr", "avg_jury_3yr", "avg_tele_3yr"]
    ]
    print(f"\nSample — Albania:")
    print(sample.to_string(index=False))

    # 2026 summary
    df26 = fe[fe["Year"] == 2026]
    nan26 = df26["avg_final_rank_3yr"].isna().sum()
    print(f"\n2026 rows: {len(df26)}, of which {nan26} have no prior final history")


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = _load()
    fe = compute_country_fixed_effects(df)

    _validate(fe, df)

    fe.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT_CSV.relative_to(ROOT)}")
    print(f"Shape: {fe.shape}")


if __name__ == "__main__":
    main()
