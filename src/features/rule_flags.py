"""
Rule-change binary flags — US-S3-02
Usage:  python src/features/rule_flags.py
Output: data/features/rule_flags.csv

Flags (0/1):
  rule_2019_semifinal_reform  — pot-based SF draw active (Year >= 2019)
  rule_2023_jury_weight_reform — decoupled jury/tele presentation + rest-of-world
                                 public vote active (Year >= 2023)

2020 was cancelled (COVID); both flags reflect the rule regime that *would*
have applied and are consistent with what was used in 2021 onwards.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_CSV = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
OUT_CSV = ROOT / "data" / "features" / "rule_flags.csv"


def compute_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with Year, Country, and two binary rule-change flags.
    Input df must contain at minimum columns: Year, Country.
    """
    out = df[["Year", "Country"]].copy()
    out["rule_2019_semifinal_reform"] = (out["Year"] >= 2019).astype(int)
    out["rule_2023_jury_weight_reform"] = (out["Year"] >= 2023).astype(int)
    return out.reset_index(drop=True)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ENRICHED_CSV, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    flags = compute_rule_flags(df)

    # Sanity check
    year_summary = (
        flags.groupby("Year")[["rule_2019_semifinal_reform", "rule_2023_jury_weight_reform"]]
        .first()
    )
    print("Rule flags by year:")
    print(year_summary.to_string())

    flags.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT_CSV.relative_to(ROOT)}  shape={flags.shape}")


if __name__ == "__main__":
    main()
