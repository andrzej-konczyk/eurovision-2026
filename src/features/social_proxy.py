"""
Social-proxy features — US-S3-05
Usage:  python src/features/social_proxy.py
Output: data/features/social_proxy.csv

Raw columns (already in enriched CSV):
  MyESB_Community  — community rank 1–43 (lower = better prediction)
  MyESB_Personal   — personal rank  1–43
  OGAE_Points      — OGAE club points 0–497 (35 NaN in dataset)

Computed features (per-year z-score, within-year normalisation):
  zscore_myesb_community
  zscore_myesb_personal
  zscore_ogae_points      — NaN rows filled with 0 after normalisation

Per-year normalisation is leakage-safe for same-year prediction: all
community/OGAE scores for a given year are published before the contest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_CSV = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
OUT_CSV = ROOT / "data" / "features" / "social_proxy.csv"

SOCIAL_COLS = ["MyESB_Community", "MyESB_Personal", "OGAE_Points"]
OUT_COLS = ["zscore_myesb_community", "zscore_myesb_personal", "zscore_ogae_points"]


def _zscore_per_year(df: pd.DataFrame, col: str, fill_na: float = 0.0) -> pd.Series:
    """
    Z-score normalise *col* within each Year group.
    Years with std == 0 (single entry or all identical) get 0.
    Missing values are filled with *fill_na* after normalisation.
    """
    def _norm(group: pd.Series) -> pd.Series:
        mu, sigma = group.mean(), group.std(ddof=1)
        if sigma == 0 or pd.isna(sigma):
            return pd.Series(0.0, index=group.index)
        return (group - mu) / sigma

    return (
        df.groupby("Year")[col]
        .transform(_norm)
        .fillna(fill_na)
    )


def compute_social_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with Year, Country, and three z-score features.
    """
    out = df[["Year", "Country"]].copy()
    out["zscore_myesb_community"] = _zscore_per_year(df, "MyESB_Community")
    out["zscore_myesb_personal"]  = _zscore_per_year(df, "MyESB_Personal")
    out["zscore_ogae_points"]     = _zscore_per_year(df, "OGAE_Points")
    return out.reset_index(drop=True)


def _validate(fe: pd.DataFrame) -> None:
    print(f"\nSocial Proxy — shape {fe.shape}")
    for col in OUT_COLS:
        s = fe[col]
        print(f"  {col}: mean={s.mean():.3f}  std={s.std():.3f}  "
              f"min={s.min():.2f}  max={s.max():.2f}  nan={s.isna().sum()}")

    # Per-year mean should be ~0 for each col
    yearly = fe.groupby(fe["Year"])[OUT_COLS].mean().round(6)
    print(f"\n  Per-year mean (should be ~0):")
    print(yearly.to_string())


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ENRICHED_CSV, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    fe = compute_social_proxy(df)
    _validate(fe)

    fe.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT_CSV.relative_to(ROOT)}  shape={fe.shape}")


if __name__ == "__main__":
    main()
