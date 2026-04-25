"""
Merge jury/tele split into the main Kaggle CSV — US-S2-01
Usage:  python src/data/merge_jury_tele.py
Output: Dataset/eurovision_2016_26_enriched.csv

Join key: [Year, Country] (one row per country per year in main CSV).
Adds columns: jury_points, tele_points (Final), jury_points_sf, tele_points_sf.
Validates jury + tele == Final_Points for all qualified finalists (Grand_Final_Ind == 1).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN_CSV  = ROOT / "Dataset" / "eurovision_2016_26_kaggle.csv"
RAW_CSV   = ROOT / "Dataset" / "jury_tele_raw.csv"
OUT_CSV   = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"

TOLS = 1  # allow ±1 point rounding difference


def _load_main() -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(MAIN_CSV, encoding=enc, low_memory=False)
            df.columns = df.columns.str.strip()
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot decode main CSV")


def _normalise(name: str) -> str:
    """Lowercase, strip, collapse whitespace for fuzzy matching."""
    return " ".join(str(name).lower().strip().split())


# Country name overrides: raw → main CSV name
COUNTRY_FIXES: dict[str, str] = {
    "czech republic": "Czech Republic",
    "czechia": "Czech Republic",
    "north macedonia": "North Macedonia",
    "f.y.r. macedonia": "North Macedonia",
    "bosnia & herzegovina": "Bosnia & Herzegovina",
    "bosnia and herzegovina": "Bosnia & Herzegovina",
    "netherlands": "Netherlands",
    "the netherlands": "Netherlands",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
}


def _align_countries(raw: pd.DataFrame, main_countries: set[str]) -> pd.DataFrame:
    """Ensure raw 'country' values match main CSV country names."""
    norm_main = {_normalise(c): c for c in main_countries}

    def _fix(name: str) -> str:
        n = _normalise(name)
        if n in COUNTRY_FIXES:
            return COUNTRY_FIXES[n]
        if n in norm_main:
            return norm_main[n]
        return name  # keep original; mismatch will surface in merge diagnostics

    raw["country"] = raw["country"].apply(_fix)
    return raw


def main() -> None:
    df_main = _load_main()
    df_raw  = pd.read_csv(RAW_CSV, encoding="utf-8")

    main_countries = set(df_main["Country"].dropna().unique())
    df_raw = _align_countries(df_raw, main_countries)

    # ── pivot raw into one row per (year, country) ───────────────────────────
    finals = df_raw[df_raw["stage"] == "final"][
        ["year", "country", "jury_points", "tele_points", "total_points"]
    ].rename(columns={
        "year": "Year",
        "country": "Country",
        "jury_points":  "jury_points",
        "tele_points":  "tele_points",
        "total_points": "_raw_total",
    })

    sf = df_raw[df_raw["stage"].str.startswith("sf")][
        ["year", "country", "jury_points", "tele_points"]
    ].rename(columns={
        "year": "Year",
        "country": "Country",
        "jury_points": "jury_points_sf",
        "tele_points": "tele_points_sf",
    })

    # ── merge ────────────────────────────────────────────────────────────────
    df = df_main.merge(finals, on=["Year", "Country"], how="left")
    df = df.merge(sf,     on=["Year", "Country"], how="left")

    # ── diagnostics: unmatched countries ─────────────────────────────────────
    raw_countries  = set(df_raw["country"].unique())
    main_countries_in_raw = set(finals["Country"].unique())
    unmatched = raw_countries - main_countries - {"Rest of world"}
    if unmatched:
        print(f"WARNING unmatched raw countries: {sorted(unmatched)}")

    # ── validation ───────────────────────────────────────────────────────────
    qualifiers = df[
        (df["Grand_Final_Ind"] == 1) &
        df["Final_Points"].notna() &
        df["jury_points"].notna() &
        df["tele_points"].notna()
    ].copy()
    qualifiers["_sum"] = qualifiers["jury_points"] + qualifiers["tele_points"]
    qualifiers["_diff"] = (qualifiers["_sum"] - qualifiers["Final_Points"]).abs()
    bad = qualifiers[qualifiers["_diff"] > TOLS]

    print(f"\nValidation: jury + tele == Final_Points")
    print(f"  Checked rows  : {len(qualifiers)}")
    print(f"  Failed (>{TOLS}pt): {len(bad)}")
    if not bad.empty:
        print(bad[["Year", "Country", "jury_points", "tele_points", "_sum", "Final_Points", "_diff"]].to_string())

    # ── coverage ─────────────────────────────────────────────────────────────
    final_rows = df[
        (df["Grand_Final_Ind"] == 1) & df["Final_Points"].notna()
    ]
    covered = final_rows["jury_points"].notna().sum()
    coverage = covered / len(final_rows) * 100 if len(final_rows) else 0
    print(f"\nCoverage (Final rows with known points):")
    print(f"  Final rows    : {len(final_rows)}")
    print(f"  Covered       : {covered}")
    print(f"  Coverage      : {coverage:.1f}%")
    if coverage < 95:
        print("  WARNING: coverage < 95% — AC not met")
    else:
        print("  OK: coverage >= 95%")

    # ── save ────────────────────────────────────────────────────────────────
    drop_cols = [c for c in ["_raw_total"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nEnriched CSV saved: {OUT_CSV.relative_to(ROOT)}")
    print(f"Shape: {df.shape}")
    new_cols = [c for c in ["jury_points", "tele_points", "jury_points_sf", "tele_points_sf"] if c in df.columns]
    print(f"New columns: {new_cols}")
    print(df[["Year", "Country", "jury_points", "tele_points"]].dropna().tail(5).to_string())


if __name__ == "__main__":
    main()
