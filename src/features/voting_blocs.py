"""
Voting-bloc features — US-S3-03
Usage:  python src/features/voting_blocs.py
Output: data/features/voting_blocs.csv
        data/features/bloc_cooccurrence.csv

Features:
  Country_Group       — bloc label (Central / Eastern / Northern /
                        South-Eastern / Southern / Western)
  avg_bloc_jury_3yr   — mean jury_points of bloc-mates across their 3
                        most recent grand-final years strictly before Year
  avg_bloc_tele_3yr   — same for tele_points

Co-occurrence matrix (bloc_cooccurrence.csv):
  Country × bloc binary membership — static, year-independent.

Leakage guarantee (PR-07): only grand-final rows with Year < current row's
Year are used. The country itself is excluded from its own bloc average.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_CSV = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
OUT_BLOCS_CSV = ROOT / "data" / "features" / "voting_blocs.csv"
OUT_COOC_CSV  = ROOT / "data" / "features" / "bloc_cooccurrence.csv"

WINDOW = 3


def _bloc_map(df: pd.DataFrame) -> dict[str, str]:
    """Return {country: Country_Group} using first non-null occurrence."""
    return (
        df[["Country", "Country_Group"]]
        .dropna(subset=["Country_Group"])
        .drop_duplicates("Country")
        .set_index("Country")["Country_Group"]
        .to_dict()
    )


def build_cooccurrence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Country × bloc binary membership matrix.
    Rows = unique countries, columns = bloc names, values ∈ {0, 1}.
    """
    bloc = _bloc_map(df)
    series = pd.Series(bloc, name="Country_Group")
    series.index.name = "Country"
    return pd.get_dummies(series).astype(int).sort_index()


def compute_voting_blocs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (Country, Year) row, compute rolling-3yr bloc-mate averages
    using only grand-final history strictly before that year.

    Returns columns: Year, Country, Country_Group,
                     avg_bloc_jury_3yr, avg_bloc_tele_3yr
    """
    bloc = _bloc_map(df)

    # Invert: bloc → set of members
    bloc_members: dict[str, set[str]] = {}
    for country, grp in bloc.items():
        bloc_members.setdefault(grp, set()).add(country)

    finals = (
        df[df["Grand_Final_Ind"] == 1]
        [["Year", "Country", "jury_points", "tele_points"]]
        .copy()
    )

    pairs = df[["Year", "Country"]].drop_duplicates()

    rows: list[dict] = []
    for country, year in zip(pairs["Country"], pairs["Year"]):
        grp = bloc.get(country)

        if grp is None:
            rows.append({
                "Year": year, "Country": country, "Country_Group": None,
                "avg_bloc_jury_3yr": np.nan, "avg_bloc_tele_3yr": np.nan,
            })
            continue

        mates = bloc_members[grp] - {country}

        # Grand-final appearances of bloc-mates strictly before this year
        prior = finals[finals["Country"].isin(mates) & (finals["Year"] < year)]

        # 3 most recent years in which any bloc-mate reached the final
        recent_years = sorted(prior["Year"].unique(), reverse=True)[:WINDOW]
        recent = prior[prior["Year"].isin(recent_years)]

        rows.append({
            "Year": year,
            "Country": country,
            "Country_Group": grp,
            "avg_bloc_jury_3yr": recent["jury_points"].mean(),
            "avg_bloc_tele_3yr": recent["tele_points"].mean(),
        })

    return pd.DataFrame(rows).sort_values(["Country", "Year"]).reset_index(drop=True)


def _validate(fe: pd.DataFrame) -> None:
    total = len(fe)
    has_jury = fe["avg_bloc_jury_3yr"].notna().sum()
    has_tele = fe["avg_bloc_tele_3yr"].notna().sum()
    print(f"\nVoting Blocs — coverage ({total} rows)")
    print(f"  avg_bloc_jury_3yr : {has_jury:>3} / {total} ({has_jury/total*100:.0f}%)")
    print(f"  avg_bloc_tele_3yr : {has_tele:>3} / {total} ({has_tele/total*100:.0f}%)")

    print(f"\nBloc sizes (unique countries per bloc):")
    print(fe.groupby("Country_Group")["Country"].nunique().sort_values(ascending=False).to_string())

    print(f"\nSample — Sweden (Northern bloc):")
    sample = fe[fe["Country"] == "Sweden"][
        ["Year", "Country_Group", "avg_bloc_jury_3yr", "avg_bloc_tele_3yr"]
    ]
    print(sample.to_string(index=False))


def main() -> None:
    OUT_BLOCS_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ENRICHED_CSV, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    cooc = build_cooccurrence_matrix(df)
    cooc.to_csv(OUT_COOC_CSV, encoding="utf-8")
    print(f"Co-occurrence matrix: {OUT_COOC_CSV.relative_to(ROOT)}  shape={cooc.shape}")
    print(cooc)

    fe = compute_voting_blocs(df)
    _validate(fe)

    fe.to_csv(OUT_BLOCS_CSV, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT_BLOCS_CSV.relative_to(ROOT)}  shape={fe.shape}")


if __name__ == "__main__":
    main()
