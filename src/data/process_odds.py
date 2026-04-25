"""
Betting odds processor — US-S2-03
Usage:  python src/data/process_odds.py [--client-file PATH]
Input:  Dataset/eurovision_odds_2018_2025.csv  (existing per-bookmaker snapshot)
        --client-file: optional CSV delivered by client (must have year, country, odds columns)
Output: Dataset/betting_odds_clean.csv
        docs/odds_ingestion_log.md  (audit trail for client-supplied files)

Schema of output CSV:
  year          | int   | Contest year
  country       | str   | Country name (from source)
  country_code  | str   | ISO 3166-1 alpha-3
  odds_open     | float | Opening odds (NULL — not available from primary source; see note)
  odds_close    | float | Consensus closing odds: harmonic mean across bookmakers >= MIN_BK_COVERAGE
  implied_prob  | float | Overround-normalised implied probability (sum to 1.0 per year)
  n_bookmakers  | int   | Number of bookmakers used for odds_close
  source        | str   | 'primary' | 'client'

NOTE on odds_open:
  No opening odds are available from the primary source (eurovision_odds_2018_2025.csv),
  which contains a single pre-contest snapshot.  If the client supplies a file with
  opening odds, re-run with --client-file.  The column is preserved as NULL to maintain
  schema compatibility and avoid silent downstream errors.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
RAW_CSV  = ROOT / "Dataset" / "eurovision_odds_2018_2025.csv"
OUT_CSV  = ROOT / "Dataset" / "betting_odds_clean.csv"
LOG_MD   = ROOT / "docs" / "odds_ingestion_log.md"
DOCS_DIR = ROOT / "docs"

# Bookmakers present in the primary CSV
BOOKMAKERS = [
    "BETSSON", "BOYLE SPORTS", "BET365", "COOL BET", "BWIN", "UNIBET",
    "BET STARS", "LAD BROKES", "888 SPORT", "CORAL", "10BET", "BETWAY",
    "SKY BET", "WILLIAM HILL", "BET FRED", "BETFAIR SPORT", "BFX",
    "OLYBET", "1XBET", "COMEON", "SMARKETS", "BETANO", "EPIC BET", "7BET", "OPTIBET",
]

# Minimum fraction of bookmakers that must have odds for a row to use consensus
MIN_BK_COVERAGE = 0.3

COUNTRY_ISO3: dict[str, str] = {
    "Albania": "ALB", "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT",
    "Azerbaijan": "AZE", "Belgium": "BEL", "Belarus": "BLR",
    "Bosnia & Herzegovina": "BIH", "Bulgaria": "BGR", "Croatia": "HRV",
    "Cyprus": "CYP", "Czech Republic": "CZE", "Denmark": "DNK", "Estonia": "EST",
    "Finland": "FIN", "France": "FRA", "Georgia": "GEO", "Germany": "DEU",
    "Greece": "GRC", "Hungary": "HUN", "Iceland": "ISL", "Ireland": "IRL",
    "Israel": "ISR", "Italy": "ITA", "Latvia": "LVA", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Malta": "MLT", "Moldova": "MDA", "Montenegro": "MNE",
    "Netherlands": "NLD", "North Macedonia": "MKD", "FYR Macedonia": "MKD",
    "Norway": "NOR", "Poland": "POL", "Portugal": "PRT", "Romania": "ROU",
    "Russia": "RUS", "San Marino": "SMR", "Serbia": "SRB", "Slovenia": "SVN",
    "Spain": "ESP", "Sweden": "SWE", "Switzerland": "CHE", "Ukraine": "UKR",
    "United Kingdom": "GBR",
    "UK": "GBR",
    "UK United Kingdom": "GBR",
    "Great Britain": "GBR",
    "Czechia": "CZE",
    "N.Macedonia North Macedonia": "MKD",
}


def _load_primary() -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(RAW_CSV, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {RAW_CSV}")


def _consensus_odds(row: pd.Series) -> tuple[float | None, int]:
    """Harmonic mean of available bookmaker decimal odds (= 1 / mean(1/odds))."""
    vals = pd.to_numeric(row[BOOKMAKERS], errors="coerce").dropna()
    vals = vals[vals > 1.0]  # sanity: valid decimal odds > 1
    if len(vals) < len(BOOKMAKERS) * MIN_BK_COVERAGE:
        return None, len(vals)
    # harmonic mean = 1 / mean(1/odds)  → unbiased best-odds estimate
    harm_mean = 1.0 / (1.0 / vals).mean()
    return round(harm_mean, 4), len(vals)


def _normalise_implied_prob(df: pd.DataFrame) -> pd.DataFrame:
    """Overround correction: scale raw 1/odds so they sum to 1.0 per year."""
    df["_raw_prob"] = df["odds_close"].apply(
        lambda x: 1.0 / x if pd.notna(x) and x > 0 else None
    )
    for yr in df["year"].unique():
        mask = (df["year"] == yr) & df["_raw_prob"].notna()
        total = df.loc[mask, "_raw_prob"].sum()
        if total > 0:
            df.loc[mask, "implied_prob"] = (df.loc[mask, "_raw_prob"] / total).round(6)
    df.drop(columns=["_raw_prob"], inplace=True)
    return df


def process_primary() -> pd.DataFrame:
    raw = _load_primary()

    records = []
    for _, row in raw.iterrows():
        odds_close, n_bk = _consensus_odds(row)
        records.append({
            "year":         int(row["year"]),
            "country":      str(row["country"]).strip(),
            "country_code": COUNTRY_ISO3.get(str(row["country"]).strip()),
            "odds_open":    None,   # not available from primary source
            "odds_close":   odds_close,
            "implied_prob": None,   # filled by normalisation
            "n_bookmakers": n_bk,
            "source":       "primary",
        })

    df = pd.DataFrame(records)
    df = _normalise_implied_prob(df)
    return df


def ingest_client_file(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Ingest a client-supplied odds file.
    Expected columns (case-insensitive): year, country, odds_open, odds_close
    Returns (DataFrame, log_entry_markdown).
    """
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = raw.columns.str.strip().str.lower()

    required = {"year", "country"}
    missing_cols = required - set(raw.columns)
    if missing_cols:
        raise ValueError(f"Client file missing columns: {missing_cols}")

    has_open  = "odds_open"  in raw.columns
    has_close = "odds_close" in raw.columns

    records = []
    for _, row in raw.iterrows():
        records.append({
            "year":         int(row["year"]),
            "country":      str(row["country"]).strip(),
            "country_code": COUNTRY_ISO3.get(str(row["country"]).strip()),
            "odds_open":    float(row["odds_open"])  if has_open  and pd.notna(row.get("odds_open"))  else None,
            "odds_close":   float(row["odds_close"]) if has_close and pd.notna(row.get("odds_close")) else None,
            "implied_prob": None,
            "n_bookmakers": 1,
            "source":       "client",
        })

    df = pd.DataFrame(records)
    df = _normalise_implied_prob(df)

    now = datetime.today().strftime("%Y-%m-%d %H:%M")
    log_entry = (
        f"## Ingestion: {path.name}\n\n"
        f"- **Date:** {now}\n"
        f"- **Rows ingested:** {len(df)}\n"
        f"- **Years covered:** {sorted(df['year'].unique())}\n"
        f"- **odds_open present:** {'yes' if has_open else 'no'}\n"
        f"- **odds_close present:** {'yes' if has_close else 'no'}\n"
        f"- **Unmapped countries:** "
        f"{sorted(df[df['country_code'].isna()]['country'].unique().tolist())}\n"
    )
    return df, log_entry


def _write_log(log_entry: str) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    header = (
        "# Odds Ingestion Log\n\n"
        "Audit trail for all client-supplied odds files.\n\n---\n\n"
    )
    if LOG_MD.exists():
        existing = LOG_MD.read_text(encoding="utf-8")
        LOG_MD.write_text(existing + "\n" + log_entry, encoding="utf-8")
    else:
        LOG_MD.write_text(header + log_entry, encoding="utf-8")
    print(f"Log updated: {LOG_MD.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-file", type=Path, default=None,
                        help="Path to a client-supplied odds CSV")
    args = parser.parse_args()

    print("Processing primary odds CSV...")
    df_primary = process_primary()

    df_final = df_primary.copy()

    if args.client_file:
        print(f"Ingesting client file: {args.client_file}")
        df_client, log_entry = ingest_client_file(args.client_file)
        # client rows override primary for matching (year, country)
        merge_keys = ["year", "country"]
        df_final = df_final.merge(
            df_client[merge_keys + ["odds_open", "odds_close", "implied_prob", "source"]],
            on=merge_keys, how="left", suffixes=("", "_client"),
        )
        for col in ("odds_open", "odds_close", "implied_prob", "source"):
            client_col = col + "_client"
            if client_col in df_final.columns:
                mask = df_final[client_col].notna()
                df_final.loc[mask, col] = df_final.loc[mask, client_col]
                df_final.drop(columns=[client_col], inplace=True)
        _write_log(log_entry)

    df_final.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved: {OUT_CSV.relative_to(ROOT)}")
    print(f"Shape: {df_final.shape}")
    print()
    print(df_final[["year","country","country_code","odds_close","implied_prob","n_bookmakers"]].head(8).to_string(index=False))
    print()
    # coverage stats
    null_close = df_final["odds_close"].isna().sum()
    null_prob  = df_final["implied_prob"].isna().sum()
    print(f"odds_close coverage: {len(df_final)-null_close}/{len(df_final)} rows")
    print(f"implied_prob coverage: {len(df_final)-null_prob}/{len(df_final)} rows")
    # sanity: sum of implied_prob per year should be ~1.0
    yr_sums = df_final.groupby("year")["implied_prob"].sum().astype(float).round(3)
    print(f"\nImplied prob sum per year (should be 1.0):\n{yr_sums.to_string()}")


if __name__ == "__main__":
    main()
