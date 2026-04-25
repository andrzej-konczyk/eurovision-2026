"""
Data gap analysis script — US-S1-03
Usage:  python src/data/gap_report.py
Output: docs/gap_report_YYYYMMDD.md

Documents:
  - Known critical gaps (jury/tele split, Genre)
  - Auto-detected high-null columns (>= WARN_PCT threshold)
  - Odds coverage gaps per bookmaker/year
  - Column naming issues
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "Dataset"
DOCS_DIR = ROOT / "docs"

PRIMARY_CSV = DATASET_DIR / "eurovision_2016_26_kaggle.csv"
ODDS_CSV    = DATASET_DIR / "eurovision_odds_2018_2025.csv"

# Columns that are legitimately null for known structural reasons
EXPECTED_NULLS = {
    "Final_Place",        # 2026 has no results; semi-final-only countries
    "Final_Points",
    "Running_Order_Final",
    "Top 5",
    "Top 10",
    "Semi_Place",         # Big6 + countries not in semis
    "Semi_Points",
    "Semi_Final_Num",
    "Running_Order_Semi",
    "Grand_Final_Ind",    # 2026 not decided yet
    "Big6_Ind",
    "Language2",          # Most songs use one language
    "Language3",
    "Language4",
    "Language5",
    "Language6",
}

# Null rate at which we raise a warning gap entry
WARN_PCT = 5.0

SEVERITY = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
}


def _load(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def _md_table(headers: list[str], rows: list[list]) -> str:
    sep = "|".join("---" for _ in headers)
    head = " | ".join(headers)
    lines = [f"| {head} |", f"|{sep}|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ── gap definitions ───────────────────────────────────────────────────────────

KNOWN_GAPS: list[dict] = [
    {
        "id": "GAP-01",
        "severity": "CRITICAL",
        "dataset": "Primary (kaggle CSV)",
        "field": "jury_points, tele_points",
        "status": "ABSENT — columns do not exist",
        "description": (
            "The dataset contains only the combined `Final_Points` score. "
            "Separate jury and televote scores are absent for all years. "
            "Since 2016 Eurovision uses a 50/50 jury/televote split; the two "
            "signals frequently diverge (e.g. Ukraine 2022 won on televote but "
            "ranked lower with juries). Both signals are strong independent "
            "predictors."
        ),
        "ml_impact": (
            "Cannot model jury vs. televote preference separately. "
            "Bloc-voting patterns differ between jury and public — "
            "losing this signal reduces prediction accuracy for countries "
            "with polarised reception (e.g. political favourites)."
        ),
        "resolution": (
            "Scrape per-year jury + televote breakdowns from the EBU results "
            "page or ESC-data GitHub dataset. Reference: DS-GAP-01."
        ),
        "owner": "Data Team",
        "sprint": "S2",
    },
    {
        "id": "GAP-02",
        "severity": "CRITICAL",
        "dataset": "Primary (kaggle CSV)",
        "field": "Genre",
        "status": "0 / 393 filled (100% null)",
        "description": (
            "The `Genre` column is entirely empty across all 393 rows and all "
            "years (2016–2026). No genre label has been provided by the client."
        ),
        "ml_impact": (
            "Genre is a potentially strong categorical feature — "
            "uptempo/dance entries have historically outperformed ballads in "
            "public vote; genre affects running-order placement decisions. "
            "Losing this feature degrades the model's ability to encode "
            "song-style effects."
        ),
        "resolution": (
            "Option A: client supplies genre labels per entry. "
            "Option B: derive genre from Spotify `track_genre` / audio features "
            "(energy, danceability, acousticness) — already planned via "
            "Spotify API (PR-03). Reference: DS-GAP-01."
        ),
        "owner": "Data Team / Client",
        "sprint": "S2",
    },
    {
        "id": "GAP-03",
        "severity": "HIGH",
        "dataset": "Primary (kaggle CSV)",
        "field": "Country, Song, Artist (header)",
        "status": "Trailing whitespace in column names",
        "description": (
            "Column names `Country `, `Song `, `Artist ` contain a trailing "
            "space. Any code referencing these columns by exact string will "
            "fail silently or require defensive `.strip()` handling everywhere."
        ),
        "ml_impact": "Low direct impact but high maintenance / bug risk.",
        "resolution": (
            "Strip column names in C-01 Data Ingestion (load step). "
            "Apply `df.columns = df.columns.str.strip()` immediately after "
            "`pd.read_csv()`."
        ),
        "owner": "Dev Team",
        "sprint": "S1",
    },
    {
        "id": "GAP-04",
        "severity": "MEDIUM",
        "dataset": "Primary (kaggle CSV)",
        "field": "Qualification_Record",
        "status": "50 / 393 null (12.7%)",
        "description": (
            "50 entries lack a `Qualification_Record` value. "
            "These are mostly debutant countries or Big6 members "
            "with no semi-final history."
        ),
        "ml_impact": (
            "`Qualification_Record` is a key semi-final risk feature. "
            "Missing values require imputation; naive mean-fill will "
            "underestimate risk for debutants."
        ),
        "resolution": (
            "Impute 0.0 for Big6 (exempt from semis) and "
            "0.5 for debutants (unknown track record). "
            "Flag imputed rows with `qual_record_imputed` boolean column."
        ),
        "owner": "Dev Team",
        "sprint": "S2",
    },
    {
        "id": "GAP-05",
        "severity": "MEDIUM",
        "dataset": "Primary (kaggle CSV)",
        "field": "OGAE_Points",
        "status": "35 / 393 null (8.9%)",
        "description": (
            "35 entries have no OGAE fan-club score. "
            "Affects mostly early years and smaller fan-club nations."
        ),
        "ml_impact": (
            "OGAE is a fan-sentiment proxy correlated with televote outcome. "
            "Missing values require imputation."
        ),
        "resolution": "Impute with per-year median. Flag imputed rows.",
        "owner": "Dev Team",
        "sprint": "S2",
    },
    {
        "id": "GAP-06",
        "severity": "LOW",
        "dataset": "Odds CSV",
        "field": "BETANO, EPIC BET, 7BET, OPTIBET",
        "status": "195 / 221 null (88.2%) — 2025 only",
        "description": (
            "These four bookmakers only appear in the 2025 dataset. "
            "They have no historical odds for 2018–2024."
        ),
        "ml_impact": (
            "Cannot be used as historical predictors. "
            "Use only the core bookmakers present across all years: "
            "BETSSON, UNIBET, LAD BROKES, SKY BET, WILLIAM HILL, BET FRED, BFX."
        ),
        "resolution": (
            "When engineering betting-odds features, restrict to bookmakers "
            "with >= 5 years of coverage. New bookmakers treated as supplementary "
            "only for 2025/2026 inference."
        ),
        "owner": "Dev Team",
        "sprint": "S2",
    },
    {
        "id": "GAP-07",
        "severity": "LOW",
        "dataset": "Primary (kaggle CSV)",
        "field": "All result columns",
        "status": "161 / 393 null (41%) — expected",
        "description": (
            "Final_Place, Final_Points, Running_Order_Final are null for: "
            "(a) 2026 entries — contest not yet held; "
            "(b) countries eliminated in the semi-finals. "
            "This is structurally expected, not a data quality issue."
        ),
        "ml_impact": "No impact — these are target/leakage columns, not features.",
        "resolution": "No action required. Document in Data Dictionary (DD-01).",
        "owner": "—",
        "sprint": "—",
    },
]


# ── auto-detect unexpected high-null columns ──────────────────────────────────

def _detect_unexpected_gaps(df: pd.DataFrame) -> list[dict]:
    gaps = []
    nulls = df.isnull().sum()
    pct   = nulls / len(df) * 100
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in EXPECTED_NULLS:
            continue
        if pct[col] >= WARN_PCT:
            gaps.append({
                "column": col,
                "null_count": int(nulls[col]),
                "pct": round(float(pct[col]), 1),
            })
    return sorted(gaps, key=lambda x: x["pct"], reverse=True)


# ── odds coverage matrix ───────────────────────────────────────────────────────

def _odds_coverage(df: pd.DataFrame) -> str:
    bk_cols = [c for c in df.columns if c not in ("year", "rank", "country", "artist", "song", "win_pct")]
    years = sorted(df["year"].unique())
    rows = []
    for bk in bk_cols:
        row = [bk]
        total_entries = 0
        total_filled = 0
        for yr in years:
            sub = df[df["year"] == yr][bk]
            n = len(sub)
            filled = int((sub != "") & sub.notna()).sum() if sub.dtype == object else int(sub.notna().sum())
            total_entries += n
            total_filled += filled
            row.append(f"{filled}/{n}" if filled < n else "✓")
        cov = round(total_filled / total_entries * 100, 0) if total_entries else 0
        row.append(f"{int(cov)}%")
        rows.append(row)

    headers = ["Bookmaker"] + [str(y) for y in years] + ["Coverage"]
    return _md_table(headers, rows)


# ── year coverage ──────────────────────────────────────────────────────────────

def _year_coverage(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Return (present_years, missing_years_in_range)."""
    present = sorted(int(y) for y in df["Year"].dropna().unique())
    expected = list(range(present[0], present[-1] + 1))
    missing = [y for y in expected if y not in present]
    return present, missing


# ── report builder ─────────────────────────────────────────────────────────────

def build_report(df_primary: pd.DataFrame, df_odds: pd.DataFrame) -> str:
    now = datetime.today().strftime("%Y-%m-%d %H:%M")
    present_years, missing_years = _year_coverage(df_primary)
    lines: list[str] = []

    lines += [
        "# Data Gap Report",
        f"Generated: {now}  ",
        f"Primary dataset: {len(df_primary)} rows × {len(df_primary.columns)} columns  ",
        f"Odds dataset: {len(df_odds)} rows × {len(df_odds.columns)} columns",
        "",
        "---",
        "",
        "## Dataset coverage",
        "",
        f"**Year range:** {present_years[0]}–{present_years[-1]} "
        f"({len(present_years)} contests present)  ",
        f"**Years present:** {', '.join(str(y) for y in present_years)}  ",
        (
            f"**Missing years:** {', '.join(str(y) for y in missing_years)} "
            "— **2020 cancelled due to COVID-19** (no contest held; "
            "structurally absent, not a data gap)."
            if missing_years
            else "**Missing years:** none — full coverage."
        ),
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    # Summary table
    summary_rows = [
        [g["id"], SEVERITY[g["severity"]] + " " + g["severity"],
         g["field"], g["status"]]
        for g in KNOWN_GAPS
    ]
    lines.append(_md_table(["ID", "Severity", "Field(s)", "Status"], summary_rows))
    lines.append("")

    # ── Known gaps ─────────────────────────────────────────────────────────────
    lines += ["---", "", "## Known gaps — detail", ""]

    for g in KNOWN_GAPS:
        icon = SEVERITY[g["severity"]]
        lines += [
            f"### {g['id']} — {g['field']} {icon}",
            "",
            f"**Severity:** {g['severity']}  ",
            f"**Dataset:** {g['dataset']}  ",
            f"**Status:** `{g['status']}`  ",
            f"**Planned fix sprint:** {g['sprint']}  ",
            f"**Owner:** {g['owner']}",
            "",
            f"**Description:** {g['description']}",
            "",
            f"**ML impact:** {g['ml_impact']}",
            "",
            f"**Resolution:** {g['resolution']}",
            "",
        ]

    # ── Auto-detected unexpected gaps (primary CSV) ────────────────────────────
    auto_gaps = _detect_unexpected_gaps(df_primary)
    lines += ["---", "", f"## Auto-detected unexpected nulls ≥ {WARN_PCT}% (primary CSV)", ""]
    if auto_gaps:
        rows = [[g["column"], g["null_count"], f"{g['pct']}%"] for g in auto_gaps]
        lines.append(_md_table(["Column", "Null count", "Null %"], rows))
    else:
        lines.append("_No unexpected high-null columns detected._")
    lines.append("")

    # ── Odds coverage matrix ───────────────────────────────────────────────────
    lines += ["---", "", "## Bookmaker odds coverage matrix", ""]
    lines.append(_odds_coverage(df_odds))
    lines.append("")
    lines += [
        "> **Core bookmakers** (≥ 80% coverage across 2018–2025):  ",
        "> BETSSON · UNIBET · LAD BROKES · SKY BET · WILLIAM HILL · BET FRED · BFX · COOL BET · BET365",
        "",
    ]

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)

    print("Loading datasets...")
    df_primary = _load(PRIMARY_CSV)
    df_odds    = _load(ODDS_CSV)

    report = build_report(df_primary, df_odds)

    date_str = datetime.today().strftime("%Y%m%d")
    out_path = DOCS_DIR / f"gap_report_{date_str}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Gap report saved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
