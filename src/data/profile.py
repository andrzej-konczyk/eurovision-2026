"""
Data profiling script — US-S1-07
Usage:  python src/data/profile.py
Output: reports/data_profile_YYYYMMDD.md

Profiles:
  - Dataset/eurovision_2016_26_kaggle.csv   (primary dataset)
  - Dataset/eurovision_odds_2018_2025.csv   (scraped bookmaker odds)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "Dataset"
REPORTS_DIR = ROOT / "reports"

PRIMARY_CSV = DATASET_DIR / "eurovision_2016_26_kaggle.csv"
ODDS_CSV = DATASET_DIR / "eurovision_odds_2018_2025.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def _null_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = df.isnull().sum()
    # Also count empty strings (odds CSV stores "" for missing)
    empty = (df == "").sum()
    combined = total + empty
    pct = combined / len(df) * 100
    result = pd.DataFrame({"missing": combined, "pct": pct.round(1)})
    return result[result["missing"] > 0].sort_values("missing", ascending=False)


def _year_dist(df: pd.DataFrame) -> pd.Series | None:
    col = next((c for c in df.columns if c.strip().lower() == "year"), None)
    return df[col].value_counts().sort_index() if col else None


def _numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    stats = num.agg(["min", "mean", "max", "std"]).T.round(3)
    stats["nulls"] = num.isnull().sum()
    return stats


def _categorical_summary(df: pd.DataFrame, max_cols: int = 20) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()[:max_cols]
    rows = []
    for col in obj_cols:
        series = df[col].replace("", np.nan).dropna()
        top = series.value_counts().index[0] if len(series) > 0 else "—"
        rows.append({
            "column": col,
            "unique": series.nunique(),
            "top_value": str(top)[:40],
            "nulls+empty": int(df[col].isnull().sum() + (df[col] == "").sum()),
        })
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def _duplicate_check(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


# ── section formatter ─────────────────────────────────────────────────────────

def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_\n"
    return df.to_markdown() + "\n"


def profile_dataset(path: Path, name: str) -> str:
    print(f"  Loading {path.name}...")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {path} with any known encoding")

    lines: list[str] = []
    lines += [f"## {name}", ""]

    # ── shape
    lines += [f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns"]
    lines += [f"**Duplicates:** {_duplicate_check(df)} rows", ""]

    # ── year distribution
    year_dist = _year_dist(df)
    if year_dist is not None:
        lines += ["### Year distribution", ""]
        lines += ["| Year | Entries |", "|------|---------|"]
        for yr, cnt in year_dist.items():
            lines.append(f"| {yr} | {cnt} |")
        lines.append("")

    # ── null / empty analysis
    null_df = _null_summary(df)
    lines += ["### Missing values (null + empty string)", ""]
    lines.append(_df_to_md(null_df))

    # ── numeric stats
    num_stats = _numeric_stats(df)
    lines += ["### Numeric columns — descriptive stats", ""]
    lines.append(_df_to_md(num_stats))

    # ── categorical summary
    cat_df = _categorical_summary(df)
    lines += ["### Categorical / text columns (first 20)", ""]
    lines.append(_df_to_md(cat_df))

    # ── full dtype list
    lines += ["### All columns & dtypes", ""]
    lines += ["| Column | dtype |", "|--------|-------|"]
    for col, dtype in df.dtypes.items():
        lines.append(f"| `{col}` | {dtype} |")
    lines.append("")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    date_str = datetime.today().strftime("%Y%m%d")
    out_path = REPORTS_DIR / f"data_profile_{date_str}.md"

    print("Eurovision 2026 — data profiling")
    print(f"  Root: {ROOT}")

    sections: list[str] = []

    header = [
        "# Data Profile Report",
        f"Generated: {datetime.today().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
    ]
    sections.append("\n".join(header))

    for path, name in [
        (PRIMARY_CSV, "Primary dataset — eurovision_2016_26_kaggle.csv"),
        (ODDS_CSV,    "Betting odds — eurovision_odds_2018_2025.csv"),
    ]:
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping")
            continue
        sections.append(profile_dataset(path, name))
        sections.append("---\n")

    report = "\n".join(sections)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
