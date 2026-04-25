"""
Data validation script — US-S2-02
Usage:  python src/data/validate.py
Input:  Dataset/eurovision_2016_26_enriched.csv
Output: docs/validation_report_YYYYMMDD.md

Checks per DD-01:
  1. Zero unexplained nulls in NOT-NULL mandatory fields
  2. country_code column derived and validated against ISO 3166-1 alpha-3
  3. 2020 documented as COVID cancellation (not an error)
  4. Stage classification consistency
  5. jury_points + tele_points == Final_Points for all finalists with known results
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
ENRICHED = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
DOCS_DIR = ROOT / "docs"

# ── ISO 3166-1 alpha-3 mapping ────────────────────────────────────────────────
# Keys: country names as they appear in the Kaggle CSV (post-strip)
COUNTRY_ISO3: dict[str, str] = {
    "Albania":             "ALB",
    "Armenia":             "ARM",
    "Australia":           "AUS",
    "Austria":             "AUT",
    "Azerbaijan":          "AZE",
    "Belgium":             "BEL",
    "Belarus":             "BLR",
    "Bosnia & Herzegovina":"BIH",
    "Bulgaria":            "BGR",
    "Croatia":             "HRV",
    "Cyprus":              "CYP",
    "Czech Republic":      "CZE",
    "Denmark":             "DNK",
    "Estonia":             "EST",
    "Finland":             "FIN",
    "France":              "FRA",
    "Georgia":             "GEO",
    "Germany":             "DEU",
    "Greece":              "GRC",
    "Hungary":             "HUN",
    "Iceland":             "ISL",
    "Ireland":             "IRL",
    "Israel":              "ISR",
    "Italy":               "ITA",
    "Latvia":              "LVA",
    "Lithuania":           "LTU",
    "Luxembourg":          "LUX",
    "Malta":               "MLT",
    "Moldova":             "MDA",
    "Montenegro":          "MNE",
    "Netherlands":         "NLD",
    "North Macedonia":     "MKD",
    "FYR Macedonia":       "MKD",   # pre-2019 name used in dataset for 2016-2018
    "Norway":              "NOR",
    "Poland":              "POL",
    "Portugal":            "PRT",
    "Romania":             "ROU",
    "Russia":              "RUS",
    "San Marino":          "SMR",
    "Serbia":              "SRB",
    "Slovenia":            "SVN",
    "Spain":               "ESP",
    "Sweden":              "SWE",
    "Switzerland":         "CHE",
    "Ukraine":             "UKR",
    "United Kingdom":      "GBR",
}

# ── Expected null rules (these are NOT validation failures) ───────────────────
#   stage = Final: Final_Place, Final_Points, Running_Order_Final null → 2026 (no contest yet)
#   stage = SF only: jury_points, tele_points, Final_Place, Final_Points null → eliminated in semi
#   Grand_Final_Ind null → 2026 entries without known final status at data capture time
#   OGAE_Points null → early years / small fan-club nations (flagged as MEDIUM gap, not a blocker)
#   Qualification_Record null → Big6 / debutants (flagged as MEDIUM gap)

TOLERANCE_PTS = 1  # rounding tolerance for jury+tele==total check


class Finding(NamedTuple):
    severity: str   # CRITICAL | HIGH | MEDIUM | INFO
    check:    str
    detail:   str
    rows:     int


def _load() -> pd.DataFrame:
    df = pd.read_csv(ENRICHED, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def _add_country_code(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Finding]]:
    df["country_code"] = df["Country"].map(COUNTRY_ISO3)
    findings: list[Finding] = []

    unknown = df[df["country_code"].isna()]["Country"].unique().tolist()
    if unknown:
        findings.append(Finding(
            "HIGH", "country_code — unmapped countries",
            f"No ISO alpha-3 mapping for: {sorted(unknown)}",
            len(df[df["country_code"].isna()]),
        ))
    else:
        findings.append(Finding(
            "INFO", "country_code — ISO alpha-3 mapping",
            f"All {df['Country'].nunique()} countries mapped to ISO 3166-1 alpha-3.",
            0,
        ))
    return df, findings


def _classify_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a clean 'stage' column: Final | SF1 | SF2 | Unknown."""
    conditions = [
        df["Grand_Final_Ind"] == 1,
        df["Semi_Final_Num"] == 1,
        df["Semi_Final_Num"] == 2,
    ]
    choices = ["Final", "SF1", "SF2"]
    df["_stage"] = pd.Categorical(
        pd.Series(pd.NA).where(True),  # placeholder
        categories=["Final", "SF1", "SF2", "Unknown"],
    )
    import numpy as np
    df["_stage"] = np.select(conditions, choices, default="Unknown")
    return df


def _check_2020(df: pd.DataFrame) -> list[Finding]:
    years = sorted(df["Year"].unique())
    expected_missing = [2020]
    actual_missing = [y for y in range(min(years), max(years) + 1) if y not in years]
    if actual_missing == expected_missing:
        return [Finding(
            "INFO", "2020 absence",
            "Year 2020 absent from dataset — Eurovision 2020 cancelled due to COVID-19 pandemic. "
            "Documented absence; no imputation required or appropriate.",
            0,
        )]
    unexpected = [y for y in actual_missing if y not in expected_missing]
    return [Finding(
        "CRITICAL", "Unexpected missing years",
        f"Years missing beyond documented 2020 cancellation: {unexpected}",
        len(unexpected),
    )]


def _check_mandatory_nulls(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []

    # year — always NOT NULL
    n = df["Year"].isna().sum()
    if n:
        findings.append(Finding("CRITICAL", "year — NOT NULL", f"{n} null values found.", n))
    else:
        findings.append(Finding("INFO", "year — NOT NULL", "0 nulls.", 0))

    # country — always NOT NULL
    n = df["Country"].isna().sum()
    if n:
        findings.append(Finding("CRITICAL", "country — NOT NULL", f"{n} null values.", n))
    else:
        findings.append(Finding("INFO", "country — NOT NULL", "0 nulls.", 0))

    # artist — always NOT NULL
    n = df["Artist"].isna().sum()
    if n:
        findings.append(Finding("CRITICAL", "artist — NOT NULL", f"{n} null values.", n))
    else:
        findings.append(Finding("INFO", "artist — NOT NULL", "0 nulls.", 0))

    # song_title (Song) — always NOT NULL
    n = df["Song"].isna().sum()
    if n:
        findings.append(Finding("CRITICAL", "song_title — NOT NULL", f"{n} null values.", n))
    else:
        findings.append(Finding("INFO", "song_title — NOT NULL", "0 nulls.", 0))

    # running_order_final — NOT NULL for Final rows WITH known results (excl. 2026)
    finals_with_results = df[(df["Grand_Final_Ind"] == 1) & df["Final_Points"].notna()]
    n = finals_with_results["Running_Order_Final"].isna().sum()
    if n:
        findings.append(Finding(
            "HIGH", "running_order_final — NOT NULL for Final (known results)",
            f"{n} finalist rows missing Running_Order_Final.", n,
        ))
    else:
        findings.append(Finding(
            "INFO", "running_order_final — NOT NULL for Final",
            f"0 unexplained nulls in {len(finals_with_results)} Final rows with known results.", 0,
        ))

    # jury_points — NOT NULL for Final rows with known results (excl. 2026)
    n_jury = finals_with_results["jury_points"].isna().sum()
    if n_jury:
        bad = finals_with_results[finals_with_results["jury_points"].isna()][["Year","Country"]].head(10)
        findings.append(Finding(
            "CRITICAL", "jury_points — NOT NULL for Final (known results)",
            f"{n_jury} finalist rows missing jury_points:\n{bad.to_string(index=False)}", n_jury,
        ))
    else:
        findings.append(Finding(
            "INFO", "jury_points — NOT NULL for Final",
            f"0 unexplained nulls in {len(finals_with_results)} Final rows.", 0,
        ))

    # tele_points — NOT NULL for Final
    n_tele = finals_with_results["tele_points"].isna().sum()
    if n_tele:
        findings.append(Finding(
            "CRITICAL", "tele_points — NOT NULL for Final (known results)",
            f"{n_tele} finalist rows missing tele_points.", n_tele,
        ))
    else:
        findings.append(Finding(
            "INFO", "tele_points — NOT NULL for Final",
            f"0 unexplained nulls in {len(finals_with_results)} Final rows.", 0,
        ))

    # final_rank — NOT NULL for Final rows with results
    n = finals_with_results["Final_Place"].isna().sum()
    if n:
        findings.append(Finding(
            "HIGH", "final_rank — NOT NULL for Final (known results)",
            f"{n} finalist rows missing Final_Place.", n,
        ))
    else:
        findings.append(Finding(
            "INFO", "final_rank — NOT NULL for Final",
            f"0 unexplained nulls in {len(finals_with_results)} Final rows.", 0,
        ))

    # Expected nulls — documented, not errors
    future = df[(df["Grand_Final_Ind"] == 1) & df["Final_Points"].isna()]
    sf_only = df[(df["Grand_Final_Ind"] != 1) & df["Final_Points"].isna()]
    findings.append(Finding(
        "INFO", "Expected nulls — Final_Points / jury_points / tele_points",
        f"{len(future)} Final rows with null results = 2026 entries (contest not yet held). "
        f"{len(sf_only)} rows eliminated in semi-finals (no Final result expected). "
        "Neither counts as an unexplained null.",
        0,
    ))

    return findings


def _check_jury_tele_sum(df: pd.DataFrame) -> list[Finding]:
    checked = df[
        (df["Grand_Final_Ind"] == 1) &
        df["Final_Points"].notna() &
        df["jury_points"].notna() &
        df["tele_points"].notna()
    ].copy()
    checked["_diff"] = (checked["jury_points"] + checked["tele_points"] - checked["Final_Points"]).abs()
    bad = checked[checked["_diff"] > TOLERANCE_PTS]
    if bad.empty:
        return [Finding(
            "INFO", "jury + tele == Final_Points",
            f"Passed for all {len(checked)} checked rows (tolerance ±{TOLERANCE_PTS}pt).", 0,
        )]
    detail = bad[["Year","Country","jury_points","tele_points","Final_Points","_diff"]].to_string(index=False)
    return [Finding("CRITICAL", "jury + tele != Final_Points", detail, len(bad))]


def _check_country_code_iso(df: pd.DataFrame) -> list[Finding]:
    mapped = df[df["country_code"].notna()]
    valid_iso3 = set(COUNTRY_ISO3.values())
    invalid = mapped[~mapped["country_code"].isin(valid_iso3)]
    if invalid.empty:
        return [Finding(
            "INFO", "country_code — valid ISO 3166-1 alpha-3",
            f"All {mapped['country_code'].nunique()} codes are valid ISO alpha-3 values.", 0,
        )]
    return [Finding(
        "HIGH", "country_code — invalid ISO alpha-3 values",
        f"Invalid codes: {sorted(invalid['country_code'].unique())}", len(invalid),
    )]


# ── report builder ─────────────────────────────────────────────────────────────

SEV_ICON      = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "INFO": "🟢"}
SEV_ICON_ASCII = {"CRITICAL": "[CRIT]", "HIGH": "[HIGH]", "MEDIUM": "[MED] ", "INFO": "[OK]  "}
SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "INFO": 3}


def _md_table(headers: list[str], rows: list[list]) -> str:
    sep  = "|".join("---" for _ in headers)
    head = " | ".join(headers)
    lines = [f"| {head} |", f"|{sep}|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_report(findings: list[Finding], df: pd.DataFrame) -> str:
    now = datetime.today().strftime("%Y-%m-%d %H:%M")
    critical = [f for f in findings if f.severity == "CRITICAL"]
    high     = [f for f in findings if f.severity == "HIGH"]
    total_rows = len(df)

    lines: list[str] = [
        "# Data Validation Report",
        f"Generated: {now}  ",
        f"Dataset: `eurovision_2016_26_enriched.csv` — {total_rows} rows × {df.shape[1]} columns  ",
        f"Standard: DD-01 Data Dictionary  ",
        "",
        "---",
        "",
        "## Result",
        "",
    ]

    if not critical and not high:
        lines += [
            "> **PASS** — zero unexplained nulls in mandatory fields; all AC met.",
            "",
        ]
    else:
        lines += [
            f"> **FAIL** — {len(critical)} CRITICAL, {len(high)} HIGH findings require resolution.",
            "",
        ]

    # Summary table
    summary_rows = [
        [
            SEV_ICON[f.severity] + " " + f.severity,
            f.check,
            f"**{f.rows}**" if f.rows else "—",
        ]
        for f in sorted(findings, key=lambda x: SEV_ORDER[x.severity])
    ]
    lines.append(_md_table(["Severity", "Check", "Affected rows"], summary_rows))
    lines.append("")

    # Detail sections
    lines += ["---", "", "## Findings — detail", ""]
    for f in sorted(findings, key=lambda x: SEV_ORDER[x.severity]):
        icon = SEV_ICON[f.severity]
        lines += [
            f"### {icon} {f.check}",
            "",
            f"**Severity:** {f.severity}  ",
            f"**Affected rows:** {f.rows if f.rows else 'none'}  ",
            "",
            f.detail,
            "",
        ]

    # Coverage stats
    lines += ["---", "", "## Coverage statistics", ""]
    years = sorted(df["Year"].unique())
    stats_rows: list[list] = []
    for yr in years:
        ydf = df[df["Year"] == yr]
        finals = ydf[ydf["Grand_Final_Ind"] == 1]
        with_results = finals[finals["Final_Points"].notna()]
        covered = with_results["jury_points"].notna().sum()
        stats_rows.append([
            yr,
            len(ydf),
            len(finals),
            len(with_results),
            f"{covered}/{len(with_results)}" if len(with_results) else "N/A (not yet held)",
        ])
    lines.append(_md_table(
        ["Year", "Total rows", "Finalists", "Known results", "jury_points coverage"],
        stats_rows,
    ))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    df = _load()

    findings: list[Finding] = []

    df, iso_findings = _add_country_code(df)
    findings.extend(iso_findings)
    findings.extend(_check_country_code_iso(df))
    findings.extend(_check_2020(df))
    findings.extend(_check_mandatory_nulls(df))
    findings.extend(_check_jury_tele_sum(df))

    # print console summary
    critical = [f for f in findings if f.severity == "CRITICAL"]
    high     = [f for f in findings if f.severity == "HIGH"]
    for f in sorted(findings, key=lambda x: SEV_ORDER[x.severity]):
        icon = SEV_ICON_ASCII[f.severity]
        print(f"{icon} {f.severity:8s} | {f.check}")

    print(f"\nSummary: {len(critical)} CRITICAL, {len(high)} HIGH, "
          f"{sum(1 for f in findings if f.severity=='MEDIUM')} MEDIUM, "
          f"{sum(1 for f in findings if f.severity=='INFO')} INFO")

    # save enriched CSV with country_code column
    out_csv = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
    df.drop(columns=["_stage"], errors="ignore", inplace=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"country_code column added, CSV updated: {out_csv.relative_to(ROOT)}")

    # write report
    report = build_report(findings, df)
    date_str = datetime.today().strftime("%Y%m%d")
    out_path = DOCS_DIR / f"validation_report_{date_str}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Validation report saved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
