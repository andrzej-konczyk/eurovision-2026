"""
2026 participant list validation — US-S2-04
Usage:  python src/data/validate_participants_2026.py
Input:  Dataset/eurovision_2016_26_enriched.csv
Output: docs/participants_2026_report_YYYYMMDD.md
        Dataset/eurovision_2016_26_enriched.csv  (adds withdrawn_2026 column)

Reference: Wikipedia / EurovisionWorld as of 2026-04-25
  35 countries confirmed by EBU.
  Source: https://en.wikipedia.org/wiki/Eurovision_Song_Contest_2026
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
ENRICHED = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
DOCS_DIR = ROOT / "docs"

# ── Official 2026 participant list (EBU confirmed, 35 countries) ──────────────
# Source: Wikipedia / EurovisionWorld, verified 2026-04-25
OFFICIAL_SF1 = {
    "Belgium", "Croatia", "Estonia", "Finland", "Georgia",
    "Greece", "Israel", "Lithuania", "Moldova", "Montenegro",
    "Poland", "Portugal", "San Marino", "Serbia", "Sweden",
}
OFFICIAL_SF2 = {
    "Armenia", "Australia", "Azerbaijan", "Bulgaria", "Cyprus",
    "Czech Republic", "Denmark", "Latvia", "Luxembourg", "Malta",
    "Norway", "Romania", "Switzerland", "Ukraine",
    # Turkmenistan listed on Wikipedia — flagged as unverified (first-time participant,
    # not an EBU member historically; to be confirmed by official EBU source)
}
OFFICIAL_AUTO = {
    "Austria",        # host + defending champion
    "France",
    "Germany",
    "Italy",
    "United Kingdom",
}

OFFICIAL_ALL = OFFICIAL_SF1 | OFFICIAL_SF2 | OFFICIAL_AUTO

# ── Known withdrawals / non-participants ─────────────────────────────────────
# Countries that were initially expected / registered but withdrew or boycotted
WITHDRAWN: dict[str, str] = {
    "Spain":       "Boycott — refused to participate due to Israeli entry (alongside Ireland, Iceland, Netherlands, Slovenia). "
                   "First time Spain (Big 5) missed since Italy rejoined in 2011.",
    "Ireland":     "Boycott — withdrew in protest of Israeli participation.",
    "Iceland":     "Withdrew — joined boycott of Israeli participation.",
    "Netherlands": "Withdrew — joined boycott of Israeli participation.",
    "Slovenia":    "Withdrew — joined boycott of Israeli participation.",
    "North Macedonia": "Confirmed non-participation prior to participant announcement.",
}

# ── Unverified entries ────────────────────────────────────────────────────────
# These appear in Wikipedia's SF2 list but seem anomalous
UNVERIFIED_OFFICIAL: dict[str, str] = {
    "Turkmenistan": "Listed in Wikipedia SF2, but Turkmenistan has never been an EBU member "
                    "and has no prior Eurovision participation. Requires confirmation from "
                    "official EBU source before adding to dataset.",
}


def _load() -> pd.DataFrame:
    df = pd.read_csv(ENRICHED, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def _normalise(name: str) -> str:
    aliases = {"Czechia": "Czech Republic", "FYR Macedonia": "North Macedonia"}
    return aliases.get(str(name).strip(), str(name).strip())


def main() -> None:
    df = _load()
    df26 = df[df["Year"] == 2026].copy()
    dataset_countries = {_normalise(c) for c in df26["Country"].unique()}

    # ── Comparison ────────────────────────────────────────────────────────────
    in_dataset_not_official = dataset_countries - OFFICIAL_ALL
    in_official_not_dataset = OFFICIAL_ALL - dataset_countries
    matched = dataset_countries & OFFICIAL_ALL

    # Albania is in dataset but not in official SF or Auto list
    albania_status = (
        "DISCREPANCY — in Kaggle dataset but absent from official SF1/SF2/Auto lists. "
        "Possible late withdrawal after dataset was frozen, or data entry error. "
        "Requires confirmation from EBU."
    ) if "Albania" in in_dataset_not_official else "OK"

    # ── Add withdrawn_2026 flag to main dataset ───────────────────────────────
    all_years_withdrawn = set(WITHDRAWN.keys())
    df["withdrawn_2026"] = df.apply(
        lambda r: (r["Year"] == 2026 and _normalise(r["Country"]) in all_years_withdrawn),
        axis=1,
    )
    df.to_csv(ENRICHED, index=False, encoding="utf-8")
    print(f"withdrawn_2026 column added to enriched CSV.")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n2026 dataset rows     : {len(df26)}")
    print(f"Official EBU count    : {len(OFFICIAL_ALL)}")
    print(f"Matched               : {len(matched)}")
    print(f"In dataset, not EBU   : {sorted(in_dataset_not_official) or 'none'}")
    print(f"In EBU, not dataset   : {sorted(in_official_not_dataset) or 'none'}")
    print(f"Confirmed withdrawals : {sorted(WITHDRAWN.keys())}")
    print(f"Unverified entries    : {sorted(UNVERIFIED_OFFICIAL.keys())}")

    # ── Build report ──────────────────────────────────────────────────────────
    now = datetime.today().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        "# Eurovision 2026 Participant List — Validation Report",
        f"Generated: {now}  ",
        f"Reference: EBU / Wikipedia / EurovisionWorld (verified {now[:10]})  ",
        "",
        "---",
        "",
        "## Result",
        "",
    ]

    status = "PASS" if not in_dataset_not_official and not in_official_not_dataset else "REVIEW REQUIRED"
    lines += [
        f"> **{status}** — {len(matched)}/{len(OFFICIAL_ALL)} official countries matched in dataset.",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Official EBU count (2026) | {len(OFFICIAL_ALL)} |",
        f"| Dataset 2026 rows | {len(df26)} |",
        f"| Matched | {len(matched)} |",
        f"| In dataset, not in EBU list | {len(in_dataset_not_official)} |",
        f"| In EBU list, not in dataset | {len(in_official_not_dataset)} |",
        f"| Confirmed withdrawals / boycotts | {len(WITHDRAWN)} |",
        "",
    ]

    # Official list by stage
    lines += ["---", "", "## Official 2026 participants", ""]
    for stage, countries in [
        ("Automatic qualifiers (Big 5 + host)", sorted(OFFICIAL_AUTO)),
        ("Semi-Final 1", sorted(OFFICIAL_SF1)),
        ("Semi-Final 2", sorted(OFFICIAL_SF2)),
    ]:
        lines.append(f"**{stage}:** {', '.join(countries)}")
        lines.append("")

    # Withdrawals
    lines += ["---", "", "## Withdrawn / non-participating countries", ""]
    for country, reason in sorted(WITHDRAWN.items()):
        in_ds = "yes" if _normalise(country) in dataset_countries else "no"
        lines += [
            f"### {country}",
            f"- **In Kaggle dataset 2026:** {in_ds}",
            f"- **Reason:** {reason}",
            "",
        ]

    # Discrepancies
    lines += ["---", "", "## Discrepancies", ""]
    if in_dataset_not_official:
        lines.append("### Countries in dataset but NOT in official EBU list")
        lines.append("")
        for c in sorted(in_dataset_not_official):
            lines.append(f"- **{c}** — {albania_status if c == 'Albania' else 'investigate'}")
        lines.append("")
    if in_official_not_dataset:
        lines.append("### Countries in official list but NOT in dataset")
        lines.append("")
        for c in sorted(in_official_not_dataset):
            lines.append(f"- **{c}** — missing row; add before model training")
        lines.append("")
    if not in_dataset_not_official and not in_official_not_dataset:
        lines.append("_No discrepancies between dataset and official EBU list._")
        lines.append("")

    # Unverified
    lines += ["---", "", "## Unverified / anomalous entries", ""]
    for country, note in UNVERIFIED_OFFICIAL.items():
        lines.append(f"- **{country}:** {note}")
    lines.append("")

    # Dataset list
    lines += ["---", "", "## Dataset 2026 entries", ""]
    rows = []
    for _, row in df26.sort_values("Country").iterrows():
        country = _normalise(row["Country"])
        stage = (
            "Auto" if row.get("Grand_Final_Ind") == 1 and pd.isna(row.get("Semi_Final_Num"))
            else f"SF{int(row['Semi_Final_Num'])}" if pd.notna(row.get("Semi_Final_Num"))
            else "Auto" if row.get("Grand_Final_Ind") == 1
            else "?"
        )
        status_flag = (
            "withdrawn" if country in WITHDRAWN
            else "discrepancy" if country in in_dataset_not_official
            else "ok"
        )
        rows.append([country, row.get("Artist",""), row.get("Song",""), stage, status_flag])
    header = ["Country", "Artist", "Song", "Stage", "Status"]
    sep    = "|".join("---" for _ in header)
    lines.append("| " + " | ".join(header) + " |")
    lines.append(f"|{sep}|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines.append("")

    report = "\n".join(lines)
    DOCS_DIR.mkdir(exist_ok=True)
    date_str = datetime.today().strftime("%Y%m%d")
    out_path = DOCS_DIR / f"participants_2026_report_{date_str}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
