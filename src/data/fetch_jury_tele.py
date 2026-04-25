"""
Fetch jury + televote split from EurovisionWorld — US-S2-01
Usage:  python src/data/fetch_jury_tele.py
Output: Dataset/jury_tele_raw.csv

Data source: EurovisionWorld year-specific JS files
  https://pix.eurovisionworld.com/scripts/js/voting/{event_id}.js

Structure of voting_table_main:
  {iso2: [run_order, place, total, jury, tele, qualified, sf_num]}
Structure of voting_table_sub (list of dicts, one per semi):
  {iso2: [run_order, place_overall, total, jury, tele, qualified, sf_num, ?]}
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "Dataset" / "jury_tele_raw.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
BASE_JS = "https://pix.eurovisionworld.com/scripts/js/voting/{event_id}.js"

# event IDs discovered by probing each year's page
YEAR_EVENT: dict[int, int] = {
    2016: 87,
    2017: 93,
    2018: 100,
    2019: 110,
    2021: 130,
    2022: 140,
    2023: 150,
    2024: 160,
    2025: 170,
}

# ISO 3166-1 alpha-2 → Country name as used in the Kaggle CSV
ISO_TO_COUNTRY: dict[str, str] = {
    "al": "Albania",
    "am": "Armenia",
    "au": "Australia",
    "at": "Austria",
    "az": "Azerbaijan",
    "be": "Belgium",
    "by": "Belarus",
    "ba": "Bosnia & Herzegovina",
    "bg": "Bulgaria",
    "hr": "Croatia",
    "cy": "Cyprus",
    "cz": "Czech Republic",
    "dk": "Denmark",
    "ee": "Estonia",
    "fi": "Finland",
    "fr": "France",
    "ge": "Georgia",
    "de": "Germany",
    "gr": "Greece",
    "hu": "Hungary",
    "is": "Iceland",
    "ie": "Ireland",
    "il": "Israel",
    "it": "Italy",
    "lv": "Latvia",
    "lt": "Lithuania",
    "lu": "Luxembourg",
    "mt": "Malta",
    "md": "Moldova",
    "me": "Montenegro",
    "nl": "Netherlands",
    "mk": "North Macedonia",
    "no": "Norway",
    "pl": "Poland",
    "pt": "Portugal",
    "ro": "Romania",
    "ru": "Russia",
    "sm": "San Marino",
    "rs": "Serbia",
    "si": "Slovenia",
    "es": "Spain",
    "se": "Sweden",
    "ch": "Switzerland",
    "ua": "Ukraine",
    "gb": "United Kingdom",
    "wld": None,  # "Rest of world" online vote — skip
}


def _extract_var(js_text: str, var_name: str) -> str | None:
    """Extract the value of a top-level JS variable assignment."""
    pattern = rf'{var_name}\s*=\s*([\[\{{].+?);\s*(?:voting_|\Z)'
    m = re.search(pattern, js_text, re.DOTALL)
    return m.group(1) if m else None


def _parse_year(year: int, event_id: int) -> list[dict]:
    url = BASE_JS.format(event_id=event_id)
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    txt = r.text

    records: list[dict] = []

    # ── Grand Final ──────────────────────────────────────────────────────────
    main_raw = _extract_var(txt, "voting_table_main")
    if main_raw:
        main: dict = json.loads(main_raw)
        for iso, vals in main.items():
            country = ISO_TO_COUNTRY.get(iso)
            if country is None:
                continue
            # vals: [run_order, place, total, jury, tele, qualified, sf_num]
            jury  = vals[3] if vals[3] != -1 else None
            tele  = vals[4] if vals[4] != -1 else None
            total = vals[2]
            records.append({
                "year":        year,
                "country":     country,
                "iso2":        iso,
                "stage":       "final",
                "place":       vals[1],
                "jury_points": jury,
                "tele_points": tele,
                "total_points": total,
            })

    # ── Semi-finals ──────────────────────────────────────────────────────────
    sub_raw = _extract_var(txt, "voting_table_sub")
    if sub_raw:
        sub_list: list[dict] = json.loads(sub_raw)
        for sf_idx, sf_dict in enumerate(sub_list, start=1):
            for iso, vals in sf_dict.items():
                country = ISO_TO_COUNTRY.get(iso)
                if country is None:
                    continue
                jury  = vals[3] if vals[3] != -1 else None
                tele  = vals[4] if vals[4] != -1 else None
                records.append({
                    "year":        year,
                    "country":     country,
                    "iso2":        iso,
                    "stage":       f"sf{sf_idx}",
                    "place":       vals[1],
                    "jury_points": jury,
                    "tele_points": tele,
                    "total_points": vals[2],
                })

    return records


def main() -> None:
    all_records: list[dict] = []

    for year, event_id in sorted(YEAR_EVENT.items()):
        try:
            records = _parse_year(year, event_id)
            print(f"{year}: {len(records)} rows fetched (event {event_id})")
            all_records.extend(records)
        except Exception as exc:
            print(f"{year}: ERROR — {exc}")
        time.sleep(0.5)  # polite crawl rate

    df = pd.DataFrame(all_records)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows -> {OUT_CSV.relative_to(ROOT)}")
    print(df.groupby("stage")["year"].nunique().rename("years_covered"))


if __name__ == "__main__":
    main()
