"""
Scraper: eurovisionworld.com odds tables, 2018–2025
Output: Dataset/eurovision_odds_2018_2025.csv

Columns: year, rank, country, artist, song, win_pct, <bookmaker>...

HTML structure of .o_table:
  thead row 1 (class otrb): empty cells
  thead row 2: td (empty x3, "winning chance"), then <th data-bm="...">BOOKMAKER</th>
  tbody <tr data-dt="...">:
    td[0]: rank
    td[1]: empty (flag)
    td[2] class=odt: <a> → country text + <span>Artist - Song</span>
    td[3]: "52%" win percentage
    td[4..]: decimal odds per bookmaker (same order as thead th elements)
"""

import re
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://eurovisionworld.com/odds/eurovision-{year}"
# 2020 was cancelled but odds were collected pre-cancellation — include it
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://eurovisionworld.com/",
}

OUTPUT_PATH = (
    r"C:\Users\andrz\OneDrive\Pulpit\Projects\Eurovision 2026"
    r"\Dataset\eurovision_odds_2018_2025.csv"
)


def get_bookmaker_names(thead) -> list[str]:
    """Extract bookmaker column names from the second thead row (th elements)."""
    rows = thead.find_all("tr")
    if len(rows) < 2:
        return []
    header_row = rows[1]
    names = []
    for th in header_row.find_all("th"):
        name = th.get_text(separator=" ").strip()
        name = re.sub(r"\s+", " ", name)
        names.append(name)
    return names


def parse_win_pct(text: str) -> str:
    text = text.strip()
    return text.replace("%", "").strip() if text not in ("", "-", "–") else ""


def parse_odd(text: str) -> str:
    text = text.strip()
    cleaned = re.sub(r"[^\d.]", "", text)
    return cleaned if cleaned else ""


def parse_country_cell(td) -> tuple[str, str, str]:
    """Return (country, artist, song) from the odt cell."""
    a = td.find("a")
    if not a:
        return td.get_text(strip=True), "", ""

    span = a.find("span")
    artist_song = span.get_text(strip=True) if span else ""
    if span:
        span.extract()

    country = a.get_text(separator=" ").strip()
    # Remove leftover icon text (fa-* classes produce no text, but clean up spaces)
    country = re.sub(r"\s+", " ", country).strip()

    artist, song = "", ""
    if " - " in artist_song:
        parts = artist_song.split(" - ", 1)
        artist, song = parts[0].strip(), parts[1].strip()
    else:
        artist = artist_song

    return country, artist, song


def scrape_year(year: int) -> list[dict]:
    url = BASE_URL.format(year=year)
    print(f"  GET {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as e:
        print(f"  ERROR: {e}")
        return []

    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code} — skipping")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", class_="o_table")
    if not table:
        print(f"  WARNING: .o_table not found for {year}")
        return []

    thead = table.find("thead")
    bookmakers = get_bookmaker_names(thead) if thead else []
    print(f"  Bookmakers ({len(bookmakers)}): {bookmakers}")

    tbody = table.find("tbody")
    if not tbody:
        print(f"  WARNING: no tbody for {year}")
        return []

    records = []
    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        rank_text = cells[0].get_text(strip=True)
        rank = int(rank_text) if re.match(r"^\d+$", rank_text) else None

        country, artist, song = parse_country_cell(cells[2])
        win_pct = parse_win_pct(cells[3].get_text(strip=True))

        record: dict = {
            "year": year,
            "rank": rank,
            "country": country,
            "artist": artist,
            "song": song,
            "win_pct": win_pct,
        }

        # Odds start at cells[4]
        for i, bm in enumerate(bookmakers):
            idx = 4 + i
            record[bm] = parse_odd(cells[idx].get_text(strip=True)) if idx < len(cells) else ""

        records.append(record)

    print(f"  Parsed {len(records)} countries")
    return records


def main() -> None:
    all_records: list[dict] = []
    for year in YEARS:
        print(f"\nYear {year}:")
        records = scrape_year(year)
        all_records.extend(records)
        if year != YEARS[-1]:
            time.sleep(2)

    if not all_records:
        print("\nNo data collected — check network or page structure.")
        return

    df = pd.DataFrame(all_records)

    fixed_cols = ["year", "rank", "country", "artist", "song", "win_pct"]
    bookie_cols = [c for c in df.columns if c not in fixed_cols]
    df = df[fixed_cols + bookie_cols]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} rows to: {OUTPUT_PATH}")
    print(f"\nPer-year country count:")
    print(df.groupby("year")["country"].count().to_string())


if __name__ == "__main__":
    main()
