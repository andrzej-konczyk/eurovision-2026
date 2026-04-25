"""
Genre features — US-S3-04  [SHOULD — target ≥90% coverage]
Usage:  python src/features/genre.py [--fetch]
Output: data/features/genre.csv
        data/features/genre_cache.json  (committed; avoids repeat API calls)

Fetch chain (per song):
  1. Spotify search (if SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET in env)
  2. MusicBrainz recording search (free, no key required)
  3. NaN (unresolved)

Broad genre categories produced (binary flags):
  genre_pop, genre_ballad, genre_dance, genre_folk, genre_rock, genre_classical

Mapping is keyword-based over raw genre tags returned by each API.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_CSV = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
CACHE_JSON   = ROOT / "data" / "features" / "genre_cache.json"
OUT_CSV      = ROOT / "data" / "features" / "genre.csv"

BROAD_GENRES = ["pop", "ballad", "dance", "folk", "rock", "classical"]

# Keyword → broad genre mapping (checked in order; first match wins)
GENRE_RULES: list[tuple[str, str]] = [
    (r"class|opera|orches|symphon|baroque",                 "classical"),
    (r"metal|punk|hard rock|indie rock|alt.rock",           "rock"),
    (r"rock(?!abil)",                                        "rock"),
    (r"folk|world|ethnic|trad|country|celtic|flam",         "folk"),
    (r"ballad|slow|power.?ballad",                           "ballad"),
    # synth-pop / electropop are pop sub-genres — check before broad dance/synth
    (r"synth.?pop|electro.?pop|indie.?pop|dream.?pop",      "pop"),
    (r"danc|electr|techno|house|edm|synth|disco|"
     r"eurodanc|trance|rave|club",                           "dance"),
    (r"pop|r.?b|soul|hip.?hop|rap|funk|rnb",                "pop"),
]

MB_BASE  = "https://musicbrainz.org/ws/2"
SP_AUTH  = "https://accounts.spotify.com/api/token"
SP_SEARCH = "https://api.spotify.com/v1/search"
HEADERS_MB = {
    "User-Agent": "Eurovision2026Predictor/1.0 (andrzej.konczyk@gmail.com)",
    "Accept": "application/json",
}


# ── Genre mapping ─────────────────────────────────────────────────────────────

def _map_genre(raw_tags: list[str]) -> Optional[str]:
    """Return the first broad genre that matches any of the raw tags."""
    joined = " ".join(raw_tags).lower()
    for pattern, genre in GENRE_RULES:
        if re.search(pattern, joined):
            return genre
    return None


def _flags(broad: Optional[str]) -> dict[str, int]:
    return {f"genre_{g}": int(broad == g) for g in BROAD_GENRES}


# ── Spotify ───────────────────────────────────────────────────────────────────

def _sp_token(client_id: str, client_secret: str) -> Optional[str]:
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post(
        SP_AUTH,
        headers={"Authorization": f"Basic {creds}"},
        data={"grant_type": "client_credentials"},
        timeout=10,
    )
    if r.ok:
        return r.json().get("access_token")
    return None


def _sp_genres(artist: str, song: str, token: str) -> list[str]:
    r = requests.get(
        SP_SEARCH,
        headers={"Authorization": f"Bearer {token}"},
        params={"q": f"track:{song} artist:{artist}", "type": "track", "limit": 1},
        timeout=10,
    )
    if not r.ok:
        return []
    items = r.json().get("tracks", {}).get("items", [])
    if not items:
        return []
    artist_id = items[0].get("artists", [{}])[0].get("id")
    if not artist_id:
        return []
    ar = requests.get(
        f"https://api.spotify.com/v1/artists/{artist_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    return ar.json().get("genres", []) if ar.ok else []


# ── MusicBrainz ───────────────────────────────────────────────────────────────

def _mb_genres(artist: str, song: str) -> list[str]:
    params = {
        "query": f'recording:"{song}" AND artist:"{artist}"',
        "fmt": "json",
        "limit": 3,
    }
    r = requests.get(f"{MB_BASE}/recording/", headers=HEADERS_MB, params=params, timeout=15)
    if not r.ok:
        return []
    recordings = r.json().get("recordings", [])
    tags: list[str] = []
    for rec in recordings[:2]:
        tags += [t["name"] for t in rec.get("tags", [])]
    return tags


# ── Fetch orchestrator ────────────────────────────────────────────────────────

def fetch_genres(
    pairs: list[tuple[str, str]],
    cache: dict,
    sp_token: Optional[str] = None,
    delay: float = 0.35,
) -> dict:
    """
    Fetch genre tags for each (artist, song) pair not already in *cache*.
    Updates *cache* in place and returns it.
    """
    for artist, song in pairs:
        key = f"{artist}||{song}"
        if key in cache:
            continue

        tags: list[str] = []

        if sp_token:
            try:
                tags = _sp_genres(artist, song, sp_token)
            except Exception:
                pass
            time.sleep(delay)

        if not tags:
            try:
                tags = _mb_genres(artist, song)
            except Exception:
                pass
            time.sleep(delay)

        broad = _map_genre(tags)
        cache[key] = {"raw_tags": tags, "broad_genre": broad}
        print(f"  {artist} - {song}: {tags[:3]} -> {broad}")

    return cache


# ── Feature builder ───────────────────────────────────────────────────────────

def compute_genre_features(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        key = f"{row['Artist']}||{row['Song']}"
        entry = cache.get(key, {})
        broad = entry.get("broad_genre")
        rows.append({
            "Year": row["Year"],
            "Country": row["Country"],
            "broad_genre": broad,
            **_flags(broad),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(do_fetch: bool = False) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env")

    df = pd.read_csv(ENRICHED_CSV, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    # 2026 entries have no result yet but have Artist/Song
    df = df.dropna(subset=["Artist", "Song"])

    cache: dict = {}
    if CACHE_JSON.exists():
        cache = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
    print(f"Cache loaded: {len(cache)} entries")

    if do_fetch:
        sp_token = None
        cid = os.getenv("SPOTIFY_CLIENT_ID", "")
        csec = os.getenv("SPOTIFY_CLIENT_SECRET", "")
        if cid and csec:
            sp_token = _sp_token(cid, csec)
            print("Spotify token:", "OK" if sp_token else "FAILED")
        else:
            print("Spotify credentials not set — using MusicBrainz only")

        pairs = list({(r["Artist"], r["Song"]) for _, r in df.iterrows()})
        print(f"Fetching {len(pairs)} unique artist/song pairs …")
        cache = fetch_genres(pairs, cache, sp_token=sp_token)

        CACHE_JSON.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Cache saved: {CACHE_JSON.relative_to(ROOT)} ({len(cache)} entries)")

    fe = compute_genre_features(df, cache)

    covered = fe["broad_genre"].notna().sum()
    pct = covered / len(fe) * 100
    status = "OK >=90%" if pct >= 90 else "WARN <90% -- run with --fetch"
    print(f"\nGenre coverage: {covered}/{len(fe)} ({pct:.1f}%)  {status}")
    print(fe["broad_genre"].value_counts(dropna=False).to_string())

    fe.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT_CSV.relative_to(ROOT)}  shape={fe.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true",
                        help="Call Spotify/MusicBrainz APIs to populate genre cache")
    args = parser.parse_args()
    main(do_fetch=args.fetch)
