"""
Patch 2026 dataset entries with official data — US-S2-04 hotfix
Usage:  python src/data/patch_2026_entries.py
Source: Official Wikipedia / Polish Eurovision page, verified 2026-04-25

Corrections applied:
  - Semi_Final_Num: corrected for all SF participants
  - Running_Order_Semi: set from official SF running order
  - Artist / Song: corrected where dataset differed from official
  - Grand_Final_Ind: set 1 for Big5 + Austria (host)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
ENRICHED = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"

# Official data structure: {country: (sf_num, running_order, artist, song)}
OFFICIAL_2026: dict[str, tuple[int, int, str, str]] = {
    # SF1 (12 May 2026)
    "Moldova":        (1,  1, "Satoshi",                          "Viva, Moldova!"),
    "Sweden":         (1,  2, "Felicia",                          "My System"),
    "Croatia":        (1,  3, "Lelek",                            "Andromeda"),
    "Greece":         (1,  4, "Akylas",                           "Ferto"),
    "Portugal":       (1,  5, "Bandidos do Cante",                "Rosa"),
    "Georgia":        (1,  6, "Bzikebi",                          "On Replay"),
    "Finland":        (1,  7, "Linda Lampenius & Pete Parkkonen", "Liekinheitin"),
    "Montenegro":     (1,  8, "Tamara Zivkovic",                  "Nova zora"),
    "Estonia":        (1,  9, "Vanilla Ninja",                    "Too Epic To Be True"),
    "Israel":         (1, 10, "No'am Bettan",                     "Michelle"),
    "Belgium":        (1, 11, "Essyla",                           "Dancing on the Ice"),
    "Lithuania":      (1, 12, "Lion Ceccah",                      "Solo quiero mas"),
    "San Marino":     (1, 13, "Senhit feat. Boy George",          "Superstar"),
    "Poland":         (1, 14, "Alicja Szemplinska",               "Pray"),
    "Serbia":         (1, 15, "Lavina",                           "Kraj mene"),
    # SF2 (14 May 2026)
    "Bulgaria":       (2,  1, "Dara",                             "Bangaranga"),
    "Azerbaijan":     (2,  2, "Jiva",                             "Just Go"),
    "Romania":        (2,  3, "Alexandra Capitanescu",            "Choke Me"),
    "Luxembourg":     (2,  4, "Eva Marija",                       "Mother Nature"),
    "Czech Republic": (2,  5, "Daniel Zizka",                     "Crossroads"),
    "Armenia":        (2,  6, "Simon",                            "Paloma Rumba"),
    "Switzerland":    (2,  7, "Veronica Fusaro",                  "Alice"),
    "Cyprus":         (2,  8, "Antigoni",                         "Jalla"),
    "Latvia":         (2,  9, "Atvara",                           "Ena"),
    "Denmark":        (2, 10, "Soren Torpegaard Lund",            "For vi gar hjem"),
    "Australia":      (2, 11, "Delta Goodrem",                    "Eclipse"),
    "Ukraine":        (2, 12, "Leleka",                           "Ridnym"),
    "Albania":        (2, 13, "Alis",                             "Nan"),
    "Malta":          (2, 14, "Aidan",                            "Bella"),
    "Norway":         (2, 15, "Jonas Lovv",                       "Ya Ya Ya"),
    # Auto-finalists (Big 5 + host)
    "Austria":        (None, None, "Cosmo",                       "Tanzschein"),
    "France":         (None, None, "Monroe",                      "Regarde!"),
    "Germany":        (None, None, "Sarah Engels",                "Fire"),
    "United Kingdom": (None, None, "Look Mum No Computer",       "Eins, zwei, drei"),
    "Italy":          (None, None, "Sal da Vinci",                "Per sempre si"),
}

AUTO_FINALISTS = {"Austria", "France", "Germany", "United Kingdom", "Italy"}


def main() -> None:
    df = pd.read_csv(ENRICHED, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    mask_2026 = df["Year"] == 2026
    before = df[mask_2026][["Country", "Semi_Final_Num", "Running_Order_Semi",
                             "Grand_Final_Ind", "Artist", "Song"]].copy()

    changes: list[str] = []

    for country, (sf_num, run_ord, artist, song) in OFFICIAL_2026.items():
        row_mask = mask_2026 & (df["Country"] == country)
        if not row_mask.any():
            changes.append(f"  MISSING: {country} not found in dataset")
            continue

        idx = df[row_mask].index[0]
        row_before = df.loc[idx]

        # Semi_Final_Num
        if sf_num is not None:
            if df.loc[idx, "Semi_Final_Num"] != float(sf_num):
                changes.append(f"  {country}: Semi_Final_Num {df.loc[idx,'Semi_Final_Num']} -> {sf_num}")
            df.loc[idx, "Semi_Final_Num"] = float(sf_num)
        else:
            df.loc[idx, "Semi_Final_Num"] = float("nan")

        # Running_Order_Semi
        if run_ord is not None:
            if df.loc[idx, "Running_Order_Semi"] != float(run_ord):
                changes.append(f"  {country}: Running_Order_Semi {df.loc[idx,'Running_Order_Semi']} -> {run_ord}")
            df.loc[idx, "Running_Order_Semi"] = float(run_ord)

        # Grand_Final_Ind for auto qualifiers
        if country in AUTO_FINALISTS:
            df.loc[idx, "Grand_Final_Ind"] = 1.0

        # Artist
        if str(row_before.get("Artist", "")).strip() != artist:
            changes.append(f"  {country}: Artist '{row_before.get('Artist','')}' -> '{artist}'")
            df.loc[idx, "Artist"] = artist

        # Song
        if str(row_before.get("Song", "")).strip() != song:
            changes.append(f"  {country}: Song '{row_before.get('Song','')}' -> '{song}'")
            df.loc[idx, "Song"] = song

    if changes:
        print(f"Changes applied ({len(changes)}):")
        for c in changes:
            print(c)
    else:
        print("No changes needed.")

    df.to_csv(ENRICHED, index=False, encoding="utf-8")
    print(f"\nSaved: {ENRICHED.relative_to(ROOT)}")

    # Verification
    df26 = df[df["Year"] == 2026]
    sf1 = df26[df26["Semi_Final_Num"] == 1.0]
    sf2 = df26[df26["Semi_Final_Num"] == 2.0]
    auto = df26[df26["Grand_Final_Ind"] == 1.0]
    print(f"\nVerification:")
    print(f"  SF1: {len(sf1)} countries (expected 15)")
    print(f"  SF2: {len(sf2)} countries (expected 15)")
    print(f"  Auto-finalists (Grand_Final_Ind=1): {len(auto)} (expected 5)")
    print(f"  SF1 countries: {sorted(sf1['Country'].tolist())}")
    print(f"  SF2 countries: {sorted(sf2['Country'].tolist())}")
    print(f"  Auto countries: {sorted(auto['Country'].tolist())}")


if __name__ == "__main__":
    main()
