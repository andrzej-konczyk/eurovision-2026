# Eurovision 2025 Data Sources

Sprint 8 data checkpoint for the 2025 holdout backtest.

## Results and Jury/Tele Split

- Official Eurovision Basel 2025 Grand Final scoreboard:
  https://www.eurovision.com/eurovision-song-contest/basel-2025/basel-2025-grand-final/
- Verification date: 2026-05-04.
- The local `jury_points` / `tele_points` fields were corrected to match the official scoreboard labels.

## Odds

- Local source file: `Dataset/eurovision_odds_2018_2025.csv`
- Processed file: `Dataset/betting_odds_clean.csv`
- 2025 coverage: 26 Grand Final countries with closing winner-market odds.
- Processing: harmonic consensus odds across available bookmakers, overround-normalised to `implied_prob`.
- External cross-check: public search on 2026-05-04 did not surface a stable Oddsportal Eurovision 2025 archive URL suitable for automated ingestion; the local 2025 odds snapshot was retained as the reproducible project source.

## Local Artifacts Updated

- `Dataset/jury_tele_raw.csv`
- `Dataset/eurovision_2016_26_enriched.csv`
- `data/features/voting_blocs.csv`
- `data/features/bloc_cooccurrence.csv`
