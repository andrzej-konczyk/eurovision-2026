# Sprint 11 - Running Order and Closing Odds Refresh

Operational checklist for the Eurovision 2026 Grand Final prediction refresh.

## Trigger 1: Grand Final Running Order Draw

The final running order is normally known after the Second Semi-Final.
Do not refresh final predictions until all 25 Grand Final countries are known.

1. Update `Dataset/eurovision_2016_26_enriched.csv`.
2. Set `Grand_Final_Ind=1` for all 25 Grand Final countries.
3. Fill `Running_Order_Final` with a unique sequence `1..25`.
4. Leave non-finalists with `Grand_Final_Ind=0` and blank `Running_Order_Final`.
5. Validate without running the model pipeline:

```powershell
py scripts\refresh_final_predictions.py
```

6. If validation passes, refresh artefacts:

```powershell
py scripts\refresh_final_predictions.py --run
```

The script runs:

```powershell
py -m src.models.train --data Dataset\eurovision_2016_26_enriched.csv
py -m src.models.confidence --data Dataset\eurovision_2016_26_enriched.csv
py -m src.models.narratives --data Dataset\eurovision_2016_26_enriched.csv
py scripts\build_predictions_json.py
```

`reports/predictions_2026.json` is rebuilt last. The Streamlit dashboard reloads it
through file mtime-based caching.

## Trigger 2: Closing Odds Refresh

The day before the Grand Final, ingest the latest closing odds and repeat the
same prediction refresh.

Expected client odds schema follows `src/data/process_odds.py`:

```text
year,country,odds_open,odds_close
```

Run:

```powershell
py scripts\refresh_final_predictions.py --odds-client-file path\to\closing_odds.csv --run
```

This first runs:

```powershell
py -m src.data.process_odds --client-file path\to\closing_odds.csv
```

Then it refreshes train, confidence intervals, narratives, and
`reports/predictions_2026.json`.

## Safety Gates

- The script fails if 2026 has anything other than 25 Grand Final rows.
- The script fails if any finalist is missing `Running_Order_Final`.
- The script fails if `Running_Order_Final` is not a unique `1..25` sequence.
- The default mode is dry-run; use `--run` only after reviewing the CSV changes.

