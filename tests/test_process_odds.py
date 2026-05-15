"""Unit tests for client odds ingestion."""

from __future__ import annotations

import pandas as pd

from src.data.process_odds import merge_client_rows


def _row(year: int, country: str, implied_prob: float, source: str) -> dict[str, object]:
    return {
        "year": year,
        "country": country,
        "country_code": country[:3].upper(),
        "odds_open": None,
        "odds_close": round(1.0 / implied_prob, 4),
        "implied_prob": implied_prob,
        "n_bookmakers": 1,
        "source": source,
    }


def test_merge_client_rows_overrides_matching_rows_and_appends_new_year() -> None:
    primary = pd.DataFrame(
        [
            _row(2025, "Finland", 0.10, "primary"),
            _row(2025, "Sweden", 0.20, "primary"),
        ]
    )
    client = pd.DataFrame(
        [
            _row(2025, "Finland", 0.30, "client"),
            _row(2026, "Finland", 0.40, "client"),
        ]
    )

    merged = merge_client_rows(primary, client)

    assert len(merged) == 3
    finland_2025 = merged[(merged["year"] == 2025) & (merged["country"] == "Finland")].iloc[0]
    finland_2026 = merged[(merged["year"] == 2026) & (merged["country"] == "Finland")].iloc[0]
    sweden_2025 = merged[(merged["year"] == 2025) & (merged["country"] == "Sweden")].iloc[0]

    assert finland_2025["implied_prob"] == 0.30
    assert finland_2025["source"] == "client"
    assert finland_2026["implied_prob"] == 0.40
    assert finland_2026["source"] == "client"
    assert sweden_2025["implied_prob"] == 0.20
    assert sweden_2025["source"] == "primary"
