"""Overview dashboard checks for Sprint 10."""

from __future__ import annotations

import json
from pathlib import Path

import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prediction_update_timestamp_formats_generated_at_as_utc():
    payload = json.loads((PROJECT_ROOT / "reports" / "predictions_2026.json").read_text(encoding="utf-8"))

    formatted = app.format_prediction_update_timestamp(payload["generated_at"])

    assert formatted == "2026-05-04"


def test_prediction_update_timestamp_handles_missing_value():
    assert app.format_prediction_update_timestamp(None) == "n/a"


def test_overview_leaderboard_returns_top_five_by_rank():
    payload = json.loads((PROJECT_ROOT / "reports" / "predictions_2026.json").read_text(encoding="utf-8"))
    predictions_df = app.countries_frame(payload)

    leaders = app.overview_leaderboard_frame(predictions_df)

    assert len(leaders) == 5
    assert leaders["rank"].tolist() == [1, 2, 3, 4, 5]
    assert {"rank", "country", "probability", "ci80_lo", "ci80_hi"}.issubset(leaders.columns)
