"""Overview dashboard checks for Sprint 10."""

from __future__ import annotations

import json
from pathlib import Path

import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prediction_update_timestamp_formats_generated_at_as_utc():
    payload = json.loads((PROJECT_ROOT / "reports" / "predictions_2026.json").read_text(encoding="utf-8"))

    formatted = app.format_prediction_update_timestamp(payload["generated_at"])

    assert formatted == "2026-05-04 15:19 UTC"


def test_prediction_update_timestamp_handles_missing_value():
    assert app.format_prediction_update_timestamp(None) == "n/a"
