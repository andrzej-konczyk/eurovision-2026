"""Dashboard checks for Sprint 8 stories."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ukraine_out_scenario_filters_and_reranks():
    payload = json.loads((PROJECT_ROOT / "reports" / "predictions_2026.json").read_text(encoding="utf-8"))
    predictions_df = app.countries_frame(payload)

    scenario = app.apply_ukraine_scenario(predictions_df, include_ukraine=False)

    assert "Ukraine" not in set(scenario["country"])
    assert len(scenario) == len(predictions_df) - 1
    assert scenario["rank"].tolist() == list(range(1, len(scenario) + 1))


def test_ukraine_in_scenario_keeps_country_count():
    payload = json.loads((PROJECT_ROOT / "reports" / "predictions_2026.json").read_text(encoding="utf-8"))
    predictions_df = app.countries_frame(payload)

    scenario = app.apply_ukraine_scenario(predictions_df, include_ukraine=True)

    assert len(scenario) == len(predictions_df)
    assert "Ukraine" in set(scenario["country"])


def test_voting_bloc_d3_html_contains_d3_and_payload():
    cooccurrence = pd.read_csv(PROJECT_ROOT / "data" / "features" / "bloc_cooccurrence.csv")

    html = app.voting_bloc_d3_html(cooccurrence)

    assert "d3@7" in html
    assert "Ukraine" in html
    assert "South-Eastern" in html


def test_bloc_cooccurrence_long_frame_shape():
    cooccurrence = pd.read_csv(PROJECT_ROOT / "data" / "features" / "bloc_cooccurrence.csv")
    long = app.bloc_cooccurrence_long_frame(cooccurrence)

    assert {"country", "bloc", "member"} == set(long.columns)
    assert len(long) == len(cooccurrence) * (len(cooccurrence.columns) - 1)
