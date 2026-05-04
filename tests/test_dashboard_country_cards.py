"""Dashboard country-card checks for US-S7-05."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def dashboard_payloads():
    predictions = json.loads((PROJECT_ROOT / "reports" / "predictions_2026.json").read_text(encoding="utf-8"))
    narratives = json.loads((PROJECT_ROOT / "reports" / "narratives_2026.json").read_text(encoding="utf-8"))
    history = app.load_csv(
        str(PROJECT_ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"),
        (PROJECT_ROOT / "Dataset" / "eurovision_2016_26_enriched.csv").stat().st_mtime_ns,
    )
    return predictions, narratives, history


def test_all_35_country_cards_have_non_empty_narratives(dashboard_payloads):
    predictions, narratives, history = dashboard_payloads
    predictions_df = app.countries_frame(predictions)

    cards = [
        app.country_card_data(country, predictions_df, narratives, history)
        for country in predictions_df["country"]
    ]

    assert len(cards) == 35
    for card in cards:
        assert card["flag"]
        assert str(card["narrative"].get("narrative", "")).strip()
        assert not card["features"].empty
        assert not card["ci"].empty
        assert {"Year", "Result", "Final_Place", "Final_Points", "Semi_Place"}.issubset(card["history"].columns)


def test_main_ranking_defaults_to_all_35_countries(dashboard_payloads):
    predictions, _, _ = dashboard_payloads
    predictions_df = app.countries_frame(predictions)
    ranking = app.main_ranking_frame(predictions_df)

    assert len(predictions_df) == 35
    assert len(ranking) == 35
    assert set(ranking["country"]) == set(predictions_df["country"])


def test_narrative_probabilities_match_current_predictions(dashboard_payloads):
    predictions, narratives, _ = dashboard_payloads
    predictions_df = app.countries_frame(predictions)
    expected = {
        row["country"]: float(row["probability"])
        for _, row in predictions_df.iterrows()
    }

    for row in narratives["countries"]:
        country = row["country"]
        assert row["probability"] == pytest.approx(round(expected[country], 4), abs=0.0001)
        match = re.search(r"model probability: ([0-9]+)%", row["narrative"])
        assert match, country
        assert int(match.group(1)) == round(expected[country] * 100)


def test_country_card_charts_render_for_all_countries(dashboard_payloads):
    predictions, narratives, history = dashboard_payloads
    predictions_df = app.countries_frame(predictions)

    for country in predictions_df["country"]:
        card = app.country_card_data(country, predictions_df, narratives, history)
        assert app.feature_bar_chart(card["features"]).data
        assert app.ci_fan_chart(card["ci"]).data
        assert app.history_chart(card["history"]).layout.title.text == "Final history 2016-2024"
