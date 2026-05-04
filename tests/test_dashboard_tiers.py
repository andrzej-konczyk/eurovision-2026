"""Dashboard tier view checks for US-S7-04."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def predictions_df():
    payload = json.loads((PROJECT_ROOT / "reports" / "predictions_2026.json").read_text(encoding="utf-8"))
    return app.countries_frame(payload)


def test_top3_position_probabilities_sum_to_one_per_column(predictions_df):
    position_df = app.position_probability_frame(predictions_df)

    column_sums = position_df.groupby("position")["probability"].sum()

    assert set(column_sums.index) == {"P1", "P2", "P3"}
    assert column_sums.to_dict() == pytest.approx({"P1": 1.0, "P2": 1.0, "P3": 1.0})


def test_top3_heatmap_has_country_by_position_shape(predictions_df):
    position_df = app.position_probability_frame(predictions_df)
    fig = app.top3_heatmap(position_df, predictions_df)

    assert fig.data[0].type == "heatmap"
    assert list(fig.data[0].x) == ["1st", "2nd", "3rd"]
    assert len(fig.data[0].y) == len(predictions_df)


def test_winner_gauge_uses_top_five_candidates(predictions_df):
    position_df = app.position_probability_frame(predictions_df)
    fig = app.winner_gauge_figure(position_df)

    assert len(fig.data) == 5
    assert all(trace.type == "indicator" for trace in fig.data)
