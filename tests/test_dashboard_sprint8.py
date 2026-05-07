"""Dashboard checks for Sprint 8 stories."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_voting_bloc_d3_html_contains_d3_and_payload():
    cooccurrence = pd.read_csv(PROJECT_ROOT / "data" / "features" / "bloc_cooccurrence.csv")

    html = app.voting_bloc_d3_html(cooccurrence)

    assert "d3@7" in html
    assert "Ukraine" in html
    assert "South-Eastern" in html
    assert '.style("background", "#ffffff")' in html
    assert '.attr("fill", "#0f172a")' in html


def test_bloc_cooccurrence_long_frame_shape():
    cooccurrence = pd.read_csv(PROJECT_ROOT / "data" / "features" / "bloc_cooccurrence.csv")
    long = app.bloc_cooccurrence_long_frame(cooccurrence)

    assert {"country", "bloc", "member"} == set(long.columns)
    assert len(long) == len(cooccurrence) * (len(cooccurrence.columns) - 1)


def test_voting_network_artifact_shape():
    network = json.loads((PROJECT_ROOT / "reports" / "voting_network_2026.json").read_text(encoding="utf-8"))

    assert len(network["nodes"]) == 35
    assert len(network["links"]) == 72


def test_voting_network_d3_html_contains_counts_and_top_edge():
    predictions = json.loads((PROJECT_ROOT / "reports" / "predictions_2026.json").read_text(encoding="utf-8"))
    network = json.loads((PROJECT_ROOT / "reports" / "voting_network_2026.json").read_text(encoding="utf-8"))
    predictions_df = app.countries_frame(predictions)

    html = app.voting_network_d3_html(network, predictions_df)

    assert 'data-node-count="35"' in html
    assert 'data-edge-count="72"' in html
    assert "d3@7" in html
    assert '"source": "Italy"' in html
    assert '"target": "Ukraine"' in html
    assert '"weight": 6' in html
    assert "top_partners" in html
    assert "d3.forceCenter(width / 2, height / 2)" in html
    assert "d3.forceCollide" in html
    assert "Math.max(r, Math.min(width - r, d.x))" in html


def test_semi_table_uses_country_flag_images():
    html = app.render_semi_table(
        [
            {
                "rank_in_semi": 1,
                "country": "Greece",
                "flag": "GR",
                "prob_qualify": 0.82,
                "ci80_lo": 0.7,
                "ci80_hi": 0.9,
            }
        ]
    )

    assert 'src="https://flagcdn.com/w40/gr.png"' in html
    assert 'alt="Greece flag"' in html
    assert ">GR<" not in html
