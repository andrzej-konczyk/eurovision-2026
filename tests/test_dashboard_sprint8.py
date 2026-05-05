"""Dashboard checks for Sprint 8 stories."""

from __future__ import annotations

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


def test_bloc_cooccurrence_long_frame_shape():
    cooccurrence = pd.read_csv(PROJECT_ROOT / "data" / "features" / "bloc_cooccurrence.csv")
    long = app.bloc_cooccurrence_long_frame(cooccurrence)

    assert {"country", "bloc", "member"} == set(long.columns)
    assert len(long) == len(cooccurrence) * (len(cooccurrence.columns) - 1)
