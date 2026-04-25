"""Tests for src/features/voting_blocs.py — US-S3-03"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.voting_blocs import build_cooccurrence_matrix, compute_voting_blocs

WINDOW = 3


def _df(records: list[dict]) -> pd.DataFrame:
    defaults = {
        "Grand_Final_Ind": 1,
        "jury_points": np.nan,
        "tele_points": np.nan,
        "Country_Group": "TestBloc",
    }
    return pd.DataFrame([{**defaults, **r} for r in records])


# ── Co-occurrence matrix ───────────────────────────────────────────────────────

def test_cooccurrence_shape():
    df = _df([
        {"Year": 2022, "Country": "Alpha", "Country_Group": "West"},
        {"Year": 2022, "Country": "Beta",  "Country_Group": "West"},
        {"Year": 2022, "Country": "Gamma", "Country_Group": "East"},
    ])
    mat = build_cooccurrence_matrix(df)
    assert mat.shape == (3, 2)  # 3 countries, 2 blocs


def test_cooccurrence_values():
    df = _df([
        {"Year": 2022, "Country": "Alpha", "Country_Group": "West"},
        {"Year": 2022, "Country": "Beta",  "Country_Group": "East"},
    ])
    mat = build_cooccurrence_matrix(df)
    assert mat.loc["Alpha", "West"] == 1
    assert mat.loc["Alpha", "East"] == 0
    assert mat.loc["Beta",  "East"] == 1


# ── No leakage ─────────────────────────────────────────────────────────────────

def test_no_leakage_current_year_excluded():
    df = _df([
        {"Year": 2021, "Country": "A", "Country_Group": "B1", "jury_points": 100.0, "tele_points": 50.0},
        {"Year": 2021, "Country": "B", "Country_Group": "B1", "jury_points":  80.0, "tele_points": 40.0},
        {"Year": 2022, "Country": "A", "Country_Group": "B1", "jury_points": 200.0, "tele_points": 90.0},
        {"Year": 2022, "Country": "B", "Country_Group": "B1", "jury_points": 160.0, "tele_points": 70.0},
    ])
    fe = compute_voting_blocs(df)
    a_2022 = fe[(fe["Country"] == "A") & (fe["Year"] == 2022)].iloc[0]
    # Only 2021 data (B's 2022 result must not be included)
    assert a_2022["avg_bloc_jury_3yr"] == pytest.approx(80.0)
    assert a_2022["avg_bloc_tele_3yr"] == pytest.approx(40.0)


# ── Country excluded from its own bloc average ────────────────────────────────

def test_self_excluded_from_bloc_average():
    df = _df([
        {"Year": 2021, "Country": "Alpha", "Country_Group": "G", "jury_points": 999.0, "tele_points": 999.0},
        {"Year": 2021, "Country": "Beta",  "Country_Group": "G", "jury_points":  50.0, "tele_points":  30.0},
        {"Year": 2022, "Country": "Alpha", "Country_Group": "G"},
        {"Year": 2022, "Country": "Beta",  "Country_Group": "G"},
    ])
    fe = compute_voting_blocs(df)
    # Alpha's 2022 feature: only Beta's 2021 result (not Alpha's own 999)
    alpha_2022 = fe[(fe["Country"] == "Alpha") & (fe["Year"] == 2022)].iloc[0]
    assert alpha_2022["avg_bloc_jury_3yr"] == pytest.approx(50.0)
    assert alpha_2022["avg_bloc_tele_3yr"] == pytest.approx(30.0)


# ── Window: 3 most recent years where bloc-mates reached the final ─────────────

def test_window_capped_at_three_years():
    # Bloc-mate in 5 different years; window should pick 3 most recent
    df = _df([
        {"Year": 2016, "Country": "M", "Country_Group": "G", "jury_points": 10.0, "tele_points": 5.0},
        {"Year": 2017, "Country": "M", "Country_Group": "G", "jury_points": 20.0, "tele_points": 10.0},
        {"Year": 2018, "Country": "M", "Country_Group": "G", "jury_points": 30.0, "tele_points": 15.0},
        {"Year": 2019, "Country": "M", "Country_Group": "G", "jury_points": 40.0, "tele_points": 20.0},
        {"Year": 2021, "Country": "M", "Country_Group": "G", "jury_points": 50.0, "tele_points": 25.0},
        {"Year": 2022, "Country": "X", "Country_Group": "G"},
    ])
    fe = compute_voting_blocs(df)
    x_2022 = fe[(fe["Country"] == "X") & (fe["Year"] == 2022)].iloc[0]
    # 3 most recent years with bloc-mate M: 2021, 2019, 2018
    assert x_2022["avg_bloc_jury_3yr"] == pytest.approx((50 + 40 + 30) / 3)
    assert x_2022["avg_bloc_tele_3yr"] == pytest.approx((25 + 20 + 15) / 3)


def test_window_fewer_than_three_uses_all():
    df = _df([
        {"Year": 2021, "Country": "M", "Country_Group": "G", "jury_points": 60.0, "tele_points": 30.0},
        {"Year": 2022, "Country": "X", "Country_Group": "G"},
    ])
    fe = compute_voting_blocs(df)
    x_2022 = fe[(fe["Country"] == "X") & (fe["Year"] == 2022)].iloc[0]
    assert x_2022["avg_bloc_jury_3yr"] == pytest.approx(60.0)


# ── Multiple bloc-mates in same year are averaged ─────────────────────────────

def test_multiple_mates_same_year_averaged():
    df = _df([
        {"Year": 2021, "Country": "M1", "Country_Group": "G", "jury_points":  60.0, "tele_points": 40.0},
        {"Year": 2021, "Country": "M2", "Country_Group": "G", "jury_points": 100.0, "tele_points": 80.0},
        {"Year": 2022, "Country": "X",  "Country_Group": "G"},
    ])
    fe = compute_voting_blocs(df)
    x_2022 = fe[(fe["Country"] == "X") & (fe["Year"] == 2022)].iloc[0]
    assert x_2022["avg_bloc_jury_3yr"] == pytest.approx(80.0)   # (60+100)/2
    assert x_2022["avg_bloc_tele_3yr"] == pytest.approx(60.0)   # (40+80)/2


# ── NaN for solo-bloc country (no mates) ─────────────────────────────────────

def test_nan_when_no_bloc_mates():
    df = _df([
        {"Year": 2021, "Country": "Lonely", "Country_Group": "Unique"},
        {"Year": 2022, "Country": "Lonely", "Country_Group": "Unique"},
    ])
    fe = compute_voting_blocs(df)
    row = fe[(fe["Country"] == "Lonely") & (fe["Year"] == 2022)].iloc[0]
    assert pd.isna(row["avg_bloc_jury_3yr"])
    assert pd.isna(row["avg_bloc_tele_3yr"])


# ── NaN for first year (no prior history) ─────────────────────────────────────

def test_nan_for_first_year():
    df = _df([
        {"Year": 2022, "Country": "A", "Country_Group": "G", "jury_points": 100.0, "tele_points": 50.0},
        {"Year": 2022, "Country": "B", "Country_Group": "G", "jury_points":  80.0, "tele_points": 40.0},
    ])
    fe = compute_voting_blocs(df)
    # First year for both — no prior data
    for country in ["A", "B"]:
        row = fe[(fe["Country"] == country) & (fe["Year"] == 2022)].iloc[0]
        assert pd.isna(row["avg_bloc_jury_3yr"])


# ── Bloc label propagated correctly ───────────────────────────────────────────

def test_country_group_in_output():
    df = _df([{"Year": 2022, "Country": "X", "Country_Group": "MyBloc"}])
    fe = compute_voting_blocs(df)
    assert fe.iloc[0]["Country_Group"] == "MyBloc"


# ── Cross-bloc isolation ───────────────────────────────────────────────────────

def test_cross_bloc_isolation():
    """Country A's bloc features are not influenced by country C in a different bloc."""
    df = _df([
        {"Year": 2021, "Country": "B",  "Country_Group": "Bloc1", "jury_points":  50.0, "tele_points": 30.0},
        {"Year": 2021, "Country": "C",  "Country_Group": "Bloc2", "jury_points": 200.0, "tele_points": 150.0},
        {"Year": 2022, "Country": "A",  "Country_Group": "Bloc1"},
        {"Year": 2022, "Country": "D",  "Country_Group": "Bloc2"},
    ])
    fe = compute_voting_blocs(df)
    a_2022 = fe[(fe["Country"] == "A") & (fe["Year"] == 2022)].iloc[0]
    assert a_2022["avg_bloc_jury_3yr"] == pytest.approx(50.0)   # only B, not C
    assert a_2022["avg_bloc_tele_3yr"] == pytest.approx(30.0)


# ── 2020 gap handled ───────────────────────────────────────────────────────────

def test_2020_gap_handled():
    df = _df([
        {"Year": 2018, "Country": "M", "Country_Group": "G", "jury_points": 30.0, "tele_points": 10.0},
        {"Year": 2019, "Country": "M", "Country_Group": "G", "jury_points": 40.0, "tele_points": 20.0},
        # 2020: skipped
        {"Year": 2021, "Country": "M", "Country_Group": "G", "jury_points": 50.0, "tele_points": 25.0},
        {"Year": 2022, "Country": "X", "Country_Group": "G"},
    ])
    fe = compute_voting_blocs(df)
    x_2022 = fe[(fe["Country"] == "X") & (fe["Year"] == 2022)].iloc[0]
    assert x_2022["avg_bloc_jury_3yr"] == pytest.approx((50 + 40 + 30) / 3)
