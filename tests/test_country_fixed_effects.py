"""
Tests for src/features/country_fixed_effects.py — US-S3-01

Verifies:
  1. No leakage: current-year data is never included in features
  2. Window cap: at most 3 prior editions are averaged
  3. Correct averages on synthetic data
  4. NaN for countries with no prior final history
  5. Handles missing jury/tele gracefully (skipna)
  6. 2020 gap (skipped year) doesn't break the window
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.country_fixed_effects import compute_country_fixed_effects


def _make_df(records: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame shaped like the enriched CSV."""
    defaults = {
        "Grand_Final_Ind": 1,
        "Final_Place": np.nan,
        "jury_points": np.nan,
        "tele_points": np.nan,
    }
    rows = [{**defaults, **r} for r in records]
    return pd.DataFrame(rows)


# ── 1. No leakage ─────────────────────────────────────────────────────────────

def test_no_leakage_same_year_excluded():
    """Feature for year Y must NOT include year Y's grand final result."""
    df = _make_df([
        {"Year": 2022, "Country": "Alpha", "Final_Place": 5.0, "jury_points": 100.0, "tele_points": 80.0},
        {"Year": 2023, "Country": "Alpha", "Final_Place": 3.0, "jury_points": 150.0, "tele_points": 90.0},
    ])
    fe = compute_country_fixed_effects(df)
    row_2023 = fe[(fe["Country"] == "Alpha") & (fe["Year"] == 2023)].iloc[0]
    # Only 2022 data should be used for 2023 features
    assert row_2023["avg_final_rank_3yr"] == pytest.approx(5.0)
    assert row_2023["avg_jury_3yr"] == pytest.approx(100.0)
    assert row_2023["avg_tele_3yr"] == pytest.approx(80.0)


def test_no_leakage_future_year_not_included():
    """Features for year Y are unaffected by data from years > Y."""
    df = _make_df([
        {"Year": 2021, "Country": "Beta", "Final_Place": 10.0, "jury_points": 60.0, "tele_points": 40.0},
        {"Year": 2022, "Country": "Beta", "Final_Place": 8.0,  "jury_points": 70.0, "tele_points": 50.0},
        {"Year": 2024, "Country": "Beta", "Final_Place": 2.0,  "jury_points": 200.0, "tele_points": 100.0},
    ])
    fe = compute_country_fixed_effects(df)
    row_2022 = fe[(fe["Country"] == "Beta") & (fe["Year"] == 2022)].iloc[0]
    # 2022 feature uses only 2021
    assert row_2022["avg_final_rank_3yr"] == pytest.approx(10.0)
    assert row_2022["avg_jury_3yr"] == pytest.approx(60.0)


# ── 2. Window cap ─────────────────────────────────────────────────────────────

def test_window_capped_at_three():
    """With 5 prior editions, only the 3 most recent are averaged."""
    df = _make_df([
        {"Year": 2016, "Country": "Gamma", "Final_Place": 20.0, "jury_points": 10.0, "tele_points": 5.0},
        {"Year": 2017, "Country": "Gamma", "Final_Place": 18.0, "jury_points": 20.0, "tele_points": 10.0},
        {"Year": 2018, "Country": "Gamma", "Final_Place": 15.0, "jury_points": 30.0, "tele_points": 15.0},
        {"Year": 2019, "Country": "Gamma", "Final_Place": 12.0, "jury_points": 40.0, "tele_points": 20.0},
        {"Year": 2021, "Country": "Gamma", "Final_Place":  9.0, "jury_points": 50.0, "tele_points": 25.0},
        {"Year": 2022, "Country": "Gamma", "Final_Place":  6.0, "jury_points": 60.0, "tele_points": 30.0},
    ])
    fe = compute_country_fixed_effects(df)
    row_2022 = fe[(fe["Country"] == "Gamma") & (fe["Year"] == 2022)].iloc[0]
    # Prior to 2022: 2016, 2017, 2018, 2019, 2021 — 3 most recent = 2021, 2019, 2018
    expected_rank = (9.0 + 12.0 + 15.0) / 3
    expected_jury = (50.0 + 40.0 + 30.0) / 3
    expected_tele = (25.0 + 20.0 + 15.0) / 3
    assert row_2022["avg_final_rank_3yr"] == pytest.approx(expected_rank)
    assert row_2022["avg_jury_3yr"] == pytest.approx(expected_jury)
    assert row_2022["avg_tele_3yr"] == pytest.approx(expected_tele)


def test_window_uses_all_when_fewer_than_three():
    """With only 2 prior editions, both are averaged (no padding)."""
    df = _make_df([
        {"Year": 2018, "Country": "Delta", "Final_Place": 10.0, "jury_points": 50.0, "tele_points": 30.0},
        {"Year": 2019, "Country": "Delta", "Final_Place":  6.0, "jury_points": 80.0, "tele_points": 60.0},
        {"Year": 2021, "Country": "Delta", "Final_Place":  4.0, "jury_points": 90.0, "tele_points": 70.0},
    ])
    fe = compute_country_fixed_effects(df)
    row_2021 = fe[(fe["Country"] == "Delta") & (fe["Year"] == 2021)].iloc[0]
    # 2 prior years: 2018, 2019
    assert row_2021["avg_final_rank_3yr"] == pytest.approx((10.0 + 6.0) / 2)
    assert row_2021["avg_jury_3yr"] == pytest.approx((50.0 + 80.0) / 2)


# ── 3. NaN for first-time / non-qualifying countries ──────────────────────────

def test_nan_for_no_prior_history():
    """Country's first appearance returns NaN for all three features."""
    df = _make_df([
        {"Year": 2022, "Country": "Epsilon", "Final_Place": 5.0, "jury_points": 100.0, "tele_points": 80.0},
    ])
    fe = compute_country_fixed_effects(df)
    row = fe[(fe["Country"] == "Epsilon") & (fe["Year"] == 2022)].iloc[0]
    assert pd.isna(row["avg_final_rank_3yr"])
    assert pd.isna(row["avg_jury_3yr"])
    assert pd.isna(row["avg_tele_3yr"])


def test_nan_for_semi_finalist_with_no_final_history():
    """Semi-finalist (Grand_Final_Ind=0) with no prior final history → NaN."""
    df = _make_df([
        {"Year": 2022, "Country": "Zeta", "Grand_Final_Ind": 0, "Final_Place": np.nan, "jury_points": np.nan, "tele_points": np.nan},
    ])
    fe = compute_country_fixed_effects(df)
    row = fe[(fe["Country"] == "Zeta") & (fe["Year"] == 2022)].iloc[0]
    assert pd.isna(row["avg_final_rank_3yr"])


# ── 4. Partial NaN in jury/tele handled gracefully ───────────────────────────

def test_skipna_in_jury_tele():
    """If some years lack jury split, avg is computed over the non-NaN years only."""
    df = _make_df([
        {"Year": 2017, "Country": "Eta", "Final_Place": 8.0, "jury_points": np.nan, "tele_points": 40.0},
        {"Year": 2018, "Country": "Eta", "Final_Place": 5.0, "jury_points": 100.0, "tele_points": 60.0},
        {"Year": 2019, "Country": "Eta", "Final_Place": 3.0, "jury_points": 120.0, "tele_points": 80.0},
    ])
    fe = compute_country_fixed_effects(df)
    row_2019 = fe[(fe["Country"] == "Eta") & (fe["Year"] == 2019)].iloc[0]
    # rank: (8+5)/2 from 2017,2018
    assert row_2019["avg_final_rank_3yr"] == pytest.approx((8.0 + 5.0) / 2)
    # jury: only 2018 is non-NaN
    assert row_2019["avg_jury_3yr"] == pytest.approx(100.0)
    # tele: both 2017, 2018
    assert row_2019["avg_tele_3yr"] == pytest.approx((40.0 + 60.0) / 2)


# ── 5. 2020 gap doesn't corrupt the 3yr window ───────────────────────────────

def test_2020_gap_handled():
    """2019, (2020 missing), 2021, 2022 — window for 2022 uses 2019, 2021 only."""
    df = _make_df([
        {"Year": 2018, "Country": "Theta", "Final_Place": 15.0, "jury_points": 30.0, "tele_points": 20.0},
        {"Year": 2019, "Country": "Theta", "Final_Place": 12.0, "jury_points": 40.0, "tele_points": 25.0},
        # 2020: skipped (COVID)
        {"Year": 2021, "Country": "Theta", "Final_Place":  9.0, "jury_points": 50.0, "tele_points": 30.0},
        {"Year": 2022, "Country": "Theta", "Final_Place":  6.0, "jury_points": 60.0, "tele_points": 35.0},
    ])
    fe = compute_country_fixed_effects(df)
    row_2022 = fe[(fe["Country"] == "Theta") & (fe["Year"] == 2022)].iloc[0]
    # 3 most recent before 2022: 2021, 2019, 2018
    assert row_2022["avg_final_rank_3yr"] == pytest.approx((9.0 + 12.0 + 15.0) / 3)
    assert row_2022["avg_jury_3yr"] == pytest.approx((50.0 + 40.0 + 30.0) / 3)


# ── 6. Semi-finalist with prior final history uses that history ───────────────

def test_semi_finalist_gets_prior_final_history():
    """A country that fails to qualify still gets fixed-effects from past finals."""
    df = _make_df([
        {"Year": 2021, "Country": "Iota", "Grand_Final_Ind": 1, "Final_Place": 10.0, "jury_points": 60.0, "tele_points": 40.0},
        {"Year": 2022, "Country": "Iota", "Grand_Final_Ind": 0, "Final_Place": np.nan, "jury_points": np.nan, "tele_points": np.nan},
    ])
    fe = compute_country_fixed_effects(df)
    row_2022 = fe[(fe["Country"] == "Iota") & (fe["Year"] == 2022)].iloc[0]
    assert row_2022["avg_final_rank_3yr"] == pytest.approx(10.0)
    assert row_2022["avg_jury_3yr"] == pytest.approx(60.0)
    assert row_2022["avg_tele_3yr"] == pytest.approx(40.0)


# ── 7. Multiple countries — no cross-contamination ────────────────────────────

def test_no_cross_country_contamination():
    """Features for country A are not influenced by country B's results."""
    df = _make_df([
        {"Year": 2021, "Country": "Alpha", "Final_Place": 5.0,  "jury_points": 100.0, "tele_points": 80.0},
        {"Year": 2021, "Country": "Beta",  "Final_Place": 20.0, "jury_points": 10.0,  "tele_points": 5.0},
        {"Year": 2022, "Country": "Alpha", "Final_Place": 3.0,  "jury_points": 150.0, "tele_points": 100.0},
        {"Year": 2022, "Country": "Beta",  "Final_Place": 18.0, "jury_points": 15.0,  "tele_points": 8.0},
    ])
    fe = compute_country_fixed_effects(df)
    alpha_2022 = fe[(fe["Country"] == "Alpha") & (fe["Year"] == 2022)].iloc[0]
    beta_2022  = fe[(fe["Country"] == "Beta")  & (fe["Year"] == 2022)].iloc[0]
    assert alpha_2022["avg_final_rank_3yr"] == pytest.approx(5.0)
    assert beta_2022["avg_final_rank_3yr"]  == pytest.approx(20.0)
