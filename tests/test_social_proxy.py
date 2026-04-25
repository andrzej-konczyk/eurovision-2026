"""Tests for src/features/social_proxy.py — US-S3-05"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.social_proxy import compute_social_proxy


def _df(records: list[dict]) -> pd.DataFrame:
    base = {"MyESB_Community": np.nan, "MyESB_Personal": np.nan, "OGAE_Points": np.nan}
    return pd.DataFrame([{**base, **r} for r in records])


# ── Per-year mean is zero ──────────────────────────────────────────────────────

def test_per_year_mean_zero():
    df = _df([
        {"Year": 2022, "Country": "A", "MyESB_Community": 10.0, "MyESB_Personal": 5.0,  "OGAE_Points": 100.0},
        {"Year": 2022, "Country": "B", "MyESB_Community": 20.0, "MyESB_Personal": 15.0, "OGAE_Points": 200.0},
        {"Year": 2022, "Country": "C", "MyESB_Community": 30.0, "MyESB_Personal": 25.0, "OGAE_Points": 300.0},
    ])
    fe = compute_social_proxy(df)
    for col in ["zscore_myesb_community", "zscore_myesb_personal", "zscore_ogae_points"]:
        assert abs(fe[col].mean()) < 1e-9, f"{col} per-year mean not zero"


# ── Per-year std is 1 (ddof=1, n>1) ──────────────────────────────────────────

def test_per_year_std_one():
    df = _df([
        {"Year": 2022, "Country": "A", "MyESB_Community": 10.0, "MyESB_Personal": 5.0,  "OGAE_Points": 100.0},
        {"Year": 2022, "Country": "B", "MyESB_Community": 20.0, "MyESB_Personal": 15.0, "OGAE_Points": 200.0},
        {"Year": 2022, "Country": "C", "MyESB_Community": 30.0, "MyESB_Personal": 25.0, "OGAE_Points": 300.0},
    ])
    fe = compute_social_proxy(df)
    assert abs(fe["zscore_myesb_community"].std(ddof=1) - 1.0) < 1e-9


# ── Relative ordering preserved ────────────────────────────────────────────────

def test_ordering_preserved():
    df = _df([
        {"Year": 2022, "Country": "A", "MyESB_Community": 5.0},
        {"Year": 2022, "Country": "B", "MyESB_Community": 15.0},
        {"Year": 2022, "Country": "C", "MyESB_Community": 25.0},
    ])
    fe = compute_social_proxy(df).sort_values("Country")
    zs = fe["zscore_myesb_community"].tolist()
    assert zs[0] < zs[1] < zs[2]


# ── Years are normalised independently ────────────────────────────────────────

def test_years_normalised_independently():
    df = _df([
        {"Year": 2021, "Country": "A", "MyESB_Community": 1.0},
        {"Year": 2021, "Country": "B", "MyESB_Community": 3.0},
        {"Year": 2022, "Country": "A", "MyESB_Community": 100.0},
        {"Year": 2022, "Country": "B", "MyESB_Community": 200.0},
    ])
    fe = compute_social_proxy(df)
    # Both years: A gets same z-score relative to its year
    a2021 = fe[(fe["Country"] == "A") & (fe["Year"] == 2021)]["zscore_myesb_community"].iloc[0]
    a2022 = fe[(fe["Country"] == "A") & (fe["Year"] == 2022)]["zscore_myesb_community"].iloc[0]
    assert pytest.approx(a2021) == a2022  # same relative position in each year


# ── OGAE NaN filled with 0 after normalisation ────────────────────────────────

def test_ogae_nan_filled_with_zero():
    df = _df([
        {"Year": 2022, "Country": "A", "OGAE_Points": 100.0},
        {"Year": 2022, "Country": "B", "OGAE_Points": np.nan},
        {"Year": 2022, "Country": "C", "OGAE_Points": 200.0},
    ])
    fe = compute_social_proxy(df)
    nan_row = fe[fe["Country"] == "B"]["zscore_ogae_points"].iloc[0]
    assert nan_row == pytest.approx(0.0)
    assert not fe["zscore_ogae_points"].isna().any()


# ── Single country in year → z-score 0 (no std) ──────────────────────────────

def test_single_country_year_zero():
    df = _df([
        {"Year": 2022, "Country": "Solo", "MyESB_Community": 7.0},
    ])
    fe = compute_social_proxy(df)
    assert fe["zscore_myesb_community"].iloc[0] == pytest.approx(0.0)


# ── Output shape and columns ───────────────────────────────────────────────────

def test_output_columns():
    df = _df([{"Year": 2022, "Country": "A", "MyESB_Community": 1.0, "MyESB_Personal": 1.0, "OGAE_Points": 1.0}])
    fe = compute_social_proxy(df)
    assert list(fe.columns) == [
        "Year", "Country",
        "zscore_myesb_community", "zscore_myesb_personal", "zscore_ogae_points",
    ]


def test_row_count_preserved():
    df = _df([
        {"Year": 2021, "Country": "A", "MyESB_Community": 1.0, "MyESB_Personal": 1.0, "OGAE_Points": 10.0},
        {"Year": 2022, "Country": "B", "MyESB_Community": 2.0, "MyESB_Personal": 2.0, "OGAE_Points": 20.0},
    ])
    assert len(compute_social_proxy(df)) == 2
