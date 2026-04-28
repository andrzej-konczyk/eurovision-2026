"""Tests for US-S4-02 — src/models/evaluate.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    HOLDOUT_YEAR,
    TARGET_DERIVED,
    TOP_K,
    add_derived_top10,
    holdout_split,
    top10_accuracy,
)


# ---------------------------------------------------------------------------
# Minimal fixture — same shape as the enriched CSV
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Synthetic enriched-CSV-like DataFrame: 3 years, 5 countries, 3 finalists/yr."""
    years = [2021, 2022, 2023, 2024]
    countries = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    rows = []
    for year in years:
        for i, country in enumerate(countries):
            finalist = i < 3
            rows.append({
                "Year": year,
                "Country": country,
                "Country_Group": ["Northern", "Western", "Eastern", "Central", "Southern"][i],
                "Grand_Final_Ind": 1 if finalist else 0,
                "Final_Place": float(i + 1) if finalist else np.nan,
                "jury_points": 100.0 if finalist else np.nan,
                "tele_points": 80.0 if finalist else np.nan,
                "Top 10": 1.0 if (finalist and i == 0) else (np.nan if not finalist else 0.0),
                "Big6_Ind": 0,
                "National_Final": 1,
                "Solo_Artist": 1,
                "Returning_Artist_Ind": 0,
                "Number of Members": 1,
                "Multiple_Language": 0,
                "EU": 1,
                "NATO": 1,
                "Qualification_Record": 0.7,
                "Semi_Final_Num": np.nan if finalist else 1.0,
                "Running_Order_Semi": np.nan if finalist else float(i + 1),
                "Running_Order_Final": float(i + 1) if finalist else np.nan,
                "MyESB_Community": 20.0,
                "MyESB_Personal": 20.0,
                "OGAE_Points": 100.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def matrix_with_target(raw_df) -> pd.DataFrame:
    from src.models.train import build_feature_matrix
    matrix = build_feature_matrix(raw_df)
    return add_derived_top10(matrix, raw_df)


# ---------------------------------------------------------------------------
# add_derived_top10
# ---------------------------------------------------------------------------


def test_add_derived_top10_creates_column(matrix_with_target) -> None:
    assert TARGET_DERIVED in matrix_with_target.columns


def test_add_derived_top10_finalist_place_leq_topk(matrix_with_target) -> None:
    """Final_Place ≤ TOP_K finalists should have target = 1."""
    mask = (
        (matrix_with_target["Grand_Final_Ind"] == 1)
        & (matrix_with_target["Final_Place"] <= TOP_K)
    )
    assert (matrix_with_target.loc[mask, TARGET_DERIVED] == 1).all()


def test_add_derived_top10_finalist_place_gt_topk(matrix_with_target) -> None:
    """Final_Place > TOP_K finalists should have target = 0."""
    mask = (
        (matrix_with_target["Grand_Final_Ind"] == 1)
        & (matrix_with_target["Final_Place"] > TOP_K)
        & matrix_with_target["Final_Place"].notna()
    )
    if mask.any():
        assert (matrix_with_target.loc[mask, TARGET_DERIVED] == 0).all()


def test_add_derived_top10_non_finalist_is_nan(matrix_with_target) -> None:
    """Non-finalists (Grand_Final_Ind == 0) should have NaN target."""
    mask = matrix_with_target["Grand_Final_Ind"] == 0
    assert matrix_with_target.loc[mask, TARGET_DERIVED].isna().all()


# ---------------------------------------------------------------------------
# holdout_split
# ---------------------------------------------------------------------------


def test_holdout_split_test_year_is_holdout(matrix_with_target) -> None:
    _, _, _, _, test_rows, _ = holdout_split(matrix_with_target, holdout_year=2024)
    assert (test_rows["Year"] == 2024).all()


def test_holdout_split_train_year_lt_holdout(matrix_with_target) -> None:
    X_train, y_train, _, _, _, _ = holdout_split(matrix_with_target, holdout_year=2024)
    # training rows come from matrix — verify all are from years < 2024
    train_rows = matrix_with_target[
        (matrix_with_target["Grand_Final_Ind"] == 1)
        & (matrix_with_target["Year"] < 2024)
        & matrix_with_target[TARGET_DERIVED].notna()
    ]
    assert len(X_train) == len(train_rows)


def test_holdout_split_no_null_target(matrix_with_target) -> None:
    _, y_train, _, y_test, _, _ = holdout_split(matrix_with_target, holdout_year=2024)
    assert y_train.notna().all()
    assert y_test.notna().all()


def test_holdout_split_grand_final_only(matrix_with_target) -> None:
    X_train, _, X_test, _, test_rows, _ = holdout_split(matrix_with_target, holdout_year=2024)
    # test_rows only includes Grand Final entries
    assert (test_rows["Grand_Final_Ind"] == 1).all()


def test_holdout_split_feat_cols_no_outcome_leakage(matrix_with_target) -> None:
    _, _, _, _, _, feat_cols = holdout_split(matrix_with_target, holdout_year=2024)
    outcome_cols = {"Final_Place", "Final_Points", "Semi_Place", "Semi_Points",
                    "Top 5", "Top 10", TARGET_DERIVED}
    assert not outcome_cols.intersection(feat_cols)


# ---------------------------------------------------------------------------
# top10_accuracy
# ---------------------------------------------------------------------------


def test_top10_accuracy_perfect() -> None:
    """All 10 predicted match all 10 actual → 1.0."""
    y = pd.Series([1] * 10 + [0] * 16)
    proba = np.array([0.9] * 10 + [0.1] * 16)
    assert top10_accuracy(y, proba, top_k=10) == 1.0


def test_top10_accuracy_zero() -> None:
    """Predicted top-10 are all actual negatives → 0.0."""
    y = pd.Series([0] * 10 + [1] * 10 + [0] * 6)
    # highest probabilities assigned to all-zero first 10
    proba = np.array([0.9] * 10 + [0.1] * 10 + [0.05] * 6)
    assert top10_accuracy(y, proba, top_k=10) == 0.0


def test_top10_accuracy_half() -> None:
    """5 of predicted top-10 match → 0.5."""
    y = pd.Series([1] * 10 + [0] * 16)
    # give high prob to first 5 positives and 5 negatives
    proba = np.zeros(26)
    proba[:5] = 0.9   # actual top-10 (correct)
    proba[10:15] = 0.8  # actual negatives (wrong)
    assert top10_accuracy(y, proba, top_k=10) == pytest.approx(0.5)


def test_top10_accuracy_range() -> None:
    """Result is always in [0, 1]."""
    rng = np.random.default_rng(0)
    y = pd.Series(([1] * 10 + [0] * 16))
    proba = rng.random(26)
    acc = top10_accuracy(y, proba, top_k=10)
    assert 0.0 <= acc <= 1.0
