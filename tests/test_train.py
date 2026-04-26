"""Tests for US-S4-02 — src/models/train.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    FEATURE_COLS,
    LGBM_GRID,
    TARGET_COL,
    XGB_GRID,
    _ENGINEERED_FEATURES,
    build_feature_matrix,
    run_grid_search,
    training_split,
)

# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------

_YEARS = [2019, 2021, 2022, 2023, 2024]
_COUNTRIES = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
_BLOCS = ["Northern", "Western", "Eastern", "Central", "Southern"]
# First 3 countries qualify to the Grand Final each year
_N_FINALISTS = 3


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for year in _YEARS:
        for i, (country, bloc) in enumerate(zip(_COUNTRIES, _BLOCS)):
            finalist = i < _N_FINALISTS
            rows.append({
                "Year": year,
                "Country": country,
                "Country_Group": bloc,
                "Grand_Final_Ind": 1 if finalist else 0,
                "Final_Place": float(i + 1) if finalist else np.nan,
                "jury_points": float(rng.integers(50, 300)) if finalist else np.nan,
                "tele_points": float(rng.integers(30, 250)) if finalist else np.nan,
                TARGET_COL: 1 if (finalist and i == 0) else (0 if finalist else np.nan),
                "Big6_Ind": 0,
                "National_Final": int(rng.integers(0, 2)),
                "Solo_Artist": 1,
                "Returning_Artist_Ind": 0,
                "Number of Members": 1,
                "Multiple_Language": 0,
                "EU": 1,
                "NATO": 1,
                "Qualification_Record": float(rng.uniform(0.3, 1.0)),
                "Semi_Final_Num": np.nan if finalist else 1.0,
                "Running_Order_Semi": np.nan if finalist else float(i + 1),
                "Running_Order_Final": float(i + 1) if finalist else np.nan,
                "MyESB_Community": float(rng.integers(1, 40)),
                "MyESB_Personal": float(rng.integers(1, 40)),
                "OGAE_Points": float(rng.integers(0, 400)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def matrix(sample_df: pd.DataFrame) -> pd.DataFrame:
    return build_feature_matrix(sample_df)


@pytest.fixture
def split(matrix: pd.DataFrame):
    return training_split(matrix)


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------


def test_build_feature_matrix_returns_dataframe(matrix: pd.DataFrame) -> None:
    assert isinstance(matrix, pd.DataFrame)


def test_build_feature_matrix_has_engineered_cols(matrix: pd.DataFrame) -> None:
    for col in _ENGINEERED_FEATURES:
        assert col in matrix.columns, f"Missing engineered column: {col}"


def test_build_feature_matrix_row_count(sample_df: pd.DataFrame, matrix: pd.DataFrame) -> None:
    assert len(matrix) == len(sample_df)


def test_build_feature_matrix_no_duplicate_keys(matrix: pd.DataFrame) -> None:
    dups = matrix.duplicated(subset=["Year", "Country"]).sum()
    assert dups == 0, f"Duplicate (Year, Country) pairs: {dups}"


# ---------------------------------------------------------------------------
# training_split
# ---------------------------------------------------------------------------


def test_training_split_excludes_non_finalists(split) -> None:
    X, y, groups, _ = split
    # Every training row must come from a Grand Final entry (Grand_Final_Ind == 1).
    # The fixture has _N_FINALISTS finalists per year × len(_YEARS) years.
    assert len(X) == _N_FINALISTS * len(_YEARS)


def test_training_split_excludes_2026(matrix: pd.DataFrame) -> None:
    # Inject a 2026 row and confirm it is excluded from training.
    row_2026 = matrix.iloc[0].copy()
    row_2026["Year"] = 2026
    row_2026["Grand_Final_Ind"] = 1
    row_2026[TARGET_COL] = 1
    extended = pd.concat([matrix, pd.DataFrame([row_2026])], ignore_index=True)
    _, _, groups, _ = training_split(extended)
    assert 2026 not in groups.values


def test_training_split_no_null_target(split) -> None:
    _, y, _, _ = split
    assert y.notna().all()


def test_training_split_target_is_binary(split) -> None:
    _, y, _, _ = split
    assert set(y.unique()).issubset({0, 1})


def test_training_split_groups_match_x_length(split) -> None:
    X, _, groups, _ = split
    assert len(groups) == len(X)


def test_training_split_feature_cols_no_outcome_leakage(split) -> None:
    _, _, _, feat_cols = split
    outcome_cols = {"Final_Place", "Final_Points", "Semi_Place", "Semi_Points", "Top 5", TARGET_COL}
    leaked = outcome_cols.intersection(feat_cols)
    assert not leaked, f"Outcome columns in feature list: {leaked}"


def test_training_split_feature_cols_subset_of_feature_cols_constant(split) -> None:
    _, _, _, feat_cols = split
    unknown = set(feat_cols) - set(FEATURE_COLS)
    assert not unknown, f"Unknown feature columns returned: {unknown}"


# ---------------------------------------------------------------------------
# run_grid_search — XGBoost (tiny grid for speed)
# ---------------------------------------------------------------------------

_TINY_XGB_GRID = {
    "model__n_estimators": [10],
    "model__max_depth": [2],
    "model__learning_rate": [0.1],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8],
}

_TINY_LGBM_GRID = {
    "model__n_estimators": [10],
    "model__num_leaves": [4],
    "model__learning_rate": [0.1],
    "model__subsample": [0.8],
    "model__min_child_samples": [2],
}


@pytest.fixture
def xgb_gs(split):
    import xgboost as xgb
    X, y, groups, _ = split
    clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=0)
    return run_grid_search(clf, _TINY_XGB_GRID, X, y, groups)


@pytest.fixture
def lgbm_gs(split):
    import lightgbm as lgb
    X, y, groups, _ = split
    clf = lgb.LGBMClassifier(objective="binary", random_state=0, verbose=-1)
    return run_grid_search(clf, _TINY_LGBM_GRID, X, y, groups)


def test_xgb_grid_search_best_score_in_range(xgb_gs) -> None:
    assert 0.0 <= xgb_gs.best_score_ <= 1.0


def test_xgb_grid_search_best_estimator_has_pipeline_steps(xgb_gs) -> None:
    assert hasattr(xgb_gs.best_estimator_, "steps")
    step_names = [name for name, _ in xgb_gs.best_estimator_.steps]
    assert "imputer" in step_names
    assert "model" in step_names


def test_lgbm_grid_search_best_score_in_range(lgbm_gs) -> None:
    assert 0.0 <= lgbm_gs.best_score_ <= 1.0


def test_lgbm_grid_search_best_estimator_has_pipeline_steps(lgbm_gs) -> None:
    assert hasattr(lgbm_gs.best_estimator_, "steps")
    step_names = [name for name, _ in lgbm_gs.best_estimator_.steps]
    assert "imputer" in step_names
    assert "model" in step_names


def test_xgb_grid_search_predict_proba_shape(xgb_gs, split) -> None:
    X, _, _, feat_cols = split
    proba = xgb_gs.best_estimator_.predict_proba(X[feat_cols])
    assert proba.shape == (len(X), 2)


def test_lgbm_grid_search_predict_proba_shape(lgbm_gs, split) -> None:
    X, _, _, feat_cols = split
    proba = lgbm_gs.best_estimator_.predict_proba(X[feat_cols])
    assert proba.shape == (len(X), 2)


# ---------------------------------------------------------------------------
# FEATURE_COLS / TARGET_COL constants
# ---------------------------------------------------------------------------


def test_feature_cols_are_all_strings() -> None:
    assert all(isinstance(c, str) for c in FEATURE_COLS)


def test_feature_cols_no_duplicates() -> None:
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_target_col_not_in_feature_cols() -> None:
    assert TARGET_COL not in FEATURE_COLS


def test_xgb_grid_keys_use_model_prefix() -> None:
    assert all(k.startswith("model__") for k in XGB_GRID)


def test_lgbm_grid_keys_use_model_prefix() -> None:
    assert all(k.startswith("model__") for k in LGBM_GRID)
