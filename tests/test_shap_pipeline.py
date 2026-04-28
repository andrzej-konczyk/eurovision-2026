"""Tests for src/models/shap_pipeline.py (US-S4-04)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb

from src.models.shap_pipeline import (
    build_explainer,
    build_top5,
    impute,
    load_pipeline,
    shap_values_pos,
    shap_pipeline,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 60
N_FEATURES = 5
FEAT_COLS = ["f1", "f2", "f3", "f4", "f5"]
COUNTRIES = [f"Country{i}" for i in range(10)]
_GROUPS = ["Western", "Eastern", "Nordic", "Balkan", "Baltic", "Other"]


@pytest.fixture()
def tiny_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = np.array([1] * 30 + [0] * 30)
    return X, y


@pytest.fixture()
def fitted_xgb_pipeline(tiny_data):
    X, y = tiny_data
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=10, max_depth=2,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42,
        )),
    ])
    pipe.fit(X, y)
    return pipe


@pytest.fixture()
def fitted_lgbm_pipeline(tiny_data):
    X, y = tiny_data
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lgb.LGBMClassifier(
            n_estimators=10, num_leaves=4,
            objective="binary", random_state=42, verbose=-1,
        )),
    ])
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------------------------
# load_pipeline
# ---------------------------------------------------------------------------

class TestLoadPipeline:
    def test_loads_existing_pkl(self, tmp_path, fitted_xgb_pipeline):
        pkl = tmp_path / "xgb_model.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(fitted_xgb_pipeline, f)
        loaded = load_pipeline("xgb", artefact_dir=tmp_path)
        assert hasattr(loaded, "predict_proba")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="xgb_model.pkl"):
            load_pipeline("xgb", artefact_dir=tmp_path)


# ---------------------------------------------------------------------------
# impute
# ---------------------------------------------------------------------------

class TestImpute:
    def test_returns_array(self, fitted_xgb_pipeline, tiny_data):
        X, _ = tiny_data
        X_df = pd.DataFrame(X, columns=FEAT_COLS)
        result = impute(fitted_xgb_pipeline, X_df)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self, fitted_xgb_pipeline, tiny_data):
        X, _ = tiny_data
        X_df = pd.DataFrame(X, columns=FEAT_COLS)
        result = impute(fitted_xgb_pipeline, X_df)
        assert result.shape == X_df.shape

    def test_no_nans_after_impute(self, fitted_xgb_pipeline):
        X_nan = pd.DataFrame(
            [[1.0, np.nan, 3.0, np.nan, 5.0]],
            columns=FEAT_COLS,
        )
        result = impute(fitted_xgb_pipeline, X_nan)
        assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# build_explainer
# ---------------------------------------------------------------------------

class TestBuildExplainer:
    def test_returns_tree_explainer_xgb(self, fitted_xgb_pipeline):
        import shap
        explainer = build_explainer(fitted_xgb_pipeline)
        assert isinstance(explainer, shap.TreeExplainer)

    def test_returns_tree_explainer_lgbm(self, fitted_lgbm_pipeline):
        import shap
        explainer = build_explainer(fitted_lgbm_pipeline)
        assert isinstance(explainer, shap.TreeExplainer)


# ---------------------------------------------------------------------------
# shap_values_pos
# ---------------------------------------------------------------------------

class TestShapValuesPos:
    def test_xgb_shape(self, fitted_xgb_pipeline, tiny_data):
        X, _ = tiny_data
        X_imp = impute(fitted_xgb_pipeline, pd.DataFrame(X, columns=FEAT_COLS))
        explainer = build_explainer(fitted_xgb_pipeline)
        sv = shap_values_pos(explainer, X_imp)
        assert sv.shape == (N_SAMPLES, N_FEATURES)

    def test_lgbm_shape(self, fitted_lgbm_pipeline, tiny_data):
        X, _ = tiny_data
        X_imp = impute(fitted_lgbm_pipeline, pd.DataFrame(X, columns=FEAT_COLS))
        explainer = build_explainer(fitted_lgbm_pipeline)
        sv = shap_values_pos(explainer, X_imp)
        assert sv.shape == (N_SAMPLES, N_FEATURES)

    def test_returns_ndarray(self, fitted_xgb_pipeline, tiny_data):
        X, _ = tiny_data
        X_imp = impute(fitted_xgb_pipeline, pd.DataFrame(X, columns=FEAT_COLS))
        sv = shap_values_pos(build_explainer(fitted_xgb_pipeline), X_imp)
        assert isinstance(sv, np.ndarray)

    def test_values_are_finite(self, fitted_xgb_pipeline, tiny_data):
        X, _ = tiny_data
        X_imp = impute(fitted_xgb_pipeline, pd.DataFrame(X, columns=FEAT_COLS))
        sv = shap_values_pos(build_explainer(fitted_xgb_pipeline), X_imp)
        assert np.all(np.isfinite(sv))


# ---------------------------------------------------------------------------
# build_top5
# ---------------------------------------------------------------------------

class TestBuildTop5:
    @pytest.fixture()
    def sample_sv(self):
        rng = np.random.default_rng(7)
        return rng.standard_normal((10, N_FEATURES))

    @pytest.fixture()
    def sample_X_imp(self):
        rng = np.random.default_rng(8)
        return rng.standard_normal((10, N_FEATURES))

    def test_output_columns(self, sample_sv, sample_X_imp):
        df = build_top5(sample_sv, sample_X_imp, FEAT_COLS, COUNTRIES)
        for col in ["country", "rank", "feature", "shap_value", "feature_value"]:
            assert col in df.columns

    def test_row_count(self, sample_sv, sample_X_imp):
        df = build_top5(sample_sv, sample_X_imp, FEAT_COLS, COUNTRIES, top_n=5)
        assert len(df) == len(COUNTRIES) * 5

    def test_top_n_respected(self, sample_sv, sample_X_imp):
        df = build_top5(sample_sv, sample_X_imp, FEAT_COLS, COUNTRIES, top_n=3)
        assert len(df) == len(COUNTRIES) * 3
        assert df["rank"].max() == 3

    def test_rank_starts_at_1(self, sample_sv, sample_X_imp):
        df = build_top5(sample_sv, sample_X_imp, FEAT_COLS, COUNTRIES)
        assert df["rank"].min() == 1

    def test_rank_ordered_by_abs_shap(self, sample_sv, sample_X_imp):
        df = build_top5(sample_sv, sample_X_imp, FEAT_COLS, COUNTRIES)
        for country in COUNTRIES:
            sub = df[df["country"] == country].sort_values("rank")
            abs_vals = sub["shap_value"].abs().tolist()
            assert abs_vals == sorted(abs_vals, reverse=True)

    def test_all_countries_present(self, sample_sv, sample_X_imp):
        df = build_top5(sample_sv, sample_X_imp, FEAT_COLS, COUNTRIES)
        assert set(df["country"].unique()) == set(COUNTRIES)

    def test_feature_names_from_feat_cols(self, sample_sv, sample_X_imp):
        df = build_top5(sample_sv, sample_X_imp, FEAT_COLS, COUNTRIES)
        assert set(df["feature"].unique()).issubset(set(FEAT_COLS))

    def test_single_country(self):
        sv = np.array([[0.5, -0.3, 0.1, -0.8, 0.2]])
        X_imp = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        df = build_top5(sv, X_imp, FEAT_COLS, ["OnlyOne"], top_n=3)
        assert len(df) == 3
        # Rank-1 must be f4 (|−0.8| largest)
        assert df[df["rank"] == 1]["feature"].iloc[0] == "f4"
        # Rank-2 must be f1 (|0.5|)
        assert df[df["rank"] == 2]["feature"].iloc[0] == "f1"


# ---------------------------------------------------------------------------
# shap_pipeline() — integration
# ---------------------------------------------------------------------------

class TestShapPipelineIntegration:
    def test_returns_both_models(self, tmp_path, fitted_xgb_pipeline, fitted_lgbm_pipeline):
        results = _run_pipeline(tmp_path, fitted_xgb_pipeline, fitted_lgbm_pipeline)
        assert set(results.keys()) == {"xgb", "lgbm"}

    def test_top5_csvs_written(self, tmp_path, fitted_xgb_pipeline, fitted_lgbm_pipeline):
        _run_pipeline(tmp_path, fitted_xgb_pipeline, fitted_lgbm_pipeline)
        assert (tmp_path / "out" / "shap_top5_xgb.csv").exists()
        assert (tmp_path / "out" / "shap_top5_lgbm.csv").exists()

    def test_summary_plots_written(self, tmp_path, fitted_xgb_pipeline, fitted_lgbm_pipeline):
        _run_pipeline(tmp_path, fitted_xgb_pipeline, fitted_lgbm_pipeline)
        assert (tmp_path / "plots" / "shap_summary_xgb.png").exists()
        assert (tmp_path / "plots" / "shap_summary_lgbm.png").exists()

    def test_meta_json_written(self, tmp_path, fitted_xgb_pipeline, fitted_lgbm_pipeline):
        _run_pipeline(tmp_path, fitted_xgb_pipeline, fitted_lgbm_pipeline)
        meta_path = tmp_path / "out" / "shap_meta.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["story"] == "US-S4-04"
        assert data["top_n"] == 5

    def test_top5_has_correct_columns(self, tmp_path, fitted_xgb_pipeline, fitted_lgbm_pipeline):
        results = _run_pipeline(tmp_path, fitted_xgb_pipeline, fitted_lgbm_pipeline)
        for df in results.values():
            for col in ["country", "rank", "feature", "shap_value", "feature_value"]:
                assert col in df.columns

    def test_invalid_target_year_raises(self, tmp_path, fitted_xgb_pipeline, fitted_lgbm_pipeline):
        # CSV contains 2022–2024 + 2026; year 1990 is genuinely absent.
        out_dir = tmp_path / "out_inv"
        plots_dir = tmp_path / "plots_inv"
        out_dir.mkdir()
        csv_path = _make_dummy_csv(tmp_path, target_year=2026)

        from src.models.train import build_feature_matrix, training_split
        _df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
        _df.columns = _df.columns.str.strip()
        X_tr, y_tr, _, _ = training_split(build_feature_matrix(_df))
        for name, clf in [
            ("xgb", xgb.XGBClassifier(n_estimators=5, max_depth=2,
                objective="binary:logistic", eval_metric="logloss", random_state=0)),
            ("lgbm", lgb.LGBMClassifier(n_estimators=5, num_leaves=4,
                objective="binary", random_state=0, verbose=-1)),
        ]:
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", clf)])
            pipe.fit(X_tr, y_tr)
            with open(out_dir / f"{name}_model.pkl", "wb") as f:
                pickle.dump(pipe, f)

        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            with pytest.raises(ValueError, match="No rows found"):
                shap_pipeline(
                    data_path=csv_path,
                    target_year=1990,
                    out_dir=out_dir,
                    plots_dir=plots_dir,
                )


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def _make_dummy_csv(path: Path, target_year: int = 2026) -> Path:
    rows = []
    for year in [2022, 2023, 2024, target_year]:
        for i in range(10):
            rows.append({
                "Year": year,
                "Country ": f"Country{i}",
                "Grand_Final_Ind": 1,
                "Top 10": (1 if i < 3 else 0) if year < target_year else float("nan"),
                "Final_Place": (i + 1) if year < target_year else float("nan"),
                "Big6_Ind": 0, "National_Final": 1, "Solo_Artist": 1,
                "Returning_Artist_Ind": 0, "Number of Members": 1,
                "Multiple_Language": 0, "EU": 1, "NATO": 1,
                "Qualification_Record": 0.5, "Semi_Final_Num": 1,
                "Running_Order_Semi": i + 1, "Running_Order_Final": i + 1,
                "MyESB_Community": 20, "MyESB_Personal": 20, "OGAE_Points": 50,
                "jury_points": 100, "tele_points": 100, "Final_Points": 200,
                "Semi_Place": i + 1, "Semi_Points": 100,
                "Country_Group": _GROUPS[i % len(_GROUPS)],
            })
    p = path / "dummy.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _run_pipeline(
    tmp_path: Path,
    xgb_pipe,  # unused — pipelines are refitted on real feature matrix below
    lgbm_pipe,
    target_year: int = 2026,
):
    out_dir = tmp_path / "out"
    plots_dir = tmp_path / "plots"
    out_dir.mkdir()

    csv_path = _make_dummy_csv(tmp_path, target_year=target_year)

    # Build the feature matrix exactly as shap_pipeline() will, then fit fresh
    # pipelines on those features so the saved pkls have the right n_features_in_.
    from src.models.train import build_feature_matrix, training_split
    _df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    _df.columns = _df.columns.str.strip()
    _matrix = build_feature_matrix(_df)
    X_tr, y_tr, _, feat_cols = training_split(_matrix)

    for name, clf in [
        ("xgb", xgb.XGBClassifier(
            n_estimators=10, max_depth=2,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42,
        )),
        ("lgbm", lgb.LGBMClassifier(
            n_estimators=10, num_leaves=4,
            objective="binary", random_state=42, verbose=-1,
        )),
    ]:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", clf),
        ])
        pipe.fit(X_tr, y_tr)
        with open(out_dir / f"{name}_model.pkl", "wb") as f:
            pickle.dump(pipe, f)

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"), \
         patch("mlflow.start_run"), \
         patch("mlflow.log_params"), \
         patch("mlflow.log_metric"), \
         patch("mlflow.log_artifact"), \
         patch("mlflow.set_tag"):
        return shap_pipeline(
            data_path=csv_path,
            target_year=target_year,
            out_dir=out_dir,
            plots_dir=plots_dir,
        )
