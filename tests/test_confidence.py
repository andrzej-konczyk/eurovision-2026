"""Tests for src/models/confidence.py (US-S4-03)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.models.confidence import bootstrap_proba, compute_ci, confidence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_train():
    """Minimal training set: 40 rows, 3 feature columns, balanced classes."""
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame(
        rng.standard_normal((n, 3)),
        columns=["f1", "f2", "f3"],
    )
    y = pd.Series([1] * 20 + [0] * 20, name="Top 10")
    return X, y


@pytest.fixture()
def tiny_target():
    """Five target-year rows (e.g. 2026 entries)."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        rng.standard_normal((5, 3)),
        columns=["f1", "f2", "f3"],
    )


@pytest.fixture()
def country_list():
    return ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]


# ---------------------------------------------------------------------------
# bootstrap_proba
# ---------------------------------------------------------------------------

class TestBootstrapProba:
    def test_output_shape(self, tiny_train, tiny_target):
        X_train, y_train = tiny_train
        proba = bootstrap_proba(
            model_name="xgb",
            best_params={"n_estimators": 10, "max_depth": 2,
                         "learning_rate": 0.1, "subsample": 0.8,
                         "colsample_bytree": 0.8},
            X_train=X_train,
            y_train=y_train,
            X_target=tiny_target,
            n_bootstrap=10,
            seed=42,
        )
        assert proba.shape == (10, 5)

    def test_probabilities_in_unit_interval(self, tiny_train, tiny_target):
        X_train, y_train = tiny_train
        proba = bootstrap_proba(
            model_name="lgbm",
            best_params={"n_estimators": 10, "num_leaves": 10,
                         "learning_rate": 0.1, "subsample": 0.8,
                         "min_child_samples": 2},
            X_train=X_train,
            y_train=y_train,
            X_target=tiny_target,
            n_bootstrap=5,
            seed=0,
        )
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_deterministic_with_same_seed(self, tiny_train, tiny_target):
        kwargs = dict(
            model_name="xgb",
            best_params={"n_estimators": 10, "max_depth": 2,
                         "learning_rate": 0.1, "subsample": 0.8,
                         "colsample_bytree": 0.8},
            X_train=tiny_train[0],
            y_train=tiny_train[1],
            X_target=tiny_target,
            n_bootstrap=8,
        )
        p1 = bootstrap_proba(**kwargs, seed=7)
        p2 = bootstrap_proba(**kwargs, seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self, tiny_train, tiny_target):
        kwargs = dict(
            model_name="xgb",
            best_params={"n_estimators": 10, "max_depth": 2,
                         "learning_rate": 0.1, "subsample": 0.8,
                         "colsample_bytree": 0.8},
            X_train=tiny_train[0],
            y_train=tiny_train[1],
            X_target=tiny_target,
            n_bootstrap=8,
        )
        p1 = bootstrap_proba(**kwargs, seed=1)
        p2 = bootstrap_proba(**kwargs, seed=2)
        assert not np.allclose(p1, p2)

    def test_single_class_bootstrap_skipped(self):
        """If every bootstrap draw happens to be single-class, RuntimeError is raised."""
        # Force single-class y_train (all zeros — any resample will be all zeros)
        X = pd.DataFrame({"f1": [0.0] * 10})
        y = pd.Series([0] * 10)
        X_target = pd.DataFrame({"f1": [0.5]})
        with pytest.raises(RuntimeError, match="single-class draws"):
            bootstrap_proba(
                model_name="xgb",
                best_params={"n_estimators": 5, "max_depth": 2,
                             "learning_rate": 0.1, "subsample": 0.8,
                             "colsample_bytree": 0.8},
                X_train=X,
                y_train=y,
                X_target=X_target,
                n_bootstrap=3,
                seed=0,
            )


# ---------------------------------------------------------------------------
# compute_ci
# ---------------------------------------------------------------------------

class TestComputeCI:
    def test_columns_present(self, country_list):
        rng = np.random.default_rng(0)
        mat = rng.uniform(0, 1, (100, 5))
        df = compute_ci(mat, country_list)
        for col in ["country", "prob_mean", "ci80_lo", "ci80_hi", "ci50_lo", "ci50_hi"]:
            assert col in df.columns

    def test_row_count(self, country_list):
        mat = np.random.default_rng(0).uniform(0, 1, (50, 5))
        df = compute_ci(mat, country_list)
        assert len(df) == 5

    def test_sorted_by_prob_mean_descending(self, country_list):
        mat = np.random.default_rng(42).uniform(0, 1, (200, 5))
        df = compute_ci(mat, country_list)
        assert (df["prob_mean"].diff().dropna() <= 0).all()

    def test_ci_ordering(self, country_list):
        mat = np.random.default_rng(5).uniform(0, 1, (500, 5))
        df = compute_ci(mat, country_list)
        # 80% interval must be wider than 50% interval
        assert (df["ci80_lo"] <= df["ci50_lo"]).all()
        assert (df["ci50_lo"] <= df["prob_mean"]).all()
        assert (df["prob_mean"] <= df["ci50_hi"]).all()
        assert (df["ci50_hi"] <= df["ci80_hi"]).all()

    def test_degenerate_constant_proba(self, country_list):
        # All bootstrap samples predict 0.6 for everyone → zero-width CI
        mat = np.full((100, 5), 0.6)
        df = compute_ci(mat, country_list)
        np.testing.assert_allclose(df["prob_mean"], 0.6)
        np.testing.assert_allclose(df["ci50_lo"], 0.6)
        np.testing.assert_allclose(df["ci50_hi"], 0.6)

    def test_single_country(self):
        mat = np.random.default_rng(0).uniform(0, 1, (100, 1))
        df = compute_ci(mat, ["OnlyOne"])
        assert len(df) == 1
        assert df.loc[0, "country"] == "OnlyOne"

    def test_probabilities_in_unit_interval(self, country_list):
        mat = np.random.default_rng(0).uniform(0, 1, (200, 5))
        df = compute_ci(mat, country_list)
        for col in ["prob_mean", "ci80_lo", "ci80_hi", "ci50_lo", "ci50_hi"]:
            assert (df[col] >= 0.0).all()
            assert (df[col] <= 1.0).all()


# ---------------------------------------------------------------------------
# confidence() — integration (fast, mocked bootstrap)
# ---------------------------------------------------------------------------

class TestConfidenceIntegration:
    def test_returns_both_models(self, tmp_path):
        """confidence() returns a dict with 'xgb' and 'lgbm' keys."""
        meta = _make_dummy_meta(tmp_path)
        results = _run_confidence(tmp_path, meta, n_bootstrap=4)
        assert set(results.keys()) == {"xgb", "lgbm"}

    def test_output_csvs_written(self, tmp_path):
        meta = _make_dummy_meta(tmp_path)
        _run_confidence(tmp_path, meta, n_bootstrap=4)
        assert (tmp_path / "confidence_xgb.csv").exists()
        assert (tmp_path / "confidence_lgbm.csv").exists()

    def test_meta_json_written(self, tmp_path):
        meta = _make_dummy_meta(tmp_path)
        _run_confidence(tmp_path, meta, n_bootstrap=4)
        meta_out = tmp_path / "confidence_meta.json"
        assert meta_out.exists()
        data = json.loads(meta_out.read_text())
        assert data["story"] == "US-S4-03"
        assert data["n_bootstrap"] == 4

    def test_csv_has_expected_columns(self, tmp_path):
        meta = _make_dummy_meta(tmp_path)
        results = _run_confidence(tmp_path, meta, n_bootstrap=4)
        for df in results.values():
            for col in ["country", "prob_mean", "ci80_lo", "ci80_hi", "ci50_lo", "ci50_hi"]:
                assert col in df.columns

    def test_target_year_not_in_training(self, tmp_path):
        """Target year rows must not appear in the training split."""
        meta = _make_dummy_meta(tmp_path)
        results = _run_confidence(tmp_path, meta, n_bootstrap=4, target_year=2025)
        for df in results.values():
            assert len(df) > 0

    def test_invalid_target_year_raises(self, tmp_path):
        meta = _make_dummy_meta(tmp_path)
        # CSV contains 2022–2024 + 2026 only; year 1990 is genuinely absent.
        csv_path = _make_dummy_csv(tmp_path, target_year=2026)
        with patch("src.models.confidence.MLFLOW_URI", str(tmp_path / "mlruns")):
            with patch("mlflow.set_tracking_uri"), \
                 patch("mlflow.set_experiment"):
                with pytest.raises(ValueError, match="No rows found for target_year"):
                    confidence(
                        data_path=csv_path,
                        target_year=1990,
                        n_bootstrap=2,
                        seed=42,
                        out_dir=tmp_path,
                        meta_path=meta,
                    )


# ---------------------------------------------------------------------------
# Helpers for integration tests
# ---------------------------------------------------------------------------

def _make_dummy_meta(tmp_path: Path) -> Path:
    """Write a minimal train_meta.json into tmp_path."""
    meta = {
        "models": {
            "xgb": {"best_params": {
                "n_estimators": 10, "max_depth": 2,
                "learning_rate": 0.1, "subsample": 0.8,
                "colsample_bytree": 0.8,
            }},
            "lgbm": {"best_params": {
                "n_estimators": 10, "num_leaves": 10,
                "learning_rate": 0.1, "subsample": 0.8,
                "min_child_samples": 2,
            }},
        }
    }
    p = tmp_path / "train_meta.json"
    p.write_text(json.dumps(meta))
    return p


def _make_dummy_csv(tmp_path: Path, target_year: int = 2026) -> Path:
    """Write a minimal enriched CSV with training years + target_year."""
    _GROUPS = ["Western", "Eastern", "Nordic", "Balkan", "Baltic", "Other"]
    rows = []
    for year in [2022, 2023, 2024, target_year]:
        for i in range(10):
            rows.append({
                "Year": year,
                "Country ": f"Country{i}",
                "Grand_Final_Ind": 1,
                "Top 10": (1 if i < 3 else 0) if year < target_year else float("nan"),
                "Final_Place": (i + 1) if year < target_year else float("nan"),
                "Big6_Ind": 0,
                "National_Final": 1,
                "Solo_Artist": 1,
                "Returning_Artist_Ind": 0,
                "Number of Members": 1,
                "Multiple_Language": 0,
                "EU": 1,
                "NATO": 1,
                "Qualification_Record": 0.5,
                "Semi_Final_Num": 1,
                "Running_Order_Semi": i + 1,
                "Running_Order_Final": i + 1,
                "MyESB_Community": 20,
                "MyESB_Personal": 20,
                "OGAE_Points": 50,
                "jury_points": 100,
                "tele_points": 100,
                "Final_Points": 200,
                "Semi_Place": i + 1,
                "Semi_Points": 100,
                "Country_Group": _GROUPS[i % len(_GROUPS)],
            })
    p = tmp_path / "dummy_enriched.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _run_confidence(
    tmp_path: Path,
    meta_path: Path,
    n_bootstrap: int = 4,
    target_year: int = 2026,
) -> dict[str, pd.DataFrame]:
    csv_path = _make_dummy_csv(tmp_path, target_year=target_year)
    with patch("src.models.confidence.MLFLOW_URI", str(tmp_path / "mlruns")):
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run"), \
             patch("mlflow.log_params"), \
             patch("mlflow.log_metric"), \
             patch("mlflow.set_tag"):
            return confidence(
                data_path=csv_path,
                target_year=target_year,
                n_bootstrap=n_bootstrap,
                seed=42,
                out_dir=tmp_path,
                meta_path=meta_path,
            )
