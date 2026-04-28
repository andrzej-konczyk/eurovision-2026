"""Tests for src/models/nn.py (US-S5-01)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.nn import (
    NNPipeline,
    _MLP,
    _cv_roc_auc,
    grid_search_cv,
    train_nn,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_GROUPS = ["Western", "Eastern", "Nordic", "Balkan", "Baltic", "Other"]
_TINY_GRID = {"hidden_dims": [(8,)], "lr": [0.05], "dropout": [0.0]}
_TINY_EPOCHS = 10
_TINY_BATCH = 16


@pytest.fixture()
def tiny_Xy():
    """Small balanced dataset for unit-level tests."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 5)).astype(np.float32)
    y = np.array([1] * 20 + [0] * 20, dtype=float)
    return X, y


@pytest.fixture()
def dummy_df():
    """Four-year dummy dataset matching the enriched CSV schema."""
    rng = np.random.default_rng(7)
    rows = []
    for year in [2022, 2023, 2024, 2026]:
        is_hist = year < 2026
        for i in range(10):
            rows.append({
                "Year": year,
                "Country": f"Country{i}",
                "Grand_Final_Ind": 1,
                "Top 10": (1 if i < 3 else 0) if is_hist else float("nan"),
                "Final_Place": (i + 1) if is_hist else float("nan"),
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
                "MyESB_Community": int(rng.integers(1, 30)),
                "MyESB_Personal": int(rng.integers(1, 30)),
                "OGAE_Points": int(rng.integers(0, 100)),
                "jury_points": int(rng.integers(50, 200)) if is_hist else 0,
                "tele_points": int(rng.integers(50, 200)) if is_hist else 0,
                "Final_Points": int(rng.integers(100, 400)) if is_hist else 0,
                "Semi_Place": i + 1,
                "Semi_Points": int(rng.integers(50, 150)),
                "Country_Group": _GROUPS[i % len(_GROUPS)],
            })
    return pd.DataFrame(rows)


def _make_dummy_csv(path: Path) -> Path:
    rng = np.random.default_rng(42)
    rows = []
    for year in [2022, 2023, 2024, 2026]:
        is_hist = year < 2026
        for i in range(10):
            rows.append({
                "Year": year,
                "Country ": f"Country{i}",
                "Grand_Final_Ind": 1,
                "Top 10": (1 if i < 3 else 0) if is_hist else float("nan"),
                "Final_Place": (i + 1) if is_hist else float("nan"),
                "Big6_Ind": 0, "National_Final": 1, "Solo_Artist": 1,
                "Returning_Artist_Ind": 0, "Number of Members": 1,
                "Multiple_Language": 0, "EU": 1, "NATO": 1,
                "Qualification_Record": 0.5, "Semi_Final_Num": 1,
                "Running_Order_Semi": i + 1, "Running_Order_Final": i + 1,
                "MyESB_Community": int(rng.integers(1, 30)),
                "MyESB_Personal": int(rng.integers(1, 30)),
                "OGAE_Points": int(rng.integers(0, 100)),
                "jury_points": int(rng.integers(50, 200)) if is_hist else 0,
                "tele_points": int(rng.integers(50, 200)) if is_hist else 0,
                "Final_Points": int(rng.integers(100, 400)) if is_hist else 0,
                "Semi_Place": i + 1, "Semi_Points": int(rng.integers(50, 150)),
                "Country_Group": _GROUPS[i % len(_GROUPS)],
            })
    p = path / "dummy.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# _MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_output_shape(self):
        model = _MLP(n_features=5, hidden_dims=(16, 8))
        x = torch.randn(10, 5)
        out = model(x)
        assert out.shape == (10, 1)

    def test_output_in_unit_interval(self):
        model = _MLP(n_features=5, hidden_dims=(16,))
        x = torch.randn(20, 5)
        out = model(x).detach().numpy()
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_dropout_zero_is_deterministic(self):
        model = _MLP(n_features=3, hidden_dims=(8,), dropout=0.0)
        model.eval()
        x = torch.randn(5, 3)
        assert torch.allclose(model(x), model(x))

    def test_single_hidden_layer(self):
        model = _MLP(n_features=10, hidden_dims=(4,))
        assert model(torch.randn(3, 10)).shape == (3, 1)

    def test_deep_architecture(self):
        model = _MLP(n_features=23, hidden_dims=(128, 64, 32))
        assert model(torch.randn(7, 23)).shape == (7, 1)


# ---------------------------------------------------------------------------
# NNPipeline
# ---------------------------------------------------------------------------

class TestNNPipeline:
    def test_fit_returns_self(self, tiny_Xy):
        X, y = tiny_Xy
        pipe = NNPipeline(hidden_dims=(8,), n_epochs=_TINY_EPOCHS)
        assert pipe.fit(X, y) is pipe

    def test_predict_proba_shape(self, tiny_Xy):
        X, y = tiny_Xy
        pipe = NNPipeline(hidden_dims=(8,), n_epochs=_TINY_EPOCHS).fit(X, y)
        out = pipe.predict_proba(X)
        assert out.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, tiny_Xy):
        X, y = tiny_Xy
        pipe = NNPipeline(hidden_dims=(8,), n_epochs=_TINY_EPOCHS).fit(X, y)
        proba = pipe.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_in_unit_interval(self, tiny_Xy):
        X, y = tiny_Xy
        pipe = NNPipeline(hidden_dims=(8,), n_epochs=_TINY_EPOCHS).fit(X, y)
        proba = pipe.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_raises_before_fit(self, tiny_Xy):
        X, _ = tiny_Xy
        pipe = NNPipeline()
        with pytest.raises(RuntimeError, match="fit"):
            pipe.predict_proba(X)

    def test_handles_dataframe_input(self, tiny_Xy):
        X, y = tiny_Xy
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        pipe = NNPipeline(hidden_dims=(8,), n_epochs=_TINY_EPOCHS).fit(X_df, y)
        out = pipe.predict_proba(X_df)
        assert out.shape == (len(X), 2)

    def test_handles_nan_input(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4))
        X[::3, 0] = np.nan  # inject NaN
        y = np.array([1] * 10 + [0] * 10, dtype=float)
        pipe = NNPipeline(hidden_dims=(4,), n_epochs=_TINY_EPOCHS).fit(X, y)
        out = pipe.predict_proba(X)
        assert not np.any(np.isnan(out))

    def test_get_params_keys(self, tiny_Xy):
        X, y = tiny_Xy
        pipe = NNPipeline(hidden_dims=(16, 8), dropout=0.1, lr=0.01)
        p = pipe.get_params()
        for key in ["hidden_dims", "dropout", "lr", "n_epochs", "batch_size"]:
            assert key in p

    def test_n_features_set_after_fit(self, tiny_Xy):
        X, y = tiny_Xy
        pipe = NNPipeline(hidden_dims=(8,), n_epochs=_TINY_EPOCHS).fit(X, y)
        assert pipe.n_features_ == X.shape[1]

    def test_seed_reproducibility(self, tiny_Xy):
        X, y = tiny_Xy
        p1 = NNPipeline(hidden_dims=(8,), n_epochs=_TINY_EPOCHS, seed=0).fit(X, y).predict_proba(X)
        p2 = NNPipeline(hidden_dims=(8,), n_epochs=_TINY_EPOCHS, seed=0).fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# _cv_roc_auc
# ---------------------------------------------------------------------------

class TestCvRocAuc:
    def test_returns_list(self, dummy_df):
        from src.models.train import build_feature_matrix, training_split
        matrix = build_feature_matrix(dummy_df)
        X, y, groups, _ = training_split(matrix)
        scores = _cv_roc_auc(
            {"hidden_dims": (8,), "lr": 0.05, "dropout": 0.0},
            X, y, groups, seed=0,
            n_epochs=_TINY_EPOCHS, batch_size=_TINY_BATCH,
        )
        assert isinstance(scores, list)

    def test_scores_in_valid_range(self, dummy_df):
        from src.models.train import build_feature_matrix, training_split
        matrix = build_feature_matrix(dummy_df)
        X, y, groups, _ = training_split(matrix)
        scores = _cv_roc_auc(
            {"hidden_dims": (8,), "lr": 0.05, "dropout": 0.0},
            X, y, groups, seed=0,
            n_epochs=_TINY_EPOCHS, batch_size=_TINY_BATCH,
        )
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_at_least_one_fold(self, dummy_df):
        from src.models.train import build_feature_matrix, training_split
        matrix = build_feature_matrix(dummy_df)
        X, y, groups, _ = training_split(matrix)
        scores = _cv_roc_auc(
            {"hidden_dims": (8,), "lr": 0.05, "dropout": 0.0},
            X, y, groups, seed=0,
            n_epochs=_TINY_EPOCHS, batch_size=_TINY_BATCH,
        )
        assert len(scores) >= 1


# ---------------------------------------------------------------------------
# grid_search_cv
# ---------------------------------------------------------------------------

class TestGridSearchCV:
    def test_returns_valid_best_params(self, dummy_df):
        from src.models.train import build_feature_matrix, training_split
        matrix = build_feature_matrix(dummy_df)
        X, y, groups, _ = training_split(matrix)
        best, _ = grid_search_cv(
            X, y, groups,
            param_grid=_TINY_GRID, seed=0,
            n_epochs=_TINY_EPOCHS, batch_size=_TINY_BATCH,
        )
        assert "hidden_dims" in best
        assert "lr" in best
        assert "dropout" in best

    def test_cv_results_length_matches_grid(self, dummy_df):
        from src.models.train import build_feature_matrix, training_split
        import itertools
        matrix = build_feature_matrix(dummy_df)
        X, y, groups, _ = training_split(matrix)
        _, results = grid_search_cv(
            X, y, groups,
            param_grid=_TINY_GRID, seed=0,
            n_epochs=_TINY_EPOCHS, batch_size=_TINY_BATCH,
        )
        n_combos = len(list(itertools.product(*_TINY_GRID.values())))
        assert len(results) == n_combos

    def test_cv_results_have_score_keys(self, dummy_df):
        from src.models.train import build_feature_matrix, training_split
        matrix = build_feature_matrix(dummy_df)
        X, y, groups, _ = training_split(matrix)
        _, results = grid_search_cv(
            X, y, groups,
            param_grid=_TINY_GRID, seed=0,
            n_epochs=_TINY_EPOCHS, batch_size=_TINY_BATCH,
        )
        for r in results:
            assert "mean_roc_auc" in r
            assert "std_roc_auc" in r


# ---------------------------------------------------------------------------
# train_nn — integration
# ---------------------------------------------------------------------------

class TestTrainNN:
    def _run(self, tmp_path):
        csv_path = _make_dummy_csv(tmp_path)
        out_dir = tmp_path / "out"
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run"), \
             patch("mlflow.log_params"), \
             patch("mlflow.log_metric"), \
             patch("mlflow.log_artifact"), \
             patch("mlflow.set_tag"), \
             patch("src.models.nn._dvc_add"):
            return train_nn(
                data_path=csv_path,
                out_dir=out_dir,
                seed=0,
                param_grid=_TINY_GRID,
                n_epochs=_TINY_EPOCHS,
                batch_size=_TINY_BATCH,
            )

    def test_returns_nn_pipeline(self, tmp_path):
        result = self._run(tmp_path)
        assert isinstance(result, NNPipeline)

    def test_pkl_written(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "out" / "nn_model.pkl").exists()

    def test_pt_written(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "out" / "nn_model.pt").exists()

    def test_config_json_written(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "out" / "nn_model_config.json").exists()

    def test_config_has_expected_keys(self, tmp_path):
        self._run(tmp_path)
        cfg = json.loads((tmp_path / "out" / "nn_model_config.json").read_text())
        for key in ["story", "best_params", "cv_results",
                    "best_cv_roc_auc_mean", "train_years", "feature_cols"]:
            assert key in cfg
        assert cfg["story"] == "US-S5-01"

    def test_pkl_is_loadable(self, tmp_path):
        self._run(tmp_path)
        with open(tmp_path / "out" / "nn_model.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, NNPipeline)
        assert loaded.model_ is not None

    def test_pt_state_dict_loadable(self, tmp_path):
        self._run(tmp_path)
        cfg = json.loads((tmp_path / "out" / "nn_model_config.json").read_text())
        n_feat = cfg["n_features"]
        hidden = tuple(cfg["best_params"]["hidden_dims"])
        model = _MLP(n_feat, hidden)
        state = torch.load(tmp_path / "out" / "nn_model.pt", weights_only=True)
        model.load_state_dict(state)  # must not raise
