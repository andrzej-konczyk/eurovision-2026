"""Tests for US-S5-05 — src/models/surrogate.py"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.surrogate import (
    MAX_INFERENCE_S,
    MAX_RANK_DELTA,
    DistilledSurrogate,
    make_ridge_pipeline,
    rank_delta,
    time_inference,
    train_surrogate,
)


# ---------------------------------------------------------------------------
# make_ridge_pipeline
# ---------------------------------------------------------------------------


def test_make_ridge_pipeline_returns_pipeline():
    pipe = make_ridge_pipeline()
    assert isinstance(pipe, Pipeline)


def test_make_ridge_pipeline_step_names():
    names = [s[0] for s in make_ridge_pipeline().steps]
    assert "imputer" in names
    assert "scaler"  in names
    assert "model"   in names


# ---------------------------------------------------------------------------
# DistilledSurrogate
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_surrogate():
    """Surrogate fitted on tiny 2-feature dataset."""
    rng = np.random.default_rng(0)
    n = 20
    X = pd.DataFrame({"f1": rng.uniform(0, 1, n), "f2": rng.uniform(0, 1, n)})
    soft = rng.uniform(0, 1, n)
    pipe = make_ridge_pipeline()
    pipe.fit(X, soft)
    return DistilledSurrogate(pipe)


def test_distilled_surrogate_predict_proba_shape(fitted_surrogate):
    X = pd.DataFrame({"f1": [0.5, 0.3], "f2": [0.2, 0.8]})
    proba = fitted_surrogate.predict_proba(X)
    assert proba.shape == (2, 2)


def test_distilled_surrogate_proba_sums_to_one(fitted_surrogate):
    X = pd.DataFrame({"f1": [0.5, 0.3], "f2": [0.2, 0.8]})
    proba = fitted_surrogate.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_distilled_surrogate_proba_clipped(fitted_surrogate):
    X = pd.DataFrame({"f1": [0.5], "f2": [0.5]})
    proba = fitted_surrogate.predict_proba(X)
    assert (proba >= 0.0).all()
    assert (proba <= 1.0).all()


def test_distilled_surrogate_predict_binary(fitted_surrogate):
    X = pd.DataFrame({"f1": [0.9, 0.1], "f2": [0.9, 0.1]})
    preds = fitted_surrogate.predict(X)
    assert set(preds).issubset({0, 1})


def test_distilled_surrogate_pickle_roundtrip(fitted_surrogate, tmp_path):
    pkl = tmp_path / "surr.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(fitted_surrogate, fh)
    with open(pkl, "rb") as fh:
        loaded = pickle.load(fh)
    X = pd.DataFrame({"f1": [0.5], "f2": [0.5]})
    np.testing.assert_allclose(
        fitted_surrogate.predict_proba(X),
        loaded.predict_proba(X),
    )


# ---------------------------------------------------------------------------
# rank_delta
# ---------------------------------------------------------------------------


def test_rank_delta_identical_probas_gives_zero():
    proba = np.array([0.9, 0.7, 0.5, 0.3])
    countries = ["A", "B", "C", "D"]
    result = rank_delta(proba, proba, countries)
    assert result["mean_abs_delta"] == 0.0
    assert result["max_abs_delta"] == 0


def test_rank_delta_reversed_order_correct_delta():
    ens  = np.array([0.9, 0.7, 0.5, 0.3])
    surr = np.array([0.3, 0.5, 0.7, 0.9])
    result = rank_delta(ens, surr, ["A", "B", "C", "D"])
    assert result["max_abs_delta"] == 3


def test_rank_delta_sorted_by_ensemble_rank():
    proba = np.array([0.8, 0.6, 0.4, 0.2])
    result = rank_delta(proba, proba, ["A", "B", "C", "D"])
    ranks = [r["ensemble_rank"] for r in result["countries"]]
    assert ranks == sorted(ranks)


def test_rank_delta_required_keys():
    proba = np.array([0.7, 0.3])
    result = rank_delta(proba, proba, ["X", "Y"])
    for key in ("mean_abs_delta", "max_abs_delta", "countries"):
        assert key in result
    for row in result["countries"]:
        for k in ("country", "ensemble_rank", "surrogate_rank", "rank_delta",
                  "ensemble_prob", "surrogate_prob"):
            assert k in row


def test_rank_delta_country_count():
    proba = np.array([0.6, 0.4, 0.2])
    result = rank_delta(proba, proba, ["A", "B", "C"])
    assert len(result["countries"]) == 3


# ---------------------------------------------------------------------------
# time_inference
# ---------------------------------------------------------------------------


def test_time_inference_returns_float(fitted_surrogate):
    X = pd.DataFrame({"f1": [0.5, 0.3], "f2": [0.2, 0.8]})
    elapsed = time_inference(fitted_surrogate, X, n_reps=5)
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


# ---------------------------------------------------------------------------
# train_surrogate — unit (tiny synthetic data)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(0)
    n = 12
    X = pd.DataFrame({
        "f1": rng.uniform(0, 1, n),
        "f2": rng.uniform(0, 1, n),
        "f3": rng.uniform(0, 1, n),
    })
    y = pd.Series((rng.uniform(0, 1, n) > 0.5).astype(int))
    groups = pd.Series([2021] * 6 + [2022] * 6)
    return X, y, groups


def test_train_surrogate_no_ensemble_fallback(tiny_data):
    X, y, groups = tiny_data
    surr, cv_info = train_surrogate(X, y, groups, seed=42, ensemble=None)
    assert isinstance(surr, DistilledSurrogate)
    assert cv_info["distillation"] == "labels"


def test_train_surrogate_with_mock_ensemble(tiny_data):
    X, y, groups = tiny_data
    mock_ens = MagicMock()
    mock_ens.predict_proba.return_value = np.column_stack(
        [np.full(len(X), 0.4), np.full(len(X), 0.6)]
    )
    surr, cv_info = train_surrogate(X, y, groups, seed=42, ensemble=mock_ens)
    assert isinstance(surr, DistilledSurrogate)
    assert cv_info["distillation"] in ("in-sample", "oof")
    assert "best_alpha" in cv_info
    assert "spearman_r_insample" in cv_info


def test_train_surrogate_predict_proba_shape(tiny_data):
    X, y, groups = tiny_data
    surr, _ = train_surrogate(X, y, groups, seed=42, ensemble=None)
    proba = surr.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_train_surrogate_best_alpha_in_grid(tiny_data):
    from src.models.surrogate import RIDGE_ALPHAS
    X, y, groups = tiny_data
    _, cv_info = train_surrogate(X, y, groups, seed=42, ensemble=None)
    assert cv_info["best_alpha"] in RIDGE_ALPHAS


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_max_rank_delta_value():
    assert MAX_RANK_DELTA == pytest.approx(2.0)


def test_max_inference_s_value():
    assert MAX_INFERENCE_S == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# run() — integration smoke test (mocked ensemble + mlflow)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    years = [2021, 2022, 2023, 2026]
    countries = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    rows = []
    for year in years:
        for i, country in enumerate(countries):
            is_hist = year < 2026
            rows.append({
                "Year": year, "Country": country,
                "Country_Group": "Western",
                "Grand_Final_Ind": 1,
                "Big6_Ind": 0, "National_Final": 1, "Solo_Artist": 1,
                "Returning_Artist_Ind": 0, "Number of Members": 1,
                "Multiple_Language": 0, "EU": 1, "NATO": 1,
                "Qualification_Record": 0.7,
                "Semi_Final_Num": float("nan"),
                "Running_Order_Semi": float("nan"),
                "Running_Order_Final": float(i + 1),
                "MyESB_Community": rng.uniform(10, 100),
                "MyESB_Personal": rng.uniform(10, 100),
                "OGAE_Points": rng.uniform(10, 500),
                "jury_points": rng.uniform(50, 200) if is_hist else float("nan"),
                "tele_points": rng.uniform(50, 200) if is_hist else float("nan"),
                "Final_Place": float(i + 1) if is_hist else float("nan"),
                "Top 10": float(1 if i < 2 else 0) if is_hist else float("nan"),
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "enriched.csv"
    df.to_csv(path, index=False)
    return path


def _mock_ensemble():
    """Return a mock pipeline whose predict_proba scales to any input size."""
    def _side_effect(X):
        n = len(X)
        proba = np.linspace(0.9, 0.1, n)
        return np.column_stack([1 - proba, proba])
    mock = MagicMock()
    mock.predict_proba.side_effect = _side_effect
    return mock


def test_run_returns_dict(synthetic_csv, tmp_path):
    with patch("src.models.surrogate.mlflow"), \
         patch("src.models.surrogate.load_pipeline",
               return_value=_mock_ensemble()):
        from src.models.surrogate import run
        result = run(data_path=synthetic_csv, target_year=2026,
                     out_dir=tmp_path, reports_dir=tmp_path, seed=42)
    assert isinstance(result, dict)
    assert result["story"] == "US-S5-05"


def test_run_writes_json(synthetic_csv, tmp_path):
    with patch("src.models.surrogate.mlflow"), \
         patch("src.models.surrogate.load_pipeline",
               return_value=_mock_ensemble()):
        from src.models.surrogate import run
        run(data_path=synthetic_csv, target_year=2026,
            out_dir=tmp_path, reports_dir=tmp_path, seed=42)
    json_path = tmp_path / "surrogate_2026.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["story"] == "US-S5-05"
    assert data["target_year"] == 2026


def test_run_writes_pkl(synthetic_csv, tmp_path):
    with patch("src.models.surrogate.mlflow"), \
         patch("src.models.surrogate.load_pipeline",
               return_value=_mock_ensemble()):
        from src.models.surrogate import run
        run(data_path=synthetic_csv, target_year=2026,
            out_dir=tmp_path, reports_dir=tmp_path, seed=42)
    pkl_path = tmp_path / "surrogate_model.pkl"
    assert pkl_path.exists()
    with open(pkl_path, "rb") as fh:
        loaded = pickle.load(fh)
    assert isinstance(loaded, DistilledSurrogate)


def test_run_kpi_fields_present(synthetic_csv, tmp_path):
    with patch("src.models.surrogate.mlflow"), \
         patch("src.models.surrogate.load_pipeline",
               return_value=_mock_ensemble()):
        from src.models.surrogate import run
        result = run(data_path=synthetic_csv, target_year=2026,
                     out_dir=tmp_path, reports_dir=tmp_path, seed=42)
    assert "kpi_delta_pass" in result
    assert "kpi_inference_pass" in result
    assert isinstance(result["kpi_delta_pass"],     bool)
    assert isinstance(result["kpi_inference_pass"], bool)


def test_run_countries_count(synthetic_csv, tmp_path):
    with patch("src.models.surrogate.mlflow"), \
         patch("src.models.surrogate.load_pipeline",
               return_value=_mock_ensemble()):
        from src.models.surrogate import run
        result = run(data_path=synthetic_csv, target_year=2026,
                     out_dir=tmp_path, reports_dir=tmp_path, seed=42)
    assert len(result["countries"]) == 5
