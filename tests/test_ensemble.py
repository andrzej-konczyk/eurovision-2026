"""Tests for US-S5-02 — src/models/ensemble.py"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import _weight_grid, _load_tree_params, _load_nn_params
from src.models.evaluate import top10_accuracy


# ---------------------------------------------------------------------------
# _weight_grid
# ---------------------------------------------------------------------------


def test_weight_grid_sums_to_one():
    for w_xgb, w_lgbm, w_nn in _weight_grid(0.1):
        assert abs(w_xgb + w_lgbm + w_nn - 1.0) < 1e-9


def test_weight_grid_non_negative():
    for triple in _weight_grid(0.1):
        assert all(w >= 0.0 for w in triple)


def test_weight_grid_step01_count():
    # C(12, 2) = 66 triples for step=0.1
    combos = list(_weight_grid(0.1))
    assert len(combos) == 66


def test_weight_grid_contains_unit_weights():
    combos = set(_weight_grid(0.1))
    assert (1.0, 0.0, 0.0) in combos
    assert (0.0, 1.0, 0.0) in combos
    assert (0.0, 0.0, 1.0) in combos


def test_weight_grid_step05_count():
    # step=0.5: (0,0,1),(0,0.5,0.5),(0,1,0),(0.5,0,0.5),(0.5,0.5,0),(1,0,0) = 6
    combos = list(_weight_grid(0.5))
    assert len(combos) == 6


# ---------------------------------------------------------------------------
# Param loading
# ---------------------------------------------------------------------------


def test_load_tree_params_missing_file(tmp_path: Path):
    result = _load_tree_params(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_tree_params_correct_prefix(tmp_path: Path):
    meta = {
        "models": {
            "xgb":  {"best_params": {"n_estimators": 100, "max_depth": 3}},
            "lgbm": {"best_params": {"n_estimators": 200, "num_leaves": 31}},
        }
    }
    p = tmp_path / "train_meta.json"
    p.write_text(json.dumps(meta))
    result = _load_tree_params(p)
    assert result["xgb"]["model__n_estimators"] == 100
    assert result["lgbm"]["model__num_leaves"] == 31


def test_load_nn_params_missing_file(tmp_path: Path):
    result = _load_nn_params(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_nn_params_hidden_dims_is_tuple(tmp_path: Path):
    cfg = {"best_params": {"hidden_dims": [64, 32], "lr": 0.001, "dropout": 0.0}}
    p = tmp_path / "nn_model_config.json"
    p.write_text(json.dumps(cfg))
    result = _load_nn_params(p)
    assert isinstance(result["hidden_dims"], tuple)
    assert result["hidden_dims"] == (64, 32)


# ---------------------------------------------------------------------------
# blend — fast smoke test with minimal MLP epochs
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small synthetic dataset matching the enriched CSV schema."""
    years = [2021, 2022, 2023, 2024]
    countries = [f"C{i:02d}" for i in range(26)]
    rows = []
    rng = np.random.default_rng(42)
    for year in years:
        for i, country in enumerate(countries):
            finalist = i < 20
            place = float(i + 1) if finalist else np.nan
            rows.append({
                "Year": year,
                "Country": country,
                "Country_Group": "Western",
                "Grand_Final_Ind": 1 if finalist else 0,
                "Final_Place": place,
                "jury_points": rng.uniform(0, 200) if finalist else np.nan,
                "tele_points": rng.uniform(0, 200) if finalist else np.nan,
                "Final_Points": rng.uniform(0, 400) if finalist else np.nan,
                "Top 10": 1.0 if (finalist and i < 10) else (np.nan if not finalist else 0.0),
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
                "MyESB_Community": rng.uniform(10, 100),
                "MyESB_Personal": rng.uniform(10, 100),
                "OGAE_Points": rng.uniform(10, 500) if i < 15 else np.nan,
            })
    return pd.DataFrame(rows)


def test_blend_returns_dict_with_required_keys(tmp_path: Path, synthetic_df: pd.DataFrame):
    """blend() returns artefact with expected keys and KPI-pass bool."""
    csv_path = tmp_path / "enriched.csv"
    synthetic_df.to_csv(csv_path, index=False)

    from src.models.ensemble import blend

    with patch("src.models.ensemble.mlflow"):
        result = blend(
            data_path=csv_path,
            out_dir=tmp_path,
            holdout_year=2024,
            seed=42,
            weight_step=0.5,   # coarse grid — keeps test fast
        )

    assert "best_weights" in result
    assert "top10_accuracy" in result
    assert "kpi_pass" in result
    assert isinstance(result["kpi_pass"], bool)
    w = result["best_weights"]
    assert abs(w["xgb"] + w["lgbm"] + w["nn"] - 1.0) < 1e-9


def test_blend_writes_json_artefact(tmp_path: Path, synthetic_df: pd.DataFrame):
    csv_path = tmp_path / "enriched.csv"
    synthetic_df.to_csv(csv_path, index=False)

    from src.models.ensemble import blend

    with patch("src.models.ensemble.mlflow"):
        blend(
            data_path=csv_path,
            out_dir=tmp_path,
            holdout_year=2024,
            seed=42,
            weight_step=0.5,
        )

    artefact_path = tmp_path / "ensemble_weights.json"
    assert artefact_path.exists()
    data = json.loads(artefact_path.read_text())
    assert data["story"] == "US-S5-02"
