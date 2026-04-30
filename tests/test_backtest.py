"""Tests for src/models/backtest.py (US-S6-01)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.backtest import (
    BACKTEST_YEARS,
    KPI_CI80_THRESHOLD,
    KPI_TOP10_THRESHOLD,
    TOP_K,
    _backtest_split,
    backtest_year,
    ci80_empirical_coverage,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_FEAT_COLS = ["f1", "f2"]


def _make_matrix(years_countries: dict[int, list[str]], *, feat_cols=_FEAT_COLS) -> pd.DataFrame:
    """Build a minimal pre-built feature matrix.

    years_countries maps year → list of country names.
    Exactly the first 10 countries per year are labelled top-10 (y=1).
    """
    rng = np.random.default_rng(0)
    rows = []
    for year, countries in years_countries.items():
        for i, c in enumerate(countries):
            y = 1 if i < 10 else 0
            rows.append({
                "Year": year,
                "Country": c,
                "Grand_Final_Ind": 1,
                "top10_derived": float(y),
                **{col: rng.standard_normal() for col in feat_cols},
            })
    return pd.DataFrame(rows)


_COUNTRIES_26 = [f"C{i:02d}" for i in range(26)]

_YEARS_COUNTRIES = {
    2016: _COUNTRIES_26,
    2017: _COUNTRIES_26,
    2018: _COUNTRIES_26,
    2019: _COUNTRIES_26,
    2021: _COUNTRIES_26,
    2022: _COUNTRIES_26,
}


@pytest.fixture()
def small_matrix():
    return _make_matrix(_YEARS_COUNTRIES)


# ---------------------------------------------------------------------------
# ci80_empirical_coverage
# ---------------------------------------------------------------------------

class TestCi80EmpiricalCoverage:

    def _make_ci_df(self, countries, ci80_lo, ci80_hi):
        return pd.DataFrame({
            "country": countries,
            "prob_mean": [0.5] * len(countries),
            "ci80_lo": ci80_lo,
            "ci80_hi": ci80_hi,
            "ci50_lo": [0.3] * len(countries),
            "ci50_hi": [0.7] * len(countries),
        })

    def test_perfect_coverage_y1(self):
        # y=1 with ci80_hi > 0.5 → covered
        ci_df = self._make_ci_df(["A", "B"], [0.2, 0.3], [0.9, 0.8])
        y = pd.Series([1, 1])
        assert ci80_empirical_coverage(ci_df, y, ["A", "B"]) == pytest.approx(1.0)

    def test_perfect_coverage_y0(self):
        # y=0 with ci80_lo < 0.5 → covered
        ci_df = self._make_ci_df(["A", "B"], [0.1, 0.2], [0.4, 0.45])
        y = pd.Series([0, 0])
        assert ci80_empirical_coverage(ci_df, y, ["A", "B"]) == pytest.approx(1.0)

    def test_zero_coverage_y0_confident_wrong(self):
        # y=0 but ci80_lo >= 0.5 → not covered (model confidently wrong)
        ci_df = self._make_ci_df(["A", "B"], [0.6, 0.7], [0.9, 0.95])
        y = pd.Series([0, 0])
        assert ci80_empirical_coverage(ci_df, y, ["A", "B"]) == pytest.approx(0.0)

    def test_zero_coverage_y1_confident_wrong(self):
        # y=1 but ci80_hi <= 0.5 → not covered (model confidently wrong)
        ci_df = self._make_ci_df(["A", "B"], [0.1, 0.2], [0.4, 0.45])
        y = pd.Series([1, 1])
        assert ci80_empirical_coverage(ci_df, y, ["A", "B"]) == pytest.approx(0.0)

    def test_partial_coverage(self):
        # A: y=1, ci80_hi=0.8 → covered
        # B: y=0, ci80_lo=0.6 → not covered
        ci_df = self._make_ci_df(["A", "B"], [0.3, 0.6], [0.8, 0.9])
        y = pd.Series([1, 0])
        assert ci80_empirical_coverage(ci_df, y, ["A", "B"]) == pytest.approx(0.5)

    def test_boundary_05_y0_not_covered(self):
        # ci80_lo == 0.5 → condition is lo < 0.5 → False → not covered
        ci_df = self._make_ci_df(["A"], [0.5], [0.9])
        y = pd.Series([0])
        assert ci80_empirical_coverage(ci_df, y, ["A"]) == pytest.approx(0.0)

    def test_boundary_05_y1_not_covered(self):
        # ci80_hi == 0.5 → condition is hi > 0.5 → False → not covered
        ci_df = self._make_ci_df(["A"], [0.1], [0.5])
        y = pd.Series([1])
        assert ci80_empirical_coverage(ci_df, y, ["A"]) == pytest.approx(0.0)

    def test_missing_country_skipped(self):
        # "B" is not in ci_df → skip it; only "A" counted
        ci_df = self._make_ci_df(["A"], [0.1], [0.8])
        y = pd.Series([1, 0])  # B would be 0 but it's absent
        cov = ci80_empirical_coverage(ci_df, y, ["A", "B"])
        # Only A counts (1/1 = 1.0)
        assert cov == pytest.approx(1.0)

    def test_empty_returns_nan(self):
        ci_df = pd.DataFrame(columns=["country", "prob_mean", "ci80_lo", "ci80_hi",
                                      "ci50_lo", "ci50_hi"])
        y = pd.Series([], dtype=int)
        assert np.isnan(ci80_empirical_coverage(ci_df, y, []))

    def test_mixed_outcomes(self):
        # A: y=1, hi=0.9 > 0.5 → covered
        # B: y=0, lo=0.1 < 0.5 → covered
        # C: y=1, hi=0.8 > 0.5 → covered
        # D: y=1, hi=0.45 ≤ 0.5 → not covered  → 3/4
        ci_df = self._make_ci_df(
            ["A", "B", "C", "D"],
            [0.2, 0.1, 0.3, 0.1],
            [0.9, 0.4, 0.8, 0.45],
        )
        y = pd.Series([1, 0, 1, 1])
        cov = ci80_empirical_coverage(ci_df, y, ["A", "B", "C", "D"])
        assert cov == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# _backtest_split
# ---------------------------------------------------------------------------

class TestBacktestSplit:

    def test_train_years_strictly_less(self, small_matrix):
        X_tr, y_tr, groups, X_te, y_te, te_rows = _backtest_split(
            small_matrix, holdout_year=2022, feat_cols=_FEAT_COLS
        )
        assert all(groups < 2022)

    def test_test_year_is_holdout(self, small_matrix):
        X_tr, y_tr, groups, X_te, y_te, te_rows = _backtest_split(
            small_matrix, holdout_year=2022, feat_cols=_FEAT_COLS
        )
        assert set(te_rows["Year"].unique()) == {2022}

    def test_train_test_disjoint(self, small_matrix):
        X_tr, y_tr, groups, X_te, y_te, te_rows = _backtest_split(
            small_matrix, holdout_year=2022, feat_cols=_FEAT_COLS
        )
        assert len(X_tr) > 0
        assert len(X_te) > 0
        assert len(X_tr) + len(X_te) <= len(small_matrix)

    def test_grand_final_only(self):
        """Non-Grand Final rows (Grand_Final_Ind=0) must not appear in either split."""
        matrix = _make_matrix({2016: _COUNTRIES_26[:15], 2017: _COUNTRIES_26[:15]})
        # Inject a non-final row into 2016
        non_final = matrix.iloc[0:1].copy()
        non_final["Grand_Final_Ind"] = 0
        matrix = pd.concat([matrix, non_final], ignore_index=True)

        X_tr, y_tr, groups, X_te, y_te, te_rows = _backtest_split(
            matrix, holdout_year=2017, feat_cols=_FEAT_COLS
        )
        assert len(X_tr) == 15  # 2016 only, Grand Final only

    def test_no_null_labels(self, small_matrix):
        X_tr, y_tr, groups, X_te, y_te, te_rows = _backtest_split(
            small_matrix, holdout_year=2022, feat_cols=_FEAT_COLS
        )
        assert y_tr.notna().all()
        assert y_te.notna().all()

    def test_feat_cols_subset(self, small_matrix):
        X_tr, y_tr, groups, X_te, y_te, te_rows = _backtest_split(
            small_matrix, holdout_year=2022, feat_cols=_FEAT_COLS
        )
        assert list(X_tr.columns) == _FEAT_COLS
        assert list(X_te.columns) == _FEAT_COLS


# ---------------------------------------------------------------------------
# backtest_year (integration — mocks slow parts)
# ---------------------------------------------------------------------------

def _make_mock_gs(X_train, y_train):
    """Return a MagicMock mimicking GridSearchCV with a DummyClassifier pipeline."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="stratified", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    mock_gs = MagicMock()
    mock_gs.best_estimator_ = pipe
    mock_gs.best_params_ = {"model__strategy": "stratified"}
    return mock_gs


class TestBacktestYear:
    """Integration tests that mock _grid_search + bootstrap_proba."""

    def _tiny_matrix(self) -> pd.DataFrame:
        # 5 training years + 1 holdout (2022), 26 countries each
        return _make_matrix({
            2016: _COUNTRIES_26,
            2017: _COUNTRIES_26,
            2018: _COUNTRIES_26,
            2019: _COUNTRIES_26,
            2021: _COUNTRIES_26,
            2022: _COUNTRIES_26,
        })

    def _dummy_raw_df(self) -> pd.DataFrame:
        # Minimal raw df — add_derived_top10 needs Year, Country, Final_Place
        rows = []
        for year in [2016, 2017, 2018, 2019, 2021, 2022]:
            for i, c in enumerate(_COUNTRIES_26):
                rows.append({
                    "Year": year,
                    "Country": c,
                    "Grand_Final_Ind": 1,
                    "Final_Place": i + 1,
                    "Top 10": 1 if i < 10 else np.nan,
                })
        return pd.DataFrame(rows)

    def _proba_matrix_fixture(self, n_countries=26, n_bootstrap=5):
        """Return a (n_bootstrap, n_countries) probability matrix."""
        rng = np.random.default_rng(42)
        return rng.random((n_bootstrap, n_countries))

    def _run(self, holdout_year=2022):
        """Run backtest_year with mocked _grid_search, bootstrap_proba, FEATURE_COLS."""
        matrix = self._tiny_matrix()
        raw_df = self._dummy_raw_df()
        proba_mat = self._proba_matrix_fixture()
        with (
            patch("src.models.backtest.FEATURE_COLS", ["f1", "f2"]),
            patch("src.models.backtest._grid_search",
                  side_effect=lambda clf, pg, X, y, g: _make_mock_gs(X, y)),
            patch("src.models.backtest.bootstrap_proba", return_value=proba_mat),
        ):
            return backtest_year(
                holdout_year=holdout_year,
                matrix=matrix,
                df_source=raw_df,
                n_bootstrap=5,
                seed=42,
            )

    def test_returns_expected_top_keys(self):
        result = self._run()
        assert result["year"] == 2022
        assert "n_train" in result
        assert "n_test" in result
        assert "train_years" in result
        assert "models" in result
        assert set(result["models"]) == {"xgb", "lgbm"}

    def test_per_model_keys(self):
        result = self._run()
        for model_name in ("xgb", "lgbm"):
            m = result["models"][model_name]
            assert "top10_accuracy" in m
            assert "ci80_empirical_coverage" in m
            assert "kpi_top10_pass" in m
            assert "kpi_ci80_pass" in m
            assert "country_detail" in m

    def test_train_years_exclude_holdout(self):
        result = self._run()
        assert 2022 not in result["train_years"]
        assert all(y < 2022 for y in result["train_years"])

    def test_top10_accuracy_in_unit_interval(self):
        result = self._run()
        for model_name in ("xgb", "lgbm"):
            acc = result["models"][model_name]["top10_accuracy"]
            assert 0.0 <= acc <= 1.0

    def test_ci80_coverage_in_unit_interval(self):
        result = self._run()
        for model_name in ("xgb", "lgbm"):
            cov = result["models"][model_name]["ci80_empirical_coverage"]
            assert 0.0 <= cov <= 1.0

    def test_country_detail_has_all_finalists(self):
        result = self._run()
        for model_name in ("xgb", "lgbm"):
            detail = result["models"][model_name]["country_detail"]
            assert len(detail) == 26  # all Grand Final rows in 2022

    def test_kpi_flags_are_bool(self):
        result = self._run()
        for model_name in ("xgb", "lgbm"):
            m = result["models"][model_name]
            assert isinstance(m["kpi_top10_pass"], bool)
            assert isinstance(m["kpi_ci80_pass"], bool)


# ---------------------------------------------------------------------------
# run_backtest (smoke test — mocks IO and bootstrap)
# ---------------------------------------------------------------------------

class TestRunBacktest:

    def _make_all_years_matrix(self) -> pd.DataFrame:
        years_countries = {
            2016: _COUNTRIES_26,
            2017: _COUNTRIES_26,
            2018: _COUNTRIES_26,
            2019: _COUNTRIES_26,
            2021: _COUNTRIES_26,
            2022: _COUNTRIES_26,
            2023: _COUNTRIES_26,
            2024: _COUNTRIES_26,
        }
        return _make_matrix(years_countries)

    def _dummy_raw_df(self) -> pd.DataFrame:
        rows = []
        for year in [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]:
            for i, c in enumerate(_COUNTRIES_26):
                rows.append({
                    "Year": year, "Country": c,
                    "Grand_Final_Ind": 1,
                    "Final_Place": i + 1,
                    "Top 10": 1 if i < 10 else np.nan,
                })
        return pd.DataFrame(rows)

    def _proba_mat(self):
        rng = np.random.default_rng(0)
        return rng.random((5, 26))

    def _ctx(self, tmp_path):
        """Return (dummy_csv, ctx_managers) for run_backtest tests."""
        matrix = self._make_all_years_matrix()
        raw_df = self._dummy_raw_df()
        dummy_csv = tmp_path / "dummy.csv"
        raw_df.to_csv(dummy_csv, index=False)
        proba_mat = self._proba_mat()
        ctx = (
            patch("src.models.backtest.FEATURE_COLS", ["f1", "f2"]),
            patch("src.models.backtest.build_feature_matrix", return_value=matrix),
            patch("src.models.backtest._grid_search",
                  side_effect=lambda clf, pg, X, y, g: _make_mock_gs(X, y)),
            patch("src.models.backtest.bootstrap_proba", return_value=proba_mat),
            patch("src.models.backtest.mlflow"),
        )
        return dummy_csv, ctx

    def test_json_written(self, tmp_path):
        dummy_csv, ctx = self._ctx(tmp_path)
        with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4]:
            run_backtest(data_path=dummy_csv, years=[2022, 2023, 2024],
                         n_bootstrap=5, seed=42, out_dir=tmp_path)
        assert (tmp_path / "backtest_2022_2024.json").exists()

    def test_markdown_written(self, tmp_path):
        dummy_csv, ctx = self._ctx(tmp_path)
        with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4]:
            run_backtest(data_path=dummy_csv, years=[2022, 2023, 2024],
                         n_bootstrap=5, seed=42, out_dir=tmp_path)
        assert (tmp_path / "backtest_2022_2024.md").exists()

    def test_result_structure(self, tmp_path):
        dummy_csv, ctx = self._ctx(tmp_path)
        with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4]:
            result = run_backtest(data_path=dummy_csv, years=[2022, 2023, 2024],
                                  n_bootstrap=5, seed=42, out_dir=tmp_path)
        assert result["story"] == "US-S6-01"
        assert set(result["years"].keys()) == {"2022", "2023", "2024"}
        assert "aggregate" in result
        assert set(result["aggregate"].keys()) == {"xgb", "lgbm"}

    def test_aggregate_keys(self, tmp_path):
        dummy_csv, ctx = self._ctx(tmp_path)
        with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4]:
            result = run_backtest(data_path=dummy_csv, years=[2022, 2023, 2024],
                                  n_bootstrap=5, seed=42, out_dir=tmp_path)
        for model_name in ("xgb", "lgbm"):
            agg = result["aggregate"][model_name]
            assert "avg_top10_accuracy" in agg
            assert "avg_ci80_empirical_coverage" in agg
            assert "all_top10_kpi_pass" in agg
            assert "all_ci80_kpi_pass" in agg
