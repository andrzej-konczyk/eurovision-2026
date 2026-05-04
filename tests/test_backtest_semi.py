"""Tests for src/models/backtest_semi.py (US-S6-01b)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.backtest_semi import (
    BACKTEST_YEARS,
    KPI_CI80_THRESHOLD,
    KPI_QUAL_THRESHOLD,
    QUALIFIERS_PER_SEMI,
    SEMI_FEATURE_COLS,
    _semi_split,
    backtest_semi_year,
    qualification_accuracy,
    run_semi_backtest,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_FEAT_COLS = ["f1", "f2"]
_COUNTRIES_SF1 = [f"A{i:02d}" for i in range(17)]   # 17 SF1 entrants
_COUNTRIES_SF2 = [f"B{i:02d}" for i in range(18)]   # 18 SF2 entrants


def _make_semi_matrix(
    years: list[int],
    *,
    feat_cols: list[str] = _FEAT_COLS,
) -> pd.DataFrame:
    """Minimal semi-final feature matrix.

    First 10 countries per semi per year are qualifiers (Grand_Final_Ind=1).
    """
    rng = np.random.default_rng(0)
    rows = []
    for year in years:
        for sf, countries in ((1, _COUNTRIES_SF1), (2, _COUNTRIES_SF2)):
            for i, c in enumerate(countries):
                rows.append({
                    "Year": year,
                    "Country": c,
                    "Semi_Final_Num": float(sf),
                    "Grand_Final_Ind": 1 if i < QUALIFIERS_PER_SEMI else 0,
                    **{col: rng.standard_normal() for col in feat_cols},
                })
    return pd.DataFrame(rows)


_ALL_TRAIN_YEARS = [2016, 2017, 2018, 2019, 2021]
_ALL_YEARS = _ALL_TRAIN_YEARS + [2022, 2023, 2024]


# ---------------------------------------------------------------------------
# qualification_accuracy
# ---------------------------------------------------------------------------

class TestQualificationAccuracy:

    def _setup(self, sf1_size=17, sf2_size=18, k=10):
        """Build synthetic proba and y arrays."""
        n = sf1_size + sf2_size
        semi_num = np.array([1] * sf1_size + [2] * sf2_size, dtype=float)
        # Perfect signal: first k in each SF get high proba
        proba = np.zeros(n)
        proba[:k] = 0.9           # SF1 top-10 → high
        proba[sf1_size:sf1_size + k] = 0.9   # SF2 top-10 → high
        y = np.zeros(n, dtype=int)
        y[:k] = 1
        y[sf1_size:sf1_size + k] = 1
        return y, proba, semi_num

    def test_perfect_accuracy(self):
        y, proba, semi_num = self._setup()
        result = qualification_accuracy(y, proba, semi_num)
        assert result["sf1"] == pytest.approx(1.0)
        assert result["sf2"] == pytest.approx(1.0)
        assert result["overall"] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        # Use equal-size semis so inversion gives all non-qualifiers in top-10
        y, proba, semi_num = self._setup(sf1_size=20, sf2_size=20)
        proba_inv = 1 - proba
        result = qualification_accuracy(y, proba_inv, semi_num)
        assert result["sf1"] == pytest.approx(0.0)
        assert result["sf2"] == pytest.approx(0.0)

    def test_half_accuracy(self):
        y, proba, semi_num = self._setup()
        # Give exactly 5 correct and 5 wrong in SF1
        proba2 = proba.copy()
        proba2[:5] = 0.1   # first 5 qualifiers get low proba → missed
        proba2[10:15] = 0.9  # 5 non-qualifiers get high proba → false positive
        result = qualification_accuracy(y, proba2, semi_num)
        assert result["sf1"] == pytest.approx(0.5)

    def test_returns_both_semis_and_overall(self):
        y, proba, semi_num = self._setup()
        result = qualification_accuracy(y, proba, semi_num)
        assert "sf1" in result
        assert "sf2" in result
        assert "overall" in result

    def test_overall_is_average_of_semis(self):
        # SF1 (17): qualifiers=0-9, non-quals=10-16
        # SF2 (18): qualifiers=17-26, non-quals=27-34
        # Want SF1=80% (8 hits), SF2=60% (6 hits) → overall=70%
        # Give missed qualifiers proba=0.01 and non-qualifiers proba=0.5
        # so non-qualifiers beat missed qualifiers deterministically.
        y, proba, semi_num = self._setup()
        n = len(y)
        proba2 = np.full(n, 0.01)
        proba2[:8] = 0.9           # SF1: 8 qualifiers (0-7) → in top-10
        proba2[10:17] = 0.5        # SF1: 7 non-quals → beat missed qualifiers 8,9
        proba2[17:17+6] = 0.9      # SF2: 6 qualifiers (0-5 local) → in top-10
        proba2[17+10:17+18] = 0.5  # SF2: 8 non-quals → beat missed qualifiers 6-9
        result = qualification_accuracy(y, proba2, semi_num)
        assert result["sf1"] == pytest.approx(0.8)
        assert result["sf2"] == pytest.approx(0.6)
        assert result["overall"] == pytest.approx(0.7)

    def test_all_values_in_unit_interval(self):
        rng = np.random.default_rng(7)
        y, proba, semi_num = self._setup()
        proba_rand = rng.random(len(y))
        result = qualification_accuracy(y, proba_rand, semi_num)
        for v in result.values():
            if not np.isnan(v):
                assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# _semi_split
# ---------------------------------------------------------------------------

class TestSemiSplit:

    def test_train_years_strictly_less(self):
        matrix = _make_semi_matrix(_ALL_YEARS)
        X_tr, y_tr, groups, X_te, y_te, te_rows = _semi_split(
            matrix, holdout_year=2022, feat_cols=_FEAT_COLS
        )
        assert all(groups < 2022)

    def test_test_year_is_holdout(self):
        matrix = _make_semi_matrix(_ALL_YEARS)
        X_tr, y_tr, groups, X_te, y_te, te_rows = _semi_split(
            matrix, holdout_year=2022, feat_cols=_FEAT_COLS
        )
        assert set(te_rows["Year"].unique()) == {2022}

    def test_only_semi_finalists(self):
        matrix = _make_semi_matrix(_ALL_YEARS)
        # Inject a Big5 row (Semi_Final_Num=NaN)
        big5_row = matrix.iloc[0:1].copy()
        big5_row["Semi_Final_Num"] = np.nan
        matrix = pd.concat([matrix, big5_row], ignore_index=True)
        X_tr, y_tr, groups, X_te, y_te, te_rows = _semi_split(
            matrix, holdout_year=2022, feat_cols=_FEAT_COLS
        )
        assert te_rows["Semi_Final_Num"].notna().all()
        assert groups.notna().all()   # groups = Year, proxy for semi presence

    def test_no_null_target(self):
        matrix = _make_semi_matrix(_ALL_YEARS)
        X_tr, y_tr, groups, X_te, y_te, te_rows = _semi_split(
            matrix, holdout_year=2022, feat_cols=_FEAT_COLS
        )
        assert y_tr.notna().all()
        assert y_te.notna().all()

    def test_feat_cols_subset(self):
        matrix = _make_semi_matrix(_ALL_YEARS)
        X_tr, y_tr, groups, X_te, y_te, te_rows = _semi_split(
            matrix, holdout_year=2022, feat_cols=_FEAT_COLS
        )
        assert list(X_tr.columns) == _FEAT_COLS
        assert list(X_te.columns) == _FEAT_COLS

    def test_running_order_final_excluded_from_semi_feature_cols(self):
        assert "Running_Order_Final" not in SEMI_FEATURE_COLS

    def test_gf_winner_odds_excluded_from_semi_feature_cols(self):
        assert "implied_prob_close" not in SEMI_FEATURE_COLS

    def test_semi_market_odds_included_in_semi_feature_cols(self):
        assert "implied_prob_semi" in SEMI_FEATURE_COLS

    def test_semi_feature_cols_use_feature_cols_plus_semi_market(self):
        from src.models.train import FEATURE_COLS
        expected = [
            c for c in FEATURE_COLS
            if c not in {"Running_Order_Final", "implied_prob_close"}
        ] + ["implied_prob_semi"]
        assert SEMI_FEATURE_COLS == expected


# ---------------------------------------------------------------------------
# backtest_semi_year (integration — mocks slow parts)
# ---------------------------------------------------------------------------

def _make_mock_gs(X_train, y_train):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="stratified", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    mock_gs = MagicMock()
    mock_gs.best_estimator_ = pipe
    mock_gs.best_params_ = {"model__strategy": "stratified"}
    return mock_gs


class TestBacktestSemiYear:

    def _matrix(self) -> pd.DataFrame:
        return _make_semi_matrix(_ALL_TRAIN_YEARS + [2022])

    def _proba_mat(self, n_test=35, n_bootstrap=5):
        rng = np.random.default_rng(0)
        return rng.random((n_bootstrap, n_test))

    def _run(self, holdout_year=2022):
        matrix = self._matrix()
        n_test = len(matrix[matrix["Year"] == holdout_year])
        proba_mat = self._proba_mat(n_test=n_test)
        with (
            patch("src.models.backtest_semi.SEMI_FEATURE_COLS", ["f1", "f2"]),
            patch("src.models.backtest_semi._grid_search",
                  side_effect=lambda clf, pg, X, y, g: _make_mock_gs(X, y)),
            patch("src.models.backtest_semi.bootstrap_proba", return_value=proba_mat),
        ):
            return backtest_semi_year(
                holdout_year=holdout_year,
                matrix=matrix,
                n_bootstrap=5,
                seed=42,
            )

    def test_returns_year_key(self):
        result = self._run()
        assert result["year"] == 2022

    def test_models_keys(self):
        result = self._run()
        assert set(result["models"]) == {"xgb", "lgbm"}

    def test_per_model_metric_keys(self):
        result = self._run()
        for m in result["models"].values():
            assert "qual_accuracy_sf1" in m
            assert "qual_accuracy_sf2" in m
            assert "qual_accuracy_overall" in m
            assert "ci80_empirical_coverage" in m
            assert "kpi_sf1_pass" in m
            assert "kpi_sf2_pass" in m
            assert "kpi_ci80_pass" in m
            assert "country_detail" in m

    def test_train_years_exclude_holdout(self):
        result = self._run()
        assert 2022 not in result["train_years"]

    def test_accuracies_in_unit_interval(self):
        result = self._run()
        for m in result["models"].values():
            for key in ("qual_accuracy_sf1", "qual_accuracy_sf2", "qual_accuracy_overall"):
                v = m[key]
                if not np.isnan(v):
                    assert 0.0 <= v <= 1.0

    def test_ci80_coverage_in_unit_interval(self):
        result = self._run()
        for m in result["models"].values():
            assert 0.0 <= m["ci80_empirical_coverage"] <= 1.0

    def test_country_detail_covers_all_semi_entries(self):
        result = self._run()
        expected = len(_COUNTRIES_SF1) + len(_COUNTRIES_SF2)
        for m in result["models"].values():
            assert len(m["country_detail"]) == expected

    def test_country_detail_has_semi_final_column(self):
        result = self._run()
        for m in result["models"].values():
            for row in m["country_detail"]:
                assert "semi_final" in row
                assert row["semi_final"] in (1.0, 2.0)

    def test_kpi_flags_are_bool(self):
        result = self._run()
        for m in result["models"].values():
            assert isinstance(m["kpi_sf1_pass"], bool)
            assert isinstance(m["kpi_sf2_pass"], bool)
            assert isinstance(m["kpi_ci80_pass"], bool)


# ---------------------------------------------------------------------------
# run_semi_backtest (smoke test)
# ---------------------------------------------------------------------------

class TestRunSemiBacktest:

    def _matrix(self) -> pd.DataFrame:
        return _make_semi_matrix(_ALL_YEARS)

    def _dummy_raw_df(self) -> pd.DataFrame:
        rows = []
        for year in _ALL_YEARS:
            for sf, countries in ((1, _COUNTRIES_SF1), (2, _COUNTRIES_SF2)):
                for i, c in enumerate(countries):
                    rows.append({
                        "Year": year, "Country": c,
                        "Semi_Final_Num": float(sf),
                        "Grand_Final_Ind": 1 if i < 10 else 0,
                    })
        return pd.DataFrame(rows)

    def _proba_mat(self, n_test=35, n_bootstrap=5):
        rng = np.random.default_rng(1)
        return rng.random((n_bootstrap, n_test))

    def _ctx(self, tmp_path):
        matrix = self._matrix()
        raw_df = self._dummy_raw_df()
        dummy_csv = tmp_path / "dummy_semi.csv"
        raw_df.to_csv(dummy_csv, index=False)
        ctx = (
            patch("src.models.backtest_semi.SEMI_FEATURE_COLS", ["f1", "f2"]),
            patch("src.models.backtest_semi.build_feature_matrix", return_value=matrix),
            patch("src.models.backtest_semi._grid_search",
                  side_effect=lambda clf, pg, X, y, g: _make_mock_gs(X, y)),
            patch("src.models.backtest_semi.bootstrap_proba",
                  return_value=self._proba_mat()),
            patch("src.models.backtest_semi.mlflow"),
        )
        return dummy_csv, ctx

    def test_json_written(self, tmp_path):
        dummy_csv, ctx = self._ctx(tmp_path)
        with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4]:
            run_semi_backtest(data_path=dummy_csv, years=[2022, 2023, 2024],
                              n_bootstrap=5, seed=42, out_dir=tmp_path)
        assert (tmp_path / "backtest_semi_2022_2024.json").exists()

    def test_markdown_written(self, tmp_path):
        dummy_csv, ctx = self._ctx(tmp_path)
        with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4]:
            run_semi_backtest(data_path=dummy_csv, years=[2022, 2023, 2024],
                              n_bootstrap=5, seed=42, out_dir=tmp_path)
        assert (tmp_path / "backtest_semi_2022_2024.md").exists()

    def test_result_structure(self, tmp_path):
        dummy_csv, ctx = self._ctx(tmp_path)
        with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4]:
            result = run_semi_backtest(data_path=dummy_csv, years=[2022, 2023, 2024],
                                       n_bootstrap=5, seed=42, out_dir=tmp_path)
        assert result["story"] == "US-S6-01b"
        assert set(result["years"].keys()) == {"2022", "2023", "2024"}
        assert "aggregate" in result
        assert set(result["aggregate"].keys()) == {"xgb", "lgbm"}

    def test_aggregate_keys(self, tmp_path):
        dummy_csv, ctx = self._ctx(tmp_path)
        with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4]:
            result = run_semi_backtest(data_path=dummy_csv, years=[2022, 2023, 2024],
                                       n_bootstrap=5, seed=42, out_dir=tmp_path)
        for model_name in ("xgb", "lgbm"):
            agg = result["aggregate"][model_name]
            assert "avg_qual_accuracy_sf1" in agg
            assert "avg_qual_accuracy_sf2" in agg
            assert "avg_qual_accuracy_overall" in agg
            assert "avg_ci80_empirical_coverage" in agg
            assert "all_sf1_kpi_pass" in agg
            assert "all_sf2_kpi_pass" in agg
            assert "all_ci80_kpi_pass" in agg
