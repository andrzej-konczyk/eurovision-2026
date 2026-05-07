"""Unit tests for src.features.market_signals."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.features.market_signals import compute_market_signals, _N_FINALISTS


class TestComputeMarketSignals:
    def _make(self, implied_prob_close, avg_final_rank_3yr):
        return pd.DataFrame(
            {"implied_prob_close": [implied_prob_close], "avg_final_rank_3yr": [avg_final_rank_3yr]}
        )

    def test_returns_dataframe_with_correct_column(self):
        df = self._make(0.20, 5.0)
        result = compute_market_signals(df)
        assert isinstance(result, pd.DataFrame)
        assert "odds_vs_history_delta" in result.columns
        assert len(result) == 1

    def test_formula_positive_delta(self):
        # implied_prob = 0.30, avg_rank = 20 → hist_perf = (27-20)/26 ≈ 0.269
        # delta ≈ 0.30 - 0.269 = 0.031 (market bullish vs weak history)
        df = self._make(0.30, 20.0)
        result = compute_market_signals(df)
        expected = 0.30 - (_N_FINALISTS + 1 - 20.0) / _N_FINALISTS
        assert math.isclose(result["odds_vs_history_delta"].iloc[0], expected, abs_tol=1e-9)

    def test_formula_negative_delta(self):
        # implied_prob = 0.05, avg_rank = 2 → hist_perf = 25/26 ≈ 0.962
        # delta ≈ 0.05 - 0.962 = -0.912 (market sceptical vs strong history)
        df = self._make(0.05, 2.0)
        result = compute_market_signals(df)
        expected = 0.05 - (_N_FINALISTS + 1 - 2.0) / _N_FINALISTS
        assert math.isclose(result["odds_vs_history_delta"].iloc[0], expected, abs_tol=1e-9)

    def test_nan_propagates_from_implied_prob(self):
        df = self._make(float("nan"), 10.0)
        result = compute_market_signals(df)
        assert math.isnan(result["odds_vs_history_delta"].iloc[0])

    def test_nan_propagates_from_avg_rank(self):
        df = self._make(0.15, float("nan"))
        result = compute_market_signals(df)
        assert math.isnan(result["odds_vs_history_delta"].iloc[0])

    def test_missing_column_returns_nan_delta(self):
        df = pd.DataFrame({"implied_prob_close": [0.2]})  # no avg_final_rank_3yr
        result = compute_market_signals(df)
        assert "odds_vs_history_delta" in result.columns
        assert math.isnan(result["odds_vs_history_delta"].iloc[0])

    def test_index_aligned_to_input(self):
        df = pd.DataFrame(
            {"implied_prob_close": [0.1, 0.2, 0.3], "avg_final_rank_3yr": [5.0, 10.0, 15.0]},
            index=[10, 20, 30],
        )
        result = compute_market_signals(df)
        assert list(result.index) == [10, 20, 30]
