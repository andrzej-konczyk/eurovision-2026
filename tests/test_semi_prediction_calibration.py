"""Tests for semi-final probability calibration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from scripts.build_semi_predictions_json import (
    apply_probability_calibrator,
    fit_probability_calibrator,
    quota_calibrate_proba_matrix,
)


def test_probability_calibrator_keeps_probabilities_in_unit_interval() -> None:
    X = pd.DataFrame({"signal": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 3})
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1] * 3)
    groups = pd.Series([2019] * 10 + [2021] * 10 + [2022] * 10)
    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="prior")),
    ])
    estimator.fit(X, y)

    calibrator, meta = fit_probability_calibrator(estimator, X, y, groups)
    raw = np.array([[0.01, 0.5, 0.99]])
    calibrated = apply_probability_calibrator(raw, calibrator)

    assert meta["method"] in {"temporal_oof_logistic", "identity"}
    assert calibrated.shape == raw.shape
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


def test_probability_calibrator_shrinks_overconfident_high_values() -> None:
    X = pd.DataFrame({"signal": list(range(40))})
    y = pd.Series([0] * 12 + [1] * 8 + [0] * 12 + [1] * 8)
    groups = pd.Series([2019] * 10 + [2021] * 10 + [2022] * 10 + [2023] * 10)
    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="prior")),
    ])
    estimator.fit(X, y)

    calibrator, meta = fit_probability_calibrator(estimator, X, y, groups)
    if calibrator is None:
        pytest.skip(f"identity fallback: {meta}")

    calibrated = apply_probability_calibrator(np.array([[0.95]]), calibrator)

    assert calibrated[0, 0] < 0.95


def test_quota_calibration_preserves_ranking_and_sums_to_qualifier_quota() -> None:
    raw = np.array([
        [0.99, 0.90, 0.80, 0.70, 0.95, 0.85, 0.75, 0.65],
        [0.60, 0.50, 0.40, 0.30, 0.70, 0.60, 0.50, 0.40],
    ])
    semi_nums = [1, 1, 1, 1, 2, 2, 2, 2]

    calibrated = quota_calibrate_proba_matrix(raw, semi_nums, qualifiers_per_semi=2)

    assert calibrated[:, :4].sum(axis=1) == pytest.approx([2.0, 2.0])
    assert calibrated[:, 4:].sum(axis=1) == pytest.approx([2.0, 2.0])
    assert list(np.argsort(raw[0, :4])[::-1]) == list(np.argsort(calibrated[0, :4])[::-1])
    assert list(np.argsort(raw[0, 4:])[::-1]) == list(np.argsort(calibrated[0, 4:])[::-1])
