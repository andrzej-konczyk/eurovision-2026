"""
Leave-last-year-out cross-validation for Eurovision time-series data.

Usage:
    cv = LeaveLastYearOut()
    for train_idx, test_idx in cv.split(X, groups=df["Year"]):
        ...
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class LeaveLastYearOut(BaseCrossValidator):
    """Time-respecting CV splitter: train on years < Y, test on year Y.

    Iterates over each unique year in ascending order. For year Y the
    training set contains every sample whose year is strictly earlier than Y.
    Folds where the training set would be empty are skipped.

    Parameters
    ----------
    min_train_years : int
        Minimum distinct training years required to include a fold.
        Default 1 skips only the very first (oldest) year.
    """

    def __init__(self, min_train_years: int = 1) -> None:
        self.min_train_years = min_train_years

    # ------------------------------------------------------------------
    # sklearn BaseCrossValidator interface
    # ------------------------------------------------------------------

    def split(self, X, y=None, groups=None):
        """Yield (train_indices, test_indices) for each temporal fold.

        Parameters
        ----------
        X : array-like, shape (n_samples, ...)
        y : ignored
        groups : array-like, shape (n_samples,)
            Contest year for each row. Required.
        """
        if groups is None:
            raise ValueError("groups must be the year array, e.g. df['Year']")

        years = np.asarray(groups)
        unique_years = np.sort(np.unique(years))

        for i, test_year in enumerate(unique_years):
            train_years = unique_years[:i]
            if len(train_years) < self.min_train_years:
                continue
            train_idx = np.where(np.isin(years, train_years))[0]
            test_idx = np.where(years == test_year)[0]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if groups is None:
            raise ValueError("groups must be the year array")
        n_unique = len(np.unique(groups))
        return max(0, n_unique - self.min_train_years)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        raise NotImplementedError("use split() directly")
