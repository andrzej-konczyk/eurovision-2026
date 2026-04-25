"""Unit tests for LeaveLastYearOut — verifies leakage-free temporal splits."""
import numpy as np
import pandas as pd
import pytest

from src.models.cv import LeaveLastYearOut


YEARS = np.array([2016, 2016, 2017, 2017, 2018, 2019, 2021, 2022])
N = len(YEARS)
X_DUMMY = np.zeros((N, 1))


def _splits(cv=None, groups=YEARS):
    cv = cv or LeaveLastYearOut()
    return list(cv.split(X_DUMMY, groups=groups))


# --- leakage ---


def test_no_year_overlap():
    """Test year in test set must not appear in train set."""
    for train_idx, test_idx in _splits():
        train_years = set(YEARS[train_idx])
        test_years = set(YEARS[test_idx])
        assert train_years.isdisjoint(test_years)


def test_train_years_strictly_before_test():
    """All training years must be strictly less than the test year."""
    for train_idx, test_idx in _splits():
        assert YEARS[train_idx].max() < YEARS[test_idx].min()


# --- split count ---


def test_n_splits_default():
    """Default min_train_years=1 skips the first year, yielding n_unique-1 folds."""
    unique_years = np.unique(YEARS)  # 6 distinct years
    cv = LeaveLastYearOut()
    assert cv.get_n_splits(groups=YEARS) == len(unique_years) - 1


def test_n_splits_matches_iteration():
    cv = LeaveLastYearOut()
    assert cv.get_n_splits(groups=YEARS) == len(_splits(cv))


def test_min_train_years_2():
    """min_train_years=2 should skip both the first and second years."""
    cv = LeaveLastYearOut(min_train_years=2)
    splits = _splits(cv)
    unique_years = np.unique(YEARS)
    assert len(splits) == len(unique_years) - 2
    # first test year must be the third unique year
    first_test_year = YEARS[splits[0][1]].min()
    assert first_test_year == unique_years[2]


# --- coverage ---


def test_all_years_appear_as_test():
    """Every year except the first must appear as a test year exactly once."""
    cv = LeaveLastYearOut()
    test_years = [YEARS[test_idx][0] for _, test_idx in _splits(cv)]
    expected = sorted(np.unique(YEARS))[1:]
    assert test_years == expected


def test_training_set_grows_monotonically():
    """Each successive fold has a strictly larger training set."""
    sizes = [len(tr) for tr, _ in _splits()]
    assert all(a < b for a, b in zip(sizes, sizes[1:]))


# --- edge cases ---


def test_single_year_yields_no_splits():
    groups = np.array([2024, 2024, 2024])
    assert _splits(groups=groups) == []


def test_two_years_yields_one_split():
    groups = np.array([2023, 2023, 2024, 2024])
    splits = _splits(groups=groups)
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert set(groups[train_idx]) == {2023}
    assert set(groups[test_idx]) == {2024}


def test_groups_required():
    with pytest.raises(ValueError, match="groups"):
        list(LeaveLastYearOut().split(X_DUMMY))


def test_get_n_splits_groups_required():
    with pytest.raises(ValueError, match="groups"):
        LeaveLastYearOut().get_n_splits()


def test_groups_as_pandas_series():
    """split() must work when groups is a pandas Series."""
    series = pd.Series(YEARS)
    splits = list(LeaveLastYearOut().split(X_DUMMY, groups=series))
    assert len(splits) == len(np.unique(YEARS)) - 1
    for train_idx, test_idx in splits:
        assert set(YEARS[train_idx]).isdisjoint(set(YEARS[test_idx]))


def test_min_train_years_exceeds_unique_warns():
    """UserWarning when min_train_years > number of unique years."""
    cv = LeaveLastYearOut(min_train_years=99)
    with pytest.warns(UserWarning, match="min_train_years"):
        n = cv.get_n_splits(groups=YEARS)
    assert n == 0
