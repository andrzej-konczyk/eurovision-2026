"""Tests for src/features/rule_flags.py — US-S3-02"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.rule_flags import compute_rule_flags


def _df(years: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"Year": years, "Country": [f"C{y}" for y in years]})


# ── rule_2019_semifinal_reform ────────────────────────────────────────────────

@pytest.mark.parametrize("year", [2016, 2017, 2018])
def test_2019_reform_off_before_2019(year):
    result = compute_rule_flags(_df([year]))
    assert result.loc[0, "rule_2019_semifinal_reform"] == 0


@pytest.mark.parametrize("year", [2019, 2021, 2022, 2023, 2024, 2025, 2026])
def test_2019_reform_on_from_2019(year):
    result = compute_rule_flags(_df([year]))
    assert result.loc[0, "rule_2019_semifinal_reform"] == 1


# ── rule_2023_jury_weight_reform ──────────────────────────────────────────────

@pytest.mark.parametrize("year", [2016, 2017, 2018, 2019, 2021, 2022])
def test_2023_reform_off_before_2023(year):
    result = compute_rule_flags(_df([year]))
    assert result.loc[0, "rule_2023_jury_weight_reform"] == 0


@pytest.mark.parametrize("year", [2023, 2024, 2025, 2026])
def test_2023_reform_on_from_2023(year):
    result = compute_rule_flags(_df([year]))
    assert result.loc[0, "rule_2023_jury_weight_reform"] == 1


# ── output shape & dtypes ─────────────────────────────────────────────────────

def test_output_columns():
    df = _df([2018, 2023])
    result = compute_rule_flags(df)
    assert list(result.columns) == ["Year", "Country",
                                    "rule_2019_semifinal_reform",
                                    "rule_2023_jury_weight_reform"]


def test_flags_are_integer():
    result = compute_rule_flags(_df([2018, 2023]))
    assert result["rule_2019_semifinal_reform"].dtype == int
    assert result["rule_2023_jury_weight_reform"].dtype == int


def test_row_count_preserved():
    df = _df([2016, 2019, 2022, 2023, 2026])
    assert len(compute_rule_flags(df)) == len(df)
