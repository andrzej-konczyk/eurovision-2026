"""
US-S4-05 — Leakage Audit.

Runs programmatic checks across all pipeline components to verify temporal
isolation (PR-07): no information from year Y or later appears in the features
or training labels for any row from year Y.

Checks
------
LA-01  FEATURE_COLS whitelist excludes all outcome columns.
LA-02  training_split() temporal filter: Year < 2026, Top-10 label known.
LA-03  LeaveLastYearOut CV: max train year < test year in every fold.
LA-04  Country fixed effects: lookback uses Year < current Year.
LA-05  Voting blocs: lookback uses Year < current Year.
LA-06  Holdout split (evaluate.py): train Year < 2024, no index overlap.
LA-07  Feature matrix X returned by training_split has no outcome columns.
LA-08  Social proxy: per-year z-score mean ≈ 0 (within-year normalisation).

CLI:
    python -m src.models.leakage_audit [--data PATH]

Exit code: 0 if all pass, 1 if any fail.
MLflow tag: leakage_check_passed = "true" | "false"
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.models.cv import LeaveLastYearOut
from src.models.train import (
    ARTEFACT_DIR,
    ENRICHED_CSV,
    EXPERIMENT_NAME,
    FEATURE_COLS,
    MLFLOW_URI,
    ROOT,
    build_feature_matrix,
    training_split,
)

load_dotenv(ROOT / ".env")

HOLDOUT_YEAR = 2024  # mirrors evaluate.py

# Columns that encode contest outcomes and must never appear as features.
_OUTCOME_COLS: frozenset[str] = frozenset({
    "Top 10",
    "Final_Place",
    "jury_points",
    "tele_points",
    "Final_Points",
    "Semi_Points",
    "Semi_Place",
})


@dataclass
class CheckResult:
    id: str
    name: str
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_la01_feature_whitelist() -> CheckResult:
    """LA-01: FEATURE_COLS contains no outcome columns."""
    leaked = set(FEATURE_COLS) & _OUTCOME_COLS
    return CheckResult(
        id="LA-01",
        name="Feature whitelist excludes outcome columns",
        passed=not leaked,
        detail=(
            f"Leaking: {sorted(leaked)}" if leaked
            else f"OK — {len(FEATURE_COLS)} features, 0 outcome columns"
        ),
    )


def check_la02_training_split(df: pd.DataFrame) -> CheckResult:
    """LA-02: training_split() temporal filter and label completeness."""
    matrix = build_feature_matrix(df)
    X, y, groups, _ = training_split(matrix)

    issues: list[str] = []
    future = sorted(int(yr) for yr in groups[groups >= 2026].unique())
    if future:
        issues.append(f"Year ≥ 2026 in training set: {future}")
    if y.isna().any():
        issues.append(f"{int(y.isna().sum())} NaN labels in y")

    return CheckResult(
        id="LA-02",
        name="training_split temporal filter and label completeness",
        passed=not issues,
        detail=(
            "; ".join(issues) if issues
            else f"OK — {len(X)} rows, years {sorted(int(yr) for yr in groups.unique())}"
        ),
    )


def check_la03_cv_splitter(df: pd.DataFrame) -> CheckResult:
    """LA-03: LeaveLastYearOut — no future data in any training fold."""
    matrix = build_feature_matrix(df)
    X, y, groups, _ = training_split(matrix)
    cv = LeaveLastYearOut()

    issues: list[str] = []
    n_folds = 0
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, groups=groups)):
        n_folds += 1
        train_yrs = groups.iloc[train_idx].unique()
        test_yrs = groups.iloc[test_idx].unique()
        if len(test_yrs) != 1:
            issues.append(f"Fold {fold_i}: expected 1 test year, got {test_yrs.tolist()}")
            continue
        test_yr = test_yrs[0]
        if train_yrs.max() >= test_yr:
            issues.append(
                f"Fold {fold_i}: train max year {int(train_yrs.max())} >= "
                f"test year {int(test_yr)}"
            )
        if set(train_idx) & set(test_idx):
            issues.append(f"Fold {fold_i}: index overlap between train and test")

    return CheckResult(
        id="LA-03",
        name="LeaveLastYearOut: no future data in any CV fold",
        passed=not issues,
        detail=(
            "; ".join(issues) if issues
            else f"OK — {n_folds} folds verified"
        ),
    )


def check_la04_country_fixed_effects(df: pd.DataFrame) -> CheckResult:
    """LA-04: Country fixed effects lookback is strictly prior-year only."""
    from src.features.country_fixed_effects import compute_country_fixed_effects

    cfe = compute_country_fixed_effects(df)
    min_year = int(df["Year"].min())
    earliest = cfe[cfe["Year"] == min_year]

    lookback_cols = [c for c in ["avg_final_rank_3yr", "avg_jury_3yr", "avg_tele_3yr"]
                     if c in earliest.columns]
    non_nan = earliest[lookback_cols].notna().any(axis=None)

    return CheckResult(
        id="LA-04",
        name="Country fixed effects: Year < current year lookback guard",
        passed=not non_nan,
        detail=(
            f"Year {min_year} has non-NaN lookback CFE features — "
            "suggests same-year data was used" if non_nan
            else f"OK — earliest year ({min_year}) CFE lookbacks are all NaN"
        ),
    )


def check_la05_voting_blocs(df: pd.DataFrame) -> CheckResult:
    """LA-05: Voting blocs lookback is strictly prior-year only."""
    from src.features.voting_blocs import compute_voting_blocs

    blocs = compute_voting_blocs(df)
    min_year = int(df["Year"].min())
    earliest = blocs[blocs["Year"] == min_year]

    lookback_cols = [c for c in ["avg_bloc_jury_3yr", "avg_bloc_tele_3yr"]
                     if c in earliest.columns]
    non_nan = earliest[lookback_cols].notna().any(axis=None)

    return CheckResult(
        id="LA-05",
        name="Voting blocs: Year < current year lookback guard",
        passed=not non_nan,
        detail=(
            f"Year {min_year} has non-NaN lookback bloc features — "
            "suggests same-year data was used" if non_nan
            else f"OK — earliest year ({min_year}) bloc lookbacks are all NaN"
        ),
    )


def check_la06_holdout_split(df: pd.DataFrame) -> CheckResult:
    """LA-06: Holdout split (evaluate.py) — train < 2024, no index overlap."""
    from src.models.evaluate import add_derived_top10, holdout_split

    matrix = build_feature_matrix(df)
    matrix_ext = add_derived_top10(matrix, df)

    if not (matrix_ext["Year"] == HOLDOUT_YEAR).any():
        return CheckResult(
            id="LA-06",
            name=f"Holdout split (train < {HOLDOUT_YEAR}, test == {HOLDOUT_YEAR})",
            passed=True,
            detail=f"SKIPPED — no {HOLDOUT_YEAR} rows present in this dataset",
        )

    try:
        X_tr, _, X_te, _, _, _ = holdout_split(matrix_ext, holdout_year=HOLDOUT_YEAR)
    except Exception as exc:
        return CheckResult(
            id="LA-06",
            name=f"Holdout split (train < {HOLDOUT_YEAR}, test == {HOLDOUT_YEAR})",
            passed=False,
            detail=f"holdout_split() raised: {exc}",
        )

    issues: list[str] = []
    overlap = set(X_tr.index) & set(X_te.index)
    if overlap:
        issues.append(f"{len(overlap)} shared indices between train and test")

    train_years = matrix_ext.loc[X_tr.index, "Year"].unique()
    bad_years = sorted(int(yr) for yr in train_years[train_years >= HOLDOUT_YEAR])
    if bad_years:
        issues.append(f"Train contains Year ≥ {HOLDOUT_YEAR}: {bad_years}")

    return CheckResult(
        id="LA-06",
        name=f"Holdout split (train < {HOLDOUT_YEAR}, test == {HOLDOUT_YEAR})",
        passed=not issues,
        detail=(
            "; ".join(issues) if issues
            else f"OK — {len(X_tr)} train rows, {len(X_te)} test rows, 0 shared indices"
        ),
    )


def check_la07_feature_matrix_columns(df: pd.DataFrame) -> CheckResult:
    """LA-07: Feature matrix X (from training_split) has no outcome columns."""
    matrix = build_feature_matrix(df)
    X, _, _, _ = training_split(matrix)

    leaked = set(X.columns) & _OUTCOME_COLS
    return CheckResult(
        id="LA-07",
        name="Feature matrix X has no outcome columns",
        passed=not leaked,
        detail=(
            f"Leaking: {sorted(leaked)}" if leaked
            else f"OK — {X.shape[1]} feature columns, none are outcomes"
        ),
    )


def check_la08_social_proxy(df: pd.DataFrame) -> CheckResult:
    """LA-08: Social proxy z-scores are within-year (per-year mean ≈ 0)."""
    from src.features.social_proxy import compute_social_proxy

    social = compute_social_proxy(df)

    # MyESB columns have no missing values → per-year mean must be ≈ 0
    # (cross-year normalisation would shift per-year means significantly)
    check_cols = [c for c in ["zscore_myesb_community", "zscore_myesb_personal"]
                  if c in social.columns]

    issues: list[str] = []
    for col in check_cols:
        bad = social.groupby("Year")[col].mean()
        bad = bad[bad.abs() > 1e-9]
        if not bad.empty:
            issues.append(
                f"{col}: per-year mean not ≈ 0 in years {bad.index.tolist()} "
                f"(max |mean| = {bad.abs().max():.2e})"
            )

    return CheckResult(
        id="LA-08",
        name="Social proxy: per-year z-score mean ≈ 0 (within-year normalisation)",
        passed=not issues,
        detail=(
            "; ".join(issues) if issues
            else f"OK — {len(check_cols)} MyESB z-score columns have per-year mean ≈ 0"
        ),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all_checks(df: pd.DataFrame) -> list[CheckResult]:
    """Run all 8 leakage checks and return results."""
    return [
        check_la01_feature_whitelist(),
        check_la02_training_split(df),
        check_la03_cv_splitter(df),
        check_la04_country_fixed_effects(df),
        check_la05_voting_blocs(df),
        check_la06_holdout_split(df),
        check_la07_feature_matrix_columns(df),
        check_la08_social_proxy(df),
    ]


def _print_results(results: list[CheckResult]) -> None:
    print("\n" + "=" * 72)
    print("LEAKAGE AUDIT — Results")
    print("=" * 72)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.id}  {r.name}")
        print(f"         {r.detail}")
    n_pass = sum(r.passed for r in results)
    print("=" * 72)
    print(f"  {n_pass}/{len(results)} checks passed")
    print("=" * 72)


def leakage_audit(
    data_path: Path = ENRICHED_CSV,
) -> list[CheckResult]:
    """Load data, run all checks, log to MLflow, return results."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    print(f"\nLeakage Audit  data={data_path.name}  rows={len(df)}")
    results = run_all_checks(df)
    _print_results(results)

    all_passed = all(r.passed for r in results)
    timestamp = datetime.now(timezone.utc).isoformat()

    with mlflow.start_run(run_name=f"leakage-audit-{timestamp[:10]}"):
        mlflow.log_params({
            "data": str(data_path.name),
            "n_checks": len(results),
            "n_passed": sum(r.passed for r in results),
        })
        for r in results:
            mlflow.log_metric(r.id.lower().replace("-", "_"), float(r.passed))
        mlflow.set_tag("story", "US-S4-05")
        mlflow.set_tag("leakage_check_passed", str(all_passed).lower())
        mlflow.set_tag("audit_timestamp", timestamp)

    if all_passed:
        print("\nAll checks PASSED. Pipeline is temporally isolated (PR-07).")
    else:
        failed = [r.id for r in results if not r.passed]
        print(f"\nFAILED: {failed}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage Audit (US-S4-05)")
    parser.add_argument("--data", type=Path, default=ENRICHED_CSV,
                        help="Enriched CSV path")
    args = parser.parse_args()
    results = leakage_audit(data_path=args.data)
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
