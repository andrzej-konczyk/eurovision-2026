"""
US-S6-01 — Backtest 2022 / 2023 / 2024.

For each holdout year Y ∈ {2022, 2023, 2024}:
  - Train XGBoost + LightGBM on Grand Final rows with known Final_Place, years < Y.
  - Grid search is re-run inside the training window (strict temporal isolation).
  - Predict top-10 probabilities for year Y; compute Top-10 accuracy.
  - Bootstrap CI (n=1_000) for year Y; compute ci80_empirical_coverage.

CI calibration definition
    ci80_empirical_coverage = fraction of holdout countries where the 80% CI is
    consistent with the actual binary outcome:
        y = 0  →  covered iff ci80_lo < 0.5  (CI doesn't confidently predict top-10)
        y = 1  →  covered iff ci80_hi > 0.5  (CI doesn't confidently rule out top-10)
    PRD KPI: ci80_empirical_coverage ≥ 0.80 per year.

Top-10 accuracy KPI: ≥ 0.70 per year.

CLI:
    python -m src.models.backtest [--years 2022 2023 2024]
                                   [--n-bootstrap N] [--seed SEED]

Outputs:
    reports/backtest_2022_2024.json  — per-year metrics + country detail
    reports/backtest_2022_2024.md    — human-readable summary table
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.models.confidence import bootstrap_proba, compute_ci
from src.models.cv import LeaveLastYearOut
from src.models.evaluate import add_derived_top10, top10_accuracy
from src.models.train import (
    ENRICHED_CSV,
    EXPERIMENT_NAME,
    FEATURE_COLS,
    LGBM_GRID,
    MLFLOW_URI,
    RANDOM_SEED,
    ROOT,
    XGB_GRID,
    build_feature_matrix,
)

load_dotenv(ROOT / ".env")

BACKTEST_YEARS: list[int] = [2022, 2023, 2024]
TOP_K = 10
KPI_TOP10_THRESHOLD = 0.70
KPI_CI80_THRESHOLD = 0.80
N_BOOTSTRAP = 1_000
REPORTS_DIR = ROOT / "reports"

_TARGET = "top10_derived"


# ---------------------------------------------------------------------------
# CI calibration
# ---------------------------------------------------------------------------


def ci80_empirical_coverage(
    ci_df: pd.DataFrame,
    y_test: pd.Series,
    countries: list[str],
) -> float:
    """Fraction of countries where the 80% CI is consistent with the actual outcome.

    For y=0 (not top-10): consistent iff ci80_lo < 0.5.
    For y=1 (top-10):     consistent iff ci80_hi > 0.5.
    """
    ci_indexed = ci_df.set_index("country")
    covered = 0
    n = 0
    for country, y in zip(countries, y_test.values):
        if country not in ci_indexed.index:
            continue
        row = ci_indexed.loc[country]
        n += 1
        if y == 0 and row["ci80_lo"] < 0.5:
            covered += 1
        elif y == 1 and row["ci80_hi"] > 0.5:
            covered += 1
    return covered / n if n > 0 else float("nan")


# ---------------------------------------------------------------------------
# Train / test split (backtest version — strict year < holdout)
# ---------------------------------------------------------------------------


def _backtest_split(
    matrix: pd.DataFrame,
    holdout_year: int,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X_train, y_train, groups, X_test, y_test, test_rows).

    train: Grand Final rows, years strictly < holdout_year, derived label not NaN.
    test:  Grand Final rows, year == holdout_year, derived label not NaN.
    """
    train_mask = (
        (matrix["Grand_Final_Ind"] == 1)
        & (matrix["Year"] < holdout_year)
        & matrix[_TARGET].notna()
    )
    test_mask = (
        (matrix["Grand_Final_Ind"] == 1)
        & (matrix["Year"] == holdout_year)
        & matrix[_TARGET].notna()
    )

    train = matrix[train_mask]
    test = matrix[test_mask]

    return (
        train[feat_cols],
        train[_TARGET].astype(int),
        train["Year"],
        test[feat_cols],
        test[_TARGET].astype(int),
        test,
    )


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])


def _grid_search(
    clf,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
) -> GridSearchCV:
    cv = LeaveLastYearOut(min_train_years=2)
    gs = GridSearchCV(
        _make_pipeline(clf),
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        refit=True,
        n_jobs=-1,
        verbose=0,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X_train, y_train, groups=groups)
    return gs


# ---------------------------------------------------------------------------
# Per-year backtest
# ---------------------------------------------------------------------------


def backtest_year(
    holdout_year: int,
    matrix: pd.DataFrame,
    df_source: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict:
    """Train on years < holdout_year, evaluate on holdout_year.

    Returns a dict with top10_accuracy, ci80_empirical_coverage, and
    per-country detail for each model (xgb, lgbm).
    """
    matrix_y = add_derived_top10(matrix, df_source)
    feat_cols = [c for c in FEATURE_COLS if c in matrix_y.columns]

    X_train, y_train, groups, X_test, y_test, test_rows = _backtest_split(
        matrix_y, holdout_year, feat_cols
    )

    train_years = sorted(groups.unique().tolist())
    countries = test_rows["Country"].reset_index(drop=True).tolist()

    print(f"\n{'='*60}")
    print(f"Backtest year : {holdout_year}")
    print(f"Train years   : {train_years}")
    print(f"Train rows    : {len(X_train)}  ({int(y_train.sum())} top-10 positives)")
    print(f"Test rows     : {len(X_test)}  ({int(y_test.sum())} actual top-10)")

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
    )
    lgbm_clf = lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        verbose=-1,
    )

    year_result: dict = {
        "year": holdout_year,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_years": train_years,
        "models": {},
    }

    model_defs = [
        ("xgb",  xgb_clf,  XGB_GRID),
        ("lgbm", lgbm_clf, LGBM_GRID),
    ]

    for model_name, clf, param_grid in model_defs:
        print(f"\n  [{model_name.upper()}] Grid search …")
        gs = _grid_search(clf, param_grid, X_train, y_train, groups)
        best_pipe = gs.best_estimator_
        print(f"    Best params : {gs.best_params_}")

        # --- Top-10 accuracy ---
        proba = best_pipe.predict_proba(X_test)[:, 1]
        acc = top10_accuracy(y_test, proba, top_k=TOP_K)
        hits = int(round(acc * TOP_K))
        print(f"    Top-10 acc  : {acc:.0%}  ({hits}/{TOP_K})")

        # Country breakdown
        pred_top = set(np.array(countries)[np.argsort(proba)[::-1][:TOP_K]])
        actual_top = set(np.array(countries)[np.where(y_test.values == 1)[0]])
        print(f"    Predicted   : {sorted(pred_top)}")
        print(f"    Actual      : {sorted(actual_top)}")
        print(f"    Hits        : {sorted(pred_top & actual_top)}")

        # --- Bootstrap CI ---
        print(f"    Bootstrap CI n={n_bootstrap} …")
        best_params_raw = {k: v for k, v in gs.best_params_.items()}
        proba_matrix = bootstrap_proba(
            model_name=model_name,
            best_params=best_params_raw,
            X_train=X_train,
            y_train=y_train,
            X_target=X_test.reset_index(drop=True),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        ci_df = compute_ci(proba_matrix, countries)

        # --- CI calibration ---
        coverage = ci80_empirical_coverage(ci_df, y_test.reset_index(drop=True), countries)
        print(f"    CI coverage : {coverage:.0%}  (KPI >={KPI_CI80_THRESHOLD:.0%})")

        kpi_top10 = acc >= KPI_TOP10_THRESHOLD
        kpi_ci80 = coverage >= KPI_CI80_THRESHOLD
        print(f"    KPI top-10  : {'PASS' if kpi_top10 else 'FAIL'}")
        print(f"    KPI ci80    : {'PASS' if kpi_ci80 else 'FAIL'}")

        # Per-country detail
        ci_df_merged = ci_df.copy()
        ci_df_merged["y_actual"] = ci_df_merged["country"].map(
            dict(zip(countries, y_test.reset_index(drop=True).tolist()))
        )
        ci_df_merged["in_predicted_top10"] = ci_df_merged["country"].isin(pred_top)
        ci_df_merged["in_actual_top10"] = ci_df_merged["country"].isin(actual_top)
        ci_df_merged["ci80_covered"] = ci_df_merged.apply(
            lambda r: (r["y_actual"] == 0 and r["ci80_lo"] < 0.5)
                      or (r["y_actual"] == 1 and r["ci80_hi"] > 0.5),
            axis=1,
        )

        year_result["models"][model_name] = {
            "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
            "top10_accuracy": float(acc),
            "top10_hits": hits,
            "ci80_empirical_coverage": float(coverage),
            "kpi_top10_pass": bool(kpi_top10),
            "kpi_ci80_pass": bool(kpi_ci80),
            "country_detail": ci_df_merged[[
                "country", "y_actual", "prob_mean",
                "ci80_lo", "ci80_hi", "ci50_lo", "ci50_hi",
                "in_predicted_top10", "in_actual_top10", "ci80_covered",
            ]].to_dict(orient="records"),
        }

    return year_result


# ---------------------------------------------------------------------------
# Aggregate + report
# ---------------------------------------------------------------------------


def _build_markdown(year_results: list[dict], timestamp: str) -> str:
    lines = [
        "# Backtest Report — 2022 / 2023 / 2024",
        "",
        f"*Generated: {timestamp[:10]}  |  n_bootstrap = {N_BOOTSTRAP}*",
        "",
        "## Top-10 Accuracy",
        "",
        "| Year | XGB | LGBM | KPI ≥ 70% |",
        "|------|-----|------|-----------|",
    ]
    for yr in year_results:
        xgb_acc = yr["models"]["xgb"]["top10_accuracy"]
        lgbm_acc = yr["models"]["lgbm"]["top10_accuracy"]
        kpi = ("PASS" if yr["models"]["xgb"]["kpi_top10_pass"]
               or yr["models"]["lgbm"]["kpi_top10_pass"] else "FAIL")
        lines.append(
            f"| {yr['year']} | {xgb_acc:.0%} ({yr['models']['xgb']['top10_hits']}/{TOP_K}) "
            f"| {lgbm_acc:.0%} ({yr['models']['lgbm']['top10_hits']}/{TOP_K}) | {kpi} |"
        )

    lines += [
        "",
        "## CI Calibration — 80% CI Empirical Coverage",
        "",
        "Coverage = fraction of countries where 80% CI is consistent with actual outcome.",
        "y=0: covered if ci80_lo < 0.5  |  y=1: covered if ci80_hi > 0.5",
        "",
        "| Year | XGB | LGBM | KPI ≥ 80% |",
        "|------|-----|------|-----------|",
    ]
    for yr in year_results:
        xgb_cov = yr["models"]["xgb"]["ci80_empirical_coverage"]
        lgbm_cov = yr["models"]["lgbm"]["ci80_empirical_coverage"]
        kpi = ("PASS" if yr["models"]["xgb"]["kpi_ci80_pass"]
               and yr["models"]["lgbm"]["kpi_ci80_pass"] else "FAIL")
        lines.append(
            f"| {yr['year']} | {xgb_cov:.0%} | {lgbm_cov:.0%} | {kpi} |"
        )

    lines += ["", "## Per-Year Country Detail", ""]
    for yr in year_results:
        lines.append(f"### {yr['year']} (train: {yr['train_years']})")
        for model_name in ("xgb", "lgbm"):
            lines.append(f"\n**{model_name.upper()}**\n")
            lines.append(
                "| Country | Actual | Prob | CI80 lo | CI80 hi | Predicted top-10 | CI covered |"
            )
            lines.append("|---------|--------|------|---------|---------|-----------------|------------|")
            for row in yr["models"][model_name]["country_detail"]:
                tick_actual = "✓" if row["in_actual_top10"] else ""
                tick_pred = "✓" if row["in_predicted_top10"] else ""
                tick_cov = "✓" if row["ci80_covered"] else "✗"
                lines.append(
                    f"| {row['country']} | {tick_actual} | {row['prob_mean']:.3f} "
                    f"| {row['ci80_lo']:.3f} | {row['ci80_hi']:.3f} "
                    f"| {tick_pred} | {tick_cov} |"
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_backtest(
    data_path: Path = ENRICHED_CSV,
    years: list[int] = BACKTEST_YEARS,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    out_dir: Path = REPORTS_DIR,
) -> dict:
    """Run full backtest for all years; write reports; log to MLflow.

    Returns the complete results dict (also written to JSON).
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    matrix = build_feature_matrix(df)

    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"\nUS-S6-01 Backtest  years={years}  n_bootstrap={n_bootstrap}  seed={seed}")

    year_results: list[dict] = []
    for year in years:
        yr = backtest_year(
            holdout_year=year,
            matrix=matrix,
            df_source=df,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        year_results.append(yr)

    # Aggregate per model across years
    aggregate: dict = {}
    for model_name in ("xgb", "lgbm"):
        accs = [yr["models"][model_name]["top10_accuracy"] for yr in year_results]
        covs = [yr["models"][model_name]["ci80_empirical_coverage"] for yr in year_results]
        aggregate[model_name] = {
            "avg_top10_accuracy": float(np.mean(accs)),
            "avg_ci80_empirical_coverage": float(np.mean(covs)),
            "all_top10_kpi_pass": all(yr["models"][model_name]["kpi_top10_pass"]
                                      for yr in year_results),
            "all_ci80_kpi_pass": all(yr["models"][model_name]["kpi_ci80_pass"]
                                     for yr in year_results),
        }

    results = {
        "story": "US-S6-01",
        "run_at": timestamp,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "backtest_years": years,
        "kpi_top10_threshold": KPI_TOP10_THRESHOLD,
        "kpi_ci80_threshold": KPI_CI80_THRESHOLD,
        "note_hyperparams": (
            "Grid search re-run inside each year's training window "
            "(strict temporal isolation — no leakage from holdout year)."
        ),
        "years": {str(yr["year"]): yr for yr in year_results},
        "aggregate": aggregate,
    }

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "backtest_2022_2024.json"
    md_path = out_dir / "backtest_2022_2024.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {json_path.relative_to(ROOT, walk_up=True)}")

    md_content = _build_markdown(year_results, timestamp)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved: {md_path.relative_to(ROOT, walk_up=True)}")

    # Summary
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"{'Year':<6}  {'Model':<6}  {'Top-10':>7}  {'CI-80':>7}  {'KPI top10':>10}  {'KPI ci80':>9}")
    print("-" * 55)
    for yr in year_results:
        for model_name in ("xgb", "lgbm"):
            m = yr["models"][model_name]
            print(
                f"{yr['year']:<6}  {model_name.upper():<6}  "
                f"{m['top10_accuracy']:>7.0%}  "
                f"{m['ci80_empirical_coverage']:>7.0%}  "
                f"{'PASS' if m['kpi_top10_pass'] else 'FAIL':>10}  "
                f"{'PASS' if m['kpi_ci80_pass'] else 'FAIL':>9}"
            )
    print("-" * 55)
    for model_name in ("xgb", "lgbm"):
        agg = aggregate[model_name]
        print(
            f"{'Avg':<6}  {model_name.upper():<6}  "
            f"{agg['avg_top10_accuracy']:>7.0%}  "
            f"{agg['avg_ci80_empirical_coverage']:>7.0%}  "
            f"{'PASS' if agg['all_top10_kpi_pass'] else 'FAIL':>10}  "
            f"{'PASS' if agg['all_ci80_kpi_pass'] else 'FAIL':>9}"
        )

    # MLflow — one run per year per model
    _log_to_mlflow(year_results, timestamp, n_bootstrap, seed)

    return results


def _log_to_mlflow(
    year_results: list[dict],
    timestamp: str,
    n_bootstrap: int,
    seed: int,
) -> None:
    for yr in year_results:
        year = yr["year"]
        for model_name in ("xgb", "lgbm"):
            m = yr["models"][model_name]
            run_name = f"{model_name}-backtest-{year}-{timestamp[:10]}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "model": model_name,
                    "holdout_year": year,
                    "n_train": yr["n_train"],
                    "n_test": yr["n_test"],
                    "n_bootstrap": n_bootstrap,
                    "seed": seed,
                    **{k: v for k, v in m["best_params"].items()},
                })
                mlflow.log_metric("top10_accuracy", m["top10_accuracy"])
                mlflow.log_metric("top10_hits", m["top10_hits"])
                mlflow.log_metric("ci80_empirical_coverage", m["ci80_empirical_coverage"])
                mlflow.set_tag("story", "US-S6-01")
                mlflow.set_tag("eval_type", "backtest")
                mlflow.set_tag("kpi_top10_pass", str(m["kpi_top10_pass"]))
                mlflow.set_tag("kpi_ci80_pass", str(m["kpi_ci80_pass"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest 2022/2023/2024 (US-S6-01)")
    parser.add_argument(
        "--years", nargs="+", type=int, default=BACKTEST_YEARS,
        help="Holdout years (default: 2022 2023 2024)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED)
    parser.add_argument("--data",        type=Path, default=ENRICHED_CSV)
    args = parser.parse_args()
    run_backtest(
        data_path=args.data,
        years=args.years,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
