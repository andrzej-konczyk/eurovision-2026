"""
US-S6-01b — Semi-final qualification backtest 2022 / 2023 / 2024.

For each holdout year Y in {2022, 2023, 2024}:
  - Train on semi-final entries (Semi_Final_Num not NaN), years < Y.
  - Target: Grand_Final_Ind  (1 = qualified, 0 = eliminated).
  - Features: FEATURE_COLS minus Running_Order_Final and implied_prob_close,
    plus implied_prob_semi from semi-final qualification markets.
    Final running order and Grand Final winner odds are not known / not
    semantically appropriate while semi-finalists are still competing.
  - Grid search re-run inside each training window (strict temporal isolation).
  - Evaluate per semi-final (SF1 and SF2 separately):
        qual_accuracy = |predicted_top_10 ∩ actual_qualifiers| / 10
  - Bootstrap CI n=1_000 -> ci80_empirical_coverage (same definition as backtest.py).

Observed structure (2016-2024):
  - 10 qualifiers per semi-final, always (K is a constant, not data-driven).
  - 15-19 entrants per semi per year.

PRD KPIs:
  - qual_accuracy >= 0.70 per semi per year.
  - ci80_empirical_coverage >= 0.80 per year.

CLI:
    python -m src.models.backtest_semi [--years 2022 2023 2024]
                                        [--n-bootstrap N] [--seed SEED]

Outputs:
    reports/backtest_semi_2022_2024.json
    reports/backtest_semi_2022_2024.md
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

from src.models.backtest import ci80_empirical_coverage
from src.models.confidence import bootstrap_proba, compute_ci
from src.models.cv import LeaveLastYearOut
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
QUALIFIERS_PER_SEMI = 10          # always 10 in each semi-final
KPI_QUAL_THRESHOLD = 0.70
KPI_CI80_THRESHOLD = 0.80
N_BOOTSTRAP = 1_000
REPORTS_DIR = ROOT / "reports"

# Semi-final feature set: drop Grand Final-only signals and add SF market odds.
SEMI_FEATURE_COLS: list[str] = [
    c for c in FEATURE_COLS
    if c not in {"Running_Order_Final", "implied_prob_close"}
] + ["implied_prob_semi"]
SEMI_TARGET = "Grand_Final_Ind"


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def qualification_accuracy(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    semi_num: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Qualification accuracy per semi-final and overall.

    For each semi (1 and 2): rank entrants by predicted probability,
    take the top-10, count overlap with actual qualifiers.

    Returns
    -------
    dict with keys "sf1", "sf2", "overall" (each a float in [0, 1]).
    """
    y = np.asarray(y_true)
    p = np.asarray(proba)
    s = np.asarray(semi_num)

    results: dict[str, float] = {}
    hits_total = 0
    qual_total = 0

    for sf in (1, 2):
        mask = s == sf
        if mask.sum() == 0:
            results[f"sf{sf}"] = float("nan")
            continue
        y_sf = y[mask]
        p_sf = p[mask]
        k = QUALIFIERS_PER_SEMI
        pred_top = set(np.argsort(p_sf)[::-1][:k])
        actual_top = set(np.where(y_sf == 1)[0])
        hits = len(pred_top & actual_top)
        results[f"sf{sf}"] = hits / k
        hits_total += hits
        qual_total += k

    results["overall"] = hits_total / qual_total if qual_total > 0 else float("nan")
    return results


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def _semi_split(
    matrix: pd.DataFrame,
    holdout_year: int,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X_train, y_train, groups, X_test, y_test, test_rows).

    train : semi-finalists (Semi_Final_Num not NaN), years < holdout_year,
            Grand_Final_Ind not NaN.
    test  : same, year == holdout_year.
    """
    train_mask = (
        matrix["Semi_Final_Num"].notna()
        & (matrix["Year"] < holdout_year)
        & matrix[SEMI_TARGET].notna()
    )
    test_mask = (
        matrix["Semi_Final_Num"].notna()
        & (matrix["Year"] == holdout_year)
        & matrix[SEMI_TARGET].notna()
    )

    train = matrix[train_mask]
    test = matrix[test_mask]

    return (
        train[feat_cols],
        train[SEMI_TARGET].astype(int),
        train["Year"],
        test[feat_cols],
        test[SEMI_TARGET].astype(int),
        test,
    )


# ---------------------------------------------------------------------------
# Pipeline helpers (same pattern as backtest.py)
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


def backtest_semi_year(
    holdout_year: int,
    matrix: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict:
    """Train on semi-final entries from years < holdout_year, evaluate on holdout_year.

    Returns a results dict with qual_accuracy and ci80_empirical_coverage per model.
    """
    feat_cols = [c for c in SEMI_FEATURE_COLS if c in matrix.columns]

    X_train, y_train, groups, X_test, y_test, test_rows = _semi_split(
        matrix, holdout_year, feat_cols
    )

    train_years = sorted(groups.unique().tolist())
    countries = test_rows["Country"].reset_index(drop=True).tolist()
    semi_nums = test_rows["Semi_Final_Num"].reset_index(drop=True).values

    n_sf1 = int((semi_nums == 1).sum())
    n_sf2 = int((semi_nums == 2).sum())
    n_qual = int(y_test.sum())

    print(f"\n{'='*60}")
    print(f"Semi backtest year : {holdout_year}")
    print(f"Train years        : {train_years}")
    print(f"Train rows         : {len(X_train)}  ({int(y_train.sum())} qualifiers)")
    print(f"Test  SF1          : {n_sf1} entries  |  SF2: {n_sf2} entries  |  qualifiers: {n_qual}")

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

    for model_name, clf, param_grid in [
        ("xgb",  xgb_clf,  XGB_GRID),
        ("lgbm", lgbm_clf, LGBM_GRID),
    ]:
        print(f"\n  [{model_name.upper()}] Grid search ...")
        gs = _grid_search(clf, param_grid, X_train, y_train, groups)
        best_pipe = gs.best_estimator_
        print(f"    Best params : {gs.best_params_}")

        # --- Qualification accuracy ---
        proba = best_pipe.predict_proba(X_test)[:, 1]
        acc_dict = qualification_accuracy(y_test.reset_index(drop=True), proba, semi_nums)

        for sf in (1, 2):
            key = f"sf{sf}"
            pct = f"{acc_dict[key]:.0%}" if not np.isnan(acc_dict[key]) else "n/a"
            kpi = "PASS" if acc_dict[key] >= KPI_QUAL_THRESHOLD else "FAIL"
            hits = int(round(acc_dict[key] * QUALIFIERS_PER_SEMI)) if not np.isnan(acc_dict[key]) else "?"
            print(f"    SF{sf} qual acc  : {pct} ({hits}/{QUALIFIERS_PER_SEMI})  KPI: {kpi}")
        print(f"    Overall acc   : {acc_dict['overall']:.0%}")

        # Country-level breakdown per SF
        for sf in (1, 2):
            sf_mask = semi_nums == sf
            if sf_mask.sum() == 0:
                continue
            sf_countries = np.array(countries)[sf_mask]
            sf_proba = proba[sf_mask]
            sf_y = y_test.reset_index(drop=True).values[sf_mask]
            pred_top = set(sf_countries[np.argsort(sf_proba)[::-1][:QUALIFIERS_PER_SEMI]])
            actual_top = set(sf_countries[sf_y == 1])
            print(f"    SF{sf} predicted : {sorted(pred_top)}")
            print(f"    SF{sf} actual    : {sorted(actual_top)}")
            print(f"    SF{sf} hits      : {sorted(pred_top & actual_top)}")

        # --- Bootstrap CI ---
        print(f"    Bootstrap CI n={n_bootstrap} ...")
        best_params_raw = dict(gs.best_params_)
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
        coverage = ci80_empirical_coverage(
            ci_df, y_test.reset_index(drop=True), countries
        )
        kpi_ci = "PASS" if coverage >= KPI_CI80_THRESHOLD else "FAIL"
        print(f"    CI coverage   : {coverage:.0%}  KPI: {kpi_ci}")

        kpi_sf1 = bool(acc_dict["sf1"] >= KPI_QUAL_THRESHOLD) if not np.isnan(acc_dict["sf1"]) else False
        kpi_sf2 = bool(acc_dict["sf2"] >= KPI_QUAL_THRESHOLD) if not np.isnan(acc_dict["sf2"]) else False

        # Per-country detail
        ci_df_out = ci_df.copy()
        y_map = dict(zip(countries, y_test.reset_index(drop=True).tolist()))
        sf_map = dict(zip(countries, semi_nums.tolist()))
        ci_df_out["y_actual"] = ci_df_out["country"].map(y_map)
        ci_df_out["semi_final"] = ci_df_out["country"].map(sf_map)
        ci_df_out["ci80_covered"] = ci_df_out.apply(
            lambda r: (r["y_actual"] == 0 and r["ci80_lo"] < 0.5)
                      or (r["y_actual"] == 1 and r["ci80_hi"] > 0.5),
            axis=1,
        )

        year_result["models"][model_name] = {
            "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
            "qual_accuracy_sf1": float(acc_dict["sf1"]),
            "qual_accuracy_sf2": float(acc_dict["sf2"]),
            "qual_accuracy_overall": float(acc_dict["overall"]),
            "ci80_empirical_coverage": float(coverage),
            "kpi_sf1_pass": kpi_sf1,
            "kpi_sf2_pass": kpi_sf2,
            "kpi_ci80_pass": bool(coverage >= KPI_CI80_THRESHOLD),
            "country_detail": ci_df_out[[
                "country", "semi_final", "y_actual", "prob_mean",
                "ci80_lo", "ci80_hi", "ci80_covered",
            ]].to_dict(orient="records"),
        }

    return year_result


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _build_markdown(year_results: list[dict], timestamp: str) -> str:
    lines = [
        "# Semi-Final Backtest Report — 2022 / 2023 / 2024",
        "",
        f"*Generated: {timestamp[:10]}  |  n_bootstrap = {N_BOOTSTRAP}  |  K = {QUALIFIERS_PER_SEMI} per semi*",
        "",
        "Target: `Grand_Final_Ind` (qualified from semi = 1).  "
        "Features `Running_Order_Final` and `implied_prob_close` excluded. "
        "`implied_prob_semi` is used for semi-final qualification market odds.",
        "",
        "## Qualification Accuracy",
        "",
        "| Year | Model | SF1 | SF2 | Overall | KPI >=70% SF1 | KPI >=70% SF2 |",
        "|------|-------|-----|-----|---------|--------------|--------------|",
    ]
    for yr in year_results:
        for model_name in ("xgb", "lgbm"):
            m = yr["models"][model_name]
            sf1 = m["qual_accuracy_sf1"]
            sf2 = m["qual_accuracy_sf2"]
            ov = m["qual_accuracy_overall"]
            lines.append(
                f"| {yr['year']} | {model_name.upper()} "
                f"| {sf1:.0%} ({int(round(sf1*10))}/{QUALIFIERS_PER_SEMI}) "
                f"| {sf2:.0%} ({int(round(sf2*10))}/{QUALIFIERS_PER_SEMI}) "
                f"| {ov:.0%} "
                f"| {'PASS' if m['kpi_sf1_pass'] else 'FAIL'} "
                f"| {'PASS' if m['kpi_sf2_pass'] else 'FAIL'} |"
            )

    lines += [
        "",
        "## CI Calibration — 80% CI Empirical Coverage",
        "",
        "| Year | XGB | LGBM | KPI >=80% |",
        "|------|-----|------|----------|",
    ]
    for yr in year_results:
        xgb_cov = yr["models"]["xgb"]["ci80_empirical_coverage"]
        lgbm_cov = yr["models"]["lgbm"]["ci80_empirical_coverage"]
        kpi = ("PASS" if yr["models"]["xgb"]["kpi_ci80_pass"]
               and yr["models"]["lgbm"]["kpi_ci80_pass"] else "FAIL")
        lines.append(f"| {yr['year']} | {xgb_cov:.0%} | {lgbm_cov:.0%} | {kpi} |")

    lines += ["", "## Per-Year Country Detail", ""]
    for yr in year_results:
        lines.append(f"### {yr['year']} (train: {yr['train_years']})")
        for model_name in ("xgb", "lgbm"):
            lines.append(f"\n**{model_name.upper()}**\n")
            lines.append(
                "| Country | SF | Actual | Prob | CI80 lo | CI80 hi | CI covered |"
            )
            lines.append("|---------|----|----|------|---------|---------|------------|")
            for row in yr["models"][model_name]["country_detail"]:
                tick_actual = "Q" if row["y_actual"] == 1 else ""
                tick_cov = "v" if row["ci80_covered"] else "x"
                lines.append(
                    f"| {row['country']} | SF{int(row['semi_final'])} "
                    f"| {tick_actual} | {row['prob_mean']:.3f} "
                    f"| {row['ci80_lo']:.3f} | {row['ci80_hi']:.3f} "
                    f"| {tick_cov} |"
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_semi_backtest(
    data_path: Path = ENRICHED_CSV,
    years: list[int] = BACKTEST_YEARS,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    out_dir: Path = REPORTS_DIR,
) -> dict:
    """Run full semi-final backtest for all years; write reports; log to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    matrix = build_feature_matrix(df)

    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"\nUS-S6-01b Semi Backtest  years={years}  n_bootstrap={n_bootstrap}  seed={seed}")
    print(
        "Features: "
        f"{len(SEMI_FEATURE_COLS)} "
        "(FEATURE_COLS minus Running_Order_Final/implied_prob_close, plus implied_prob_semi)"
    )

    year_results: list[dict] = []
    for year in years:
        yr = backtest_semi_year(
            holdout_year=year,
            matrix=matrix,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        year_results.append(yr)

    # Aggregate per model
    aggregate: dict = {}
    for model_name in ("xgb", "lgbm"):
        sf1_accs = [yr["models"][model_name]["qual_accuracy_sf1"] for yr in year_results]
        sf2_accs = [yr["models"][model_name]["qual_accuracy_sf2"] for yr in year_results]
        ovr_accs = [yr["models"][model_name]["qual_accuracy_overall"] for yr in year_results]
        covs = [yr["models"][model_name]["ci80_empirical_coverage"] for yr in year_results]
        aggregate[model_name] = {
            "avg_qual_accuracy_sf1": float(np.nanmean(sf1_accs)),
            "avg_qual_accuracy_sf2": float(np.nanmean(sf2_accs)),
            "avg_qual_accuracy_overall": float(np.nanmean(ovr_accs)),
            "avg_ci80_empirical_coverage": float(np.mean(covs)),
            "all_sf1_kpi_pass": all(yr["models"][model_name]["kpi_sf1_pass"]
                                    for yr in year_results),
            "all_sf2_kpi_pass": all(yr["models"][model_name]["kpi_sf2_pass"]
                                    for yr in year_results),
            "all_ci80_kpi_pass": all(yr["models"][model_name]["kpi_ci80_pass"]
                                     for yr in year_results),
        }

    results = {
        "story": "US-S6-01b",
        "run_at": timestamp,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "backtest_years": years,
        "qualifiers_per_semi": QUALIFIERS_PER_SEMI,
        "kpi_qual_threshold": KPI_QUAL_THRESHOLD,
        "kpi_ci80_threshold": KPI_CI80_THRESHOLD,
        "semi_feature_cols": SEMI_FEATURE_COLS,
        "note_hyperparams": (
            "Grid search re-run inside each year's training window "
            "(strict temporal isolation)."
        ),
        "years": {str(yr["year"]): yr for yr in year_results},
        "aggregate": aggregate,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "backtest_semi_2022_2024.json"
    md_path = out_dir / "backtest_semi_2022_2024.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {json_path.relative_to(ROOT, walk_up=True)}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(year_results, timestamp))
    print(f"Saved: {md_path.relative_to(ROOT, walk_up=True)}")

    # Summary table
    print(f"\n{'='*70}")
    print("SEMI BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"{'Year':<6}  {'Model':<6}  {'SF1':>6}  {'SF2':>6}  {'Ovr':>6}  {'CI-80':>6}  "
          f"{'KPI sf1':>8}  {'KPI sf2':>8}  {'KPI ci80':>9}")
    print("-" * 70)
    for yr in year_results:
        for model_name in ("xgb", "lgbm"):
            m = yr["models"][model_name]
            print(
                f"{yr['year']:<6}  {model_name.upper():<6}  "
                f"{m['qual_accuracy_sf1']:>6.0%}  "
                f"{m['qual_accuracy_sf2']:>6.0%}  "
                f"{m['qual_accuracy_overall']:>6.0%}  "
                f"{m['ci80_empirical_coverage']:>6.0%}  "
                f"{'PASS' if m['kpi_sf1_pass'] else 'FAIL':>8}  "
                f"{'PASS' if m['kpi_sf2_pass'] else 'FAIL':>8}  "
                f"{'PASS' if m['kpi_ci80_pass'] else 'FAIL':>9}"
            )
    print("-" * 70)
    for model_name in ("xgb", "lgbm"):
        agg = aggregate[model_name]
        print(
            f"{'Avg':<6}  {model_name.upper():<6}  "
            f"{agg['avg_qual_accuracy_sf1']:>6.0%}  "
            f"{agg['avg_qual_accuracy_sf2']:>6.0%}  "
            f"{agg['avg_qual_accuracy_overall']:>6.0%}  "
            f"{agg['avg_ci80_empirical_coverage']:>6.0%}  "
            f"{'PASS' if agg['all_sf1_kpi_pass'] else 'FAIL':>8}  "
            f"{'PASS' if agg['all_sf2_kpi_pass'] else 'FAIL':>8}  "
            f"{'PASS' if agg['all_ci80_kpi_pass'] else 'FAIL':>9}"
        )

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
            run_name = f"{model_name}-semi-backtest-{year}-{timestamp[:10]}"
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
                mlflow.log_metric("qual_accuracy_sf1", m["qual_accuracy_sf1"])
                mlflow.log_metric("qual_accuracy_sf2", m["qual_accuracy_sf2"])
                mlflow.log_metric("qual_accuracy_overall", m["qual_accuracy_overall"])
                mlflow.log_metric("ci80_empirical_coverage", m["ci80_empirical_coverage"])
                mlflow.set_tag("story", "US-S6-01b")
                mlflow.set_tag("eval_type", "semi_backtest")
                mlflow.set_tag("kpi_sf1_pass", str(m["kpi_sf1_pass"]))
                mlflow.set_tag("kpi_sf2_pass", str(m["kpi_sf2_pass"]))
                mlflow.set_tag("kpi_ci80_pass", str(m["kpi_ci80_pass"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semi-final qualification backtest 2022/2023/2024 (US-S6-01b)"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=BACKTEST_YEARS,
        help="Holdout years (default: 2022 2023 2024)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED)
    parser.add_argument("--data",        type=Path, default=ENRICHED_CSV)
    args = parser.parse_args()
    run_semi_backtest(
        data_path=args.data,
        years=args.years,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
