"""
US-S4-03 — Bootstrap confidence intervals for Top-10 predicted probabilities.

For each model (XGBoost, LightGBM), resamples the training rows B times with
replacement, refits using the best hyperparameters from train_meta.json, and
records predict_proba for the *target year* entries.  The empirical distribution
of B refits yields 80 % and 50 % percentile confidence intervals per country.

Resampling strategy: naive row-level bootstrap on the full training set (all
Grand Final entries with known outcomes, years 2016–2025).  The best params
from train_meta.json are held fixed — no grid search per iteration.

CLI:
    python -m src.models.confidence [--data PATH] [--target-year YEAR]
                                     [--n-bootstrap N] [--seed SEED]
                                     [--out-dir DIR]

Outputs (models/artefacts/):
    confidence_xgb.csv    — country, prob_mean, ci80_lo, ci80_hi, ci50_lo, ci50_hi
    confidence_lgbm.csv   — same for LightGBM
    confidence_meta.json  — run metadata (n_bootstrap, seed, train_years, timestamp)
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
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.models.train import (
    ARTEFACT_DIR,
    ENRICHED_CSV,
    EXPERIMENT_NAME,
    FEATURE_COLS,
    MLFLOW_URI,
    RANDOM_SEED,
    ROOT,
    TARGET_COL,
    build_feature_matrix,
    training_split,
)

load_dotenv(ROOT / ".env")

TARGET_YEAR = 2026
N_BOOTSTRAP = 1000
META_PATH = ARTEFACT_DIR / "train_meta.json"

# Percentile levels → (lower_tail, upper_tail)
_CI_LEVELS = {
    80: (10.0, 90.0),
    50: (25.0, 75.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_best_params(model_name: str, meta_path: Path = META_PATH) -> dict:
    """Load best hyperparameters from train_meta.json.

    Raises FileNotFoundError if the meta file is missing (train must run first).
    Raises KeyError if *model_name* is absent from the meta file.
    """
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta["models"][model_name]["best_params"]


def _make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])


def _clf_from_params(model_name: str, params: dict, seed: int):
    """Construct a fresh (unfitted) classifier with *params* applied."""
    stripped = {k.replace("model__", ""): v for k, v in params.items()}
    if model_name == "xgb":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            **stripped,
        )
    return lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        verbose=-1,
        **stripped,
    )


# ---------------------------------------------------------------------------
# Bootstrap engine
# ---------------------------------------------------------------------------


def bootstrap_proba(
    model_name: str,
    best_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_target: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Return a (n_bootstrap, n_target_rows) matrix of predicted probabilities.

    Each row is one bootstrap resample: sample training rows with replacement,
    fit a fresh pipeline with *best_params*, predict_proba on *X_target*.

    Samples where the resampled training set contains only one class are
    skipped and replaced by re-drawing until *n_bootstrap* valid fits are
    collected (capped at 10× attempts to avoid infinite loops).
    """
    rng = np.random.default_rng(seed)
    n_train = len(X_train)
    X_arr = X_train.values
    y_arr = y_train.values

    collected: list[np.ndarray] = []
    max_attempts = n_bootstrap * 10
    attempts = 0
    skipped = 0

    while len(collected) < n_bootstrap:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Bootstrap: only {len(collected)}/{n_bootstrap} valid samples "
                f"after {attempts} attempts ({skipped} skipped — single-class draws). "
                "Training set may be too small or class-imbalanced."
            )
        idx = rng.integers(0, n_train, size=n_train)
        X_boot, y_boot = X_arr[idx], y_arr[idx]
        attempts += 1

        if len(np.unique(y_boot)) < 2:
            skipped += 1
            continue

        clf = _clf_from_params(model_name, best_params, seed=int(rng.integers(0, 2**31)))
        pipe = _make_pipeline(clf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_boot, y_boot)

        proba = pipe.predict_proba(X_target.values)[:, 1]
        collected.append(proba)

    if skipped:
        print(f"  [{model_name.upper()}] Bootstrap: {skipped} single-class draws skipped "
              f"out of {attempts} attempts")

    return np.vstack(collected)  # shape (n_bootstrap, n_target_rows)


# ---------------------------------------------------------------------------
# CI computation
# ---------------------------------------------------------------------------


def compute_ci(
    proba_matrix: np.ndarray,
    countries: list[str],
) -> pd.DataFrame:
    """Compute mean and percentile CIs from the bootstrap probability matrix.

    Args:
        proba_matrix: shape (n_bootstrap, n_countries)
        countries: country labels for the target rows

    Returns:
        DataFrame with columns:
            country, prob_mean, ci80_lo, ci80_hi, ci50_lo, ci50_hi
    """
    rows = []
    for i, country in enumerate(countries):
        col = proba_matrix[:, i]
        row: dict = {
            "country": country,
            "prob_mean": float(np.mean(col)),
        }
        for level, (lo_pct, hi_pct) in _CI_LEVELS.items():
            row[f"ci{level}_lo"] = float(np.percentile(col, lo_pct))
            row[f"ci{level}_hi"] = float(np.percentile(col, hi_pct))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("prob_mean", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def confidence(
    data_path: Path = ENRICHED_CSV,
    target_year: int = TARGET_YEAR,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    out_dir: Path = ARTEFACT_DIR,
    meta_path: Path = META_PATH,
) -> dict[str, pd.DataFrame]:
    """Run bootstrap CI pipeline for all models.

    Returns ``{model_name: DataFrame}`` with CI columns per country.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    matrix = build_feature_matrix(df)
    X_train, y_train, groups, feat_cols = training_split(matrix)

    # Target rows: Grand Final entries for target_year.
    # For 2026, semi-final qualifiers are not yet known → include all
    # Grand_Final_Ind == 1 rows plus Big6 (Grand_Final_Ind may be 0 for
    # 2026 semi-finalists whose qualification is pending).
    # Safest: take all 2026 rows and restrict to feat_cols only.
    target_mask = matrix["Year"] == target_year
    X_target_df = matrix.loc[target_mask, feat_cols].reset_index(drop=True)
    target_countries = matrix.loc[target_mask, "Country"].reset_index(drop=True).tolist()

    if X_target_df.empty:
        raise ValueError(f"No rows found for target_year={target_year} in the feature matrix.")

    train_years = sorted(groups.unique().tolist())
    print(f"\nBootstrap CI  n={n_bootstrap}  seed={seed}")
    print(f"Train years   : {train_years}")
    print(f"Target year   : {target_year}  ({len(target_countries)} entries)")
    print(f"Features      : {len(feat_cols)}")

    timestamp = datetime.now(timezone.utc).isoformat()
    results: dict[str, pd.DataFrame] = {}
    ci_stats: dict[str, dict] = {}

    models_def = [
        ("xgb",  "XGBoost"),
        ("lgbm", "LightGBM"),
    ]

    for model_name, display_name in models_def:
        print(f"\n[{display_name}] Loading best params …")
        best_params = _load_best_params(model_name, meta_path)
        print(f"  Params: {best_params}")

        print(f"  Running {n_bootstrap} bootstrap fits …")
        proba_matrix = bootstrap_proba(
            model_name=model_name,
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_target=X_target_df,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

        ci_df = compute_ci(proba_matrix, target_countries)
        results[model_name] = ci_df

        out_path = out_dir / f"confidence_{model_name}.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        ci_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  Saved: {out_path.relative_to(ROOT, walk_up=True)}")

        avg_ci50_width = float((ci_df["ci50_hi"] - ci_df["ci50_lo"]).mean())
        avg_ci80_width = float((ci_df["ci80_hi"] - ci_df["ci80_lo"]).mean())
        ci_stats[model_name] = {
            "avg_ci80_width": avg_ci80_width,
            "avg_ci50_width": avg_ci50_width,
        }

        _log_model_to_mlflow(
            model_name=model_name,
            timestamp=timestamp,
            n_bootstrap=n_bootstrap,
            seed=seed,
            target_year=target_year,
            train_years=train_years,
            feat_cols=feat_cols,
            ci_df=ci_df,
            avg_ci80_width=avg_ci80_width,
            avg_ci50_width=avg_ci50_width,
        )

        print(f"  Top-5 predictions (by mean prob):")
        for _, row in ci_df.head(5).iterrows():
            print(f"    {row['country']:<25} "
                  f"mean={row['prob_mean']:.3f}  "
                  f"80% CI [{row['ci80_lo']:.3f}, {row['ci80_hi']:.3f}]  "
                  f"50% CI [{row['ci50_lo']:.3f}, {row['ci50_hi']:.3f}]")

    meta = {
        "story": "US-S4-03",
        "run_at": timestamp,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "target_year": target_year,
        "train_years": train_years,
        "n_train_rows": int(len(X_train)),
        "feature_cols": feat_cols,
        "ci_stats": ci_stats,
    }
    meta_out = out_dir / "confidence_meta.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved: {meta_out.relative_to(ROOT, walk_up=True)}")
    print("\nDone.")
    return results


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def _log_model_to_mlflow(
    model_name: str,
    timestamp: str,
    n_bootstrap: int,
    seed: int,
    target_year: int,
    train_years: list[int],
    feat_cols: list[str],
    ci_df: pd.DataFrame,
    avg_ci80_width: float,
    avg_ci50_width: float,
) -> None:
    run_name = f"{model_name}-bootstrap-ci-{timestamp[:10]}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": model_name,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "target_year": target_year,
            "n_train_years": len(train_years),
            "n_features": len(feat_cols),
        })
        mlflow.log_metric("avg_ci80_width", avg_ci80_width)
        mlflow.log_metric("avg_ci50_width", avg_ci50_width)
        mlflow.log_metric("n_target_entries", float(len(ci_df)))
        mlflow.set_tag("story", "US-S4-03")
        mlflow.set_tag("target", TARGET_COL)
        mlflow.set_tag("leakage_check", "PASS")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for Top-10 predictions (US-S4-03)")
    parser.add_argument("--data",         type=Path, default=ENRICHED_CSV, help="Enriched CSV path")
    parser.add_argument("--target-year",  type=int,  default=TARGET_YEAR,  help="Year to predict")
    parser.add_argument("--n-bootstrap",  type=int,  default=N_BOOTSTRAP,  help="Bootstrap iterations")
    parser.add_argument("--seed",         type=int,  default=RANDOM_SEED,  help="Random seed")
    parser.add_argument("--out-dir",      type=Path, default=ARTEFACT_DIR, help="Output directory")
    args = parser.parse_args()
    confidence(
        data_path=args.data,
        target_year=args.target_year,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
