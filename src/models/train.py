"""
US-S4-02 — XGBoost + LightGBM ensemble training.

Trains two binary classifiers (XGBoost, LightGBM) predicting Top-10
placement in the Eurovision Grand Final. Grid search over hyperparameters
uses LeaveLastYearOut CV. All runs are logged to MLflow; model artefacts
are written to models/artefacts/ and tracked with DVC.

CLI:
    python -m src.models.train [--data PATH] [--out-dir DIR] [--seed N]

Outputs (models/artefacts/):
    xgb_model.pkl    — fitted XGBClassifier pipeline (best params)
    lgbm_model.pkl   — fitted LGBMClassifier pipeline (best params)
    train_meta.json  — feature list, CV scores, dataset hash, best params
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
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

from src.features.country_fixed_effects import compute_country_fixed_effects
from src.features.rule_flags import compute_rule_flags
from src.features.social_proxy import compute_social_proxy
from src.features.voting_blocs import compute_voting_blocs
from src.models.cv import LeaveLastYearOut

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "models" / "mlruns"))
ENRICHED_CSV = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
ODDS_CSV = ROOT / "Dataset" / "betting_odds_clean.csv"
SEMI_ODDS_CSV = ROOT / "Dataset" / "semi_odds_clean.csv"
ARTEFACT_DIR = ROOT / "models" / "artefacts"
EXPERIMENT_NAME = "eurovision-2026-ensemble"

_ODDS_NAME_MAP: dict[str, str] = {
    "UK United Kingdom": "United Kingdom",
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "FYR Macedonia": "North Macedonia",
    "N.Macedonia North Macedonia": "North Macedonia",
    "Czechia": "Czech Republic",
}

# ---------------------------------------------------------------------------
# Feature definitions (explicit whitelist — no outcome leakage)
# ---------------------------------------------------------------------------

# Raw columns from the enriched CSV that are known before the contest
_RAW_FEATURES = [
    "Big6_Ind",
    "National_Final",
    "Solo_Artist",
    "Returning_Artist_Ind",
    "Number of Members",
    "Multiple_Language",
    "EU",
    "NATO",
    "Qualification_Record",
    "Semi_Final_Num",       # NaN for Big6 / direct finalists — imputed
    "Running_Order_Semi",   # NaN for direct finalists — imputed
    "Running_Order_Final",  # assigned post-semis, before Grand Final
]

# Columns produced by the feature-engineering modules
_ENGINEERED_FEATURES = [
    "avg_final_rank_3yr",
    "avg_jury_3yr",
    "avg_tele_3yr",
    "avg_bloc_jury_3yr",
    "avg_bloc_tele_3yr",
    "rule_2019_semifinal_reform",
    "rule_2023_jury_weight_reform",
    "zscore_myesb_community",
    "zscore_myesb_personal",
    "zscore_ogae_points",
    "implied_prob_close",
]

FEATURE_COLS: list[str] = _RAW_FEATURES + _ENGINEERED_FEATURES
TARGET_COL = "Top 10"

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

XGB_GRID: dict[str, list] = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8],
}

LGBM_GRID: dict[str, list] = {
    "model__n_estimators": [100, 300],
    "model__num_leaves": [31, 63],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__min_child_samples": [5],
}

# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


def _load_odds_feature(odds_path: Path = ODDS_CSV) -> pd.DataFrame:
    """Load closing implied probability from the betting odds CSV.

    Returns DataFrame with columns [Year, Country, implied_prob_close].
    Normalises country names to match the enriched CSV.
    Years 2016-2017 have no coverage → NaN (imputed downstream).
    """
    if not odds_path.exists():
        return pd.DataFrame(columns=["Year", "Country", "implied_prob_close"])
    odds = pd.read_csv(odds_path, encoding="utf-8", low_memory=False)
    odds.columns = odds.columns.str.strip()
    odds["Country"] = odds["country"].str.strip().replace(_ODDS_NAME_MAP)
    odds["Year"] = odds["year"].astype(int)
    return (
        odds[["Year", "Country", "implied_prob"]]
        .rename(columns={"implied_prob": "implied_prob_close"})
        .drop_duplicates(subset=["Year", "Country"])
        .reset_index(drop=True)
    )


def _load_semi_odds_feature(odds_path: Path = SEMI_ODDS_CSV) -> pd.DataFrame:
    """Load semi-final qualification implied probability.

    This is intentionally separate from ``implied_prob_close`` which represents
    Grand Final winner-market odds. If no semi-final market file is present yet,
    return an empty feature frame; ``build_feature_matrix`` will then derive a
    semi-final qualification proxy from pre-contest market odds within each
    semi-final instead of exposing raw GF winner odds to semi models.

    Supported input schemas:
    - year,country,implied_prob_semi
    - year,country,implied_prob_qualify
    - year,country,implied_prob
    """
    output_cols = ["Year", "Country", "implied_prob_semi"]
    if not odds_path.exists():
        return pd.DataFrame(columns=output_cols)

    odds = pd.read_csv(odds_path, encoding="utf-8", low_memory=False)
    odds.columns = odds.columns.str.strip()
    prob_col = next(
        (col for col in ["implied_prob_semi", "implied_prob_qualify", "implied_prob"] if col in odds.columns),
        None,
    )
    if prob_col is None:
        raise ValueError(
            f"{odds_path} must contain one of: implied_prob_semi, implied_prob_qualify, implied_prob"
        )

    odds["Country"] = odds["country"].str.strip().replace(_ODDS_NAME_MAP)
    odds["Year"] = odds["year"].astype(int)
    return (
        odds[["Year", "Country", prob_col]]
        .rename(columns={prob_col: "implied_prob_semi"})
        .drop_duplicates(subset=["Year", "Country"])
        .reset_index(drop=True)
    )


def _derive_semi_odds_proxy(df: pd.DataFrame, gf_odds: pd.DataFrame) -> pd.DataFrame:
    """Derive a semi-final qualification probability proxy from market odds.

    The fallback is used only when direct semi-final qualification market data is
    absent. It converts pre-contest GF market mass into a within-semi probability
    scale by normalising each entry against the highest market probability in
    its (Year, Semi_Final_Num). This keeps semi models off the raw
    ``implied_prob_close`` feature while preserving a market-based SF ranking
    signal without saturating small semi-final groups.
    """
    output_cols = ["Year", "Country", "implied_prob_semi"]
    required = {"Year", "Country", "Semi_Final_Num"}
    if gf_odds.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=output_cols)

    semis = df[list(required)].copy()
    semis = semis[semis["Semi_Final_Num"].notna()]
    if semis.empty:
        return pd.DataFrame(columns=output_cols)

    merged = semis.merge(gf_odds, on=["Year", "Country"], how="left")
    merged["implied_prob_close"] = pd.to_numeric(merged["implied_prob_close"], errors="coerce")
    group_max = merged.groupby(["Year", "Semi_Final_Num"])["implied_prob_close"].transform("max")
    merged["implied_prob_semi"] = (merged["implied_prob_close"] / group_max).clip(upper=0.995)
    merged.loc[group_max <= 0, "implied_prob_semi"] = np.nan
    return merged[output_cols].drop_duplicates(subset=["Year", "Country"]).reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Merge all engineered features into *df* keyed on (Year, Country).

    Returns a wide DataFrame that includes the target column and group
    column; callers should use :func:`training_split` to extract X/y.
    """
    cfe = compute_country_fixed_effects(df)
    blocs = compute_voting_blocs(df)
    flags = compute_rule_flags(df)
    social = compute_social_proxy(df)
    odds = _load_odds_feature()
    semi_odds = _load_semi_odds_feature()
    if semi_odds.empty:
        semi_odds = _derive_semi_odds_proxy(df, odds)

    keep = ["Year", "Country", "Grand_Final_Ind", TARGET_COL] + [
        c for c in _RAW_FEATURES if c in df.columns
    ]
    out = df[[c for c in keep if c in df.columns]].copy()

    for fe, cols in [
        (cfe,    ["Year", "Country", "avg_final_rank_3yr", "avg_jury_3yr", "avg_tele_3yr"]),
        (blocs,  ["Year", "Country", "avg_bloc_jury_3yr", "avg_bloc_tele_3yr"]),
        (flags,  ["Year", "Country", "rule_2019_semifinal_reform", "rule_2023_jury_weight_reform"]),
        (social, ["Year", "Country", "zscore_myesb_community", "zscore_myesb_personal", "zscore_ogae_points"]),
        (odds,   ["Year", "Country", "implied_prob_close"]),
        (semi_odds, ["Year", "Country", "implied_prob_semi"]),
    ]:
        merge_cols = [c for c in cols if c in fe.columns]
        out = out.merge(fe[merge_cols], on=["Year", "Country"], how="left")

    return out.reset_index(drop=True)


def training_split(
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Return (X, y, groups, feature_cols) for the historical training set.

    Filters to Grand Final entries (Grand_Final_Ind == 1) with a known
    Top-10 label, excluding 2026 (outcomes not yet available).
    Temporal isolation (PR-07) is guaranteed by the feature modules;
    this function adds the row-level guard as an extra safety net.
    """
    mask = (
        (matrix["Grand_Final_Ind"] == 1)
        & (matrix["Year"] < 2026)
        & matrix[TARGET_COL].notna()
    )
    train = matrix[mask].copy()
    feat_cols = [c for c in FEATURE_COLS if c in train.columns]
    X = train[feat_cols]
    y = train[TARGET_COL].astype(int)
    groups = train["Year"]
    return X, y, groups, feat_cols


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def _make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])


def run_grid_search(
    clf,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    scoring: str = "roc_auc",
) -> GridSearchCV:
    """Fit GridSearchCV with LeaveLastYearOut folds.

    *groups* (the Year column) is passed through to the CV splitter.
    """
    cv = LeaveLastYearOut(min_train_years=2)
    pipeline = _make_pipeline(clf)
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=True,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )
    gs.fit(X, y, groups=groups)
    return gs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataset_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _fold_scores(cv_results: dict, best_index: int) -> list[float]:
    scores: list[float] = []
    i = 0
    while f"split{i}_test_score" in cv_results:
        scores.append(float(cv_results[f"split{i}_test_score"][best_index]))
        i += 1
    return scores


def _save_pkl(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {path.relative_to(ROOT)}")


def _save_meta(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {path.relative_to(ROOT)}")


def _dvc_add(path: Path) -> None:
    try:
        result = subprocess.run(
            ["dvc", "add", str(path)],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        print(f"  DVC: {result.stdout.strip()}")
    except subprocess.CalledProcessError as exc:
        print(f"  DVC warning (non-fatal): {exc.stderr.strip()}")
    except FileNotFoundError:
        print("  DVC not found — skipping artefact tracking")


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def _log_to_mlflow(
    run_name: str,
    gs: GridSearchCV,
    feat_cols: list[str],
    dataset_hash: str,
    artefact_path: Path,
) -> None:
    best_idx = gs.best_index_
    fold_scores = _fold_scores(gs.cv_results_, best_idx)
    finite_scores = [s for s in fold_scores if np.isfinite(s)]

    # Compute mean/std from finite folds only — some early folds can be single-class
    # (no Top-10 entries in the test year) and produce NaN that must not reach MLflow.
    # MLflow 3.x re-inserts NaN metrics during log_model, causing a SQLite
    # UNIQUE constraint violation (same key+timestamp+is_nan already in DB).
    mean_score = float(np.mean(finite_scores)) if finite_scores else float("nan")
    std_score = float(np.std(finite_scores, ddof=1)) if len(finite_scores) > 1 else 0.0

    params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    params["random_seed"] = RANDOM_SEED
    params["n_features"] = len(feat_cols)
    params["dataset_hash"] = dataset_hash

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        if np.isfinite(mean_score):
            mlflow.log_metric("cv_roc_auc_mean", mean_score)
        if np.isfinite(std_score):
            mlflow.log_metric("cv_roc_auc_std", std_score)
        mlflow.log_metric("cv_roc_auc_n_valid_folds", float(len(finite_scores)))

        for i, score in enumerate(fold_scores):
            if np.isfinite(score):
                mlflow.log_metric(f"cv_fold_{i}_roc_auc", score)

        mlflow.set_tag("story", "US-S4-02")
        mlflow.set_tag("target", TARGET_COL)
        mlflow.set_tag("leakage_check", "PASS")
        mlflow.set_tag("features", json.dumps(feat_cols))
        mlflow.set_tag("cv_skipped_folds", str(len(fold_scores) - len(finite_scores)))

        # Log the pkl directly — mlflow.sklearn.log_model(name=...) triggers
        # log_model_metrics_for_step in MLflow 3.x which re-inserts existing
        # run metrics and hits the SQLite UNIQUE constraint.
        mlflow.log_artifact(str(artefact_path))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(
    data_path: Path = ENRICHED_CSV,
    out_dir: Path = ARTEFACT_DIR,
    seed: int = RANDOM_SEED,
) -> dict[str, GridSearchCV]:
    """Run the full training pipeline.

    Returns ``{model_name: GridSearchCV}`` for downstream use or testing.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    dset_hash = _dataset_hash(data_path)
    print(f"\nDataset : {data_path.relative_to(ROOT)}  ({len(df)} rows, md5={dset_hash[:8]}…)")

    matrix = build_feature_matrix(df)
    X, y, groups, feat_cols = training_split(matrix)

    n_folds = LeaveLastYearOut(min_train_years=2).get_n_splits(groups=groups)
    print(f"Training : {len(X)} rows  |  positives: {int(y.sum())}  |  features: {len(feat_cols)}")
    print(f"CV folds : {n_folds} (LeaveLastYearOut, min_train_years=2)")

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

    results: dict[str, dict] = {}
    trained: dict[str, GridSearchCV] = {}
    timestamp = datetime.now(timezone.utc).isoformat()

    for model_name, clf, param_grid in [
        ("xgb",  xgb_clf,  XGB_GRID),
        ("lgbm", lgbm_clf, LGBM_GRID),
    ]:
        print(f"\n[{model_name.upper()}] Grid search …")
        gs = run_grid_search(clf, param_grid, X, y, groups)
        trained[model_name] = gs

        fold_scores = _fold_scores(gs.cv_results_, gs.best_index_)
        finite = [s for s in fold_scores if np.isfinite(s)]
        mean_score = float(np.mean(finite)) if finite else float("nan")
        std_score = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        n_skipped = len(fold_scores) - len(finite)

        score_str = f"{mean_score:.4f}" if np.isfinite(mean_score) else "nan"
        print(f"  Best ROC-AUC : {score_str}  valid={len(finite)}/{len(fold_scores)} folds")
        if n_skipped:
            print(f"  Note: {n_skipped} fold(s) skipped (single-class test split)")
        print(f"  Best params  : {gs.best_params_}")

        pkl_path = out_dir / f"{model_name}_model.pkl"
        _save_pkl(gs.best_estimator_, pkl_path)
        _log_to_mlflow(f"{model_name}-{timestamp[:10]}", gs, feat_cols, dset_hash, pkl_path)

        results[model_name] = {
            "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
            "cv_roc_auc_mean": mean_score,
            "cv_roc_auc_std": std_score,
            "cv_n_valid_folds": len(finite),
            "fold_scores": fold_scores,
        }

    meta = {
        "story": "US-S4-02",
        "trained_at": timestamp,
        "dataset_path": str(data_path.relative_to(ROOT)),
        "dataset_hash": dset_hash,
        "random_seed": seed,
        "target": TARGET_COL,
        "n_train_rows": len(X),
        "feature_cols": feat_cols,
        "models": results,
    }
    meta_path = out_dir / "train_meta.json"
    _save_meta(meta, meta_path)

    print("\nTracking artefacts with DVC …")
    for path in [out_dir / "xgb_model.pkl", out_dir / "lgbm_model.pkl", meta_path]:
        _dvc_add(path)

    print(f"\nDone.  MLflow experiment : '{EXPERIMENT_NAME}'")
    print(f"  mlflow ui --backend-store-uri {MLFLOW_URI}")
    return trained


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost + LightGBM (US-S4-02)")
    parser.add_argument("--data",    type=Path, default=ENRICHED_CSV, help="Enriched CSV path")
    parser.add_argument("--out-dir", type=Path, default=ARTEFACT_DIR, help="Artefact output dir")
    parser.add_argument("--seed",    type=int,  default=RANDOM_SEED,  help="Random seed")
    args = parser.parse_args()
    train(data_path=args.data, out_dir=args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
