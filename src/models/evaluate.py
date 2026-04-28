"""
US-S4-02 — Holdout evaluation: Top-10 accuracy on the 2024 Grand Final.

The `Top 10` column in the enriched CSV is only populated for actual
top-10 entries; positions 11–26 carry NaN. This script derives the binary
target from `Final_Place ≤ 10` instead, giving complete labels for all
finalists with a known placement.

Train window : 2016–2023  (Grand Final entries, Final_Place known)
Holdout      : 2024  (26 finalists; Netherlands excluded — NaN Final_Place)

PRD KPI: ≥ 70% top-10 accuracy (target 80%).

    top10_accuracy = |predicted_top10 ∩ actual_top10| / 10

Usage:
    python -m src.models.evaluate
"""
from __future__ import annotations

import json
import os
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
    build_feature_matrix,
)

load_dotenv(ROOT / ".env")

HOLDOUT_YEAR = 2024
TOP_K = 10
KPI_THRESHOLD = 0.70
TARGET_DERIVED = "top10_derived"


# ---------------------------------------------------------------------------
# Target derivation
# ---------------------------------------------------------------------------


def add_derived_top10(matrix: pd.DataFrame, df_source: pd.DataFrame) -> pd.DataFrame:
    """Merge Final_Place into *matrix* and derive top10_derived = (Final_Place ≤ 10).

    *df_source* is the raw enriched DataFrame; *matrix* is the output of
    build_feature_matrix which does not carry outcome columns.
    """
    fp = (
        df_source[["Year", "Country", "Final_Place"]]
        .drop_duplicates(subset=["Year", "Country"])
    )
    out = matrix.merge(fp, on=["Year", "Country"], how="left")
    out[TARGET_DERIVED] = np.where(
        (out["Grand_Final_Ind"] == 1) & out["Final_Place"].notna(),
        (out["Final_Place"] <= TOP_K).astype(float),
        np.nan,
    )
    return out


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def holdout_split(
    matrix: pd.DataFrame,
    holdout_year: int = HOLDOUT_YEAR,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Return (X_train, y_train, X_test, y_test, test_rows, feat_cols)."""
    feat_cols = [c for c in FEATURE_COLS if c in matrix.columns]

    train_mask = (
        (matrix["Grand_Final_Ind"] == 1)
        & (matrix["Year"] < holdout_year)
        & matrix[TARGET_DERIVED].notna()
    )
    test_mask = (
        (matrix["Grand_Final_Ind"] == 1)
        & (matrix["Year"] == holdout_year)
        & matrix[TARGET_DERIVED].notna()
    )

    train = matrix[train_mask]
    test = matrix[test_mask]

    return (
        train[feat_cols],
        train[TARGET_DERIVED].astype(int),
        test[feat_cols],
        test[TARGET_DERIVED].astype(int),
        test,
        feat_cols,
    )


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def top10_accuracy(y_true: pd.Series, proba_pos: np.ndarray, top_k: int = TOP_K) -> float:
    """Fraction of actual top-k entries that appear in the predicted top-k."""
    arr = np.asarray(y_true)
    pred_top = set(np.argsort(proba_pos)[::-1][:top_k])
    actual_top = set(np.where(arr == 1)[0])
    return len(pred_top & actual_top) / top_k


# ---------------------------------------------------------------------------
# Best-param loading
# ---------------------------------------------------------------------------


def _load_best_params(model_name: str, meta_path: Path) -> dict | None:
    """Return {model__param: value} from train_meta.json, or None if unavailable."""
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = meta["models"][model_name]["best_params"]
        return {f"model__{k}": v for k, v in raw.items()}
    except (KeyError, json.JSONDecodeError):
        return None


def _make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    data_path: Path = ENRICHED_CSV,
    holdout_year: int = HOLDOUT_YEAR,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Train on years < holdout_year, report top-10 accuracy on holdout_year.

    Returns ``{model_name: top10_accuracy}``  (values in [0, 1]).
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    matrix = build_feature_matrix(df)
    matrix = add_derived_top10(matrix, df)

    X_train, y_train, X_test, y_test, test_rows, feat_cols = holdout_split(
        matrix, holdout_year
    )

    n_pos = int(y_test.sum())
    print(f"\nHoldout year : {holdout_year}")
    print(f"Train rows   : {len(X_train)}  (Grand Final, {min(df['Year'])}–{holdout_year - 1})")
    print(f"Test rows    : {len(X_test)} finalists  ({n_pos} actual top-10)")

    if n_pos == 0:
        print(f"  ERROR: no Top-10 labels derived for {holdout_year} — check Final_Place column.")
        return {}

    meta_path = ARTEFACT_DIR / "train_meta.json"
    results: dict[str, float] = {}

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

    for model_name, clf in [("xgb", xgb_clf), ("lgbm", lgbm_clf)]:
        best_params = _load_best_params(model_name, meta_path)

        pipeline = _make_pipeline(clf)
        if best_params:
            pipeline.set_params(**best_params)
            print(f"\n  [{model_name.upper()}] Fitting with params from train_meta.json …")
        else:
            print(f"\n  [{model_name.upper()}] train_meta.json not found — using defaults …")
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        acc = top10_accuracy(y_test, proba)
        hits = int(acc * TOP_K)

        print(f"  Top-10 accuracy : {acc:.0%}  ({hits}/{TOP_K})")

        # Country-level breakdown
        if "Country" in test_rows.columns:
            countries = test_rows["Country"].values
            pred_top = set(countries[np.argsort(proba)[::-1][:TOP_K]])
            actual_top = set(countries[np.where(y_test.values == 1)[0]])
            print(f"  Predicted : {sorted(pred_top)}")
            print(f"  Actual    : {sorted(actual_top)}")
            print(f"  Hits      : {sorted(pred_top & actual_top)}")
            print(f"  Missed    : {sorted(actual_top - pred_top)}")

        kpi = "PASS" if acc >= KPI_THRESHOLD else f"FAIL (need >={KPI_THRESHOLD:.0%})"
        print(f"  KPI       : {kpi}")

        with mlflow.start_run(run_name=f"{model_name}-holdout-{holdout_year}"):
            if best_params:
                mlflow.log_params({k.replace("model__", ""): v for k, v in best_params.items()})
            mlflow.log_param("holdout_year", holdout_year)
            mlflow.log_param("top_k", TOP_K)
            mlflow.log_metric("top10_accuracy", acc)
            mlflow.log_metric("top10_hits", hits)
            mlflow.set_tag("story", "US-S4-02")
            mlflow.set_tag("eval_type", "holdout")
            mlflow.set_tag("kpi_pass", str(acc >= KPI_THRESHOLD))

        results[model_name] = acc

    print(f"\n{'='*50}")
    print(f"PRD KPI threshold : >={KPI_THRESHOLD:.0%}")
    for name, acc in results.items():
        status = "PASS" if acc >= KPI_THRESHOLD else "FAIL"
        print(f"  {name.upper():5s}: {acc:.0%}  ({int(acc * TOP_K)}/{TOP_K})  {status}")

    return results


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
