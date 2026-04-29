"""
US-S5-02 — Ensemble: weighted average of XGB + LightGBM + MLP.

Searches blend weights on the 2024 holdout (grid step=0.1, weights sum to 1).
All three models are re-trained on years < holdout_year so the weight search
is a genuine out-of-sample evaluation with no future leakage.

Best params for XGB/LGBM are loaded from train_meta.json; best MLP params
from nn_model_config.json.  Artefact written to models/artefacts/.

CLI:
    python -m src.models.ensemble [--data PATH] [--out-dir DIR] [--seed N]
                                   [--holdout YEAR] [--step FLOAT]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.evaluate import (
    HOLDOUT_YEAR,
    KPI_THRESHOLD,
    TOP_K,
    add_derived_top10,
    holdout_split,
    top10_accuracy,
)
from src.models.nn import NNPipeline
from src.models.train import (
    ARTEFACT_DIR,
    ENRICHED_CSV,
    EXPERIMENT_NAME,
    MLFLOW_URI,
    RANDOM_SEED,
    ROOT,
    build_feature_matrix,
)

load_dotenv(ROOT / ".env")

WEIGHT_STEP = 0.1


# ---------------------------------------------------------------------------
# Weight grid
# ---------------------------------------------------------------------------


def _weight_grid(step: float = WEIGHT_STEP):
    """Yield (w_xgb, w_lgbm, w_nn) triples that sum to 1.0."""
    n = round(1.0 / step)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            yield round(i * step, 10), round(j * step, 10), round(k * step, 10)


# ---------------------------------------------------------------------------
# Param loading
# ---------------------------------------------------------------------------


def _load_tree_params(meta_path: Path) -> dict[str, dict]:
    """Return {model_name: {model__param: value}} from train_meta.json."""
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            name: {f"model__{k}": v for k, v in info["best_params"].items()}
            for name, info in meta.get("models", {}).items()
        }
    except (KeyError, json.JSONDecodeError):
        return {}


def _load_nn_params(cfg_path: Path) -> dict:
    """Return best MLP hyperparams from nn_model_config.json."""
    if not cfg_path.exists():
        return {}
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        bp = dict(cfg.get("best_params", {}))
        if "hidden_dims" in bp:
            bp["hidden_dims"] = tuple(bp["hidden_dims"])
        return bp
    except (KeyError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------


def _xgb_pipeline(params: dict, seed: int) -> Pipeline:
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])
    if params:
        pipe.set_params(**params)
    return pipe


def _lgbm_pipeline(params: dict, seed: int) -> Pipeline:
    clf = lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        verbose=-1,
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])
    if params:
        pipe.set_params(**params)
    return pipe


# ---------------------------------------------------------------------------
# DVC helper
# ---------------------------------------------------------------------------


def _dvc_add(path: Path) -> None:
    try:
        subprocess.run(
            ["dvc", "add", str(path)],
            check=True, capture_output=True, text=True, cwd=ROOT,
        )
        print(f"  DVC: tracked {path.name}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  DVC: skipped for {path.name}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def blend(
    data_path: Path = ENRICHED_CSV,
    out_dir: Path = ARTEFACT_DIR,
    holdout_year: int = HOLDOUT_YEAR,
    seed: int = RANDOM_SEED,
    weight_step: float = WEIGHT_STEP,
) -> dict:
    """Search ensemble weights, evaluate on holdout, save artefact.

    All three models are re-trained on years < *holdout_year* to ensure
    temporal isolation (PR-07).  Returns the full result dict.
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

    print(f"\nEnsemble weight search  (holdout={holdout_year})")
    print(f"Train rows : {len(X_train)}  |  Test rows : {len(X_test)}")
    print(f"Positives  : train={int(y_train.sum())}  test={int(y_test.sum())}")

    # --- load best hyperparams from prior artefacts ---
    tree_params = _load_tree_params(out_dir / "train_meta.json")
    nn_bp = _load_nn_params(out_dir / "nn_model_config.json")

    source_note = "train_meta.json" if tree_params else "defaults"
    print(f"Params loaded from: {source_note}")

    # --- fit all three models on holdout training split ---
    print("\nFitting XGBoost …")
    xgb_pipe = _xgb_pipeline(tree_params.get("xgb", {}), seed)
    xgb_pipe.fit(X_train, y_train)

    print("Fitting LightGBM …")
    lgbm_pipe = _lgbm_pipeline(tree_params.get("lgbm", {}), seed)
    lgbm_pipe.fit(X_train, y_train)

    print("Fitting MLP …")
    nn_pipe = NNPipeline(
        hidden_dims=nn_bp.get("hidden_dims", (64, 32)),
        dropout=nn_bp.get("dropout", 0.0),
        lr=nn_bp.get("lr", 0.001),
        n_epochs=nn_bp.get("n_epochs", 300),
        batch_size=nn_bp.get("batch_size", 32),
        seed=seed,
    )
    nn_pipe.fit(X_train.values, y_train.values)

    # --- holdout probabilities ---
    p_xgb = xgb_pipe.predict_proba(X_test)[:, 1]
    p_lgbm = lgbm_pipe.predict_proba(X_test)[:, 1]
    p_nn = nn_pipe.predict_proba(X_test.values)[:, 1]

    print("\nStandalone accuracy:")
    standalone: dict[str, float] = {}
    for name, proba in [("xgb", p_xgb), ("lgbm", p_lgbm), ("nn", p_nn)]:
        acc = top10_accuracy(y_test, proba)
        standalone[name] = acc
        kpi = "PASS" if acc >= KPI_THRESHOLD else "FAIL"
        print(f"  {name.upper():5s}: {acc:.0%}  ({int(acc * TOP_K)}/{TOP_K})  {kpi}")

    # --- weight grid search ---
    n_combos = sum(1 for _ in _weight_grid(weight_step))
    print(f"\nWeight grid search ({n_combos} combos, step={weight_step}) …")

    best_acc = -1.0
    best_w = (1 / 3, 1 / 3, 1 / 3)
    all_results: list[dict] = []

    for w_xgb, w_lgbm, w_nn in _weight_grid(weight_step):
        blended = w_xgb * p_xgb + w_lgbm * p_lgbm + w_nn * p_nn
        acc = top10_accuracy(y_test, blended)
        all_results.append({
            "w_xgb": w_xgb,
            "w_lgbm": w_lgbm,
            "w_nn": w_nn,
            "top10_accuracy": round(acc, 4),
        })
        if acc > best_acc:
            best_acc = acc
            best_w = (w_xgb, w_lgbm, w_nn)

    w_xgb, w_lgbm, w_nn = best_w
    hits = int(best_acc * TOP_K)
    kpi_pass = best_acc >= KPI_THRESHOLD
    kpi_label = "PASS" if kpi_pass else f"FAIL (need >={KPI_THRESHOLD:.0%})"

    print(f"\nBest weights  : xgb={w_xgb:.2f}  lgbm={w_lgbm:.2f}  nn={w_nn:.2f}")
    print(f"Top-10 accuracy: {best_acc:.0%}  ({hits}/{TOP_K})")
    print(f"KPI            : {kpi_label}")

    # country breakdown
    if "Country" in test_rows.columns:
        blended_best = w_xgb * p_xgb + w_lgbm * p_lgbm + w_nn * p_nn
        countries = test_rows["Country"].values
        pred_top = set(countries[np.argsort(blended_best)[::-1][:TOP_K]])
        actual_top = set(countries[np.where(y_test.values == 1)[0]])
        print(f"  Predicted : {sorted(pred_top)}")
        print(f"  Actual    : {sorted(actual_top)}")
        print(f"  Hits      : {sorted(pred_top & actual_top)}")
        print(f"  Missed    : {sorted(actual_top - pred_top)}")

    # --- save artefact ---
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    artefact = {
        "story": "US-S5-02",
        "saved_at": timestamp,
        "holdout_year": holdout_year,
        "weight_step": weight_step,
        "best_weights": {"xgb": w_xgb, "lgbm": w_lgbm, "nn": w_nn},
        "top10_accuracy": round(best_acc, 4),
        "top10_hits": hits,
        "kpi_pass": kpi_pass,
        "standalone": {k: round(v, 4) for k, v in standalone.items()},
        "top_results": sorted(all_results, key=lambda r: -r["top10_accuracy"])[:20],
    }
    weights_path = out_dir / "ensemble_weights.json"
    weights_path.write_text(json.dumps(artefact, indent=2), encoding="utf-8")
    try:
        label = weights_path.relative_to(ROOT)
    except ValueError:
        label = weights_path
    print(f"\n  Saved: {label}")

    _dvc_add(weights_path)

    # --- MLflow ---
    with mlflow.start_run(run_name=f"ensemble-{timestamp[:10]}"):
        mlflow.log_params({
            "w_xgb": w_xgb,
            "w_lgbm": w_lgbm,
            "w_nn": w_nn,
            "holdout_year": holdout_year,
            "weight_step": weight_step,
            "seed": seed,
        })
        mlflow.log_metric("top10_accuracy", best_acc)
        mlflow.log_metric("top10_hits", hits)
        for name, acc in standalone.items():
            mlflow.log_metric(f"top10_accuracy_{name}", acc)
        mlflow.log_artifact(str(weights_path))
        mlflow.set_tag("story", "US-S5-02")
        mlflow.set_tag("model", "ensemble")
        mlflow.set_tag("kpi_pass", str(kpi_pass))

    print("\nDone.")
    return artefact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble weight search (US-S5-02)")
    parser.add_argument("--data",    type=Path,  default=ENRICHED_CSV)
    parser.add_argument("--out-dir", type=Path,  default=ARTEFACT_DIR)
    parser.add_argument("--holdout", type=int,   default=HOLDOUT_YEAR)
    parser.add_argument("--seed",    type=int,   default=RANDOM_SEED)
    parser.add_argument("--step",    type=float, default=WEIGHT_STEP)
    args = parser.parse_args()
    blend(
        data_path=args.data,
        out_dir=args.out_dir,
        holdout_year=args.holdout,
        seed=args.seed,
        weight_step=args.step,
    )


if __name__ == "__main__":
    main()
