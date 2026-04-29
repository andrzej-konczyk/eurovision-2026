"""
US-S5-05 — Linear surrogate model for the ensemble.

Trains a polynomial Ridge regression surrogate via knowledge distillation:
  - Impute → StandardScale → degree-2 interactions → StandardScale → RidgeCV
  - Fitted to reproduce the ensemble's in-sample soft probabilities

With 23 raw features, degree-2 interaction terms give 276 total features,
allowing the surrogate to capture cross-feature effects (e.g. jury history ×
community enthusiasm) that the full LGBM ensemble exploits non-linearly.

Key acceptance criteria:

  1. Mean absolute rank delta vs best-blend ensemble < 2.0 positions
     on the 2026 inference set.  Achieved on this dataset: ~5–7 positions.
     The gap is structural: for 2026, implied_prob_close + Running_Order_Final
     + OGAE_Points are ALL constant (NaN → median imputation), so ranking is
     determined by jury/tele history + social scores whose interaction the linear
     surrogate approximates but cannot fully replicate.  Documented as KL-07.
  2. Inference time on 2026 set < 2.0 seconds.  ✓ (< 2 ms)

Artefacts produced:
    models/artefacts/surrogate_model.pkl
    reports/surrogate_YYYY.json

CLI:
    python -m src.models.surrogate [--target-year YEAR] [--data PATH]
                                   [--out-dir DIR] [--reports-dir DIR]
                                   [--seed INT] [--max-delta FLOAT]
                                   [--max-inference-s FLOAT]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.models.shap_pipeline import load_pipeline
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

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
REPORTS_DIR = ROOT / "reports"
TARGET_YEAR = 2026
ENSEMBLE_MODEL = "lgbm"   # best blend is lgbm=1.0

MAX_RANK_DELTA = 2.0      # acceptance gate (structural limitation: see KL-07)
MAX_INFERENCE_S = 2.0     # acceptance gate: wall-clock seconds

RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ---------------------------------------------------------------------------
# Predict-proba wrapper for Ridge (regression output → pseudo-probabilities)
# ---------------------------------------------------------------------------


class DistilledSurrogate:
    """Wraps a Ridge regression pipeline with a predict_proba interface.

    The Ridge output is clipped to [0, 1] and treated as the positive-class
    probability.  Ranking quality (Spearman / rank_delta) is the primary
    evaluation metric; calibration is not guaranteed.
    """

    def __init__(self, pipe: Pipeline) -> None:
        self._pipe = pipe

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = np.clip(self._pipe.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def make_ridge_pipeline() -> Pipeline:
    """Return an unfitted pipeline: impute → scale → poly(2) interactions → scale → RidgeCV.

    23 raw features + C(23,2)=253 interaction pairs = 276 total features.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler",  StandardScaler()),
        ("poly",    PolynomialFeatures(degree=2, interaction_only=True,
                                       include_bias=False)),
        ("scaler2", StandardScaler()),
        ("model",   RidgeCV(alphas=RIDGE_ALPHAS, scoring="neg_mean_squared_error")),
    ])


# ---------------------------------------------------------------------------
# Distillation training
# ---------------------------------------------------------------------------


def train_surrogate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int = RANDOM_SEED,
    ensemble: object | None = None,
) -> tuple[DistilledSurrogate, dict]:
    """Train Ridge surrogate via in-sample distillation.

    Fits the surrogate to reproduce *ensemble*'s soft probability outputs on
    X_train so the surrogate learns the ensemble's decision surface.
    Falls back to binary labels *y* when *ensemble* is None (for unit tests).
    """
    if ensemble is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            soft_targets = ensemble.predict_proba(X)[:, 1]
        distillation = "in-sample"
    else:
        soft_targets = y.astype(float).values
        distillation = "labels"

    pipe = make_ridge_pipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, soft_targets)

    best_alpha = float(pipe.named_steps["model"].alpha_)
    train_pred = np.clip(pipe.predict(X), 0.0, 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho_result = spearmanr(soft_targets, train_pred)
    rho = float(rho_result.statistic if hasattr(rho_result, "statistic") else rho_result[0])
    if np.isnan(rho):
        rho = 0.0

    cv_info = {
        "best_alpha": best_alpha,
        "spearman_r_insample": round(rho, 4),
        "distillation": distillation,
    }
    return DistilledSurrogate(pipe), cv_info


# ---------------------------------------------------------------------------
# Rank-delta evaluation
# ---------------------------------------------------------------------------


def rank_delta(
    ensemble_proba: np.ndarray,
    surrogate_proba: np.ndarray,
    countries: list[str],
) -> dict:
    """Compare rankings produced by *ensemble_proba* and *surrogate_proba*.

    Returns dict with mean_abs_delta, max_abs_delta, and per-country rows.
    Ranks are 1-based (1 = highest probability).
    """
    ens_rank  = pd.Series(ensemble_proba,  index=countries).rank(ascending=False, method="min")
    surr_rank = pd.Series(surrogate_proba, index=countries).rank(ascending=False, method="min")
    diff = (ens_rank - surr_rank).abs()

    rows = [
        {
            "country":        c,
            "ensemble_rank":  int(ens_rank[c]),
            "surrogate_rank": int(surr_rank[c]),
            "rank_delta":     int(diff[c]),
            "ensemble_prob":  round(float(ensemble_proba[i]), 4),
            "surrogate_prob": round(float(surrogate_proba[i]), 4),
        }
        for i, c in enumerate(countries)
    ]
    rows.sort(key=lambda r: r["ensemble_rank"])

    return {
        "mean_abs_delta": round(float(diff.mean()), 4),
        "max_abs_delta":  int(diff.max()),
        "countries": rows,
    }


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------


def time_inference(
    pipeline: DistilledSurrogate,
    X: pd.DataFrame,
    n_reps: int = 10,
) -> float:
    """Return median wall-clock seconds for predict_proba over *n_reps* runs."""
    times = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_reps):
            t0 = time.perf_counter()
            pipeline.predict_proba(X)
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    data_path: Path = ENRICHED_CSV,
    target_year: int = TARGET_YEAR,
    out_dir: Path = ARTEFACT_DIR,
    reports_dir: Path = REPORTS_DIR,
    seed: int = RANDOM_SEED,
    max_delta: float = MAX_RANK_DELTA,
    max_inference_s: float = MAX_INFERENCE_S,
    ensemble_model: str = ENSEMBLE_MODEL,
) -> dict:
    """Train distilled surrogate, evaluate KPIs, save artefact and JSON report."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    matrix = build_feature_matrix(df)

    X, y, groups, feat_cols = training_split(matrix)

    print(f"\nSurrogate  target_year={target_year}  seed={seed}")
    print(f"Training rows : {len(X)}  features : {len(feat_cols)}")

    # --- load ensemble for distillation targets ---
    ensemble = load_pipeline(ensemble_model, out_dir)

    # --- train surrogate ---
    surrogate, cv_info = train_surrogate(X, y, groups, seed, ensemble=ensemble)
    print(f"Distillation  : {cv_info['distillation']}"
          f"  Ridge alpha={cv_info['best_alpha']}"
          f"  Spearman r (in-sample)={cv_info['spearman_r_insample']:.4f}")

    # --- 2026 inference ---
    target_mask = matrix["Year"] == target_year
    if not target_mask.any():
        raise ValueError(f"No rows for Year={target_year} in {data_path}.")
    X_target  = matrix.loc[target_mask, feat_cols].reset_index(drop=True)
    countries = matrix.loc[target_mask, "Country"].reset_index(drop=True).tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ens_proba  = ensemble.predict_proba(X_target)[:, 1]
        surr_proba = surrogate.predict_proba(X_target)[:, 1]

    # --- rank delta ---
    delta_info = rank_delta(ens_proba, surr_proba, countries)
    mean_delta = delta_info["mean_abs_delta"]
    kpi_delta  = mean_delta < max_delta

    # --- inference timing ---
    inf_s   = time_inference(surrogate, X_target)
    kpi_inf = inf_s < max_inference_s

    print(f"Rank delta    : mean={mean_delta:.2f}  max={delta_info['max_abs_delta']}"
          f"  KPI (<{max_delta}): {'PASS' if kpi_delta else 'FAIL (see KL-07)'}")
    print(f"Inference     : {inf_s*1000:.1f} ms"
          f"  KPI (<{max_inference_s}s): {'PASS' if kpi_inf else 'FAIL'}")

    print("\nTop-10 comparison (by ensemble rank):")
    for row in delta_info["countries"][:10]:
        marker = "!" if row["rank_delta"] >= 2 else " "
        print(f"  {marker} {row['ensemble_rank']:2d}. {row['country']:<22}"
              f"  ens={row['ensemble_prob']:.3f} surr={row['surrogate_prob']:.3f}"
              f"  delta={row['rank_delta']}")

    # --- save artefact ---
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "surrogate_model.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(surrogate, fh)
    _pkl_label = pkl_path.relative_to(ROOT) if pkl_path.is_relative_to(ROOT) else pkl_path
    print(f"  Saved: {_pkl_label}")

    # --- save JSON report ---
    timestamp = datetime.now(timezone.utc).isoformat()
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "story": "US-S5-05",
        "generated_at": timestamp,
        "target_year": target_year,
        "ensemble_model": ensemble_model,
        "surrogate_type": "RidgeCV poly-2 (distilled)",
        "distillation": cv_info["distillation"],
        "best_alpha": cv_info["best_alpha"],
        "spearman_r_insample": cv_info["spearman_r_insample"],
        "mean_rank_delta": mean_delta,
        "max_rank_delta": delta_info["max_abs_delta"],
        "inference_time_s": round(inf_s, 6),
        "kpi_delta_pass":        kpi_delta,
        "kpi_inference_pass":    kpi_inf,
        "kpi_delta_note": (
            "Structural limitation (KL-07): implied_prob_close, Running_Order_Final "
            "and OGAE_Points are NaN (constant) for all 2026 entries; ranking is "
            "determined by jury/tele/social features whose non-linear tree interactions "
            "cannot be replicated by a linear model."
        ) if not kpi_delta else None,
        "max_delta_threshold":   max_delta,
        "max_inference_threshold": max_inference_s,
        "countries": delta_info["countries"],
    }
    json_path = reports_dir / f"surrogate_{target_year}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _json_label = json_path.relative_to(ROOT) if json_path.is_relative_to(ROOT) else json_path
    print(f"  Saved: {_json_label}")

    # --- MLflow ---
    with mlflow.start_run(run_name=f"surrogate-{target_year}-{timestamp[:10]}"):
        mlflow.log_params({
            "target_year":    target_year,
            "ensemble_model": ensemble_model,
            "distillation":   cv_info["distillation"],
            "best_alpha":     cv_info["best_alpha"],
            "seed":           seed,
        })
        mlflow.log_metrics({
            "spearman_r_insample": cv_info["spearman_r_insample"],
            "mean_rank_delta":     mean_delta,
            "max_rank_delta":      float(delta_info["max_abs_delta"]),
            "inference_time_s":    inf_s,
        })
        mlflow.log_artifact(str(json_path))
        mlflow.set_tag("story",              "US-S5-05")
        mlflow.set_tag("kpi_delta_pass",     str(kpi_delta))
        mlflow.set_tag("kpi_inference_pass", str(kpi_inf))

    print("\nDone.")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear surrogate model (US-S5-05)")
    parser.add_argument("--data",            type=Path,  default=ENRICHED_CSV)
    parser.add_argument("--target-year",     type=int,   default=TARGET_YEAR)
    parser.add_argument("--out-dir",         type=Path,  default=ARTEFACT_DIR)
    parser.add_argument("--reports-dir",     type=Path,  default=REPORTS_DIR)
    parser.add_argument("--seed",            type=int,   default=RANDOM_SEED)
    parser.add_argument("--max-delta",       type=float, default=MAX_RANK_DELTA,
                        help="Acceptance gate: mean absolute rank delta")
    parser.add_argument("--max-inference-s", type=float, default=MAX_INFERENCE_S,
                        help="Acceptance gate: inference wall-clock seconds")
    args = parser.parse_args()
    run(
        data_path=args.data,
        target_year=args.target_year,
        out_dir=args.out_dir,
        reports_dir=args.reports_dir,
        seed=args.seed,
        max_delta=args.max_delta,
        max_inference_s=args.max_inference_s,
    )


if __name__ == "__main__":
    main()
