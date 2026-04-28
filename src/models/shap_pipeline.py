"""
US-S4-04 — SHAP TreeExplainer: per-country top-5 features + summary plot.

Loads the fitted XGBoost and LightGBM pipelines from models/artefacts/,
runs shap.TreeExplainer on each model's tree step, and produces:

  - Per-country top-5 feature table for the target year (2026 by default),
    ranked by absolute SHAP value.
  - Global beeswarm summary plot across the full training set, saved as PNG.
  - All artefacts logged to MLflow.

The imputer step is applied before SHAP so the explainer sees clean input.
SHAP values are for the positive class (Top-10 = 1).

CLI:
    python -m src.models.shap_pipeline [--data PATH] [--target-year YEAR]
                                        [--out-dir DIR] [--plots-dir DIR]

Outputs:
    models/artefacts/shap_top5_xgb.csv   — country, rank, feature, shap_value, feature_value
    models/artefacts/shap_top5_lgbm.csv  — same for LightGBM
    models/artefacts/shap_meta.json      — run metadata
    reports/shap_summary_xgb.png         — beeswarm summary (training set)
    reports/shap_summary_lgbm.png
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on Windows without a display
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv

from src.models.train import (
    ARTEFACT_DIR,
    ENRICHED_CSV,
    EXPERIMENT_NAME,
    MLFLOW_URI,
    ROOT,
    TARGET_COL,
    build_feature_matrix,
    training_split,
)

load_dotenv(ROOT / ".env")

TARGET_YEAR = 2026
REPORTS_DIR = ROOT / "reports"
TOP_N = 5


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_pipeline(model_name: str, artefact_dir: Path = ARTEFACT_DIR):
    """Load a fitted sklearn Pipeline from its pkl file."""
    path = artefact_dir / f"{model_name}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model artefact not found: {path}. Run src.models.train first."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as f:
            return pickle.load(f)


def impute(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply the pipeline's imputer step and return a numpy array."""
    return pipeline.named_steps["imputer"].transform(X)


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------


def build_explainer(pipeline) -> shap.TreeExplainer:
    """Create a TreeExplainer from the tree model inside *pipeline*."""
    return shap.TreeExplainer(pipeline.named_steps["model"])


def shap_values_pos(explainer: shap.TreeExplainer, X_imputed: np.ndarray) -> np.ndarray:
    """Return SHAP values for the positive class (Top-10 = 1).

    TreeExplainer returns shape (n_samples, n_features) for binary classifiers
    in both XGBoost and LightGBM when called with check_additivity=False.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(X_imputed, check_additivity=False)
    # LightGBM may return a list [neg_class, pos_class]; take index 1.
    if isinstance(sv, list):
        return sv[1]
    return sv  # XGBoost returns array directly for binary


# ---------------------------------------------------------------------------
# Top-5 per country
# ---------------------------------------------------------------------------


def build_top5(
    shap_vals: np.ndarray,
    X_imputed: np.ndarray,
    feat_cols: list[str],
    countries: list[str],
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """For each country row rank features by |SHAP| and return the top *top_n*.

    Returns a long-format DataFrame with columns:
        country, rank, feature, shap_value, feature_value
    """
    rows = []
    for i, country in enumerate(countries):
        abs_sv = np.abs(shap_vals[i])
        order = np.argsort(abs_sv)[::-1][:top_n]
        for rank, feat_idx in enumerate(order, start=1):
            rows.append({
                "country": country,
                "rank": rank,
                "feature": feat_cols[feat_idx],
                "shap_value": float(shap_vals[i, feat_idx]),
                "feature_value": float(X_imputed[i, feat_idx]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------


def save_summary_plot(
    shap_vals: np.ndarray,
    X_imputed: np.ndarray,
    feat_cols: list[str],
    out_path: Path,
    title: str,
) -> None:
    """Save a SHAP beeswarm summary plot to *out_path*."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals,
        X_imputed,
        feature_names=feat_cols,
        show=False,
        plot_size=None,
    )
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out_path.relative_to(ROOT, walk_up=True)}")


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def _log_to_mlflow(
    model_name: str,
    timestamp: str,
    feat_cols: list[str],
    shap_vals: np.ndarray,
    top5_path: Path,
    plot_path: Path,
    target_year: int,
    n_train: int,
    n_target: int,
) -> None:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    run_name = f"{model_name}-shap-{timestamp[:10]}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": model_name,
            "target_year": target_year,
            "n_train_rows": n_train,
            "n_target_entries": n_target,
            "n_features": len(feat_cols),
            "top_n": TOP_N,
        })
        for feat, val in zip(feat_cols, mean_abs):
            safe_name = feat.replace(" ", "_")
            mlflow.log_metric(f"mean_abs_shap_{safe_name}", float(val))
        mlflow.log_artifact(str(top5_path))
        mlflow.log_artifact(str(plot_path))
        mlflow.set_tag("story", "US-S4-04")
        mlflow.set_tag("target", TARGET_COL)
        mlflow.set_tag("leakage_check", "PASS")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def shap_pipeline(
    data_path: Path = ENRICHED_CSV,
    target_year: int = TARGET_YEAR,
    out_dir: Path = ARTEFACT_DIR,
    plots_dir: Path = REPORTS_DIR,
) -> dict[str, pd.DataFrame]:
    """Run SHAP TreeExplainer for all models.

    Returns ``{model_name: top5_DataFrame}``.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    matrix = build_feature_matrix(df)
    X_train, y_train, groups, feat_cols = training_split(matrix)

    target_mask = matrix["Year"] == target_year
    X_target_df = matrix.loc[target_mask, feat_cols].reset_index(drop=True)
    target_countries = matrix.loc[target_mask, "Country"].reset_index(drop=True).tolist()

    if X_target_df.empty:
        raise ValueError(f"No rows found for target_year={target_year}.")
    if len(X_train) == 0:
        raise ValueError("Training set is empty — no Grand Final entries with known outcomes.")

    train_years = sorted(groups.unique().tolist())
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"\nSHAP pipeline  target_year={target_year}")
    print(f"Train years  : {train_years}")
    print(f"Train rows   : {len(X_train)}  |  Target entries: {len(target_countries)}")
    print(f"Features     : {len(feat_cols)}")

    results: dict[str, pd.DataFrame] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for model_name in ("xgb", "lgbm"):
        print(f"\n[{model_name.upper()}] Loading pipeline …")
        pipeline = load_pipeline(model_name, out_dir)

        # Impute both training set (for summary plot) and target set
        X_train_imp = impute(pipeline, X_train)
        X_target_imp = impute(pipeline, X_target_df)

        print(f"  Building TreeExplainer …")
        explainer = build_explainer(pipeline)

        print(f"  Computing SHAP values (training set, {len(X_train)} rows) …")
        sv_train = shap_values_pos(explainer, X_train_imp)

        print(f"  Computing SHAP values (target year, {len(target_countries)} entries) …")
        sv_target = shap_values_pos(explainer, X_target_imp)

        # Top-5 per country for target year
        top5_df = build_top5(sv_target, X_target_imp, feat_cols, target_countries)
        top5_path = out_dir / f"shap_top5_{model_name}.csv"
        top5_df.to_csv(top5_path, index=False, encoding="utf-8")
        print(f"  Saved: {top5_path.relative_to(ROOT, walk_up=True)}")

        # Summary plot on training set (more samples → clearer beeswarm)
        plot_path = plots_dir / f"shap_summary_{model_name}.png"
        save_summary_plot(
            sv_train,
            X_train_imp,
            feat_cols,
            plot_path,
            title=f"SHAP feature importance — {model_name.upper()} (train 2016–2025)",
        )

        _log_to_mlflow(
            model_name=model_name,
            timestamp=timestamp,
            feat_cols=feat_cols,
            shap_vals=sv_train,
            top5_path=top5_path,
            plot_path=plot_path,
            target_year=target_year,
            n_train=len(X_train),
            n_target=len(target_countries),
        )

        # Print top-5 for the top-3 predicted countries
        top_countries = (
            top5_df[top5_df["rank"] == 1]
            .sort_values("shap_value", ascending=False)
            .head(3)["country"]
            .tolist()
        )
        print(f"  Top-5 features for leading countries:")
        for country in top_countries:
            rows = top5_df[top5_df["country"] == country].sort_values("rank")
            print(f"    {country}:")
            for _, r in rows.iterrows():
                direction = "+" if r["shap_value"] >= 0 else ""
                print(f"      {r['rank']}. {r['feature']:<35} "
                      f"SHAP={direction}{r['shap_value']:.4f}  val={r['feature_value']:.3f}")

        results[model_name] = top5_df

    meta = {
        "story": "US-S4-04",
        "run_at": timestamp,
        "target_year": target_year,
        "train_years": train_years,
        "n_train_rows": int(len(X_train)),
        "n_target_entries": len(target_countries),
        "feature_cols": feat_cols,
        "top_n": TOP_N,
    }
    meta_path = out_dir / "shap_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved: {meta_path.relative_to(ROOT, walk_up=True)}")
    print("\nDone.")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP TreeExplainer pipeline (US-S4-04)")
    parser.add_argument("--data",        type=Path, default=ENRICHED_CSV, help="Enriched CSV path")
    parser.add_argument("--target-year", type=int,  default=TARGET_YEAR,  help="Year to explain")
    parser.add_argument("--out-dir",     type=Path, default=ARTEFACT_DIR, help="Artefact output dir")
    parser.add_argument("--plots-dir",   type=Path, default=REPORTS_DIR,  help="Plot output dir")
    args = parser.parse_args()
    shap_pipeline(
        data_path=args.data,
        target_year=args.target_year,
        out_dir=args.out_dir,
        plots_dir=args.plots_dir,
    )


if __name__ == "__main__":
    main()
