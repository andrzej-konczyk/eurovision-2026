"""
US-S5-01 — PyTorch MLP: third ensemble member.

Architecture: SimpleImputer(median) → StandardScaler → fully-connected MLP
(ReLU activations, optional dropout) → sigmoid output.  Same 23 features as
XGBoost and LightGBM.

Cross-validation: LeaveLastYearOut (temporal, no future leakage).
Hyperparameter grid: hidden_dims × lr × dropout.
Artefacts (DVC-tracked):
    models/artefacts/nn_model.pkl        — full NNPipeline (pickle)
    models/artefacts/nn_model.pt         — PyTorch state dict
    models/artefacts/nn_model_config.json — metadata + CV results

CLI:
    python -m src.models.nn [--data PATH] [--out-dir DIR] [--seed N]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models.cv import LeaveLastYearOut
from src.models.train import (
    ARTEFACT_DIR,
    ENRICHED_CSV,
    EXPERIMENT_NAME,
    MLFLOW_URI,
    RANDOM_SEED,
    ROOT,
    build_feature_matrix,
    training_split,
)

load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_N_EPOCHS: int = 300
_BATCH_SIZE: int = 32

_NN_GRID: dict[str, list] = {
    "hidden_dims": [(64, 32), (128, 64), (32,)],
    "lr": [0.01, 0.001],
    "dropout": [0.0, 0.2],
}


# ---------------------------------------------------------------------------
# MLP module
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Fully-connected MLP with ReLU activations and optional dropout.

    Output is a single sigmoid unit (binary probability for positive class).
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: tuple[int, ...],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [n_features, *hidden_dims]
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


# ---------------------------------------------------------------------------
# sklearn-compatible pipeline wrapper
# ---------------------------------------------------------------------------


class NNPipeline:
    """Impute → scale → MLP with sklearn-style fit / predict_proba."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.0,
        lr: float = 0.001,
        n_epochs: int = _N_EPOCHS,
        batch_size: int = _BATCH_SIZE,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        # keep_empty_features=True prevents all-NaN columns (e.g. implied_prob_close
        # when betting odds are absent) from being silently dropped, keeping the
        # feature count consistent with FEATURE_COLS across train and predict.
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_ = StandardScaler()
        self.model_: _MLP | None = None
        self.n_features_: int | None = None

    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "NNPipeline":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        X_imp = self.imputer_.fit_transform(X_arr)
        X_sc = self.scaler_.fit_transform(X_imp).astype(np.float32)
        # Zero-variance columns (all-NaN after imputation) produce NaN/inf after
        # scaling — replace with 0 so gradients remain finite.
        X_sc = np.nan_to_num(X_sc, nan=0.0, posinf=0.0, neginf=0.0)
        self.n_features_ = X_sc.shape[1]

        torch.manual_seed(self.seed)
        self.model_ = _MLP(self.n_features_, self.hidden_dims, self.dropout)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        X_t = torch.from_numpy(X_sc)
        y_t = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)

        n = len(X_t)
        rng = np.random.default_rng(self.seed)

        self.model_.train()
        for _ in range(self.n_epochs):
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                Xb, yb = X_t[batch], y_t[batch]
                optimizer.zero_grad()
                criterion(self.model_(Xb), yb).backward()
                optimizer.step()
        self.model_.eval()
        return self

    def predict_proba(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X_arr = np.asarray(X, dtype=float)
        X_imp = self.imputer_.transform(X_arr)
        X_sc = self.scaler_.transform(X_imp).astype(np.float32)
        X_sc = np.nan_to_num(X_sc, nan=0.0, posinf=0.0, neginf=0.0)
        X_t = torch.from_numpy(X_sc)
        with torch.no_grad():
            proba = self.model_(X_t).detach().numpy().flatten()
        return np.column_stack([1.0 - proba, proba])

    def get_params(self) -> dict:
        return {
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "lr": self.lr,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
        }


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------


def _cv_roc_auc(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
    n_epochs: int,
    batch_size: int,
) -> list[float]:
    """Run LeaveLastYearOut CV for *params*; return per-fold ROC-AUC scores."""
    cv = LeaveLastYearOut()
    X_arr = X.values
    y_arr = y.values
    scores: list[float] = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, groups=groups)):
        pipe = NNPipeline(
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"],
            lr=params["lr"],
            n_epochs=n_epochs,
            batch_size=batch_size,
            seed=seed + fold_i,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pipe.fit(X_arr[train_idx], y_arr[train_idx])
            except Exception:
                continue

        y_test = y_arr[test_idx]
        if len(np.unique(y_test)) < 2:
            continue  # single-class test fold — skip

        proba = pipe.predict_proba(X_arr[test_idx])[:, 1]
        scores.append(float(roc_auc_score(y_test, proba)))

    return scores


def grid_search_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    param_grid: dict | None = None,
    seed: int = RANDOM_SEED,
    n_epochs: int = _N_EPOCHS,
    batch_size: int = _BATCH_SIZE,
) -> tuple[dict, list[dict]]:
    """Exhaustive grid search with LeaveLastYearOut CV.

    Returns ``(best_params, cv_results)`` where each entry in cv_results
    contains the hyperparameters plus ``mean_roc_auc`` and ``std_roc_auc``.
    """
    if param_grid is None:
        param_grid = _NN_GRID

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    cv_results: list[dict] = []
    best_score = -np.inf
    best_params: dict = {}

    for combo in combos:
        params = dict(zip(keys, combo))
        scores = _cv_roc_auc(params, X, y, groups, seed, n_epochs, batch_size)
        mean_auc = float(np.mean(scores)) if scores else 0.0
        std_auc = float(np.std(scores)) if scores else 0.0
        cv_results.append({**params, "mean_roc_auc": mean_auc, "std_roc_auc": std_auc})
        print(
            f"  hidden={params['hidden_dims']}  lr={params['lr']}  "
            f"dropout={params['dropout']}  ->  "
            f"ROC-AUC={mean_auc:.4f} +/- {std_auc:.4f}"
        )
        if mean_auc > best_score:
            best_score = mean_auc
            best_params = params

    return best_params, cv_results


# ---------------------------------------------------------------------------
# DVC helper
# ---------------------------------------------------------------------------


def _dvc_add(path: Path) -> None:
    try:
        subprocess.run(
            ["dvc", "add", str(path)],
            check=True, capture_output=True, text=True,
        )
        print(f"  DVC: tracked {path.name}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  DVC: skipped for {path.name} (dvc unavailable or error)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_nn(
    data_path: Path = ENRICHED_CSV,
    out_dir: Path = ARTEFACT_DIR,
    seed: int = RANDOM_SEED,
    param_grid: dict | None = None,
    n_epochs: int = _N_EPOCHS,
    batch_size: int = _BATCH_SIZE,
) -> NNPipeline:
    """Train MLP with CV grid search, save artefacts, log to MLflow.

    Returns the final fitted NNPipeline.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    matrix = build_feature_matrix(df)
    X, y, groups, feat_cols = training_split(matrix)

    train_years = sorted(int(yr) for yr in groups.unique())
    print(f"\nMLP training  seed={seed}")
    print(f"Train years  : {train_years}")
    print(f"Train rows   : {len(X)}")
    print(f"Features     : {len(feat_cols)}")

    print(f"\nGrid search (LeaveLastYearOut CV, epochs={n_epochs}) …")
    best_params, cv_results = grid_search_cv(
        X, y, groups,
        param_grid=param_grid,
        seed=seed,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    best_entry = next(
        r for r in cv_results
        if r["hidden_dims"] == best_params["hidden_dims"]
        and r["lr"] == best_params["lr"]
        and r["dropout"] == best_params["dropout"]
    )
    best_mean = best_entry["mean_roc_auc"]
    best_std = best_entry["std_roc_auc"]
    print(f"\nBest  : {best_params}")
    print(f"CV ROC-AUC : {best_mean:.4f} ± {best_std:.4f}")

    # Refit on full training set
    print("\nRefitting on full training set …")
    torch.manual_seed(seed)
    final_pipe = NNPipeline(
        hidden_dims=best_params["hidden_dims"],
        dropout=best_params["dropout"],
        lr=best_params["lr"],
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipe.fit(X.values, y.values)

    # Save artefacts
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "nn_model.pkl"
    pt_path = out_dir / "nn_model.pt"
    cfg_path = out_dir / "nn_model_config.json"

    with open(pkl_path, "wb") as f:
        pickle.dump(final_pipe, f)

    assert final_pipe.model_ is not None
    torch.save(final_pipe.model_.state_dict(), pt_path)

    timestamp = datetime.now(timezone.utc).isoformat()
    config = {
        "story": "US-S5-01",
        "run_at": timestamp,
        "best_params": {**best_params, "hidden_dims": list(best_params["hidden_dims"])},
        "cv_results": [
            {**r, "hidden_dims": list(r["hidden_dims"])} for r in cv_results
        ],
        "best_cv_roc_auc_mean": best_mean,
        "best_cv_roc_auc_std": best_std,
        "train_years": train_years,
        "n_train_rows": int(len(X)),
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for p in (pkl_path, pt_path):
        _dvc_add(p)

    # MLflow
    run_name = f"nn-mlp-{timestamp[:10]}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "hidden_dims": str(list(best_params["hidden_dims"])),
            "dropout": best_params["dropout"],
            "lr": best_params["lr"],
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "n_train_rows": int(len(X)),
            "n_features": len(feat_cols),
            "seed": seed,
        })
        mlflow.log_metric("cv_roc_auc_mean", best_mean)
        mlflow.log_metric("cv_roc_auc_std", best_std)
        mlflow.log_artifact(str(cfg_path))
        mlflow.set_tag("story", "US-S5-01")
        mlflow.set_tag("model", "mlp")

    print(f"  Saved: {pkl_path.relative_to(ROOT, walk_up=True)}")
    print(f"  Saved: {pt_path.relative_to(ROOT, walk_up=True)}")
    print(f"  Saved: {cfg_path.relative_to(ROOT, walk_up=True)}")
    print("\nDone.")
    return final_pipe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP training pipeline (US-S5-01)")
    parser.add_argument("--data",       type=Path, default=ENRICHED_CSV)
    parser.add_argument("--out-dir",    type=Path, default=ARTEFACT_DIR)
    parser.add_argument("--seed",       type=int,  default=RANDOM_SEED)
    parser.add_argument("--n-epochs",   type=int,  default=_N_EPOCHS)
    parser.add_argument("--batch-size", type=int,  default=_BATCH_SIZE)
    args = parser.parse_args()
    train_nn(
        data_path=args.data,
        out_dir=args.out_dir,
        seed=args.seed,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
