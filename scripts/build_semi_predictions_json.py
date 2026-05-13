"""Build 2026 semi-final qualification predictions for the Streamlit dashboard."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backtest_semi import (
    QUALIFIERS_PER_SEMI,
    SEMI_FEATURE_COLS,
    SEMI_TARGET,
)
from src.models.confidence import bootstrap_proba, compute_ci
from src.models.cv import LeaveLastYearOut
from src.models.train import (
    ENRICHED_CSV,
    LGBM_GRID,
    RANDOM_SEED,
    ROOT,
    XGB_GRID,
    build_feature_matrix,
)


REPORTS_DIR = ROOT / "reports"
OUTPUT_JSON = REPORTS_DIR / "semi_predictions_2026.json"
TARGET_YEAR = 2026
DEFAULT_N_BOOTSTRAP = 1000
CALIBRATION_EPS = 1e-6

COUNTRY_FLAGS = {
    "Albania": "🇦🇱",
    "Armenia": "🇦🇲",
    "Australia": "🇦🇺",
    "Azerbaijan": "🇦🇿",
    "Belgium": "🇧🇪",
    "Bulgaria": "🇧🇬",
    "Croatia": "🇭🇷",
    "Cyprus": "🇨🇾",
    "Czech Republic": "🇨🇿",
    "Denmark": "🇩🇰",
    "Estonia": "🇪🇪",
    "Finland": "🇫🇮",
    "Georgia": "🇬🇪",
    "Greece": "🇬🇷",
    "Israel": "🇮🇱",
    "Latvia": "🇱🇻",
    "Lithuania": "🇱🇹",
    "Luxembourg": "🇱🇺",
    "Malta": "🇲🇹",
    "Moldova": "🇲🇩",
    "Montenegro": "🇲🇪",
    "Norway": "🇳🇴",
    "Poland": "🇵🇱",
    "Portugal": "🇵🇹",
    "Romania": "🇷🇴",
    "San Marino": "🇸🇲",
    "Serbia": "🇷🇸",
    "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭",
    "Ukraine": "🇺🇦",
}


def grid_search_single_process(
    clf,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
) -> GridSearchCV:
    cv = LeaveLastYearOut(min_train_years=2)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        refit=True,
        n_jobs=1,
        verbose=0,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train, y_train, groups=groups)
    return grid_search


def ranked_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = frame.sort_values(["semi_final", "prob_mean"], ascending=[True, False]).copy()
    out["rank_in_semi"] = out.groupby("semi_final").cumcount() + 1
    return out.to_dict(orient="records")


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), CALIBRATION_EPS, 1.0 - CALIBRATION_EPS)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_probability_calibrator(
    best_estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
) -> tuple[LogisticRegression | None, dict[str, Any]]:
    """Fit a monotonic Platt-style calibrator from temporal OOF predictions."""
    cv = LeaveLastYearOut(min_train_years=2)
    oof_proba: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []

    for train_idx, valid_idx in cv.split(X_train, y_train, groups):
        estimator = clone(best_estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof_proba.append(estimator.predict_proba(X_train.iloc[valid_idx])[:, 1])
        oof_y.append(y_train.iloc[valid_idx].to_numpy())

    if not oof_proba:
        return None, {"method": "identity", "reason": "no_oof_folds"}

    raw = np.concatenate(oof_proba)
    actual = np.concatenate(oof_y)
    if len(np.unique(actual)) < 2:
        return None, {"method": "identity", "reason": "single_class_oof"}

    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(_logit(raw), actual)
    coefficient = float(calibrator.coef_[0][0])
    intercept = float(calibrator.intercept_[0])
    if coefficient <= 0:
        return None, {
            "method": "identity",
            "reason": "non_monotonic_fit",
            "coefficient": coefficient,
            "intercept": intercept,
        }

    calibrated = calibrator.predict_proba(_logit(raw))[:, 1]
    return calibrator, {
        "method": "temporal_oof_logistic",
        "n_oof_rows": int(len(raw)),
        "raw_mean": float(np.mean(raw)),
        "calibrated_mean": float(np.mean(calibrated)),
        "actual_rate": float(np.mean(actual)),
        "coefficient": coefficient,
        "intercept": intercept,
    }


def apply_probability_calibrator(
    proba_matrix: np.ndarray,
    calibrator: LogisticRegression | None,
) -> np.ndarray:
    if calibrator is None:
        return proba_matrix
    flat = proba_matrix.reshape(-1)
    calibrated = calibrator.predict_proba(_logit(flat))[:, 1]
    return calibrated.reshape(proba_matrix.shape)


def quota_calibrate_proba_matrix(
    proba_matrix: np.ndarray,
    semi_nums: list[int],
    qualifiers_per_semi: int = QUALIFIERS_PER_SEMI,
) -> np.ndarray:
    """Constrain each bootstrap draw so expected qualifiers per semi equals the contest quota."""
    calibrated = np.asarray(proba_matrix, dtype=float).copy()
    semi_array = np.asarray(semi_nums)
    for row_idx in range(calibrated.shape[0]):
        for semi_final in sorted(set(semi_nums)):
            mask = semi_array == semi_final
            current_sum = float(np.nansum(calibrated[row_idx, mask]))
            if current_sum <= 0:
                calibrated[row_idx, mask] = qualifiers_per_semi / int(mask.sum())
                continue
            calibrated[row_idx, mask] = np.clip(
                calibrated[row_idx, mask] * (qualifiers_per_semi / current_sum),
                0.0,
                1.0,
            )
    return calibrated


def build_semi_predictions(n_bootstrap: int, seed: int) -> dict[str, Any]:
    df = pd.read_csv(ENRICHED_CSV, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    matrix = build_feature_matrix(df)
    feat_cols = [column for column in SEMI_FEATURE_COLS if column in matrix.columns]

    train_mask = (
        matrix["Semi_Final_Num"].notna()
        & (matrix["Year"] < TARGET_YEAR)
        & matrix[SEMI_TARGET].notna()
    )
    target_mask = (
        matrix["Semi_Final_Num"].notna()
        & (matrix["Year"] == TARGET_YEAR)
    )
    train = matrix[train_mask].copy()
    target = matrix[target_mask].copy()

    if train.empty:
        raise ValueError("No historical semi-final rows available for training.")
    if target.empty:
        raise ValueError(f"No semi-final rows available for target_year={TARGET_YEAR}.")

    X_train = train[feat_cols]
    y_train = train[SEMI_TARGET].astype(int)
    groups = train["Year"]
    X_target = target[feat_cols].reset_index(drop=True)
    countries = target["Country"].reset_index(drop=True).tolist()
    semi_nums = target["Semi_Final_Num"].reset_index(drop=True).astype(int).tolist()
    running_order = target.get("Running_Order_Semi", pd.Series([None] * len(target))).reset_index(drop=True).tolist()

    model_outputs: dict[str, pd.DataFrame] = {}
    best_params: dict[str, dict[str, Any]] = {}
    calibration_meta: dict[str, dict[str, Any]] = {}
    models = [
        (
            "xgb",
            xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
            ),
            XGB_GRID,
        ),
        (
            "lgbm",
            lgb.LGBMClassifier(
                objective="binary",
                random_state=seed,
                verbose=-1,
            ),
            LGBM_GRID,
        ),
    ]

    for model_name, clf, param_grid in models:
        print(f"[{model_name.upper()}] grid search")
        gs = grid_search_single_process(clf, param_grid, X_train, y_train, groups)
        params = {key.replace("model__", ""): value for key, value in gs.best_params_.items()}
        best_params[model_name] = params
        calibrator, calibration_meta[model_name] = fit_probability_calibrator(
            gs.best_estimator_,
            X_train,
            y_train.reset_index(drop=True),
            groups.reset_index(drop=True),
        )
        print(
            f"[{model_name.upper()}] calibration "
            f"{calibration_meta[model_name].get('method')} "
            f"raw_mean={calibration_meta[model_name].get('raw_mean', float('nan')):.3f} "
            f"cal_mean={calibration_meta[model_name].get('calibrated_mean', float('nan')):.3f}"
        )
        print(f"[{model_name.upper()}] bootstrap n={n_bootstrap}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba_matrix = bootstrap_proba(
                model_name=model_name,
                best_params=gs.best_params_,
                X_train=X_train,
                y_train=y_train,
                X_target=X_target,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        proba_matrix = apply_probability_calibrator(proba_matrix, calibrator)
        pre_quota_sum = {
            f"sf{semi_final}": float(np.mean(np.sum(proba_matrix[:, np.asarray(semi_nums) == semi_final], axis=1)))
            for semi_final in sorted(set(semi_nums))
        }
        proba_matrix = quota_calibrate_proba_matrix(proba_matrix, semi_nums)
        calibration_meta[model_name]["quota_calibration"] = {
            "method": "per_bootstrap_semi_sum_to_quota",
            "qualifiers_per_semi": QUALIFIERS_PER_SEMI,
            "mean_sum_before": pre_quota_sum,
            "mean_sum_after": {
                f"sf{semi_final}": float(np.mean(np.sum(proba_matrix[:, np.asarray(semi_nums) == semi_final], axis=1)))
                for semi_final in sorted(set(semi_nums))
            },
        }
        ci_df = compute_ci(proba_matrix, countries)
        ci_df["semi_final"] = ci_df["country"].map(dict(zip(countries, semi_nums)))
        ci_df["running_order"] = ci_df["country"].map(dict(zip(countries, running_order)))
        ci_df["rank_in_semi"] = ci_df.groupby("semi_final")["prob_mean"].rank(
            method="first",
            ascending=False,
        ).astype(int)
        model_outputs[model_name] = ci_df

    xgb_df = model_outputs["xgb"].set_index("country")
    lgbm_df = model_outputs["lgbm"].set_index("country")
    rows: list[dict[str, Any]] = []
    for country in countries:
        xgb_row = xgb_df.loc[country]
        lgbm_row = lgbm_df.loc[country]
        prob_qualify = float(np.nanmean([xgb_row["prob_mean"], lgbm_row["prob_mean"]]))
        ci80_lo = float(np.nanmean([xgb_row["ci80_lo"], lgbm_row["ci80_lo"]]))
        ci80_hi = float(np.nanmean([xgb_row["ci80_hi"], lgbm_row["ci80_hi"]]))
        rows.append(
            {
                "country": country,
                "flag": COUNTRY_FLAGS.get(country, ""),
                "semi_final": int(xgb_row["semi_final"]),
                "running_order": None if pd.isna(xgb_row["running_order"]) else int(xgb_row["running_order"]),
                "prob_qualify": prob_qualify,
                "ci80_lo": ci80_lo,
                "ci80_hi": ci80_hi,
                "xgb_prob": float(xgb_row["prob_mean"]),
                "xgb_ci80_lo": float(xgb_row["ci80_lo"]),
                "xgb_ci80_hi": float(xgb_row["ci80_hi"]),
                "lgbm_prob": float(lgbm_row["prob_mean"]),
                "lgbm_ci80_lo": float(lgbm_row["ci80_lo"]),
                "lgbm_ci80_hi": float(lgbm_row["ci80_hi"]),
            }
        )

    rows = sorted(rows, key=lambda row: (row["semi_final"], -row["prob_qualify"]))
    for semi_final in (1, 2):
        sf_rows = [row for row in rows if row["semi_final"] == semi_final]
        for rank, row in enumerate(sf_rows, start=1):
            row["rank_in_semi"] = rank
            row["predicted_qualifier"] = rank <= QUALIFIERS_PER_SEMI

    return {
        "story": "US-S7-02",
        "generated_at": datetime.now(UTC).isoformat(),
        "target_year": TARGET_YEAR,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "qualifiers_per_semi": QUALIFIERS_PER_SEMI,
        "n_train_rows": int(len(train)),
        "train_years": sorted(int(year) for year in groups.unique().tolist()),
        "n_countries": len(rows),
        "feature_cols": feat_cols,
        "source_file": str(ENRICHED_CSV.relative_to(ROOT)),
        "models": {
            "xgb": {
                "best_params": best_params["xgb"],
                "calibration": calibration_meta["xgb"],
                "countries": ranked_rows(model_outputs["xgb"]),
            },
            "lgbm": {
                "best_params": best_params["lgbm"],
                "calibration": calibration_meta["lgbm"],
                "countries": ranked_rows(model_outputs["lgbm"]),
            },
        },
        "calibration": {
            "method": "temporal_oof_logistic_plus_quota",
            "note": (
                "Bootstrap probabilities are passed through a monotonic logistic calibrator "
                "fitted on historical leave-last-year-out predictions, then quota-calibrated so each "
                "semi-final bootstrap draw sums to 10 expected qualifiers before CI summaries are computed."
            ),
        },
        "countries": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2026 semi-final qualification predictions.")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_semi_predictions(n_bootstrap=args.n_bootstrap, seed=args.seed)
    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Wrote {OUTPUT_JSON.relative_to(ROOT)} with {payload['n_countries']} countries")


if __name__ == "__main__":
    main()
