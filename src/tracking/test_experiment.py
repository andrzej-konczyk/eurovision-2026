"""
MLflow smoke test — US-S1-04
Usage:  python src/tracking/test_experiment.py
Logs a dummy classifier run to verify the local tracking server is wired up.
Then run:  mlflow ui --backend-store-uri ./mlruns  (opens on localhost:5000)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
mlflow.set_tracking_uri(TRACKING_URI)

EXPERIMENT_NAME = "eurovision-2026-smoke-test"


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {"C": 1.0, "max_iter": 200, "solver": "lbfgs"}
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    with mlflow.start_run(run_name="smoke-test-iris"):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.set_tag("story", "US-S1-04")
        mlflow.sklearn.log_model(model, name="model")

    print(f"Run logged to: {TRACKING_URI}")
    print(f"Experiment:    {EXPERIMENT_NAME}")
    print(f"Accuracy:      {acc:.4f}")
    print()
    print("Start UI with:")
    print(f"  mlflow ui --backend-store-uri {TRACKING_URI}")
    print("  -> http://localhost:5000")


if __name__ == "__main__":
    main()
