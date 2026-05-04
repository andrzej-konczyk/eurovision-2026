"""Shared MLflow tracking configuration."""

from __future__ import annotations

import os
from pathlib import Path


def tracking_uri(root: Path) -> str:
    """Return the local MLflow tracking URI.

    The project standard is a SQLite backend at repo root. Relative SQLite
    paths are expanded to absolute file URIs so scripts behave the same from
    every working directory and do not silently fall back to the file store.
    """
    configured = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db").strip()
    if configured.startswith("sqlite:///"):
        db_path = configured.removeprefix("sqlite:///")
        path = Path(db_path)
        if not path.is_absolute():
            path = root / path
        return f"sqlite:///{path.as_posix()}"
    return configured
