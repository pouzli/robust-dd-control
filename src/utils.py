"""General utility helpers for reproducible experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Set NumPy random seed."""
    np.random.seed(seed)


def get_project_root(marker_files: tuple[str, ...] = ("pyproject.toml", "setup.py", "README.md")) -> Path:
    """Return the project root directory.

    The project root is detected by walking up from the current working directory
    until a known marker file (e.g., pyproject.toml) is found.

    This is useful for notebooks that may be executed from a subdirectory (e.g., ``notebooks/``).
    """

    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    return cwd


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root if it is not absolute."""

    p = Path(path)
    if p.is_absolute():
        return p
    return (get_project_root() / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path object.

    Relative paths are resolved against the project root (via :func:`get_project_root`).
    """

    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save dataframe as CSV with parent directory creation."""

    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Save dict to JSON file."""

    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load dataframe from CSV file.

    Relative paths are resolved against the project root.
    """

    p = resolve_path(path)
    return pd.read_csv(p)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load dict from JSON file.

    Relative paths are resolved against the project root.
    """

    p = resolve_path(path)
    return json.loads(p.read_text(encoding="utf-8"))
