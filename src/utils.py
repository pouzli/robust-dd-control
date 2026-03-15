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


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save dataframe as CSV with parent directory creation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Save dict to JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
