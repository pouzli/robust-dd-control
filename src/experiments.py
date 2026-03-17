"""Helpers for synthetic experiment sweeps and summary metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def add_gaussian_noise(values: Array, sigma: float, rng: np.random.Generator) -> Array:
    """Add i.i.d. Gaussian noise with standard deviation sigma."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return values.copy()
    return values + rng.normal(loc=0.0, scale=sigma, size=values.shape)


def residual_statistics(residuals: Array) -> dict[str, float]:
    """Compute residual norm statistics used in experiment tables."""
    norms = np.linalg.norm(residuals, axis=1)
    return {
        "max_residual_norm": float(np.max(norms)),
        "mean_residual_norm": float(np.mean(norms)),
        "median_residual_norm": float(np.median(norms)),
        "q90_residual_norm": float(np.quantile(norms, 0.90)),
        "q95_residual_norm": float(np.quantile(norms, 0.95)),
        "q99_residual_norm": float(np.quantile(norms, 0.99)),
    }


def is_hurwitz(a_matrix: Array, tol: float = 1e-9) -> bool:
    """Check if all real parts of eigenvalues are strictly negative."""
    eigvals = np.linalg.eigvals(a_matrix)
    return bool(np.all(np.real(eigvals) < -tol))


@dataclass
class TailMetrics:
    """Tail norm metrics for robust boundedness diagnostics."""

    final_norm: float
    mean_norm_over_tail: float
    max_norm_over_tail: float


def compute_tail_metrics(states: Array, tail_fraction: float = 0.2) -> TailMetrics:
    """Compute final and tail norms for a trajectory states array (T,n)."""
    norms = np.linalg.norm(states, axis=1)
    tail_start = int((1.0 - tail_fraction) * len(norms))
    tail = norms[tail_start:]
    return TailMetrics(
        final_norm=float(norms[-1]),
        mean_norm_over_tail=float(np.mean(tail)),
        max_norm_over_tail=float(np.max(tail)),
    )


def summarize_tail_metrics(metrics: list[TailMetrics]) -> dict[str, float]:
    """Aggregate list of TailMetrics into robust-radius style summaries."""
    tail_means = np.array([m.mean_norm_over_tail for m in metrics], dtype=float)
    tail_maxes = np.array([m.max_norm_over_tail for m in metrics], dtype=float)
    return {
        "mean_tail_radius": float(np.mean(tail_means)),
        "max_tail_radius": float(np.max(tail_maxes)),
        "ultimate_radius_estimate": float(np.quantile(tail_maxes, 0.95)),
    }
