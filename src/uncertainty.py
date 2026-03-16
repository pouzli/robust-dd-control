"""Residual-based deterministic uncertainty estimation tools."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def compute_residuals(xdot_true: Array, xdot_pred: Array) -> Array:
    """Compute residual vectors r = xdot_true - xdot_pred."""
    return xdot_true - xdot_pred


def residual_norms(residuals: Array) -> Array:
    """Compute Euclidean norm of each residual vector."""
    return np.linalg.norm(residuals, axis=1)


def estimate_epsilon(res_norms: Array, q: float = 0.95) -> float:
    """Estimate deterministic uncertainty bound via high quantile."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0,1)")
    return float(np.quantile(res_norms, q))


def compute_bounding_box(x_samples: Array) -> tuple[Array, Array]:
    """Axis-aligned box Omega from sample min and max coordinates."""
    return np.min(x_samples, axis=0), np.max(x_samples, axis=0)


def is_inside_omega(x: Array, lower: Array, upper: Array) -> bool:
    """Check if x belongs to axis-aligned bounding box Omega."""
    return bool(np.all(x >= lower) and np.all(x <= upper))


def bounded_disturbance(t: float, epsilon: float) -> Array:
    """Smooth bounded disturbance with ||Delta|| <= epsilon."""
    signal = np.array([np.sin(3.0 * t), np.cos(2.0 * t)], dtype=float)
    norm = np.linalg.norm(signal)
    if norm < 1e-12:
        return np.zeros_like(signal)
    return epsilon * 0.9 * signal / norm
