"""Data-driven identification routines for vector fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .basis import build_theta

Array = NDArray[np.float64]


@dataclass
class IdentificationResult:
    """Container for fitted model coefficients and metadata."""

    coefficients: Array
    basis_name: str
    ridge_alpha: float


def fit_least_squares(theta: Array, xdot: Array) -> Array:
    """Solve min ||Theta C - Xdot||_F using least squares."""
    c, *_ = np.linalg.lstsq(theta, xdot, rcond=None)
    return c


def fit_ridge(theta: Array, xdot: Array, alpha: float = 1e-6) -> Array:
    """Solve ridge regression closed-form: (T^T T + aI)^-1 T^T y."""
    n_features = theta.shape[1]
    regularized = theta.T @ theta + alpha * np.eye(n_features)
    rhs = theta.T @ xdot
    return np.linalg.solve(regularized, rhs)


def fit_identified_model(
    x_samples: Array,
    xdot_samples: Array,
    basis_fn: Callable[[Array], Array],
    basis_name: str,
    method: str = "least_squares",
    ridge_alpha: float = 1e-6,
) -> IdentificationResult:
    """Fit coefficients matrix C for f_hat(x)=Theta(x)C."""
    theta = build_theta(x_samples, basis_fn)
    if method == "least_squares":
        c = fit_least_squares(theta, xdot_samples)
    elif method == "ridge":
        c = fit_ridge(theta, xdot_samples, alpha=ridge_alpha)
    else:
        raise ValueError("method must be one of {'least_squares', 'ridge'}")
    return IdentificationResult(coefficients=c, basis_name=basis_name, ridge_alpha=ridge_alpha)


def predict_vector_field(x_samples: Array, basis_fn: Callable[[Array], Array], coefficients: Array) -> Array:
    """Predict xdot for rows of x_samples."""
    theta = build_theta(x_samples, basis_fn)
    return theta @ coefficients


def rmse(y_true: Array, y_pred: Array) -> float:
    """Root mean square error across all states and samples."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
