"""Basis dictionaries for representing nonlinear vector fields."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
BasisFn = Callable[[Array], Array]


def linear_basis(x: Array) -> Array:
    """Return linear features [x1, x2, ..., xn]."""
    return x.astype(float)


def linear_with_constant_basis(x: Array) -> Array:
    """Return [1, x1, x2, ..., xn]."""
    return np.concatenate((np.array([1.0]), x.astype(float)))


def quadratic_basis(x: Array) -> Array:
    """Return quadratic polynomial features without constant term.

    For n states, returns [x_i, x_i^2, x_i*x_j (i<j)].
    """
    n = x.shape[0]
    features: list[float] = []
    features.extend(x.tolist())
    features.extend((x**2).tolist())
    for i in range(n):
        for j in range(i + 1, n):
            features.append(float(x[i] * x[j]))
    return np.asarray(features, dtype=float)


def quadratic_with_constant_basis(x: Array) -> Array:
    """Return [1, linear, quadratic, cross] features."""
    return np.concatenate((np.array([1.0]), quadratic_basis(x)))


def reduced_quadratic_no_cross_basis(x: Array) -> Array:
    """Return [1, x1, x2, ..., xn, x1^2, ..., xn^2] without cross terms."""
    return np.concatenate((np.array([1.0]), x.astype(float), (x**2).astype(float)))


def reduced_cross_only_basis(x: Array) -> Array:
    """Return [1, linear terms, pairwise cross terms], no squared terms."""
    n = x.shape[0]
    features: list[float] = [1.0]
    features.extend(x.tolist())
    for i in range(n):
        for j in range(i + 1, n):
            features.append(float(x[i] * x[j]))
    return np.asarray(features, dtype=float)


def basis_registry() -> dict[str, BasisFn]:
    """Registry of all available basis functions."""
    return {
        "linear": linear_basis,
        "linear_with_constant": linear_with_constant_basis,
        "quadratic": quadratic_basis,
        "quadratic_full": quadratic_with_constant_basis,
        "quadratic_with_constant": quadratic_with_constant_basis,
        "reduced_quadratic_no_cross": reduced_quadratic_no_cross_basis,
        "reduced_cross_only": reduced_cross_only_basis,
    }


def get_basis(name: str) -> BasisFn:
    """Fetch basis function by name."""
    mapping = basis_registry()
    if name not in mapping:
        available = ", ".join(mapping)
        raise ValueError(f"Unknown basis '{name}'. Available: {available}")
    return mapping[name]


def build_theta(x_samples: Array, basis_fn: BasisFn) -> Array:
    """Build feature matrix Theta(X) for sample matrix X of shape (N, n)."""
    return np.vstack([basis_fn(x) for x in x_samples])
