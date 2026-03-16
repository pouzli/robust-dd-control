"""Lyapunov-analysis helpers for identified models."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_lyapunov

Array = NDArray[np.float64]


def numerical_jacobian(f: Callable[[Array], Array], x_eq: Array, h: float = 1e-6) -> Array:
    """Finite-difference Jacobian of vector field f at equilibrium x_eq."""
    n = x_eq.shape[0]
    jac = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = h
        jac[:, i] = (f(x_eq + dx) - f(x_eq - dx)) / (2.0 * h)
    return jac


def solve_lyapunov(a_matrix: Array, q_matrix: Array) -> Array:
    """Solve A^T P + P A = -Q for P."""
    p = solve_continuous_lyapunov(a_matrix.T, -q_matrix)
    return 0.5 * (p + p.T)


def lyapunov_value(x: Array, p_matrix: Array) -> float:
    """Compute V(x)=x^T P x."""
    return float(x.T @ p_matrix @ x)


def evaluate_lyapunov_grid(
    p_matrix: Array,
    x1_range: tuple[float, float],
    x2_range: tuple[float, float],
    points: int = 100,
) -> tuple[Array, Array, Array]:
    """Evaluate Lyapunov function on 2D meshgrid for contour plots."""
    x1 = np.linspace(x1_range[0], x1_range[1], points)
    x2 = np.linspace(x2_range[0], x2_range[1], points)
    xx, yy = np.meshgrid(x1, x2)
    vv = np.zeros_like(xx)
    for i in range(points):
        for j in range(points):
            state = np.array([xx[i, j], yy[i, j]], dtype=float)
            vv[i, j] = lyapunov_value(state, p_matrix)
    return xx, yy, vv


def estimate_ultimate_bound_radius(epsilon: float, lambda_min_q: float, lambda_max_p: float) -> float:
    """Conservative ultimate bound radius proxy for additive bounded disturbance."""
    if lambda_min_q <= 0 or lambda_max_p <= 0:
        raise ValueError("Lyapunov eigenvalue terms must be positive")
    return float(2.0 * epsilon * np.sqrt(lambda_max_p) / lambda_min_q)
