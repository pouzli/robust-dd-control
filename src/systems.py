"""Reference nonlinear systems for data-driven identification and control."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def cross_coupled_uncontrolled(t: float, x: Array, gamma: float = 0.8, delta: float = 0.6) -> Array:
    """Cross-coupled nonlinear benchmark dynamics without control input.

    Args:
        t: Time argument (unused but kept for ODE solver compatibility).
        x: State vector of shape (2,).
        gamma: Cross-coupling coefficient for x1*x2 term.
        delta: Coefficient for x1^2 term in second equation.

    Returns:
        Time derivative vector of shape (2,).
    """
    x1, x2 = x
    return np.array(
        [
            -x1 + x2 + gamma * x1 * x2,
            -2.0 * x2 + delta * (x1**2),
        ],
        dtype=float,
    )


def cross_coupled_controlled(
    t: float,
    x: Array,
    u: Array,
    b_matrix: Array,
    gamma: float = 0.8,
    delta: float = 0.6,
) -> Array:
    """Cross-coupled dynamics with additive control channel B u."""
    return cross_coupled_uncontrolled(t=t, x=x, gamma=gamma, delta=delta) + b_matrix @ u


def linear_stable_system(t: float, x: Array) -> Array:
    """Auxiliary stable linear system for basic debugging."""
    a = np.array([[-1.0, 0.2], [-0.1, -1.5]], dtype=float)
    return a @ x
