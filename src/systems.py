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


def linear_damped_system(t: float, x: Array, a: float = 1.0, b: float = 1.0) -> Array:
    """Linear damped oscillator (Example 1).

    ẋ₁ = -a x₁ + x₂
    ẋ₂ = -b x₂
    """

    x1, x2 = x
    return np.array([-a * x1 + x2, -b * x2], dtype=float)


def quadratic_nonlinear_oscillator(t: float, x: Array, omega: float = 1.0, alpha: float = 0.5, beta: float = 0.2) -> Array:
    """Nonlinear oscillator with quadratic right-hand side (Example 2).

    ẋ₁ = x₂
    ẋ₂ = -ω² x₁ - α x₂ + β x₁²
    """

    x1, x2 = x
    return np.array([x2, -omega**2 * x1 - alpha * x2 + beta * x1**2], dtype=float)


def van_der_pol_oscillator(t: float, x: Array, mu: float = 1.0) -> Array:
    """Van der Pol oscillator (Example 3).

    ẋ₁ = x₂
    ẋ₂ = μ (1 - x₁²) x₂ - x₁
    """

    x1, x2 = x
    return np.array([x2, mu * (1.0 - x1**2) * x2 - x1], dtype=float)


def saturated_nonlinearity_system(t: float, x: Array) -> Array:
    """System with saturation-like nonlinearity (Example 5).

    ẋ₁ = -x₁ + tanh(x₂)
    ẋ₂ = -x₂

    This system is useful to check robustness of polynomial-based identification.
    """

    x1, x2 = x
    return np.array([-x1 + np.tanh(x2), -x2], dtype=float)


def get_example_system(name: str):
    """Return a (callable, metadata) tuple for a named example system.

    The returned callable has signature (t, x) -> xdot and can be used with
    the simulation suite.
    """

    mapping = {
        "linear_damped": linear_damped_system,
        "quadratic_oscillator": quadratic_nonlinear_oscillator,
        "van_der_pol": van_der_pol_oscillator,
        "cross_coupled": cross_coupled_uncontrolled,
        "saturated": saturated_nonlinearity_system,
    }

    if name not in mapping:
        raise ValueError(
            f"Unknown example system '{name}'. Available: {', '.join(sorted(mapping))}"
        )
    return mapping[name]
