"""Simulation helpers for true, identified, and uncertain systems."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

Array = NDArray[np.float64]


def simulate_trajectory(
    dynamics: Callable[[float, Array], Array],
    x0: Array,
    t_span: tuple[float, float],
    t_eval: Array,
) -> tuple[Array, Array]:
    """Integrate trajectory for a system xdot=dynamics(t,x)."""
    sol = solve_ivp(dynamics, t_span=t_span, y0=x0, t_eval=t_eval, rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.t, sol.y.T


def simulate_batch(
    dynamics: Callable[[float, Array], Array],
    initials: Array,
    t_span: tuple[float, float],
    t_eval: Array,
) -> list[tuple[Array, Array]]:
    """Simulate multiple initial conditions."""
    return [simulate_trajectory(dynamics, x0, t_span, t_eval) for x0 in initials]


def uncertain_dynamics(
    base_dynamics: Callable[[float, Array], Array],
    disturbance: Callable[[float], Array],
) -> Callable[[float, Array], Array]:
    """Build uncertain dynamics xdot=f(x,t)+Delta(t)."""

    def dyn(t: float, x: Array) -> Array:
        return base_dynamics(t, x) + disturbance(t)

    return dyn
