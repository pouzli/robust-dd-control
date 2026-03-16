"""Linear state-feedback helpers for the benchmark system."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import place_poles

Array = NDArray[np.float64]


def design_state_feedback(a_matrix: Array, b_matrix: Array, desired_poles: Array) -> Array:
    """Return K such that A + B K has desired poles (u = Kx convention)."""
    placement = place_poles(a_matrix, b_matrix, desired_poles)
    k_standard = placement.gain_matrix
    return -k_standard


def closed_loop_jacobian(a_matrix: Array, b_matrix: Array, k_matrix: Array) -> Array:
    """Compute closed-loop linearization A_cl = A + B K."""
    return a_matrix + b_matrix @ k_matrix


def candidate_gains() -> dict[str, Array]:
    """Small set of hand-crafted gain matrices for 2D analysis."""
    return {
        "mild": np.array([[-0.8, -0.2]], dtype=float),
        "medium": np.array([[-1.5, -0.6]], dtype=float),
        "aggressive": np.array([[-2.5, -1.0]], dtype=float),
    }
