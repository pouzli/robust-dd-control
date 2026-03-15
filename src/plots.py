"""Plotting utilities for trajectories, residuals, and Lyapunov contours."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def set_plot_style() -> None:
    """Set a clean default plotting style."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (7, 5),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
        }
    )


def save_figure(path: str | Path) -> None:
    """Save current figure with tight layout and 300 DPI."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=300)


def plot_phase_trajectories(trajectories: list[tuple[Array, Array]], title: str) -> None:
    """Plot 2D phase trajectories from batch simulation output."""
    for _, states in trajectories:
        plt.plot(states[:, 0], states[:, 1], lw=1.3)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)


def plot_residual_histogram(norms: Array, epsilon: float, bins: int = 30) -> None:
    """Histogram of residual norms with epsilon threshold marker."""
    plt.hist(norms, bins=bins, alpha=0.8, color="steelblue")
    plt.axvline(epsilon, color="crimson", linestyle="--", linewidth=2, label=f"epsilon={epsilon:.3f}")
    plt.xlabel("||r||")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Residual norms and deterministic uncertainty bound")


def plot_lyapunov_contours(xx: Array, yy: Array, vv: Array, levels: int = 15) -> None:
    """Contour plot for Lyapunov level sets."""
    contours = plt.contour(xx, yy, vv, levels=levels, cmap="viridis")
    plt.clabel(contours, inline=True, fontsize=8)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Lyapunov level sets")
