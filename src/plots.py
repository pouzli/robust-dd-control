"""Plotting utilities for trajectories, residuals, and Lyapunov contours."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .utils import resolve_path

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
    """Save current figure with tight layout and 300 DPI.

    Relative paths are resolved against the project root so that notebooks
    always write outputs to the `results/` directory in the repository root.
    """

    output = resolve_path(path)
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


def plot_residual_error_distributions(residuals: Array, epsilon: float, bins: int = 30) -> None:
    """Plot residual error distributions for each component, norm, and empirical CDF."""
    if residuals.ndim != 2:
        raise ValueError("residuals must be a 2D array of shape (n_samples, n_components)")

    norms = np.linalg.norm(residuals, axis=1)
    components = residuals.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for idx in range(min(components, 2)):
        axes[idx].hist(residuals[:, idx], bins=bins, alpha=0.8, color="steelblue", edgecolor="white")
        axes[idx].axvline(0.0, color="black", linestyle=":", linewidth=1.5)
        axes[idx].set_xlabel(f"r{idx + 1}")
        axes[idx].set_ylabel("Count")
        axes[idx].set_title(f"Residual component r{idx + 1}")

    if components < 2:
        axes[1].axis("off")

    axes[2].hist(norms, bins=bins, alpha=0.8, color="darkorange", edgecolor="white")
    axes[2].axvline(epsilon, color="crimson", linestyle="--", linewidth=2, label=f"epsilon={epsilon:.3f}")
    axes[2].set_xlabel("||r||")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Residual norm distribution")
    axes[2].legend()

    sorted_norms = np.sort(norms)
    cdf = np.arange(1, len(sorted_norms) + 1, dtype=float) / len(sorted_norms)
    axes[3].plot(sorted_norms, cdf, color="seagreen", linewidth=2)
    axes[3].axvline(epsilon, color="crimson", linestyle="--", linewidth=2, label=f"epsilon={epsilon:.3f}")
    axes[3].set_xlabel("||r||")
    axes[3].set_ylabel("Empirical CDF")
    axes[3].set_title("Residual norm empirical CDF")
    axes[3].legend()

    fig.suptitle("Residual error distributions")


def plot_lyapunov_contours(xx: Array, yy: Array, vv: Array, levels: int = 15) -> None:
    """Contour plot for Lyapunov level sets."""
    contours = plt.contour(xx, yy, vv, levels=levels, cmap="viridis")
    plt.clabel(contours, inline=True, fontsize=8)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Lyapunov level sets")
