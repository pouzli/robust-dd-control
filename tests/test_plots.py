import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.plots import plot_residual_error_distributions


def test_plot_residual_error_distributions_creates_four_axes() -> None:
    residuals = np.array([[0.1, -0.2], [0.3, 0.0], [-0.1, 0.4]], dtype=float)
    plot_residual_error_distributions(residuals, epsilon=0.35, bins=5)
    fig = plt.gcf()
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_residual_error_distributions_rejects_non_matrix_input() -> None:
    with pytest.raises(ValueError, match="residuals must be a 2D array"):
        plot_residual_error_distributions(np.array([0.1, 0.2, 0.3]), epsilon=0.2)
