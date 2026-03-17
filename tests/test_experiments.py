import numpy as np

from src.experiments import add_gaussian_noise, compute_tail_metrics, residual_statistics, summarize_tail_metrics


def test_noise_reproducibility() -> None:
    x = np.ones((4, 2), dtype=float)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    n1 = add_gaussian_noise(x, 0.05, rng1)
    n2 = add_gaussian_noise(x, 0.05, rng2)
    assert np.allclose(n1, n2)


def test_residual_statistics_keys() -> None:
    residuals = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 2.0]], dtype=float)
    stats = residual_statistics(residuals)
    required = {
        "max_residual_norm",
        "mean_residual_norm",
        "median_residual_norm",
        "q90_residual_norm",
        "q95_residual_norm",
        "q99_residual_norm",
    }
    assert required.issubset(stats.keys())


def test_tail_summary_structure() -> None:
    states = np.column_stack([np.linspace(1.0, 0.2, 100), np.linspace(0.5, 0.1, 100)])
    tm = compute_tail_metrics(states)
    summary = summarize_tail_metrics([tm, tm])
    assert summary["mean_tail_radius"] > 0
    assert summary["max_tail_radius"] > 0
    assert summary["ultimate_radius_estimate"] > 0
