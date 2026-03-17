"""Run full pipeline for a set of example nonlinear dynamical systems.

This script implements the same steps as the notebooks, but loops over a set of
predefined example right-hand sides (the five examples from the report).

Usage:
    python scripts/run_all_examples.py  # runs all examples
    python scripts/run_all_examples.py --examples cross_coupled van_der_pol
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.basis import get_basis
from src.control import candidate_gains
from src.identification import fit_identified_model, predict_vector_field
from src.lyapunov import numerical_jacobian, solve_lyapunov
from src.plots import (
    plot_phase_trajectories,
    plot_residual_error_distributions,
    plot_residual_histogram,
    save_figure,
    set_plot_style,
)
from src.simulation import simulate_batch, uncertain_dynamics
from src.uncertainty import (
    bounded_disturbance,
    compute_residuals,
    estimate_epsilon,
    residual_norms,
)
from src.systems import get_example_system
from src.utils import ensure_dir, save_dataframe, save_json, set_seed


def run_example(example_name: str, root_dir: Path) -> None:
    """Run the full data-driven identification and robustness pipeline for one example."""

    print(f"\n=== Running example: {example_name} ===")
    set_seed(42)
    set_plot_style()

    data_dir = ensure_dir(root_dir / "data" / "processed")
    results_dir = ensure_dir(root_dir / "results" / example_name)
    figs_dir = ensure_dir(results_dir / "figures")
    metrics_dir = ensure_dir(results_dir / "metrics")

    dataset_path = data_dir / f"{example_name}_dataset.csv"

    dynamics = get_example_system(example_name)

    # 1) Generate training data
    t_eval = np.linspace(0.0, 10.0, 400)
    initials = np.array(
        [[-1.2, -0.8], [-1.0, 0.7], [-0.6, 1.1], [0.5, -1.0], [1.0, 0.9], [1.3, -0.4]],
        dtype=float,
    )

    traj_true = simulate_batch(dynamics, initials, (0.0, 10.0), t_eval)
    x = np.vstack([states for _, states in traj_true])
    xdot = np.vstack(
        [
            np.array([dynamics(float(t), s) for t, s in zip(ts, states)])
            for ts, states in traj_true
        ]
    )

    save_dataframe(
        pd.DataFrame(np.hstack([x, xdot]), columns=["x1", "x2", "xdot1", "xdot2"]),
        dataset_path,
    )

    # 2) Identification
    basis_fn = get_basis("quadratic_with_constant")
    model = fit_identified_model(x, xdot, basis_fn=basis_fn, basis_name="quadratic_with_constant")
    xdot_hat = predict_vector_field(x, basis_fn, model.coefficients)
    res = compute_residuals(xdot, xdot_hat)
    eps = estimate_epsilon(residual_norms(res), q=0.95)

    metrics = {
        "example": example_name,
        "epsilon": float(eps),
        "residual_mean": float(np.mean(residual_norms(res))),
        "residual_max": float(np.max(residual_norms(res))),
    }

    # 3) Lyapunov / uncertain simulation
    fhat = lambda t, x: predict_vector_field(x[None, :], basis_fn, model.coefficients)[0]

    A = numerical_jacobian(lambda x_: fhat(0.0, x_), np.zeros(2))
    P = solve_lyapunov(A, np.eye(2))

    unc = uncertain_dynamics(fhat, lambda t: bounded_disturbance(t, eps))
    traj_unc = simulate_batch(unc, initials, (0.0, 12.0), np.linspace(0, 12, 500))

    # 4) Closed-loop simulation
    B = np.array([[0.0], [1.0]])
    K = candidate_gains()["medium"]
    cl = lambda t, x: dynamics(t, x) + (B @ (K @ x)).reshape(-1) + bounded_disturbance(t, eps)
    traj_cl = simulate_batch(cl, initials, (0.0, 12.0), np.linspace(0, 12, 500))

    # Save results
    save_json(metrics, metrics_dir / "summary.json")

    plt = __import__("matplotlib.pyplot").pyplot
    plt.figure()
    plot_residual_histogram(residual_norms(res), eps)
    save_figure(figs_dir / "residual_hist.png")
    plt.close()

    plt.figure()
    plot_residual_error_distributions(res, eps)
    save_figure(figs_dir / "residual_error_distributions.png")
    plt.close()

    plt.figure()
    plot_phase_trajectories(traj_true, "True system")
    save_figure(figs_dir / "phase_true.png")
    plt.close()

    plt.figure()
    plot_phase_trajectories(traj_unc, "Uncertain uncontrolled")
    save_figure(figs_dir / "uncertain_uncontrolled.png")
    plt.close()

    plt.figure()
    plot_phase_trajectories(traj_cl, "Uncertain controlled")
    save_figure(figs_dir / "uncertain_controlled.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline for a set of example dynamical systems.")
    parser.add_argument(
        "--examples",
        nargs="*",
        default=[
            "linear_damped",
            "quadratic_oscillator",
            "van_der_pol",
            "cross_coupled",
            "saturated",
        ],
        help="List of example names to run.",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    for name in args.examples:
        run_example(name, root_dir)


if __name__ == "__main__":
    main()
