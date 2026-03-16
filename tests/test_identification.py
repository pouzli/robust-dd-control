import numpy as np

from src.basis import get_basis
from src.identification import fit_identified_model, predict_vector_field


def test_least_squares_shape() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(50, 2))
    xdot = np.column_stack((-x[:, 0] + x[:, 1], -2.0 * x[:, 1]))
    basis_fn = get_basis("quadratic_with_constant")
    res = fit_identified_model(x, xdot, basis_fn=basis_fn, basis_name="quadratic_with_constant")
    pred = predict_vector_field(x, basis_fn, res.coefficients)
    assert res.coefficients.shape == (6, 2)
    assert pred.shape == xdot.shape
