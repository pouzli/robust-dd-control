import numpy as np

from src.basis import get_basis
from src.identification import fit_identified_model, predict_vector_field
from src.lyapunov import numerical_jacobian, solve_lyapunov


def test_lyapunov_matrix_symmetric() -> None:
    a = np.array([[-1.0, 0.5], [0.0, -2.0]])
    q = np.eye(2)
    p = solve_lyapunov(a, q)
    assert np.allclose(p, p.T, atol=1e-10)


def test_jacobian_near_zero_for_polynomial_basis() -> None:
    x = np.array([[0.1, -0.2], [0.2, 0.3], [-0.1, 0.2], [0.0, 0.0]])
    xdot = np.column_stack((-x[:, 0] + x[:, 1] + 0.8 * x[:, 0] * x[:, 1], -2.0 * x[:, 1] + 0.6 * x[:, 0] ** 2))
    basis_fn = get_basis("quadratic_with_constant")
    res = fit_identified_model(x, xdot, basis_fn=basis_fn, basis_name="quadratic_with_constant")

    def fhat(state: np.ndarray) -> np.ndarray:
        return predict_vector_field(state[None, :], basis_fn, res.coefficients)[0]

    jac = numerical_jacobian(fhat, np.zeros(2))
    expected = np.array([[-1.0, 1.0], [0.0, -2.0]])
    assert np.allclose(jac, expected, atol=1e-2)
