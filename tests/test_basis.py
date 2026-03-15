import numpy as np

from src.basis import build_theta, get_basis, quadratic_with_constant_basis


def test_quadratic_with_constant_shape() -> None:
    x = np.array([1.0, -2.0])
    phi = quadratic_with_constant_basis(x)
    assert phi.shape == (6,)


def test_build_theta_shape() -> None:
    x_samples = np.array([[0.0, 1.0], [1.0, 2.0], [-1.0, 0.5]])
    theta = build_theta(x_samples, get_basis("quadratic_with_constant"))
    assert theta.shape == (3, 6)
