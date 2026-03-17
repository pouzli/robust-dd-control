import numpy as np

from src.basis import basis_registry, build_theta, get_basis, quadratic_with_constant_basis


def test_quadratic_with_constant_shape() -> None:
    x = np.array([1.0, -2.0])
    phi = quadratic_with_constant_basis(x)
    assert phi.shape == (6,)


def test_build_theta_shape() -> None:
    x_samples = np.array([[0.0, 1.0], [1.0, 2.0], [-1.0, 0.5]])
    theta = build_theta(x_samples, get_basis("quadratic_with_constant"))
    assert theta.shape == (3, 6)


def test_new_basis_registry_entries_and_shapes() -> None:
    x = np.array([0.2, -0.4], dtype=float)
    reg = basis_registry()
    for required in [
        "linear",
        "linear_with_constant",
        "quadratic_full",
        "reduced_quadratic_no_cross",
        "reduced_cross_only",
    ]:
        assert required in reg

    assert get_basis("linear")(x).shape == (2,)
    assert get_basis("linear_with_constant")(x).shape == (3,)
    assert get_basis("quadratic_full")(x).shape == (6,)
    assert get_basis("reduced_quadratic_no_cross")(x).shape == (5,)
    assert get_basis("reduced_cross_only")(x).shape == (4,)
