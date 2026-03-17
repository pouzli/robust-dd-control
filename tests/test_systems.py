import numpy as np

from src.systems import get_example_system


def test_get_example_system_names() -> None:
    names = [
        "linear_damped",
        "quadratic_oscillator",
        "van_der_pol",
        "cross_coupled",
        "saturated",
    ]
    for name in names:
        fn = get_example_system(name)
        xdot = fn(0.0, np.array([0.1, -0.2], dtype=float))
        assert xdot.shape == (2,)


def test_get_example_system_bad_name() -> None:
    try:
        get_example_system("unknown_system")
        assert False, "Expected ValueError for unknown system name"
    except ValueError:
        pass
