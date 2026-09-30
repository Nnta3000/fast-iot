from __future__ import annotations

import numpy as np

from iot import SimplexTheta


def test_simplex_sample_lies_in_the_set() -> None:
    rng = np.random.default_rng(0)
    simplex = SimplexTheta(4)
    theta = simplex.sample(rng)
    assert theta.shape == (4,)
    assert simplex.contains(theta)
    assert abs(float(theta.sum()) - 1.0) < 1e-12


def test_simplex_contains_rejects_outside_points() -> None:
    simplex = SimplexTheta(3)
    assert not simplex.contains(np.ones(3))  # sum is 3
    assert not simplex.contains(np.array([-0.1, 0.6, 0.5]))
    assert not simplex.contains(np.ones(4))  # wrong length
    assert simplex.contains(np.array([0.0, 1.0, 0.0]))


def test_simplex_linear_constraints_shape() -> None:
    constraints = SimplexTheta(3).linear_constraints()
    assert constraints.dimension == 3
    np.testing.assert_allclose(constraints.a_eq, np.ones((1, 3)))
    np.testing.assert_allclose(constraints.b_eq, np.ones(1))
    np.testing.assert_allclose(constraints.lower, np.zeros(3))
    assert np.all(np.isinf(constraints.upper))
    assert constraints.bounds() == [(0.0, float("inf"))] * 3


def test_simplex_rejects_invalid_dimension() -> None:
    import pytest

    with pytest.raises(ValueError, match=">= 1"):
        SimplexTheta(0)
