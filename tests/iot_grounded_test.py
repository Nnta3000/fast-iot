from __future__ import annotations

import numpy as np

from iot import (
    fixed_dual_theta_ranges,
    is_optimal_plan,
    make_diagonal_grounded_instance,
)
from iot.generators.grounded import GroundedInstance


def test_grounded_diagonal_instance_is_consistent(simplex_instance: GroundedInstance) -> None:
    instance = simplex_instance
    problem = instance.problem
    assert problem.n_features == 3
    assert problem.support_size >= 3
    assert problem.theta_set.contains(instance.theta_star)

    theta_range = fixed_dual_theta_ranges(problem, instance.source_dual, instance.target_dual)
    assert theta_range.contains(instance.theta_star)

    check = is_optimal_plan(problem, instance.theta_star)
    assert check.is_optimal
    assert abs(check.relative_gap) < 1e-9


def test_grounded_instance_accepts_explicit_theta_star() -> None:
    rng = np.random.default_rng(4)
    theta_star = np.array([0.1, 0.6, 0.3])
    instance = make_diagonal_grounded_instance(
        n_source=10,
        n_target=11,
        n_features=3,
        rng=rng,
        theta_star=theta_star,
    )
    np.testing.assert_allclose(instance.theta_star, theta_star)
    assert is_optimal_plan(instance.problem, instance.theta_star).is_optimal
