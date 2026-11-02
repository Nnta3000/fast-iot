from __future__ import annotations

import numpy as np

from iot import make_diagonal_grounded_instance, optimal_theta_ranges, solve_inverse_ot
from iot.core.parameters import ExactObjective, SolverParameters


def test_optimal_theta_ranges_contain_theta_star_and_max_margin_theta() -> None:
    rng = np.random.default_rng(11)
    instance = make_diagonal_grounded_instance(n_source=12, n_target=12, n_features=3, rng=rng)
    theta_range = optimal_theta_ranges(instance.problem)
    assert theta_range.dimension == int(instance.problem.n_features)
    assert theta_range.contains(instance.theta_star)
    assert bool(np.all(theta_range.lower <= theta_range.upper + 1e-12))

    result = solve_inverse_ot(instance.problem, params=SolverParameters(max_iterations=80), objective=ExactObjective(tie_break="lex"))
    assert theta_range.contains(result.theta)
    assert result.is_optimal is True
