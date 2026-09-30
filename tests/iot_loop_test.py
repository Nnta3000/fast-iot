from __future__ import annotations

import numpy as np
import pytest

from iot import (
    EdgeSet,
    InverseOTError,
    InverseOTProblem,
    LinearMasterSolver,
    RegularizedMasterSolver,
    is_optimal_plan,
    make_diagonal_grounded_instance,
    solve_inverse_ot,
)


def _all_edges(problem: InverseOTProblem) -> EdgeSet:
    m, n = int(problem.n_source), int(problem.n_target)
    rows = np.repeat(np.arange(m, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), m)
    return EdgeSet.from_arrays(rows, cols, n_target=n)


def test_loop_converges_to_an_optimal_theta() -> None:
    rng = np.random.default_rng(5)
    instance = make_diagonal_grounded_instance(n_source=12, n_target=12, n_features=3, rng=rng)
    result = solve_inverse_ot(instance.problem, tie_break="lex", max_iterations=60)
    assert result.converged
    assert result.is_optimal is True
    assert instance.problem.theta_set.contains(result.theta, tol=1e-6)
    assert result.delta >= -1e-9
    assert result.iterations[-1].violation_count == 0


def test_master_without_connectors_is_unbounded() -> None:
    rng = np.random.default_rng(6)
    instance = make_diagonal_grounded_instance(n_source=9, n_target=9, n_features=2, rng=rng)
    rows, cols, _values = instance.problem.support()
    support_only = EdgeSet.from_arrays(rows, cols, n_target=int(instance.problem.n_target))
    # CN: 只用 S 时 delta 无上界；连接边 R 是防止这一点所必需的。
    # EN: With only S, delta is unbounded; the connector set R is what prevents this.
    with pytest.raises(InverseOTError, match="infeasible or unbounded"):
        LinearMasterSolver().solve(instance.problem, support_only, tie_break="lex")


def test_loop_matches_full_master_and_uses_far_fewer_edges() -> None:
    rng = np.random.default_rng(7)
    instance = make_diagonal_grounded_instance(n_source=10, n_target=10, n_features=3, rng=rng)
    loop = solve_inverse_ot(instance.problem, tie_break="lex", max_iterations=80)
    full = LinearMasterSolver().solve(instance.problem, _all_edges(instance.problem), tie_break="lex")
    assert abs(loop.delta - full.delta) < 1e-6
    assert loop.support_size < int(instance.problem.n_source) * int(instance.problem.n_target)
    assert is_optimal_plan(instance.problem, loop.theta).is_optimal


def test_regularized_master_with_hysteresis_converges() -> None:
    rng = np.random.default_rng(9)
    instance = make_diagonal_grounded_instance(n_source=12, n_target=12, n_features=3, rng=rng)
    result = solve_inverse_ot(
        instance.problem,
        solver=RegularizedMasterSolver(epsilon=1e-4, random_seed=0),
        tie_break="none",
        max_iterations=120,
    )
    assert result.converged
    assert result.is_optimal is True
    assert result.final_max_violation <= 1e-9
