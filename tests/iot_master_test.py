from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from iot import (
    DiagonalWeightFeatures,
    EdgeSet,
    InverseOTError,
    InverseOTProblem,
    LinearMasterSolver,
    SimplexTheta,
    is_optimal_plan,
    make_diagonal_grounded_instance,
    mandatory_edges,
)

SOLVER = LinearMasterSolver()


def _all_edges(problem: InverseOTProblem) -> EdgeSet:
    m, n = int(problem.n_source), int(problem.n_target)
    rows = np.repeat(np.arange(m, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), m)
    return EdgeSet.from_arrays(rows, cols, n_target=n)


def _bad_plan_problem() -> InverseOTProblem:
    features = DiagonalWeightFeatures(np.array([[0.0], [1.0]]), np.array([[1.0], [0.0]]))
    return InverseOTProblem(
        features=features, theta_set=SimplexTheta(1), plan=sparse.csr_matrix(np.eye(2) * 0.5)
    )


def test_master_constraints_hold_on_mandatory_edges() -> None:
    rng = np.random.default_rng(1)
    instance = make_diagonal_grounded_instance(n_source=10, n_target=10, n_features=3, rng=rng)
    mandatory, _ = mandatory_edges(instance.problem)
    solution = SOLVER.solve(instance.problem, mandatory, tie_break="lex")
    assert solution.delta >= -1e-9
    assert instance.problem.theta_set.contains(solution.theta, tol=1e-6)

    support_keys = instance.problem.support_keys()
    in_support = np.isin(mandatory.keys, support_keys)
    costs = instance.problem.features.costs(solution.theta, mandatory.rows, mandatory.cols)
    residual = solution.source_dual[mandatory.rows] + solution.target_dual[mandatory.cols] - costs
    np.testing.assert_allclose(residual[in_support], 0.0, atol=1e-7)
    assert bool(np.all(residual[~in_support] <= -solution.delta + 1e-7))


def test_lemma_1_full_edge_master_certifies_optimality() -> None:
    rng = np.random.default_rng(2)
    instance = make_diagonal_grounded_instance(n_source=10, n_target=10, n_features=3, rng=rng)
    all_edges = _all_edges(instance.problem)
    solution = SOLVER.solve(instance.problem, all_edges, tie_break="lex")
    # CN: 全边 master 可行且 delta >= 0 意味着 pi_hat 最优（Lemma 1）。
    # EN: A full-edge master feasible with delta >= 0 implies pi_hat is optimal (Lemma 1).
    assert solution.delta >= -1e-9
    assert is_optimal_plan(instance.problem, solution.theta).is_optimal


def test_infeasible_observation_raises() -> None:
    problem = _bad_plan_problem()
    all_edges = _all_edges(problem)
    with pytest.raises(InverseOTError, match="infeasible or unbounded"):
        SOLVER.solve(problem, all_edges, tie_break="lex")


def test_lex_and_random_agree_on_max_margin_delta() -> None:
    rng = np.random.default_rng(3)
    instance = make_diagonal_grounded_instance(n_source=12, n_target=12, n_features=3, rng=rng)
    mandatory, _ = mandatory_edges(instance.problem)
    lex = SOLVER.solve(instance.problem, mandatory, tie_break="lex")
    randomized = LinearMasterSolver(random_seed=7).solve(instance.problem, mandatory, tie_break="random")
    # CN: 在同一（松弛）边集上，不同 tie-breaking 必须给出相同的 max-margin delta。
    # EN: On the same (relaxed) edge set, different tie-breaking must reach the same
    #     max-margin delta.
    assert abs(lex.delta - randomized.delta) < 1e-7
    assert instance.problem.theta_set.contains(lex.theta, tol=1e-6)
    assert instance.problem.theta_set.contains(randomized.theta, tol=1e-6)
