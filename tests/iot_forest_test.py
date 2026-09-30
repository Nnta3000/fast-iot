"""
CN: `SpanningForestMasterSolver`（f,g 消元，新增选项）的正确性与通用性测试。
EN: Correctness and generality tests for `SpanningForestMasterSolver` (f,g elimination, a new
    option).

CN: 它必须与整表 master 给出相同的 delta 与 theta（约化是 (delta,theta) 投影不变的可逆仿射
    重参数化），并在任意 `LinearThetaSet`（含 d^2 稠密特征族）上成立。
EN: It must agree with the full-tableau master in delta and theta (the reduction is an
    invertible affine reparameterisation with an unchanged (delta, theta) projection), and work
    for any `LinearThetaSet` (including a d^2 dense feature family).
"""

from __future__ import annotations

import numpy as np
import ot
import pytest
from scipy import sparse

from iot import (
    DenseFeatureTensor,
    EdgeSet,
    InverseOTProblem,
    RegularizedMasterSolver,
    SimplexTheta,
    SpanningForestMasterSolver,
    build_forest_representation,
    dense_cost_matrix,
    is_optimal_plan,
    make_diagonal_grounded_instance,
    mandatory_edges,
    solve_inverse_ot,
)


def _all_edges(problem: InverseOTProblem) -> EdgeSet:
    m, n = int(problem.n_source), int(problem.n_target)
    rows = np.repeat(np.arange(m, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), m)
    return EdgeSet.from_arrays(rows, cols, n_target=n)


def _dense_quadratic_problem(m: int, n: int, k: int, seed: int) -> InverseOTProblem:
    rng = np.random.default_rng(seed)
    features = DenseFeatureTensor(rng.random((m, n, k)))
    theta_set = SimplexTheta(k)
    theta_star = theta_set.sample(rng)
    cost = dense_cost_matrix(features, theta_star)
    plan = ot.emd(np.full(m, 1.0 / m), np.full(n, 1.0 / n), cost)
    return InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))


def _assert_matches_full_master(problem: InverseOTProblem, edges: EdgeSet) -> None:
    cold = RegularizedMasterSolver(epsilon=1e-4, random_seed=0).solve(
        problem, edges, tie_break="none"
    )
    forest = SpanningForestMasterSolver(epsilon=1e-4, random_seed=0).solve(
        problem, edges, tie_break="none"
    )
    assert abs(cold.delta - forest.delta) <= 1e-7
    np.testing.assert_allclose(cold.theta, forest.theta, atol=1e-6)
    # CN: 重构的 (f,g) 必须满足 S 上的等式与 S 外的不等式（证书）。
    # EN: The reconstructed (f,g) must satisfy the S equalities and off-S inequalities.
    in_support = np.isin(edges.keys, problem.support_keys())
    costs = problem.features.costs(forest.theta, edges.rows, edges.cols)
    residual = forest.source_dual[edges.rows] + forest.target_dual[edges.cols] - costs
    np.testing.assert_allclose(residual[in_support], 0.0, atol=1e-7)
    assert bool(np.all(residual[~in_support] <= -forest.delta + 1e-7))


def test_forest_matches_full_master_on_diagonal() -> None:
    rng = np.random.default_rng(41)
    instance = make_diagonal_grounded_instance(n_source=12, n_target=12, n_features=3, rng=rng)
    mandatory, _ = mandatory_edges(instance.problem)
    _assert_matches_full_master(instance.problem, mandatory)
    _assert_matches_full_master(instance.problem, _all_edges(instance.problem))


def test_forest_matches_full_master_on_dense_d2_family() -> None:
    # CN: d=3 -> 6 维二次特征族；证明它不是只为 diagonal weights 定制的。
    # EN: d=3 -> 6-dimensional quadratic family; proves it is not diagonal-only.
    problem = _dense_quadratic_problem(8, 8, 6, seed=11)
    _assert_matches_full_master(problem, _all_edges(problem))


def test_forest_representation_is_exact_on_support() -> None:
    rng = np.random.default_rng(42)
    instance = make_diagonal_grounded_instance(n_source=10, n_target=10, n_features=4, rng=rng)
    representation = build_forest_representation(instance.problem)
    rows, cols, _ = instance.problem.support()
    theta = instance.theta_star
    tau = np.arange(representation.n_components, dtype=np.float64) * 0.1
    source = representation.coefficients[rows] @ theta + representation.signs[rows] * tau[
        representation.component[rows]
    ]
    target = representation.coefficients[int(instance.problem.n_source) + cols] @ theta + (
        representation.signs[int(instance.problem.n_source) + cols]
        * tau[representation.component[int(instance.problem.n_source) + cols]]
    )
    costs = instance.problem.features.costs(theta, rows, cols)
    np.testing.assert_allclose(source + target, costs, atol=1e-9)


def test_forest_drives_the_loop_to_an_optimal_theta() -> None:
    rng = np.random.default_rng(43)
    instance = make_diagonal_grounded_instance(n_source=12, n_target=12, n_features=3, rng=rng)
    result = solve_inverse_ot(
        instance.problem,
        solver=SpanningForestMasterSolver(epsilon=1e-4, random_seed=0),
        tie_break="none",
        max_iterations=150,
    )
    assert result.converged
    assert result.is_optimal is True
    assert is_optimal_plan(instance.problem, result.theta).is_optimal
