"""
CN: `WarmStartMasterSolver`（增量热启动 HiGHS，新增选项）的正确性与通用性测试。
EN: Correctness and generality tests for `WarmStartMasterSolver` (incremental warm-started
    HiGHS, a new option).

CN: 它必须对任意 `LinearThetaSet` 成立（这里同时用 diagonal weights 与 d^2 维稠密特征族），
    并与整表重建的 `RegularizedMasterSolver` 给出相同的 delta 与证书。
EN: It must work for any `LinearThetaSet` (we use both diagonal weights and a d^2-dimensional
    dense feature family) and give the same delta and certificate as the tableau-rebuilding
    `RegularizedMasterSolver`.
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
    WarmStartMasterSolver,
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
    """CN: d^2 维稠密线性特征族（非 diagonal weights）的可行实例。"""
    rng = np.random.default_rng(seed)
    tensor = rng.random((m, n, k))
    features = DenseFeatureTensor(tensor)
    theta_set = SimplexTheta(k)
    theta_star = theta_set.sample(rng)
    cost = dense_cost_matrix(features, theta_star)
    plan = ot.emd(np.full(m, 1.0 / m), np.full(n, 1.0 / n), cost)
    return InverseOTProblem(
        features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan)
    )


def _assert_same_delta(problem: InverseOTProblem, edges: EdgeSet, *, tol: float = 1e-7) -> None:
    cold = RegularizedMasterSolver(epsilon=1e-4, random_seed=0).solve(
        problem, edges, tie_break="none"
    )
    warm = WarmStartMasterSolver(epsilon=1e-4, random_seed=0).solve(
        problem, edges, tie_break="none"
    )
    assert abs(cold.delta - warm.delta) <= tol
    # CN: S 上的等式必须成立（可行性与证书），不要求 (f,g) 逐位相同（退化顶点）。
    # EN: S equalities must hold (feasibility/certificate); (f,g) need not be bit-identical.
    in_support = np.isin(edges.keys, problem.support_keys())
    costs = problem.features.costs(warm.theta, edges.rows, edges.cols)
    residual = warm.source_dual[edges.rows] + warm.target_dual[edges.cols] - costs
    np.testing.assert_allclose(residual[in_support], 0.0, atol=1e-7)
    assert bool(np.all(residual[~in_support] <= -warm.delta + 1e-7))


def test_warmstart_matches_cold_on_mandatory_and_full_edges() -> None:
    rng = np.random.default_rng(31)
    instance = make_diagonal_grounded_instance(n_source=12, n_target=12, n_features=3, rng=rng)
    mandatory, _ = mandatory_edges(instance.problem)
    _assert_same_delta(instance.problem, mandatory)
    _assert_same_delta(instance.problem, _all_edges(instance.problem))


def test_warmstart_works_on_dense_d2_feature_family() -> None:
    # CN: d=3 的二次特征族 -> 6 维参数；证明它不是只为 diagonal weights 定制的。
    # EN: d=3 quadratic feature family -> 6 parameters; proves it is not diagonal-only.
    problem = _dense_quadratic_problem(8, 8, 6, seed=7)
    _assert_same_delta(problem, _all_edges(problem))
    warm = WarmStartMasterSolver(epsilon=1e-4, random_seed=0).solve(
        problem, _all_edges(problem), tie_break="none"
    )
    assert warm.delta >= -1e-9


def test_warmstart_incremental_sequence_matches_cold_each_step() -> None:
    rng = np.random.default_rng(32)
    instance = make_diagonal_grounded_instance(n_source=10, n_target=10, n_features=3, rng=rng)
    mandatory, _ = mandatory_edges(instance.problem)
    all_edges = _all_edges(instance.problem)
    warm_solver = WarmStartMasterSolver(epsilon=1e-4, random_seed=0)
    # CN: 递增的边集序列，模拟约束生成的多轮 F 变化，每轮与冷启一致。
    # EN: An increasing sequence of edge sets simulates the loop's changing F; each step must
    #     agree with a cold solve.
    for size in (mandatory.size, mandatory.size + 5, mandatory.size + 10, all_edges.size):
        subset = EdgeSet.from_arrays(
            all_edges.rows[:size], all_edges.cols[:size], n_target=int(instance.problem.n_target)
        ).union(mandatory)
        cold = RegularizedMasterSolver(epsilon=1e-4, random_seed=0).solve(
            instance.problem, subset, tie_break="none"
        )
        warm = warm_solver.solve(instance.problem, subset, tie_break="none")
        assert abs(cold.delta - warm.delta) <= 1e-7


def test_warmstart_drives_the_loop_to_an_optimal_theta() -> None:
    rng = np.random.default_rng(33)
    instance = make_diagonal_grounded_instance(n_source=12, n_target=12, n_features=3, rng=rng)
    result = solve_inverse_ot(
        instance.problem,
        solver=WarmStartMasterSolver(epsilon=1e-4, random_seed=0),
        tie_break="none",
        max_iterations=120,
    )
    assert result.converged
    assert result.is_optimal is True
    assert is_optimal_plan(instance.problem, result.theta).is_optimal
