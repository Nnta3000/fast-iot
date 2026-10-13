"""
CN: `iot.cutting_plane` 的不变量测试：割有效性（theta* 不被割掉）、master 的 LP 定义、
    终止证书、投影/确定性/注入求解器，以及对非线性 Theta 的显式拒绝。
EN: Invariant tests for `iot.cutting_plane`: cut validity (theta* is never cut off), the
    master's LP definition, the termination certificate, projection/determinism/injected
    solvers, and the explicit rejection of a non-linear Theta.
"""

from __future__ import annotations

import numpy as np
import ot
import pytest
from scipy import sparse

from iot import (
    BilinearPSDFeatures,
    BilinearPSDTheta,
    CuttingPlaneConfig,
    DiagonalWeightFeatures,
    InverseOTError,
    InverseOTProblem,
    SimplexTheta,
    dense_cost_matrix,
    is_optimal_plan,
    solve_cut_master,
    solve_cutting_plane,
)
from iot.subgradient import make_pot_solver, observed_feature_moment, support_moment


def _problem(
    n: int = 24, k: int = 3, seed: int = 0
) -> tuple[InverseOTProblem, np.ndarray]:
    """CN: 与 iot_subgradient_test 相同的小实例构造。EN: Same small construction as iot_subgradient_test."""
    rng = np.random.default_rng(int(seed))
    theta_set = SimplexTheta(int(k))
    theta_star = theta_set.sample(rng)
    source = rng.normal(size=(int(n), int(k)))
    target = rng.normal(size=(int(n), int(k)))
    features = DiagonalWeightFeatures(source, target)
    cost = dense_cost_matrix(features, theta_star)
    mass = np.full(int(n), 1.0 / float(n))
    plan = ot.emd(mass, mass, cost, numItermax=2_000_000)
    return InverseOTProblem(features, theta_set, sparse.csr_matrix(plan)), theta_star


def test_reaches_optimal_plan_and_simplex() -> None:
    problem, _ = _problem(n=30, k=3, seed=0)
    result = solve_cutting_plane(
        problem, config=CuttingPlaneConfig(max_iterations=200, tol=1e-9)
    )
    assert result.converged
    assert result.final_gap <= 1e-9
    assert problem.theta_set.contains(result.theta, tol=1e-9)
    assert is_optimal_plan(problem, result.theta).is_optimal
    assert result.n_oracle_calls == len(result.iterations)
    assert len(result.cuts) == result.n_oracle_calls - 1


@pytest.mark.parametrize("mode", ["max_margin", "kelley"])
def test_master_modes_reach_optimal_plan(mode: str) -> None:
    problem, _ = _problem(n=24, k=3, seed=1)
    result = solve_cutting_plane(
        problem, config=CuttingPlaneConfig(master_mode=mode, max_iterations=300, tol=1e-9)  # type: ignore[arg-type]
    )
    assert result.converged
    assert is_optimal_plan(problem, result.theta).is_optimal


def test_ground_truth_is_never_cut_off() -> None:
    # CN: 每个 OT 方案给出的 g 都是 L0 的次梯度，因此可容许的 theta* 必满足所有割。
    # EN: Every g from an OT plan is a subgradient of L0, so the admissible theta* satisfies all cuts.
    problem, theta_star = _problem(n=28, k=3, seed=2)
    result = solve_cutting_plane(
        problem, config=CuttingPlaneConfig(max_iterations=200, tol=1e-9)
    )
    assert len(result.cuts) >= 1
    for cut in result.cuts:
        assert float(cut.gradient @ theta_star) <= 1e-9
        # The gradient is exactly the moment difference at the queried point.
        plan = make_pot_solver(problem)(cut.theta)
        expected = observed_feature_moment(problem) - support_moment(problem.features, plan)
        np.testing.assert_allclose(cut.gradient, expected, atol=1e-10)
        assert cut.gap == pytest.approx(float(cut.theta @ cut.gradient))


def test_master_solution_satisfies_its_cut_bank() -> None:
    problem, _ = _problem(n=24, k=3, seed=3)
    result = solve_cutting_plane(
        problem, config=CuttingPlaneConfig(max_iterations=60, tol=-1.0)
    )
    # CN: 轨迹中每条记录都对应一次真实 OT 查询，割按此顺序加入银行。
    # EN: Each record is a real OT query and cuts are added in that order.
    assert len(result.cuts) == len(result.iterations)
    matrix = np.vstack([cut.gradient for cut in result.cuts])
    theta = result.theta
    assert problem.theta_set.contains(theta, tol=1e-9)
    for row in range(matrix.shape[0]):
        assert float(matrix[row] @ theta) <= 1e-7
    # CN: 银行非空时，任意时刻的 master 解都应满足当时已加入的割。
    # EN: With a non-empty bank, every master solution satisfies the cuts present at that time.
    for step in range(1, len(result.cuts) + 1):
        master = solve_cut_master(
            matrix[:step], theta_set=problem.theta_set, mode="max_margin"
        )
        assert problem.theta_set.contains(master.theta, tol=1e-8)
        for row in range(step):
            assert float(matrix[row] @ master.theta) <= 1e-7


def _feasible_gradients(rng: np.random.Generator, n_cuts: int, dim: int) -> np.ndarray:
    """CN: 构造在均匀点上严格满足的割（<g, 1/dim> < 0），保证 max_margin master 可行。
    EN: Build cuts strictly satisfied at the uniform point (<g, 1/dim> < 0), so the max_margin
        master is feasible."""
    raw = rng.normal(size=(int(n_cuts), int(dim)))
    shift = raw.mean(axis=1, keepdims=True) + 1.0
    return raw - shift


def test_max_margin_and_kelley_agree_by_definition() -> None:
    # CN: max_margin 的 m* = max(0, -min_theta max_s <g_s, theta>) = max(0, -kelley 模型值)。
    # EN: max_margin's m* = max(0, -min_theta max_s <g_s, theta>) = max(0, -kelley model value).
    rng = np.random.default_rng(7)
    gradients = _feasible_gradients(rng, 6, 4)
    theta_set = SimplexTheta(4)
    max_margin = solve_cut_master(gradients, theta_set=theta_set, mode="max_margin")
    kelley = solve_cut_master(gradients, theta_set=theta_set, mode="kelley")
    assert max_margin.margin == pytest.approx(max(0.0, -kelley.model_value), abs=1e-8)


def test_normalized_master_returns_unit_scale_margin() -> None:
    rng = np.random.default_rng(11)
    gradients = _feasible_gradients(rng, 5, 4) * np.array([[1.0], [100.0], [0.01], [3.0], [0.2]])
    theta_set = SimplexTheta(4)
    master = solve_cut_master(
        gradients, theta_set=theta_set, mode="max_margin", normalize_cuts=True
    )
    assert master.margin >= 0.0
    assert theta_set.contains(master.theta, tol=1e-8)
    unit = gradients / np.linalg.norm(gradients, axis=1)[:, None]
    for row in range(unit.shape[0]):
        assert float(unit[row] @ master.theta) <= 1e-7


def test_zero_gradient_cuts_are_ignored() -> None:
    # CN: 恒真割 0·theta<=0 不应把 margin 压成 0，也不应让 master 失败。
    # EN: The always-true cut 0 theta <= 0 must not force margin to 0 nor fail the master.
    gradients = np.vstack([np.full(4, -1.0), np.zeros(4), np.full(4, -2.0)])
    master = solve_cut_master(gradients, theta_set=SimplexTheta(4), mode="max_margin")
    assert master.margin > 0.0
    with pytest.raises(InverseOTError):
        solve_cut_master(np.zeros((3, 4)), theta_set=SimplexTheta(4), mode="max_margin")


def test_learner_is_deterministic() -> None:
    problem, _ = _problem(n=20, k=3, seed=5)
    config = CuttingPlaneConfig(max_iterations=80, tol=1e-9)
    first = solve_cutting_plane(problem, config=config)
    second = solve_cutting_plane(problem, config=config)
    np.testing.assert_array_equal(first.theta, second.theta)
    assert [item.gap for item in first.iterations] == [item.gap for item in second.iterations]
    assert len(first.cuts) == len(second.cuts)


def test_injected_solver_is_used_once_per_oracle_call() -> None:
    problem, _ = _problem(n=16, k=3, seed=6)
    calls: list[int] = []
    pot = make_pot_solver(problem)

    def solver(theta: np.ndarray) -> sparse.spmatrix:
        calls.append(int(theta.size))
        return pot(theta)

    result = solve_cutting_plane(
        problem,
        config=CuttingPlaneConfig(max_iterations=5, tol=-1.0),
        exact_plan_solver=solver,
    )
    assert len(calls) == result.n_oracle_calls == len(result.iterations) == 5


def test_best_gap_never_exceeds_final_gap() -> None:
    problem, _ = _problem(n=20, k=3, seed=7)
    result = solve_cutting_plane(
        problem, config=CuttingPlaneConfig(max_iterations=50, tol=-1.0)
    )
    assert result.best_gap <= result.final_gap + 1e-12
    np.testing.assert_array_equal(result.best_theta.shape, result.theta.shape)


def test_rejects_non_linear_theta() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(6, 2))
    target = rng.normal(size=(6, 2))
    features = BilinearPSDFeatures(source, target)
    theta_set = BilinearPSDTheta(2)
    plan = sparse.csr_matrix(np.full((6, 6), 1.0 / 36.0))
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=plan)
    with pytest.raises(InverseOTError):
        solve_cutting_plane(problem)


def test_bad_config_and_master_validation() -> None:
    with pytest.raises(ValueError):
        CuttingPlaneConfig(master_mode="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CuttingPlaneConfig(max_iterations=0)
    with pytest.raises(ValueError):
        solve_cut_master(np.zeros((0, 3)), theta_set=SimplexTheta(3))
    with pytest.raises(ValueError):
        solve_cut_master(np.ones((2, 3)), theta_set=SimplexTheta(2))
