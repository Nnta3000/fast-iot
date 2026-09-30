from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from iot import (
    DenseFeatureTensor,
    DiagonalWeightFeatures,
    InverseOTError,
    InverseOTProblem,
    SimplexTheta,
    dual_optimality_certificate,
    fixed_dual_theta_ranges,
    is_optimal_plan,
    make_diagonal_grounded_instance,
    plan_cost,
)


def _two_point_problem(plan: np.ndarray) -> InverseOTProblem:
    # CN: 一维 x=[0,1], y=[1,0]；theta=1 时代价是 [[1,0],[0,1]]，反角线最优。
    # EN: 1D x=[0,1], y=[1,0]; at theta=1 the cost is [[1,0],[0,1]] and the
    #     anti-diagonal is optimal.
    features = DiagonalWeightFeatures(np.array([[0.0], [1.0]]), np.array([[1.0], [0.0]]))
    return InverseOTProblem(features=features, theta_set=SimplexTheta(1), plan=sparse.csr_matrix(plan))


def _identified_problem() -> tuple[InverseOTProblem, np.ndarray, np.ndarray, np.ndarray]:
    # CN: 构造一个固定对偶势把 theta 唯一确定的实例。
    # EN: Construct an instance whose fixed dual pair pins theta uniquely.
    tensor = np.zeros((2, 2, 2), dtype=np.float64)
    tensor[0, 0] = [1.0, 0.0]  # c_00 = theta_0
    tensor[1, 1] = [0.0, 1.0]  # c_11 = theta_1
    tensor[0, 1] = [5.0, 0.0]  # off-support, large
    tensor[1, 0] = [0.0, 5.0]  # off-support, large
    features = DenseFeatureTensor(tensor)
    plan = sparse.csr_matrix(np.diag([0.5, 0.5]))
    problem = InverseOTProblem(features=features, theta_set=SimplexTheta(2), plan=plan)
    theta_star = np.array([0.3, 0.7])
    source_dual = np.array([0.3, 0.7])
    target_dual = np.array([0.0, 0.0])
    return problem, theta_star, source_dual, target_dual


def test_is_optimal_plan_separates_optimal_and_suboptimal_plans() -> None:
    theta = np.array([1.0])
    optimal_plan = np.array([[0.0, 0.5], [0.5, 0.0]])
    identity_plan = np.array([[0.5, 0.0], [0.0, 0.5]])

    optimal = is_optimal_plan(_two_point_problem(optimal_plan), theta)
    suboptimal = is_optimal_plan(_two_point_problem(identity_plan), theta)

    assert optimal.is_optimal
    assert abs(optimal.relative_gap) < 1e-12
    assert not suboptimal.is_optimal
    assert suboptimal.relative_gap > 1e-6


def test_plan_cost_matches_explicit_matrix() -> None:
    problem = _two_point_problem(np.array([[0.0, 0.5], [0.5, 0.0]]))
    assert abs(plan_cost(problem, np.array([1.0]))) < 1e-12


def test_is_optimal_plan_rejects_theta_outside_theta_set() -> None:
    problem = _two_point_problem(np.array([[0.0, 0.5], [0.5, 0.0]]))
    with pytest.raises(InverseOTError, match="inside the parameter set"):
        is_optimal_plan(problem, np.array([0.0]))


def test_identified_instance_is_optimal_at_theta_star() -> None:
    problem, theta_star, _source_dual, _target_dual = _identified_problem()
    check = is_optimal_plan(problem, theta_star)
    assert check.is_optimal
    assert abs(check.absolute_gap) < 1e-12


def test_fixed_dual_region_is_pinned_by_identifying_equalities() -> None:
    problem, theta_star, source_dual, target_dual = _identified_problem()
    theta_range = fixed_dual_theta_ranges(
        problem, source_dual, target_dual, equality_tol=0.0, feasibility_tol=0.0
    )
    np.testing.assert_allclose(theta_range.lower, theta_star, atol=1e-7)
    np.testing.assert_allclose(theta_range.upper, theta_star, atol=1e-7)
    assert theta_range.contains(theta_star)
    assert float(theta_range.width().max()) < 1e-7


def test_is_optimal_plan_raises_when_pot_hits_iteration_cap() -> None:
    # CN: 用极小的 num_itermax 模拟「POT 默认上限被截断」；此时必须抛错而不是静默给出
    #     被截断的“最优”代价。真实 n=3200 上 POT 默认上限确实会这样失败。
    # EN: Use a tiny num_itermax to emulate "POT's default cap is hit"; this must raise
    #     rather than silently return an iteration-capped "optimal" cost. Real n=3200
    #     instances do hit the default cap this way.
    rng = np.random.default_rng(21)
    instance = make_diagonal_grounded_instance(
        n_source=64, n_target=64, n_features=3, rng=rng
    )
    with pytest.raises(InverseOTError, match="did not reach optimality"), pytest.warns(
        UserWarning, match="numItermax"
    ):
        is_optimal_plan(instance.problem, instance.theta_star, num_itermax=1)
    good = is_optimal_plan(instance.problem, instance.theta_star, num_itermax=2_000_000)
    assert good.is_optimal


def test_dual_certificate_detects_capped_duals() -> None:
    rng = np.random.default_rng(22)
    instance = make_diagonal_grounded_instance(
        n_source=64, n_target=64, n_features=3, rng=rng
    )
    good = dual_optimality_certificate(
        instance.problem, instance.theta_star, num_itermax=2_000_000
    )
    assert good.certified
    assert good.max_dual_violation <= 1e-6
    assert good.support_residual <= 1e-6

    with pytest.warns(UserWarning, match="numItermax"):
        capped = dual_optimality_certificate(
            instance.problem, instance.theta_star, num_itermax=1
        )
    # CN: 被截断的 (u, v) 不再对偶可行，证书必须失败（且不依赖 result_code）。
    # EN: The capped (u, v) are no longer dual feasible, so the certificate must fail
    #     (and does not rely on result_code).
    assert not capped.certified
    assert capped.max_dual_violation > 1e-3


def test_dual_certificate_rejects_suboptimal_plan() -> None:
    # CN: theta=1 时最优是反对角线；单位矩阵方案的支撑与对偶最优面不相交，证书失败。
    # EN: At theta=1 the anti-diagonal is optimal; the identity plan's support is not in
    #     the dual-optimal face, so the certificate fails.
    theta = np.array([1.0])
    optimal_plan = np.array([[0.0, 0.5], [0.5, 0.0]])
    identity_plan = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert dual_optimality_certificate(_two_point_problem(optimal_plan), theta).certified
    assert not dual_optimality_certificate(_two_point_problem(identity_plan), theta).certified
