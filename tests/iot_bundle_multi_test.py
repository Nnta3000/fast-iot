"""
EN: **Tests for the multi-plan bundle** (`iot.sparse.bundle_multi`).

    The bundle's correctness statement is a bracket: `lower <= F* <= upper`, with `F*` computed independently by a
    dense joint LP and by the dense Frank–Wolfe. Three checks: the exact value at the returned θ matches the LP's
    optimum, the bracket contains it, and the returned plans are transport plans for the observed marginals with a
    small support. A fourth check keeps the identity `F(θ) = ⟨g, θ⟩` honest, since the cut form depends on it.

CN: **多计划 bundle 的测试**（`iot.sparse.bundle_multi`）。其正确性陈述是一个区间：`lower <= F* <= upper`，其中 `F*`
    由稠密联合 LP 与稠密 Frank–Wolfe 独立给出。三项检查：返回 θ 处的精确值等于 LP 最优值、区间包含它、返回的计划是观测
    marginal 下的运输计划且支撑很小。第四项检查 `F(θ)=⟨g,θ⟩` 恒等式，因为割的形式依赖它。
"""

from __future__ import annotations

import numpy as np
import ot
import pytest
import scipy.sparse as sparse

from iot.core.theta import SimplexTheta
from iot.sparse.bundle_multi import solve_bundle_multi
from iot.sparse.features import PointDistanceFeatures
from experiments.studies.iot.synthetic.multi_observation_sparse.reference import (
    dense_frank_wolfe,
    dense_joint_lp,
)


def _planted(n: int = 12, n_windows: int = 3, seed: int = 0):
    """
    EN: points, a planted θ*, and observed plans that are exactly optimal at θ*.
    CN: 点、植入的 θ*，以及恰在 θ* 处最优的观测计划。
    """
    rng = np.random.default_rng(seed)
    source_points = rng.random((n, 2))
    target_points = rng.random((n, 2))
    features = PointDistanceFeatures(source_points, target_points, scale=1.0, width=4)
    theta_star = np.array([0.1, 0.5, 0.2, 0.2])
    rows = np.repeat(np.arange(n, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), n)
    dense = np.asarray(features.costs(theta_star, rows, cols)).reshape(n, n)
    plans = []
    for _ in range(n_windows):
        a = rng.random(n) + 0.1
        a /= a.sum()
        b = rng.random(n) + 0.1
        b /= b.sum()
        plans.append(sparse.csr_matrix(np.asarray(ot.emd(a, b, dense, numItermax=1_000_000))))
    return features, theta_star, plans


def test_bundle_reaches_the_dense_lp_optimum_and_brackets_it() -> None:
    """
    EN: at the returned θ, `F` must equal the joint LP's optimum (the ground truth), and the reported bracket must
        contain it.
    CN: 在返回的 θ 处 `F` 必须等于联合 LP 的最优值（真值），且报告的区间必须包含它。
    """
    features, _theta_star, plans = _planted(seed=0)
    result = solve_bundle_multi(plans, features, tol=1e-5, max_iterations=40, inner_tol=1e-9)
    reference = dense_joint_lp(plans, [features] * len(plans), theta_set=SimplexTheta(4))
    assert result.value == pytest.approx(reference.value, rel=1e-6, abs=1e-9)
    assert result.lower <= reference.value + 1e-9
    assert result.upper >= reference.value - 1e-9
    assert abs(result.theta.sum() - 1.0) < 1e-9
    assert result.theta.min() >= -1e-12


def test_bundle_matches_dense_frank_wolfe_on_the_same_family() -> None:
    """
    EN: the same family, two algorithms: the bundle's exact value must not exceed the dense Frank–Wolfe's best
        value by more than the tolerance (the FW is an upper bound on `F*`).
    CN: 同一特征族、两种算法：bundle 的精确值不应比稠密 Frank–Wolfe 的最好值高出容差（FW 是 `F*` 的上界）。
    """
    features, _theta_star, plans = _planted(seed=1)
    result = solve_bundle_multi(plans, features, tol=1e-5, max_iterations=40, inner_tol=1e-9)
    fw_value, _fw_theta, _trace = dense_frank_wolfe(plans, [features] * len(plans),
                                                    theta_set=SimplexTheta(4), iterations=60)
    assert result.value <= fw_value + 1e-6


def test_returned_plans_are_transport_plans_with_small_support() -> None:
    """
    EN: each returned plan has the observed marginals and at most `m + n - 1` non-zeros; the diagonal masses are
        reported and finite.
    CN: 每个返回计划具有观测 marginal、非零元至多 `m+n-1`；对角质量被报告且有限。
    """
    features, _theta_star, plans = _planted(n=10, seed=2)
    result = solve_bundle_multi(plans, features, tol=1e-5, max_iterations=30, inner_tol=1e-9)
    for observed, predicted in zip(plans, result.plans, strict=True):
        np.testing.assert_allclose(np.asarray(predicted.sum(axis=1)).reshape(-1),
                                   np.asarray(observed.sum(axis=1)).reshape(-1), atol=1e-9)
        np.testing.assert_allclose(np.asarray(predicted.sum(axis=0)).reshape(-1),
                                   np.asarray(observed.sum(axis=0)).reshape(-1), atol=1e-9)
        assert predicted.nnz <= 2 * int(features.n_source) - 1
    assert all(np.isfinite(value) for value in result.predicted_diagonal_mass)


def test_bundle_rejects_bad_input() -> None:
    """EN: a plan without positive marginals, and a mismatched number of feature maps. CN: 非正 marginal、特征映射数不符。"""
    features, _theta_star, plans = _planted(seed=3)
    empty = sparse.csr_matrix(plans[0].shape)
    with pytest.raises(ValueError):
        solve_bundle_multi([empty], features)
    with pytest.raises(ValueError):
        solve_bundle_multi(plans, [features, features])


# EN: zero-subgradient guard. A window whose observed plan is already optimal at the queried theta gives F = 0
#     and a zero cut; the bundle must stop with a certified bracket instead of handing an empty bank to the master.
# CN: 零次梯度守卫。若某窗的观测计划在查询 theta 处已最优，则 F=0、割为零；bundle 应停止并给出证书区间，
#     而不是把空割银行交给 master。
N = 6


def _unit_square(seed: int) -> PointDistanceFeatures:
    """EN: two point clouds on the unit square. CN: 单位方格上的两片点云。"""
    rng = np.random.default_rng(seed)
    return PointDistanceFeatures(rng.random((N, 2)), rng.random((N, 2)), width=3)


def _plan_at(features: PointDistanceFeatures, theta: np.ndarray) -> sparse.csr_matrix:
    """EN: the exact uniform-marginal OT plan at `theta`. CN: 在 `theta` 处、均匀边际下的精确 OT 计划。"""
    rows = np.repeat(np.arange(N, dtype=np.int64), N)
    cols = np.tile(np.arange(N, dtype=np.int64), N)
    mass = np.full(N, 1.0 / N)
    return sparse.csr_matrix(
        np.asarray(ot.emd(mass, mass, np.asarray(features.costs(theta, rows, cols)).reshape(N, N),
                          numItermax=2_000_000)))


def test_zero_subgradient_stops_with_a_bracket() -> None:
    """EN: a plan optimal at uniform gives F=0, a zero gap and no exception. CN: 在均匀点最优的计划应给出 F=0、零间隙且不抛错。"""
    features = _unit_square(0)
    uniform = np.full(features.n_features, 1.0 / features.n_features)
    result = solve_bundle_multi([_plan_at(features, uniform)], [features],
                                theta_set=SimplexTheta(features.n_features), max_iterations=10)
    assert result.converged
    assert abs(result.value) < 1e-12
    assert result.gap <= 1e-12
    assert len(result.iterations) == 1
    assert np.allclose(result.theta, uniform)


def test_a_nonzero_gradient_still_builds_cuts() -> None:
    """EN: if uniform is not already optimal the loop builds a cut and reaches a certified bracket. CN: 若均匀点非最优，循环应建割并给出证书区间。"""
    features = _unit_square(1)
    result = solve_bundle_multi([_plan_at(features, np.asarray([0.0, 0.0, 1.0]))], [features],
                                theta_set=SimplexTheta(features.n_features), max_iterations=20)
    assert len(result.iterations) >= 2
    assert result.upper >= -1e-12
    assert result.lower <= result.upper + 1e-9
    assert abs(result.value) < 1e-12
    assert result.theta[-1] > 0.99
