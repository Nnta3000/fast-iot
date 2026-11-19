"""
EN: **Tests for the shared-theta arm** (`iot.relaxed.multi_plan`).

    Three properties matter for a comparison arm and none of them is visible from a single number:
      1. the returned objective is the function the docstring defines (recomputed independently);
      2. the optimiser does not increase it, and it goes to zero when the observed plans are exactly optimal at
         one common `theta` (the planted case) — i.e. the arm can *find* a shared explanation when one exists;
      3. the prediction is an exact OT plan at the fitted cost and the given marginals, so the flow comparison is
         not confounded by a decoder.

CN: **共享 theta 臂的测试**（`iot.relaxed.multi_plan`）。比较臂需要三条性质，单看一个数字都看不出来：返回的目标就是文档
    所定义的那个函数（独立重算）；优化不使其增大，且在观测计划确实由同一个 `theta` 最优生成（植入情形）时趋于 0 —— 即
    存在共享解释时该臂能找到；预测是拟合代价与给定 marginal 下的精确 OT 计划，故流量比较不被解码器混淆。
"""

from __future__ import annotations

import numpy as np
import ot
import pytest

from iot.relaxed.multi_plan import (SharedThetaFit, distance_features, fit_shared_theta,
                                    mixed_features, regret_of)


def _geometry(n: int = 9, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EN: a 1-D point set, its distance matrix, and two marginals. CN: 一维点集、距离矩阵与两组 marginal。"""
    rng = np.random.default_rng(seed)
    points = rng.random(n)
    distance = np.abs(points[:, None] - points[None, :])
    source = rng.random(n) + 0.2
    source /= source.sum()
    target = rng.random(n) + 0.2
    target /= target.sum()
    return points, distance, source, target


def test_features_are_non_negative_and_exclude_the_constant_by_default() -> None:
    """
    EN: `theta` lives on the simplex, so every feature must be non-negative. The constant column is off by default:
        under `min_theta R` a constant cost makes every plan optimal, so that column would let the fit report
        `R = 0` on any observation (measured on the NYC instance).

    CN: `theta` 在单纯形上，故每个特征必须非负。常数列默认关闭：在 `min_theta R` 下常数代价使任何计划最优，该列会让
        拟合对任何观测都报 `R=0`（已在纽约实例上实测）。
    """
    _points, distance, _source, _target = _geometry()
    features = distance_features(distance)
    assert features.shape == (9, 9, 4)
    assert float(features.min()) >= 0.0
    with_constant = distance_features(distance, include_constant=True)
    assert with_constant.shape == (9, 9, 5)
    np.testing.assert_allclose(with_constant[:, :, 0], 1.0)


def test_planted_shared_theta_drives_the_objective_to_zero() -> None:
    """
    EN: plant two plans that are both optimal at one `theta`, then refit from the uniform start: the summed regret
        must fall to (numerically) zero, which is the arm's correctness statement.
    CN: 植入两个都由同一个 `theta` 最优生成的计划，再从均匀点重新拟合：求和 regret 必须降到（数值上的）0，这就是该臂的
        正确性陈述。
    """
    _points, distance, source, target = _geometry()
    features = distance_features(distance)
    planted = np.array([0.25, 0.15, 0.10, 0.50])
    cost = np.tensordot(features, planted, axes=([2], [0]))
    plan = np.asarray(ot.emd(source, target, cost, numItermax=1_000_000))
    flows = np.stack([plan, plan])
    fit = fit_shared_theta(flows, distance, iterations=60)
    assert isinstance(fit, SharedThetaFit)
    assert fit.objective < 1e-9
    assert fit.history[0] > fit.objective
    assert all(b <= a + 1e-12 for a, b in zip(fit.history, fit.history[1:])), "the objective must not increase"
    # EN: `R = 0` says the observed plan is *optimal* at the fitted cost, not that it is the only optimal one, so the
    #     check is that the predicted plan is optimal too: its cost must equal the forward OT value.
    # CN: `R=0` 只说明观测计划在拟合代价下**最优**，并不说明它是唯一最优；故检查预测计划同样最优：其代价等于前向 OT 值。
    predicted = fit.predict(source, target)
    cost = fit.cost()
    np.testing.assert_allclose(float(np.sum(cost * predicted)),
                               float(ot.emd2(source, target, cost, numItermax=2_000_000)),
                               rtol=1e-9, atol=1e-9)


def test_objective_is_recomputed_the_same_way_it_is_reported() -> None:
    """
    EN: recompute `F(theta)` at the returned point from the definition; a mismatch would mean the optimiser is
        minimising something other than what the table reports.
    CN: 按定义在返回点重算 `F(theta)`；不一致意味着优化器最小化的不是表格报告的东西。
    """
    _points, distance, source, target = _geometry()
    features = distance_features(distance)
    rng = np.random.default_rng(1)
    flows = np.stack([
        np.asarray(ot.emd(source, target, rng.random((9, 9)), numItermax=1_000_000)),
        np.asarray(ot.emd(target, source, rng.random((9, 9)), numItermax=1_000_000)),
    ])
    fit = fit_shared_theta(flows, distance, iterations=15)
    cost = np.tensordot(features, fit.theta, axes=([2], [0]))
    total = 0.0
    for plan in flows:
        rows, cols = plan.sum(axis=1), plan.sum(axis=0)
        total += float(np.sum(cost * plan) - ot.emd2(rows, cols, cost, numItermax=2_000_000))
    assert fit.objective == pytest.approx(total, rel=1e-9, abs=1e-12)
    assert float(fit.theta.sum()) == pytest.approx(1.0)
    assert float(fit.theta.min()) >= 0.0


def test_bad_inputs_are_rejected() -> None:
    """EN: negative flows, a non-square distance, an unknown family. CN: 负流量、非方阵距离、未知族。"""
    _points, distance, source, target = _geometry()
    with pytest.raises(ValueError):
        fit_shared_theta(-np.ones((1, 9, 9)), distance)
    # EN: a square but identically-zero distance has no scale to normalise by.
    # CN: 方阵但恒为零的距离没有可归一化的尺度。
    with pytest.raises(ValueError):
        distance_features(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        distance_features(distance, family="no_such_family")
    with pytest.raises(ValueError):
        fit_shared_theta(np.ones((1, 9, 9)), distance, iterations=0)


def test_a_constant_feature_makes_the_regret_trivially_zero() -> None:
    """
    EN: regression for the degeneracy that forced the constant column off by default.

        A constant cost makes every plan that matches the marginals optimal, so `R = 0` on *any* observation. The
        check is direct arithmetic rather than an optimisation: evaluate `R` at the constant vertex for a rank-one
        (not optimal) flow and require exactly zero, then require the default family to be positive there. If
        somebody re-adds the constant column as the default, the second assertion fails.

    CN: 迫使常数列默认关闭的退化的回归测试。常数代价使任何满足 marginal 的计划都最优，故对**任何**观测 `R=0`。检查用直接
        算术而不是优化：在一个秩一（非最优）流量上于常数顶点处求 `R`，要求恰为 0；再要求默认族在该处为正。若有人把常数列
        改回默认，第二条断言会失败。
    """
    _points, distance, source, target = _geometry()
    rank_one = np.outer(source, target)
    with_constant = distance_features(distance, include_constant=True)
    constant_vertex = np.zeros(with_constant.shape[2])
    constant_vertex[0] = 1.0
    assert regret_of(constant_vertex, with_constant, rank_one) == pytest.approx(0.0, abs=1e-9)
    without_constant = distance_features(distance)
    assert without_constant.shape[2] == 4
    # EN: every vertex of the default family gives a strictly positive regret on a rank-one flow.
    # CN: 默认族的每个顶点在该秩一流量上都给出严格正的 regret。
    for k in range(without_constant.shape[2]):
        vertex = np.zeros(without_constant.shape[2])
        vertex[k] = 1.0
        assert regret_of(vertex, without_constant, rank_one) > 0.0


def test_lowrank_family_adds_non_negative_interaction_columns_from_the_given_flows() -> None:
    """
    EN: the low-rank block is built from the flows passed in, is non-negative, and adds exactly `rank` columns.

        A per-zone term `a_i + b_j` would be absorbed by the OT duals and change no plan, so the block must be an
        *interaction* `u_i v_j`, which is why the factors are not summed into per-zone potentials. The test also
        checks the leakage property that is checkable without a held-out window: two different flow tensors give two
        different feature tensors.

    CN: 低秩块由传入流量构造、非负、且恰好增加 `rank` 列。形如 `a_i + b_j` 的逐 zone 项会被 OT 对偶吸收、不改变任何计划，
        故该块必须是**交互** `u_i v_j`，这也是因子不被求和成逐 zone 势的原因。测试还检查一个无需留出窗即可检查的不泄漏性质：
        不同的流量张量给出不同的特征张量。
    """
    _points, distance, source, target = _geometry()
    first = np.stack([np.outer(source, target)])
    second = np.stack([np.outer(target, source)])
    features_first = mixed_features(distance, first, family="distance_lowrank", rank=3)
    features_second = mixed_features(distance, second, family="distance_lowrank", rank=3)
    assert features_first.shape[2] == 4 + 3
    assert float(features_first.min()) >= 0.0
    # EN: each appended column has unit Frobenius norm (the gauge), so theta's split is a shape choice.
    # CN: 追加的每列是单位 Frobenius 范数（规范），故 theta 的分配是形状选择。
    for k in range(4, 7):
        assert float(np.linalg.norm(features_first[:, :, k])) == pytest.approx(1.0)
    assert not np.allclose(features_first, features_second), "the factors must depend on the flows passed in"
