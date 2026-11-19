"""
EN: **Tests for the related-work reproduction** (`studies/related/inverse_cost_recovery`).

    The study reports one scientific reading — minimising the exact-OT regret on Ma et al.'s discrete benchmark
    selects a family member that is fixed by the cost's shape — so the tests pin the three things that reading
    depends on:

      1. the scale handling (`_best_scale`, `_relative`) really is invariant to a positive rescaling of the cost,
         which is the equivalence the OT plan itself has;
      2. the gauge (`_unit_norm`) puts every feature at unit Frobenius norm, so `cost_rel_scaled` is not
         dominated by a magnitude mismatch between family members;
      3. the master's `theta` is the **global** minimiser of `R` on a small instance, checked against a brute-force
         grid over the simplex; without this the "the criterion selects member k" reading could be a solver
         artefact.

CN: **相关工作报告实验的测试**。该研究只有一条科学读数 —— 在 Ma 等的离散基准上最小化精确 OT 的 regret 会选出由代价
    形状（而非真值成员）决定的族内成员 —— 测试固定这条读数所依赖的三件事：尺度处理对代价的正缩放不变；规范使每个
    特征的单位 Frobenius 范数；以及 master 给出的 `theta` 确实是 `R` 的**全局**最小点（用小实例的单纯形网格搜索核对）。
"""

from __future__ import annotations

import numpy as np
import ot
import pytest
import scipy.sparse as sparse

from iot_experiments.studies.related.inverse_cost_recovery.run import (
    _best_scale,
    _dense_cost,
    _family,
    _fit,
    _regret,
    _relative,
    _unit_norm,
)


def test_best_scale_is_invariant_to_a_positive_rescaling_of_the_cost() -> None:
    """
    EN: a cost and `3 * cost` induce the same OT plan, so the scaled cost error must be 0 in both cases.
    CN: 代价与 `3*代价` 诱导同一 OT 计划，故两者的缩放后代价误差都必须为 0。
    """
    rng = np.random.default_rng(0)
    truth = rng.random((6, 6))
    cost = truth / 2.5
    np.testing.assert_allclose(_best_scale(cost, truth) * cost, truth, rtol=1e-12, atol=1e-12)
    assert _relative(_best_scale(cost, truth) * cost, truth) < 1e-12
    # EN: the raw relative error is nonzero exactly because the scale differs.
    # CN: 原始相对误差非零，正是因为尺度不同。
    assert _relative(cost, truth) > 0.1


def test_unit_norm_puts_every_feature_at_unit_frobenius_norm() -> None:
    """
    EN: the gauge is applied per feature column, not to the tensor as a whole.
    CN: 规范按特征列施加，而不是整体施加。
    """
    rng = np.random.default_rng(1)
    tensor = np.abs(rng.normal(size=(7, 7, 3))) * np.array([1.0, 1e3, 1e-3])[None, None, :]
    scaled = _unit_norm(tensor)
    flat = scaled.reshape(-1, 3)
    np.testing.assert_allclose(np.linalg.norm(flat, axis=0), 1.0, rtol=1e-12, atol=1e-12)
    # EN: within each column the pattern is unchanged -- every entry is divided by the same number -- so the ratio
    #     to that column's own maximum is preserved. (A ratio BETWEEN columns is not preserved: each column is
    #     scaled by its own norm, which is the point of the gauge.)
    # CN: 每列内部的形状不变（同除一个数），故与该列最大值的比值不变。**列间**比值不被保留 —— 每列按自身范数缩放，
    #     这正是规范的目的。
    for k in range(3):
        np.testing.assert_allclose(scaled[:, :, k] / scaled[:, :, k].max(),
                                   tensor[:, :, k] / tensor[:, :, k].max(), rtol=1e-12, atol=0.0)


def test_fit_returns_a_global_minimiser_of_the_regret() -> None:
    """
    EN: the master's `theta` must be at least as good as every point of a brute-force grid over the simplex.

        This is the test that separates "the criterion prefers member k" from "the solver stopped at member k".
        The instance is tiny (`n = 8`, `K = 2`), so the full grid at step 0.02 (51 points) is cheap.

    CN: master 的 `theta` 必须不劣于单纯形网格搜索的每个点。这条测试把"准则偏好成员 k"与"求解器停在成员 k"区分开。
        实例很小（`n=8, K=2`），步长 0.02 的完整网格（51 点）很便宜。
    """
    n = 8
    index = np.arange(n, dtype=np.float64) / float(n)
    distance = np.abs(index[:, None] - index[None, :])
    truth = distance ** 2.0
    features, member = _family("p_basis", n, 1.0, (1.0, 2.0))
    assert member == 0

    rng = np.random.default_rng(2)
    mu = rng.dirichlet(np.ones(n))
    nu = rng.dirichlet(np.ones(n))
    plan = np.asarray(ot.sinkhorn(mu, nu, truth, reg=0.05, method="sinkhorn_log", numItermax=20_000))

    theta, regret, status = _fit(features, sparse.csr_matrix(plan), backend="highs")
    assert status.lower().startswith("optim") or "optimal" in status.lower()
    np.testing.assert_allclose(regret, _regret(_dense_cost(features, theta, n), plan, mu, nu),
                               rtol=1e-9, atol=1e-12)

    grid = np.linspace(0.0, 1.0, 51)
    best = min((_regret(_dense_cost(features, np.array([t, 1.0 - t]), n), plan, mu, nu) for t in grid))
    # EN: the LP is exact, so it may only be *below* the grid's best (a grid point need not hit the optimum).
    # CN: LP 是精确的，故它只可能不高于网格最优（网格点未必命中最优）。
    assert regret <= best + 1e-12, f"LP regret {regret!r} is worse than the grid's {best!r}"
    assert np.isclose(theta.sum(), 1.0, atol=1e-9)
    assert np.all(theta >= -1e-9)


def test_family_rejects_an_unknown_name() -> None:
    """EN: an unknown family is an error, never a silent default. CN: 未知族直接报错，不静默回退。"""
    with pytest.raises(ValueError):
        _family("no_such_family", 5, 1.0, (1.0, 2.0))
