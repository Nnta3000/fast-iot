"""
EN: **Tests for the sparse exact-OT oracle** (`iot.sparse.sparse_ot`).

    The oracle replaces a dense POT solve; the only thing that matters is that it returns the *same* optimum on the
    full grid, with a plan whose support is small. Three checks: the value against POT's dense network simplex, the
    plan's feasibility, and the certificate (`max sigma` outside the final edge set at zero). The warm-start check is
    there because the outer bundle keeps edge sets between iterations, and a warm set that changes the answer would
    silently make the bundle's cuts wrong.

CN: **稀疏精确 OT oracle 的测试**（`iot.sparse.sparse_ot`）。该 oracle 取代稠密 POT 求解，唯一重要的是它在**全网格**上
    给出相同最优值，且计划的支撑很小。三项检查：与 POT 稠密网络单纯形的值一致、计划可行、证书（最终边集之外的最大 `sigma`
    为零）。热启动检查是必要的：外层 bundle 在轮间保留边集，若热启动集改变答案，bundle 的割就会静默出错。
"""

from __future__ import annotations

import numpy as np
import ot
import pytest

from iot.core.edges import EdgeSet
from iot.sparse.features import PointDistanceFeatures
from iot.sparse.sparse_ot import solve_sparse_ot


def _instance(n: int = 12, seed: int = 0):
    """EN: points, marginals, features and a strictly positive θ. CN: 点、marginal、特征与严格正的 θ。"""
    rng = np.random.default_rng(seed)
    source_points = rng.random((n, 2))
    target_points = rng.random((n, 2))
    source_mass = rng.random(n) + 0.1
    source_mass /= source_mass.sum()
    target_mass = rng.random(n) + 0.1
    target_mass /= target_mass.sum()
    features = PointDistanceFeatures(source_points, target_points, scale=1.0, width=4)
    theta = np.array([0.1, 0.5, 0.2, 0.2])
    return features, source_mass, target_mass, theta


def test_sparse_ot_matches_dense_pot_and_is_feasible() -> None:
    """
    EN: same optimum as POT's dense solve, a feasible plan with at most `m + n - 1` non-zeros, and a closed
        certificate.
    CN: 与 POT 稠密求解同最优值，计划可行且非零元至多 `m+n-1`，证书闭合。
    """
    features, a, b, theta = _instance()
    n_source = int(features.n_source)
    n_target = int(features.n_target)
    result = solve_sparse_ot(features, a, b, theta=theta, gamma=2, tol=1e-9)
    rows = np.repeat(np.arange(n_source, dtype=np.int64), n_target)
    cols = np.tile(np.arange(n_target, dtype=np.int64), n_source)
    dense = np.asarray(features.costs(theta, rows, cols)).reshape(n_source, n_target)
    reference = np.asarray(ot.emd(a, b, dense, numItermax=1_000_000))
    assert result.value == pytest.approx(float(np.sum(dense * reference)), rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(np.asarray(result.plan.sum(axis=1)).reshape(-1), a, atol=1e-10)
    np.testing.assert_allclose(np.asarray(result.plan.sum(axis=0)).reshape(-1), b, atol=1e-10)
    assert result.n_nonzeros <= n_source + n_target - 1
    assert result.inner_gap <= 1e-9


def test_warm_edge_set_keeps_the_answer_and_saves_sweeps() -> None:
    """
    EN: re-solving from the converged edge set must return the same value in one sweep — the property the bundle
        relies on when it keeps `E_l` between θ updates.
    CN: 从收敛边集重解必须在一次扫描内给出相同值 —— 这正是 bundle 在 θ 更新之间保留 `E_l` 所依赖的性质。
    """
    features, a, b, theta = _instance(seed=1)
    first = solve_sparse_ot(features, a, b, theta=theta, tol=1e-9)
    warm = solve_sparse_ot(features, a, b, theta=theta, edges=first.edges, tol=1e-9)
    assert warm.value == pytest.approx(first.value, rel=1e-12, abs=1e-14)
    assert warm.sweeps == 1
    assert warm.n_edges <= first.n_edges


def test_sparse_ot_rejects_bad_input() -> None:
    """EN: mismatched masses, a wrong θ length, and a negative mass. CN: 质量不匹配、θ 长度错误、负质量。"""
    features, a, b, theta = _instance()
    with pytest.raises(ValueError):
        solve_sparse_ot(features, a[:-1], b, theta=theta)
    with pytest.raises(ValueError):
        solve_sparse_ot(features, a, b, theta=theta[:3])
    with pytest.raises(ValueError):
        solve_sparse_ot(features, -a, b, theta=theta)


def test_initial_edge_set_must_be_feasible_after_sweeps() -> None:
    """
    EN: a deliberately tiny starting edge set (one edge per row, wrong masses) is repaired by the sweeps and still
        returns the dense optimum.
    CN: 刻意很小的起始边集（每行一条边、质量不对）会被扫描修补，仍返回稠密最优值。
    """
    features, a, b, theta = _instance(seed=2)
    n_source = int(features.n_source)
    n_target = int(features.n_target)
    rows = np.arange(n_source, dtype=np.int64)
    cols = np.arange(n_source, dtype=np.int64) % n_target
    start = EdgeSet.from_arrays(rows, cols, n_target=n_target)
    result = solve_sparse_ot(features, a, b, theta=theta, edges=start, tol=1e-9)
    full_rows = np.repeat(np.arange(n_source, dtype=np.int64), n_target)
    full_cols = np.tile(np.arange(n_target, dtype=np.int64), n_source)
    dense = np.asarray(features.costs(theta, full_rows, full_cols)).reshape(n_source, n_target)
    reference = float(np.sum(dense * np.asarray(ot.emd(a, b, dense, numItermax=1_000_000))))
    assert result.value == pytest.approx(reference, rel=1e-9, abs=1e-12)
    assert result.inner_gap <= 1e-9
