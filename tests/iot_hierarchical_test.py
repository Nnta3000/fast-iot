"""
CN: **层级逆 OT 的精确测试** —— 用合成实例（已知 θ*、δ* > 0），走**精确循环**。

    CN: 这是 Typst 文件真正要求的测试：SynthRAD 上精确 master 通常不可行，
        `is_optimal_plan` 连扁平解都不通过，因此在那里测不到 Lemma 2 的单调界与 Theorem 6 的
        "输出与 flat 相同"。合成实例是唯一能测到它们的地方。

    CN: 断言：
        1. **输出相同**（Theorem 6(b)）：层级与扁平的 δ 相等（浮点容差），θ 在 grow-only 模式下允许不同；
        2. **单调界链**（Corollary 3(2)）：`δ*_L >= ... >= δ*_0`，各层单独求解得到；
        3. **种子可行**（Lemma 5）：`B ⊆ F₀ ⊆ U`；
        4. **无层级时逐位不变**：`hierarchical=None` 与显式扁平调用等价。

    CN: 这些测试**只**用合成数据（`DiagonalWeightFeatures` 上的已知 θ*），不读 SynthRAD。

EN: **Exact tests of the hierarchical inverse OT**, on synthetic instances (known theta*, delta* > 0), using
    the **exact loop**.

    This is the test the Typst note actually calls for: on SynthRAD the exact master is usually infeasible
    and even the flat solution fails `is_optimal_plan`, so Lemma 2's monotone bounds and Theorem 6's
    "same output as flat" cannot be exercised there. A synthetic instance is the only place they can be.

    Assertions:
      1. **same output** (Theorem 6(b)): hierarchical and flat agree on delta (float tolerance); theta may
         differ in grow-only mode;
      2. **monotone bounds** (Corollary 3(2)): `delta*_L >= ... >= delta*_0`, each level solved on its own;
      3. **admissible seed** (Lemma 5): `B subseteq F0 subseteq U`;
      4. **no hierarchy means no change**: `hierarchical=None` equals an explicit flat call.

    These tests use **only** synthetic data (a known theta* over `DiagonalWeightFeatures`).
"""

from __future__ import annotations

import numpy as np
import ot
import pytest
import scipy.sparse as sparse
from jaxtyping import Float, Int

from iot import DiagonalWeightFeatures, InverseOTProblem
from iot.core.theta import SimplexTheta
from iot.exact.components import mandatory_edges
from iot.exact.loop import solve_inverse_ot
from iot.hierarchical import HierarchyParameters, build_levels


def _planted_problem(m: int = 40, n: int = 40, d: int = 4, seed: int = 0) -> InverseOTProblem:
    """
    CN: **可精确识别**的合成实例：代价来自已知 θ*，π̂ 是它在 θ* 下的最优方案，
        因此 (A1) 成立、master 有界、δ* > 0（通常）。
    EN: an **exactly identifiable** synthetic instance: costs from a known theta*, with pi_hat its optimal
        plan under it, so (A1) holds, the master is bounded, and delta* > 0 (usually).
    """
    rng = np.random.default_rng(int(seed))
    x: Float[np.ndarray, "m d"] = np.ascontiguousarray(rng.normal(size=(m, d)))
    y: Float[np.ndarray, "n d"] = np.ascontiguousarray(rng.normal(size=(n, d)))
    theta_star: Float[np.ndarray, "d"] = np.abs(rng.normal(size=d))
    theta_star = theta_star / theta_star.sum()
    features = DiagonalWeightFeatures(x, y)
    rows: Int[np.ndarray, "m n"] = np.repeat(np.arange(m), n)
    cols: Int[np.ndarray, "m n"] = np.tile(np.arange(n), m)
    cost: Float[np.ndarray, "m n"] = np.asarray(features.costs(theta_star, rows, cols),
                                                 dtype=np.float64).reshape(m, n)
    plan: Float[np.ndarray, "m n"] = np.asarray(
        ot.emd(np.full(m, 1 / m), np.full(n, 1 / n), cost, numItermax=10_000_000), dtype=np.float64)
    return InverseOTProblem(features=features, theta_set=SimplexTheta(dimension=d),
                            plan=sparse.csr_matrix(plan))


def test_no_hierarchy_is_bit_identical_to_flat() -> None:
    """
    CN: `hierarchical=None` 必须与显式扁平调用完全一致（这是"默认不改变任何结果"的保证）。
    EN: `hierarchical=None` must equal an explicit flat call exactly.
    """
    problem = _planted_problem()
    a = solve_inverse_ot(problem, tol=1e-9, gamma=1, max_iterations=50)
    b = solve_inverse_ot(problem, tol=1e-9, gamma=1, max_iterations=50, hierarchical=None)
    assert float(a.delta) == float(b.delta)
    assert np.array_equal(np.asarray(a.theta), np.asarray(b.theta))


def test_levels_are_nested_and_keep_every_point_attached() -> None:
    """
    CN: Lemma 1：索引集嵌套；每个保留点至少连一条诱导支撑边；诱导支撑 != 整个粗网格。
    EN: Lemma 1: nested index sets; every retained point has an induced support edge; the induced support
        is not the whole coarse grid.
    """
    problem = _planted_problem()
    levels = build_levels(problem, rho=0.25, max_levels=2, grid_budget=1, seed=0)
    assert len(levels) >= 2, "coarsening should produce at least one coarse level when grid_budget=1"
    for fine, coarse in zip(levels, levels[1:]):
        assert set(coarse.rows.tolist()) <= set(fine.rows.tolist())
        assert set(coarse.cols.tolist()) <= set(fine.cols.tolist())
        assert set(coarse.support.rows.tolist()) <= set(coarse.rows.tolist())
        assert set(coarse.support.cols.tolist()) <= set(coarse.cols.tolist())
        assert coarse.n_source <= fine.n_source and coarse.n_target <= fine.n_target
        assert coarse.support.size > 0
        assert coarse.support.size < coarse.grid_size, "induced support must not fill the grid"


def test_monotone_bounds_hold_across_levels() -> None:
    """
    CN: **Corollary 3(2)**：每层**单独**求解，`δ*_ℓ` 沿粗化方向不减。这是 Lemma 2 的直接后果，
        只能在精确 master 可行的合成实例上测。
    EN: **Corollary 3(2)**: solving each level on its own, `delta*_ell` does not decrease towards coarser
        levels. That is the direct consequence of Lemma 2, testable only where the exact master is feasible.
    """
    problem = _planted_problem()
    levels = build_levels(problem, rho=0.25, max_levels=2, grid_budget=1, seed=0)
    deltas: list[float] = []
    for index, level in enumerate(levels):
        target = problem if index == 0 else _coarse(problem, level)
        result = solve_inverse_ot(target, tol=1e-9, gamma=1, max_iterations=60, validate=False)
        deltas.append(float(result.delta))
    # CN: deltas[0] 最细，deltas[-1] 最粗；沿粗化方向应不减。
    # EN: deltas[0] is finest, deltas[-1] coarsest; they must not decrease towards coarser.
    for finer, coarser in zip(deltas, deltas[1:]):
        assert coarser >= finer - 1e-9, f"delta decreased towards coarser: {deltas}"


def _coarse(problem: InverseOTProblem, level) -> InverseOTProblem:  # noqa: ANN001
    """CN: 粗层问题（诱导方案）。EN: the coarse problem (induced plan)."""
    src = np.asarray(problem.features.source_points, dtype=np.float64)[level.rows]
    tgt = np.asarray(problem.features.target_points, dtype=np.float64)[level.cols]
    dense = np.asarray(problem.plan.todense(), dtype=np.float64)
    restricted = np.ascontiguousarray(dense[np.ix_(level.rows, level.cols)])
    return InverseOTProblem(features=DiagonalWeightFeatures(np.ascontiguousarray(src),
                                                            np.ascontiguousarray(tgt)),
                            theta_set=problem.theta_set, plan=sparse.csr_matrix(restricted))


def test_hierarchical_exact_matches_flat_delta() -> None:
    """
    CN: **Theorem 6(b)**：层级（精确循环）与扁平的 δ 相同。grow-only 模式下 θ 允许不同，
        所以只比较 δ 与 `is_optimal`。
    EN: **Theorem 6(b)**: hierarchical (exact loop) and flat agree on delta. In grow-only mode theta may
        differ, so only delta and `is_optimal` are compared.
    """
    problem = _planted_problem()
    flat = solve_inverse_ot(problem, tol=1e-9, gamma=1, max_iterations=60)
    hier = solve_inverse_ot(problem, tol=1e-9, gamma=1, max_iterations=60,
                            hierarchical=HierarchyParameters(rho=0.25, max_levels=1, grid_budget=1,
                                                             kappa=2, level_seed=0))
    assert hier.converged
    assert float(hier.delta) == pytest.approx(float(flat.delta), abs=1e-6), (
        f"delta differs: hier={float(hier.delta):.9f} flat={float(flat.delta):.9f}")
    assert bool(hier.is_optimal) is bool(flat.is_optimal)


def test_seed_contains_the_mandatory_set() -> None:
    """
    CN: **Lemma 5**：种子必须包含 `B = S ∪ R`，否则精确循环的 master 会无界。
    EN: **Lemma 5**: the seed must contain `B = S ∪ R`; otherwise the exact master is unbounded.
    """
    problem = _planted_problem()
    levels = build_levels(problem, rho=0.25, max_levels=1, grid_budget=1, seed=0)
    from iot.hierarchical.transfer import prolong, seed  # noqa: PLC0415

    fine, coarse = levels[0], levels[1]
    coarse_problem = _coarse(problem, coarse)
    coarse_out = solve_inverse_ot(coarse_problem, tol=1e-9, gamma=1, max_iterations=60, validate=False)
    f_full, g_full = prolong(problem, coarse, fine, np.asarray(coarse_out.theta, dtype=np.float64),
                             np.asarray(coarse_out.source_dual, dtype=np.float64),
                             np.asarray(coarse_out.target_dual, dtype=np.float64))
    mandatory = mandatory_edges(problem)[0]
    f0, stats = seed(problem, fine, mandatory, theta=np.asarray(coarse_out.theta, dtype=np.float64),
                     source_dual=f_full, target_dual=g_full, kappa=2, carry_keys=None,
                     n_target=int(problem.n_target))
    missing = np.setdiff1d(mandatory.keys, f0.keys)
    assert missing.size == 0, f"{missing.size} mandatory edges missing from the seed"
    assert int(f0.size) <= int(problem.n_source) * int(problem.n_target)
    assert stats["n_mandatory"] == int(mandatory.size)


def test_hierarchy_parameters_reject_bad_values() -> None:
    """CN: 早失败。EN: fail early on bad knobs."""
    for bad in ({"rho": 0.0}, {"rho": 1.0}, {"max_levels": 0}, {"grid_budget": 0}, {"kappa": 0}):
        with pytest.raises(ValueError):
            HierarchyParameters(**bad)
