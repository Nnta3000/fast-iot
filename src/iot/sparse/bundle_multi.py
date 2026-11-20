"""
EN: **One θ for many sparse plans — a cutting-plane (Kelley) bundle with a certified gap.**

    Objective, unchanged from the dense reference: `F(θ) = Σ_l [⟨c(θ), π̂_l⟩ − OT_l(c(θ))]`, `c(θ) = Σ_k θ_k φ_k`,
    `θ ∈ Θ`. `F` is convex (linear minus a concave OT value) and positively homogeneous of degree one in the cost
    (`OT(αc) = α OT(c)`), so for an exact optimal plan `P_l` at `θ` the vector `g = Σ_l (M(π̂_l) − M(P_l))`,
    `M(P) = φᵀ P`, is a subgradient **and** `F(θ) = ⟨g, θ⟩`. That is what makes the cut `⟨g, θ'⟩` a valid linear
    underestimator with no constant term, and it is the same identity the single-plan `alternatives/cutting_plane`
    module relies on.

    The oracle is `sparse_ot.solve_sparse_ot`: exact OT per window on a growing edge set with blockwise pricing.
    The master is the existing `solve_cut_master(..., mode="kelley")` — `K + 1` variables, independent of `n`.
    Bracket: `lower =` master value, `upper = min_i F(θ_i)` over the exact evaluations. No approximation is assumed:
    the inner solves are exact (certified by their own pricing gap) and the outer bracket is valid for the true `F*`.

    Memory: `O(Σ_l |E_l| + K·t + (m + n) L)`; no `m n` array. **Time stays quadratic**: every pricing sweep visits
    all `m n` pairs.

CN: **一个 θ 对应多个稀疏计划 —— 带证书间隙的割平面（Kelley）bundle。** 目标与稠密参考相同：
    `F(θ) = Σ_l [⟨c(θ),π̂_l⟩ − OT_l(c(θ))]`。`F` 凸且对代价一次正齐次，故在最优计划 `P_l` 处
    `g = Σ_l (M(π̂_l) − M(P_l))` 是次梯度且 `F(θ)=⟨g,θ⟩` —— 这正是割 `⟨g,θ'⟩` 无需常数项的原因，也是单计划
    `cutting_plane` 模块所依赖的同一恒等式。oracle 为 `sparse_ot.solve_sparse_ot`（增长的边集 + 分块定价，精确）；master
    为既有的 `solve_cut_master(mode="kelley")`（`K+1` 个变量，与 `n` 无关）。区间：`lower` = master 值，
    `upper` = 各精确评估 `F(θ_i)` 的最小值。内存 `O(Σ|E_l| + K·t + (m+n)L)`，无 `m n` 数组；**时间仍是二次的**，
    因为每次定价扫描都遍历全部 `m n` 对。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse
from jaxtyping import Float

from ..alternatives.cutting_plane import solve_cut_master
from ..core.edges import EdgeSet
from ..core.features import FeatureMap
from ..core.theta import LinearThetaSet, SimplexTheta
from .sparse_ot import SparseOTResult, solve_sparse_ot

__all__ = ["BundleIteration", "BundleResult", "solve_bundle_multi"]


@dataclass(frozen=True)
class BundleIteration:
    """
    EN: one outer iteration: the exact objective at the queried θ, the master's lower bound, and the diagnostics.
    CN: 一轮外层：被查询 θ 处的精确目标、master 的下界，以及诊断量。
    """

    index: int
    theta: tuple[float, ...]
    value: float
    lower: float
    upper: float
    gap: float
    inner_gaps: tuple[float, ...]
    edges: tuple[int, ...]
    sweeps: tuple[int, ...]
    seconds: float
    master_seconds: float


@dataclass(frozen=True)
class BundleResult:
    """
    EN: the bundle result: the bracket `[lower, upper]` around the true `F*`, the best θ, and per-window artefacts.

        `value` is `F(best_theta)`; `gap = upper - lower >= 0`; `converged` says whether `gap <= tol·max(1,|upper|)`.
        `observed_diagonal_mass` and `predicted_diagonal_mass` are reported per window because a diagonal-heavy
        plan is the signature of the same-zone degeneracy.

    CN: bundle 结果：真值 `F*` 两侧的区间 `[lower, upper]`、最优 θ，以及逐窗产物。`value` 为 `F(best_theta)`；
        `gap = upper - lower >= 0`；`converged` 表示是否 `gap <= tol·max(1,|upper|)`。逐窗报告观测与预测计划的
        对角质量，因为对角占优正是同 zone 退化的特征。
    """

    theta: Float[np.ndarray, "k"]
    value: float
    lower: float
    upper: float
    gap: float
    converged: bool
    plans: tuple[sparse.csr_matrix, ...]
    edge_sets: tuple[EdgeSet, ...]
    iterations: tuple[BundleIteration, ...]
    observed_diagonal_mass: tuple[float, ...]
    predicted_diagonal_mass: tuple[float, ...]
    n_cuts: int
    wall_time: float
    total_inner_seconds: float = 0.0
    stopped_at_zero: bool = False


def _moment(features: FeatureMap, plan: sparse.csr_matrix) -> Float[np.ndarray, "k"]:
    """EN: `M(P) = φᵀ P` over the plan's nonzeros only. CN: 只在计划非零元上计算 `M(P)=φᵀP`。"""
    coo = plan.tocoo()
    phi = features.features(coo.row.astype(np.int64), coo.col.astype(np.int64))
    return np.ascontiguousarray(phi.T @ coo.data.astype(np.float64), dtype=np.float64)


def _diagonal_mass(plan: sparse.csr_matrix) -> float:
    """EN: mass that stays in the same index, as a share of the total. CN: 停留在同一索引的质量占比。"""
    total = float(plan.sum())
    if total <= 0.0:
        return float("nan")
    diagonal = float(plan.diagonal().sum())
    return diagonal / total


def solve_bundle_multi(
    plans: tuple[sparse.csr_matrix, ...] | list[sparse.csr_matrix],
    features: FeatureMap | tuple[FeatureMap, ...] | list[FeatureMap],
    *,
    theta_set: LinearThetaSet | None = None,
    gamma: int = 2,
    tol: float = 1e-6,
    max_iterations: int = 60,
    inner_tol: float = 1e-9,
    inner_max_sweeps: int = 60,
    max_edges_per_window: int | None = None,
    prune_slack: float | None = None,
) -> BundleResult:
    """
    EN: minimise `F(θ)` by Kelley cuts, each cut built from exact per-window OT solves.

        `plans` are the observed sparse transport plans; `features` is one map shared by all windows, or one per
        window (fresh points per observation). `theta_set` defaults to the simplex. `tol` bounds the **certified**
        outer gap; `inner_tol` bounds each forward solve's pricing gap and is reported per iteration.

    CN: 用 Kelley 割最小化 `F(θ)`，每个割由逐窗精确 OT 求解得到。`plans` 为观测稀疏运输计划；`features` 可为一
        个共享特征映射或每窗一个。`theta_set` 默认单纯形；`tol` 约束**带证书**的外层间隙，`inner_tol` 约束每次前向求解
        的定价间隙，并逐轮报告。
    """
    observed = list(plans)
    if len(observed) < 1:
        raise ValueError("need at least one observed plan")
    maps = list(features) if isinstance(features, (tuple, list)) else [features] * len(observed)
    if len(maps) != len(observed):
        raise ValueError(f"got {len(maps)} feature maps for {len(observed)} plans")
    first = observed[0]
    if first.shape != (int(maps[0].n_source), int(maps[0].n_target)):
        raise ValueError(f"plan shape {first.shape} does not match features "
                         f"({maps[0].n_source}, {maps[0].n_target})")
    for plan in observed:
        if np.any(plan.data < 0.0):
            raise ValueError("observed plans must be non-negative")
        rows = np.asarray(plan.sum(axis=1)).reshape(-1)
        cols = np.asarray(plan.sum(axis=0)).reshape(-1)
        if np.any(rows <= 0.0) or np.any(cols <= 0.0):
            raise ValueError("every row and column of an observed plan must have positive mass")
    k = int(maps[0].n_features)
    set_theta = SimplexTheta(k) if theta_set is None else theta_set
    theta = np.full(k, 1.0 / k, dtype=np.float64)

    observed_moments = [_moment(map_l, plan) for map_l, plan in zip(maps, observed, strict=True)]
    edge_sets: list[EdgeSet | None] = [
        EdgeSet.from_arrays(plan.tocoo().row.astype(np.int64), plan.tocoo().col.astype(np.int64),
                            n_target=int(map_l.n_target))
        for plan, map_l in zip(observed, maps, strict=True)]
    masses = [(np.asarray(plan.sum(axis=1)).reshape(-1), np.asarray(plan.sum(axis=0)).reshape(-1))
              for plan in observed]

    cuts: list[np.ndarray] = []
    records: list[BundleIteration] = []
    upper = float("inf")
    best_theta = theta.copy()
    best_value = float("inf")
    best_plans: tuple[sparse.csr_matrix, ...] = ()
    total_inner = 0.0
    started = time.perf_counter()
    plans_now: tuple[sparse.csr_matrix, ...] = ()
    stopped_at_zero = False
    for iteration in range(1, int(max_iterations) + 1):
        iteration_started = time.perf_counter()
        gradient = np.zeros(k, dtype=np.float64)
        value = 0.0
        results: list[SparseOTResult] = []
        for position, (map_l, plan_l) in enumerate(zip(maps, observed, strict=True)):
            source_mass, target_mass = masses[position]
            result = solve_sparse_ot(map_l, source_mass, target_mass, theta=theta,
                                     edges=edge_sets[position], gamma=int(gamma), tol=float(inner_tol),
                                     max_sweeps=int(inner_max_sweeps), max_edges=max_edges_per_window,
                                     prune_slack=prune_slack)
            results.append(result)
            total_inner += result.seconds
            edge_sets[position] = result.edges
            fitted = _moment(map_l, result.plan)
            gradient += observed_moments[position] - fitted
            value += float(observed_moments[position] @ theta) - result.value
        plans_now = tuple(result.plan for result in results)

        # EN: homogeneity identity: F(theta) = <g, theta>. A mismatch means the forward solves are not exact.
        # CN: 齐次恒等式：F(theta)=<g,theta>。不一致说明前向求解不精确。
        identity = float(gradient @ theta)
        if abs(identity - value) > 1e-6 * max(1.0, abs(value)):
            raise RuntimeError(f"homogeneity identity violated: F={value!r} <g,theta>={identity!r}")

        upper = min(upper, value)
        if value < best_value:
            best_value = value
            best_theta = theta.copy()
            best_plans = plans_now
        # EN: A zero subgradient means every observed plan is already optimal at this theta, so F(theta) = 0.
        #     Since F >= 0 that is the global minimum and the cut would be the always-true halfspace
        #     0·theta' <= 0; `solve_cut_master` rejects an all-zero bank. Stop instead of raising.
        # CN: 零次梯度表示每个观测计划在该 theta 处已最优，故 F(theta)=0；因 F>=0 这是全局最小，
        #     割为恒真半空间 0·theta'<=0，而 `solve_cut_master` 会拒绝全零割银行。直接停止而非报错。
        if float(np.linalg.norm(gradient)) <= 1e-12:
            records.append(BundleIteration(
                index=iteration, theta=tuple(float(v) for v in theta), value=value, lower=value, upper=upper,
                gap=float(upper - value), inner_gaps=tuple(result.inner_gap for result in results),
                edges=tuple(result.n_edges for result in results),
                sweeps=tuple(result.sweeps for result in results),
                seconds=float(time.perf_counter() - iteration_started), master_seconds=0.0))
            stopped_at_zero = True
            break
        cuts.append(gradient)
        master_started = time.perf_counter()
        master = solve_cut_master(np.stack(cuts), theta_set=set_theta, mode="kelley")
        master_seconds = float(time.perf_counter() - master_started)
        lower = float(master.model_value)
        record = BundleIteration(
            index=iteration, theta=tuple(float(v) for v in theta), value=value, lower=lower, upper=upper,
            gap=upper - lower, inner_gaps=tuple(result.inner_gap for result in results),
            edges=tuple(result.n_edges for result in results), sweeps=tuple(result.sweeps for result in results),
            seconds=float(time.perf_counter() - iteration_started), master_seconds=master_seconds)
        records.append(record)
        if upper - lower <= float(tol) * max(1.0, abs(upper)):
            break
        # EN: `solve_cut_master` already raises on a failed LP; `status` is its free-text message.
        # CN: `solve_cut_master` 在 LP 失败时已经抛出；`status` 只是它的文本消息。
        theta = np.asarray(master.theta, dtype=np.float64)

    gap = float(records[-1].upper - records[-1].lower)
    converged = bool(gap <= float(tol) * max(1.0, abs(records[-1].upper)))
    observed_diagonal = tuple(_diagonal_mass(plan) for plan in observed)
    predicted_diagonal = tuple(_diagonal_mass(plan) for plan in best_plans) if best_plans else ()
    return BundleResult(
        theta=best_theta, value=float(best_value), lower=float(records[-1].lower), upper=float(records[-1].upper),
        gap=gap, converged=converged, plans=best_plans, edge_sets=tuple(
            edge if edge is not None else EdgeSet.from_arrays(np.zeros(0, dtype=np.int64),
                                                             np.zeros(0, dtype=np.int64), n_target=int(m.n_target))
            for edge, m in zip(edge_sets, maps, strict=True)),
        iterations=tuple(records), observed_diagonal_mass=observed_diagonal,
        predicted_diagonal_mass=predicted_diagonal, n_cuts=len(cuts),
        wall_time=float(time.perf_counter() - started), total_inner_seconds=total_inner,
        stopped_at_zero=bool(stopped_at_zero))
