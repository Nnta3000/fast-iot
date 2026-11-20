"""
EN: **Exact OT on a growing edge set, with blockwise pricing and a certificate.**

    The forward problem `min_{P in Pi(a,b)} <c(theta), P>` is solved on a restricted edge set `E`, then every pair
    *outside* `E` is priced by streaming blocks of the reduced cost `sigma_ij = u_i + v_j - c_ij(theta)`. If no
    outside edge has `sigma > tol`, the restricted plan is optimal for the **full** grid: `(u, v)` is dual feasible
    everywhere (weak duality) and complementary slackness holds on `E`. Otherwise the top-`gamma` violators per row
    and per column are added and the restricted problem is re-solved.

    Memory: `O(|E| + max_tile_elements)` doubles; no `m x n` array is created. Time: each restricted LP is a sparse
    LP with `|E|` variables and `m + n` equality rows, and each pricing sweep costs `O(m n K)` because it must look
    at every pair — **time stays quadratic, only memory is linear**.

    The restricted LP is solved with `scipy.optimize.linprog(method="highs")` on a sparse `A_eq`. This is a labelled
    **reference** path: AGENTS.md retires HiGHS from the production *master* (CuPDLPx is the production master); here
    it is used because it is the only exact sparse transport solver available in the environment (POT is dense,
    HELLO-OT is squared-Euclidean only, `networkx.network_simplex` needs integer capacities). The study measures how
    much of the runtime it takes before any replacement is proposed.

CN: **在增长边集上的精确 OT，带分块定价与证书。** 前向问题在受限边集 `E` 上求解，然后对 `E` 之外的每一对用分块的
    约化代价 `sigma = u+v-c` 定价。若外面没有 `sigma > tol`，则受限计划对**全网格**最优：`(u,v)` 处处对偶可行，
    且 `E` 上满足互补松弛；否则把每行/每列 top-`gamma` 违例加入后重解。内存 `O(|E| + max_tile_elements)`，从不创建
    `m x n` 数组；时间上每次定价扫描仍是 `O(m n K)` —— **只有内存是线性的，时间仍是二次的**。受限 LP 用
    `scipy.optimize.linprog(method="highs")`（稀疏 `A_eq`），标注为**参考**路径：AGENTS.md 把 HiGHS 从生产 master
    退役（生产 master 是 CuPDLPx），但环境里没有其它精确稀疏运输求解器（POT 稠密、HELLO-OT 仅平方欧氏、
    `networkx` 只支持整数容量）。实验会先测量它占多少运行时间，再决定是否提出替代实现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sparse
from jaxtyping import Float, Int
from scipy.optimize import linprog

from ..core.edges import EdgeSet
from ..core.features import FeatureMap
from ..core.problem import InverseOTProblem
from ..core.theta import SimplexTheta
from ..exact.master import MasterSolution
from ..exact.oracle import scan_violations

__all__ = ["SparseOTResult", "solve_sparse_ot"]

DEFAULT_LINPROG_OPTIONS: dict[str, float] = {
    # EN: tighter than HiGHS's defaults so the OT value is accurate enough to be a subgradient step.
    # CN: 比 HiGHS 默认更紧，使 OT 值精确到可以作为次梯度步。
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
}


@dataclass(frozen=True)
class SparseOTResult:
    """
    EN: one exact forward solve: the plan on `E`, its value, the duals that certify it, and the diagnostics.

        `inner_gap` is `max(0, max sigma outside E)` at termination — the certificate. `value` is `<c, P>`.
        `n_edges` is `|E|` after pruning; `n_nonzeros` is the plan's support size (at most `m + n - 1`).

    CN: 一次精确前向求解：`E` 上的计划、其值、给出证书的对偶，以及诊断量。`inner_gap` 是终止时 `E` 之外的最大
        `sigma`（截到 0），即证书；`value` 为 `<c,P>`；`n_edges` 为剪枝后的 `|E|`，`n_nonzeros` 为计划支撑大小
        （至多 `m+n-1`）。
    """

    plan: sparse.csr_matrix
    value: float
    source_dual: Float[np.ndarray, "m"]
    target_dual: Float[np.ndarray, "n"]
    edges: EdgeSet
    sweeps: int
    inner_gap: float
    n_edges: int
    max_edges: int
    n_nonzeros: int
    seconds: float = 0.0
    lp_seconds: float = 0.0
    price_seconds: float = 0.0
    history: tuple[float, ...] = field(default=())


def _north_west(source_mass: Float[np.ndarray, "m"], target_mass: Float[np.ndarray, "n"]
                ) -> tuple[Int[np.ndarray, "edges"], Int[np.ndarray, "edges"], Float[np.ndarray, "edges"]]:
    """
    EN: the north-west-corner plan: a feasible transport plan with at most `m + n - 1` edges, `O(m + n)` time.

        Used only as a starting edge set when no warm set is given; it is feasible for any marginals, which a
        cheapest-per-row start is not.

    CN: 西北角计划：可行运输计划，至多 `m+n-1` 条边，`O(m+n)` 时间。仅在无热启动时用作起始边集；它对任何 marginal
        都可行（"每行最便宜"的起点则不一定）。
    """
    a = np.asarray(source_mass, dtype=np.float64).reshape(-1)
    b = np.asarray(target_mass, dtype=np.float64).reshape(-1)
    row = 0
    col = 0
    remaining_source = float(a[0])
    remaining_target = float(b[0])
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    while row < a.size and col < b.size:
        mass = min(remaining_source, remaining_target)
        if mass > 0.0:
            rows.append(row)
            cols.append(col)
            values.append(mass)
        remaining_source -= mass
        remaining_target -= mass
        if remaining_source <= 1e-15:
            row += 1
            if row < a.size:
                remaining_source = float(a[row])
        if remaining_target <= 1e-15:
            col += 1
            if col < b.size:
                remaining_target = float(b[col])
    return (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64),
            np.asarray(values, dtype=np.float64))


def _assemble(n_source: int, n_target: int, rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> sparse.csr_matrix:
    """EN: the sparse `(m+n) x |E|` equality matrix of the transport constraints. CN: 运输约束的稀疏等式矩阵。"""
    n_edges = int(rows.size)
    index = np.arange(n_edges, dtype=np.int64)
    data = np.ones(2 * n_edges, dtype=np.float64)
    constraint = np.concatenate([rows, n_source + cols]).astype(np.int64)
    column = np.concatenate([index, index]).astype(np.int64)
    return sparse.csr_matrix((data, (constraint, column)), shape=(n_source + n_target, n_edges))


def solve_sparse_ot(
    features: FeatureMap,
    source_mass: Float[np.ndarray, "m"],
    target_mass: Float[np.ndarray, "n"],
    *,
    theta: Float[np.ndarray, "k"],
    edges: EdgeSet | None = None,
    gamma: int = 2,
    tol: float = 1e-7,
    max_sweeps: int = 60,
    max_edges: int | None = None,
    prune_slack: float | None = None,
    max_tile_elements: int = 1 << 20,
    linprog_options: dict[str, float] | None = None,
) -> SparseOTResult:
    """
    EN: solve the forward OT exactly, on a growing edge set, and certify it.

        `edges` is the warm starting set (kept across calls by the caller); `None` starts from the north-west-corner
        plan. `prune_slack` drops converged edges whose slack `c - u - v` exceeds it and that carry no mass, except
        those added in the final sweep; `None` keeps everything (the safer default while measuring growth).

    CN: 在增长边集上精确求解前向 OT 并给出证书。`edges` 是热启动集（由调用方跨轮保留），`None` 时从西北角计划开始。
        `prune_slack` 在收敛后删除松弛 `c-u-v` 超过它且不携带质量的边（最后一轮新加的除外）；`None` 全保留（测量增长
        时的更安全默认）。
    """
    started = time.perf_counter()
    a = np.ascontiguousarray(np.asarray(source_mass, dtype=np.float64).reshape(-1))
    b = np.ascontiguousarray(np.asarray(target_mass, dtype=np.float64).reshape(-1))
    n_source = int(features.n_source)
    n_target = int(features.n_target)
    if a.size != n_source or b.size != n_target:
        raise ValueError(f"masses ({a.size}, {b.size}) do not match features ({n_source}, {n_target})")
    if np.any(a < 0.0) or np.any(b < 0.0):
        raise ValueError("masses must be non-negative")
    if abs(float(a.sum()) - float(b.sum())) > 1e-9 * max(1.0, float(a.sum())):
        raise ValueError(f"masses are incompatible: sum(a)={a.sum()!r} sum(b)={b.sum()!r}")
    vector = np.asarray(theta, dtype=np.float64).reshape(-1)
    if vector.size != int(features.n_features):
        raise ValueError(f"theta has {vector.size} entries but there are {features.n_features} features")
    if int(gamma) < 1:
        raise ValueError("gamma must be at least 1")
    budget = None if max_edges is None else int(max_edges)
    options = dict(DEFAULT_LINPROG_OPTIONS if linprog_options is None else linprog_options)
    target = b

    if edges is None:
        rows, cols, _values = _north_west(a, target)
    else:
        rows, cols = edges.rows, edges.cols
    keys = np.unique(rows.astype(np.int64) * n_target + cols.astype(np.int64))
    rows = (keys // n_target).astype(np.int64)
    cols = (keys % n_target).astype(np.int64)
    support = rows * n_target + cols

    # EN: the problem object is only needed to give `scan_violations` the feature map and the dimensions.
    # CN: problem 对象只是用来把特征与维度交给 `scan_violations`。
    reference_plan = sparse.csr_matrix(
        (np.full(rows.size, 1.0 / max(1, rows.size), dtype=np.float64), (rows, cols)),
        shape=(n_source, n_target))
    problem = InverseOTProblem(features=features, theta_set=SimplexTheta(int(features.n_features)),
                               plan=reference_plan)

    lp_seconds = 0.0
    price_seconds = 0.0
    history: list[float] = []
    value = float("nan")
    source_dual = np.zeros(n_source, dtype=np.float64)
    target_dual = np.zeros(n_target, dtype=np.float64)
    repaired = False
    peak_edges = int(rows.size)
    for sweep in range(1, int(max_sweeps) + 1):
        cost_edges = np.ascontiguousarray(features.costs(vector, rows, cols), dtype=np.float64)
        matrix = _assemble(n_source, n_target, rows, cols)
        right_hand_side = np.concatenate([a, target])
        lp_started = time.perf_counter()
        result = linprog(c=cost_edges, A_eq=matrix, b_eq=right_hand_side, bounds=(0.0, None),
                         method="highs", options=options)
        lp_seconds += float(time.perf_counter() - lp_started)
        if result.status != 0 or result.x is None:
            # EN: an infeasible restricted problem means the edge set cannot carry these marginals (a pruned or
            #     hand-made warm set). Repair once by adding the north-west-corner edges, which are always feasible;
            #     a second failure is a real error and is raised.
            # CN: 受限问题不可行说明该边集承载不了这组 marginal（剪枝过度或手工热启动）。用总可行的西北角边修补一次；
            #     第二次失败是真错误，直接抛出。
            if repaired:
                raise RuntimeError(f"restricted OT LP failed: status={result.status} message={result.message!r} "
                                   f"|E|={rows.size}")
            nw_rows, nw_cols, _nw_values = _north_west(a, target)
            support = np.unique(np.concatenate([support, nw_rows * n_target + nw_cols]))
            rows = (support // n_target).astype(np.int64)
            cols = (support % n_target).astype(np.int64)
            repaired = True
            continue
        mass = np.asarray(result.x, dtype=np.float64)
        marginals = np.asarray(result.eqlin.marginals, dtype=np.float64).reshape(-1)
        u = np.ascontiguousarray(marginals[:n_source])
        v = np.ascontiguousarray(marginals[n_source:])
        # EN: HiGHS returns duals with `A^T y <= c`; verify, and flip once if the sign convention differs.
        # CN: HiGHS 返回满足 `A^T y <= c` 的对偶；先验证，符号相反时翻转一次。
        sigma_edges = u[rows] + v[cols] - cost_edges
        if float(sigma_edges.max()) > float(tol) + 1e-9:
            u, v = -u, -v
            sigma_edges = u[rows] + v[cols] - cost_edges
            if float(sigma_edges.max()) > float(tol) + 1e-9:
                raise RuntimeError(f"restricted LP duals are infeasible on E: max sigma={sigma_edges.max():.3e}")
        value = float(cost_edges @ mass)
        source_dual, target_dual = u, v

        master = MasterSolution(delta=0.0, theta=vector, source_dual=u, target_dual=v, status=str(result.status))
        price_started = time.perf_counter()
        scan = scan_violations(problem, master, excluded=EdgeSet.from_arrays(rows, cols, n_target=n_target),
                               tol=float(tol), gamma=int(gamma), max_tile_elements=int(max_tile_elements))
        price_seconds += float(time.perf_counter() - price_started)
        gap = max(0.0, float(scan.max_violation)) if np.isfinite(scan.max_violation) else 0.0
        history.append(gap)
        if gap <= float(tol):
            break
        if scan.rows.size == 0:
            raise RuntimeError(f"pricing reports gap {gap:.3e} but returned no violators")
        new_keys = np.unique(np.concatenate([support, scan.rows.astype(np.int64) * n_target
                                            + scan.cols.astype(np.int64)]))
        if budget is not None and new_keys.size > budget:
            raise RuntimeError(f"|E| would reach {new_keys.size} > max_edges={budget}; the memory claim needs |E| "
                               f"= O(n + m)")
        support = new_keys
        rows = (support // n_target).astype(np.int64)
        cols = (support % n_target).astype(np.int64)
        peak_edges = max(peak_edges, int(support.size))
    else:
        raise RuntimeError(f"sparse OT did not certify in {max_sweeps} sweeps (last gap {history[-1]:.3e})")

    plan = sparse.csr_matrix((mass, (rows, cols)), shape=(n_source, n_target))
    plan.eliminate_zeros()
    keep_rows, keep_cols = rows, cols
    if prune_slack is not None:
        slack = cost_edges - (source_dual[rows] + target_dual[cols])
        in_plan = np.asarray(plan[rows, cols]).reshape(-1) > 0.0
        keep = in_plan | (slack <= float(prune_slack))
        keep_rows, keep_cols = rows[keep], cols[keep]
    kept_edges = EdgeSet.from_arrays(keep_rows, keep_cols, n_target=n_target)
    return SparseOTResult(
        plan=plan, value=value, source_dual=source_dual, target_dual=target_dual, edges=kept_edges,
        sweeps=int(sweep), inner_gap=float(history[-1]), n_edges=int(kept_edges.size),
        max_edges=int(peak_edges),
        n_nonzeros=int(plan.nnz), seconds=float(time.perf_counter() - started),
        lp_seconds=lp_seconds, price_seconds=price_seconds, history=tuple(history))
