"""
EN: **Dense references for the sparse bundle** — used only to check it, never on the production path.

    Two references, both of which materialise an `m x n` cost (that is their purpose):

      `dense_joint_lp`   the exact problem as **one LP**: variables `(θ, {u_l, v_l})`, constraints
                         `u_l,i + v_l,j - c_ij(θ) <= 0` for every pair and every window. Its optimum is `F*`
                         exactly, so it is the ground truth for the bracket.
      `dense_frank_wolfe` the same Frank–Wolfe scheme as `iot.relaxed.multi_plan`, but over an arbitrary
                         `FeatureMap` so it can be compared with the bundle on the *same* feature family.

    CN: **稀疏 bundle 的稠密参考** —— 只用于核对，从不在生产路径上。两个参考都会物化 `m x n` 代价（这正是它们的用途）：
        `dense_joint_lp` 把精确问题写成**一个 LP**（变量 `(θ,{u_l,v_l})`，约束 `u+v-c(θ) ≤ 0`），其最优值就是 `F*`；
        `dense_frank_wolfe` 与 `iot.relaxed.multi_plan` 相同的 Frank–Wolfe，但作用于任意 `FeatureMap`，以便与 bundle 在
        **同一特征族**上比较。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot
import scipy.sparse as sparse
from jaxtyping import Float
from scipy.optimize import linprog

from iot.core.features import FeatureMap
from iot.core.theta import LinearThetaSet

__all__ = ["DenseLPResult", "dense_frank_wolfe", "dense_joint_lp"]


@dataclass(frozen=True)
class DenseLPResult:
    """EN: the joint LP's optimum, its θ, and the solver status. CN: 联合 LP 的最优值、θ 与状态。"""

    value: float
    theta: Float[np.ndarray, "k"]
    status: str
    n_variables: int
    n_constraints: int


def _moment(features: FeatureMap, plan: sparse.csr_matrix) -> np.ndarray:
    """EN: `φᵀ P` on the plan's support. CN: 在计划支撑上计算 `φᵀP`。"""
    coo = plan.tocoo()
    phi = features.features(coo.row.astype(np.int64), coo.col.astype(np.int64))
    return np.asarray(phi.T @ coo.data.astype(np.float64), dtype=np.float64).reshape(-1)


def dense_joint_lp(plans: list[sparse.csr_matrix], features: list[FeatureMap], *,
                   theta_set: LinearThetaSet, linprog_options: dict[str, float] | None = None
                   ) -> DenseLPResult:
    """
    EN: `min_{θ, u, v} Σ_l [⟨M(π̂_l), θ⟩ − ⟨a_l, u_l⟩ − ⟨b_l, v_l⟩]` s.t. `u_l,i + v_l,j ≤ c_ij(θ)` for all `l,i,j`.

        For fixed θ the inner maximisation over `(u_l, v_l)` is the OT dual, so the LP value is `Σ_l [⟨c,π̂_l⟩ −
        OT_l(c)]` at the optimal θ — i.e. `F*`. Constraints: `L m n`, each with `K + 2` non-zeros.

    CN: 上式即 `F*`；约束 `L m n` 条，每条 `K+2` 个非零。
    """
    observed = list(plans)
    maps = list(features)
    if len(observed) != len(maps):
        raise ValueError("one feature map per plan is required")
    k = int(maps[0].n_features)
    m = int(maps[0].n_source)
    n = int(maps[0].n_target)
    n_windows = len(observed)
    moments = [_moment(map_l, plan_l) for map_l, plan_l in zip(maps, observed, strict=True)]
    observed_sum = np.sum(np.stack(moments, axis=0), axis=0)

    n_variables = k + n_windows * (m + n)
    objective = np.zeros(n_variables, dtype=np.float64)
    objective[:k] = observed_sum
    offset = k
    for plan_l in observed:
        masses_source = np.asarray(plan_l.sum(axis=1)).reshape(-1)
        masses_target = np.asarray(plan_l.sum(axis=0)).reshape(-1)
        objective[offset:offset + m] = -masses_source
        objective[offset + m:offset + m + n] = -masses_target
        offset += m + n

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    row_index = 0
    for position, map_l in enumerate(maps):
        base = k + position * (m + n)
        grid_rows = np.repeat(np.arange(m, dtype=np.int64), n)
        grid_cols = np.tile(np.arange(n, dtype=np.int64), m)
        phi = np.asarray(map_l.features(grid_rows, grid_cols), dtype=np.float64)   # (m n, k)
        n_edges = m * n
        # EN: θ block: -φ (row block of the constraint matrix); u and v blocks: +1 on their column.
        # CN: θ 块为 -φ；u、v 块在各自列上取 +1。
        for column in range(k):
            rows.append(np.arange(row_index, row_index + n_edges, dtype=np.int64))
            cols.append(np.full(n_edges, column, dtype=np.int64))
            data.append(-phi[:, column])
        rows.append(np.arange(row_index, row_index + n_edges, dtype=np.int64))
        cols.append(base + grid_rows)
        data.append(np.ones(n_edges, dtype=np.float64))
        rows.append(np.arange(row_index, row_index + n_edges, dtype=np.int64))
        cols.append(base + m + grid_cols)
        data.append(np.ones(n_edges, dtype=np.float64))
        row_index += n_edges
    matrix = sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(row_index, n_variables))
    constraints = theta_set.linear_constraints()
    a_eq = sparse.hstack([
        sparse.csr_matrix(np.asarray(constraints.a_eq, dtype=np.float64)),
        sparse.csr_matrix((constraints.a_eq.shape[0], n_variables - k)),
    ], format="csr")
    b_eq = np.asarray(constraints.b_eq, dtype=np.float64).reshape(-1)
    bounds = [*constraints.bounds()] + [(-np.inf, np.inf)] * (n_variables - k)
    result = linprog(objective, A_ub=matrix, b_ub=np.zeros(row_index, dtype=np.float64),
                     A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs",
                     options={} if linprog_options is None else dict(linprog_options))
    if not result.success or result.x is None:
        raise RuntimeError(f"dense joint LP failed: {result.message}")
    return DenseLPResult(value=float(result.fun), theta=np.ascontiguousarray(result.x[:k]),
                         status=str(result.message), n_variables=int(n_variables),
                         n_constraints=int(row_index))


def dense_frank_wolfe(plans: list[sparse.csr_matrix], features: list[FeatureMap], *,
                      theta_set: LinearThetaSet, iterations: int = 200,
                      num_itermax: int = 1_000_000,
                      stop: object | None = None) -> tuple[float, np.ndarray, tuple[float, ...]]:
    """
    EN: the dense Frank–Wolfe of `iot.relaxed.multi_plan`, over arbitrary feature maps.

        Each iteration builds an `m x n` cost per window, solves the forward OT with POT's dense network simplex, and
        takes the standard `2/(t+2)` step. Returns `(best F, best θ, trace)`. It is deliberately the same algorithm as
        the dense reference arm so the comparison isolates the memory strategy.

    CN: `iot.relaxed.multi_plan` 的稠密 Frank–Wolfe，作用于任意特征映射：每轮每窗物化 `m x n` 代价、用 POT 稠密网络
        单纯形求解前向 OT，并取标准 `2/(t+2)` 步长。返回 `(最优 F, 最优 θ, 轨迹)`。
    """
    observed = list(plans)
    maps = list(features)
    k = int(maps[0].n_features)
    theta = np.full(k, 1.0 / k, dtype=np.float64)
    trace: list[float] = []

    def value_of(vector: np.ndarray) -> float:
        total = 0.0
        for map_l, plan_l in zip(maps, observed, strict=True):
            coo = plan_l.tocoo()
            phi = map_l.features(coo.row.astype(np.int64), coo.col.astype(np.int64))
            mass_source = np.asarray(plan_l.sum(axis=1)).reshape(-1)
            mass_target = np.asarray(plan_l.sum(axis=0)).reshape(-1)
            rows = np.repeat(np.arange(int(map_l.n_source), dtype=np.int64), int(map_l.n_target))
            cols = np.tile(np.arange(int(map_l.n_target), dtype=np.int64), int(map_l.n_source))
            dense = np.asarray(map_l.costs(vector, rows, cols)).reshape(int(map_l.n_source),
                                                                        int(map_l.n_target))
            transport = float(ot.emd2(mass_source, mass_target, dense, numItermax=int(num_itermax)))
            total += float(phi.T @ coo.data @ vector) - transport
        return float(total)

    best = value_of(theta)
    trace.append(best)
    best_theta = theta.copy()
    for step in range(1, int(iterations) + 1):
        # EN: `stop(step, best, theta)` may end the run early (the benchmark uses it for the time/memory cap).
        # CN: `stop(step, best, theta)` 可提前结束（基准用它实现时间/内存上限）。
        if stop is not None and not bool(stop(step, best, best_theta)):  # type: ignore[operator]
            break
        gradient = np.zeros(k, dtype=np.float64)
        for map_l, plan_l in zip(maps, observed, strict=True):
            coo = plan_l.tocoo()
            phi = map_l.features(coo.row.astype(np.int64), coo.col.astype(np.int64))
            rows = np.repeat(np.arange(int(map_l.n_source), dtype=np.int64), int(map_l.n_target))
            cols = np.tile(np.arange(int(map_l.n_target), dtype=np.int64), int(map_l.n_source))
            dense = np.asarray(map_l.costs(theta, rows, cols)).reshape(int(map_l.n_source),
                                                                        int(map_l.n_target))
            mass_source = np.asarray(plan_l.sum(axis=1)).reshape(-1)
            mass_target = np.asarray(plan_l.sum(axis=0)).reshape(-1)
            forward = np.asarray(ot.emd(mass_source, mass_target, dense, numItermax=int(num_itermax)))
            # EN: moment of the forward plan over its own nonzeros only; the cost was dense, the features need not be.
            # CN: 前向计划的矩只在其非零元上计算；代价是稠密的，特征不必是。
            forward_rows, forward_cols = np.nonzero(forward)
            plan_phi = map_l.features(forward_rows.astype(np.int64), forward_cols.astype(np.int64))
            gradient += (phi.T @ coo.data) - (plan_phi.T @ forward[forward_rows, forward_cols])
        vertex = np.zeros(k, dtype=np.float64)
        vertex[int(np.argmin(gradient))] = 1.0
        step_size = 2.0 / (step + 2.0)
        candidate = (1.0 - step_size) * theta + step_size * vertex
        candidate_value = value_of(candidate)
        if candidate_value <= best:
            theta, best = candidate, candidate_value
            if candidate_value < trace[-1]:
                best_theta = theta.copy()
        trace.append(best)
    return float(best), best_theta, tuple(trace)
