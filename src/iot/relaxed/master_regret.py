"""
CN: **松弛（最小 regret）master** —— 逆 OT 的一个**替代**形式，与现有 max-margin master 并存、互不影响。

    变量 `w = (theta, f, g)`，在边集 `E`（`S ⊆ E`）上求解
        minimise   R(w) = theta·phi_bar - a^T f - b^T g,   phi_bar = sum_{e in S} pi_e phi_e
        s.t.       sigma_e = f_i + g_j - theta·phi_ij <= 0   for ALL e in E
                   theta in Theta (单纯形),   g_1 = 0
    没有 delta、没有 delta>=0、S 上没有等式约束。

    CN: **为什么有界**（与 max-margin 的关键差别）：`g_1 = 0` 钉住全局平移；`B = S ∪ R` 的连接弧钉住
        分量间平移；`S ⊆ E` 且 `pi_e > 0` 使 `R = -sum_S pi_e sigma_e >= 0` 在整个可行集上成立，
        所以 R 有下界、stage 1 不可能无界。**注意 sigma 的符号**：`sigma = f+g-c`，
        因此 `A_ub` 的 S 行是 `[+1 at f_i, +1 at g_j, -phi at theta] <= 0`。
        编码成 `f+g >= c` 会让 R 无界地变负——这个符号错误在开发中实际发生过，必须由测试锁住。

EN: **Relaxed (minimum-regret) master** -- an **alternative** inverse-OT formulation, coexisting with the
    existing max-margin master and leaving it untouched.

    Variables `w = (theta, f, g)`; on an edge set `E` (`S ⊆ E`):
        minimise   R(w) = theta·phi_bar - a^T f - b^T g,   phi_bar = sum_{e in S} pi_e phi_e
        s.t.       sigma_e = f_i + g_j - theta·phi_ij <= 0   for ALL e in E
                   theta in Theta (simplex),   g_1 = 0
    No delta, no delta >= 0, no equality constraints on S.

    **Why it is bounded:** `g_1 = 0` pins the global shift and the connector arcs pin the component shifts;
    `S ⊆ E` with `pi_e > 0` gives `R = -sum_S pi_e sigma_e >= 0` on the whole feasible set, so R is bounded
    below and stage 1 cannot be unbounded. **Mind the sign of sigma:** `sigma = f+g-c`, so each S row of
    `A_ub` is `[+1 at f_i, +1 at g_j, -phi at theta] <= 0`. Encoding `f+g >= c` makes R unbounded below;
    that exact sign bug happened during development and is locked down by a test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from ..core.edges import EdgeSet
from ..core.problem import InverseOTError, InverseOTProblem
from ..core.theta import LinearThetaSet

__all__ = [
    "RegretSolution",
    "RegretMasterSolver",
    "build_regret_lp",
    "regret_objective",
    "support_duals",
]


@dataclass(frozen=True)
class RegretSolution:
    """
    CN: 一次 regret master 的结果：w=(theta,f,g)、目标值 R、以及诊断。
    EN: One regret-master result: w=(theta,f,g), the objective R, and diagnostics.
    """

    theta: np.ndarray
    source_dual: np.ndarray
    target_dual: np.ndarray
    regret: float
    status: str
    n_edges: int
    box_bound_active: bool = False
    regret_at_solution: float = 0.0


@dataclass(frozen=True)
class _RegretSystem:
    """CN: 稀疏 LP 的矩阵视图。EN: Matrix view of the sparse LP."""

    a_ub: sparse.csr_matrix
    b_ub: np.ndarray
    a_eq: sparse.csr_matrix
    b_eq: np.ndarray
    bounds: list[tuple[float | None, float | None]]
    n_features: int
    n_source: int
    n_target: int

    @property
    def size(self) -> int:
        return int(self.n_features) + int(self.n_source) + int(self.n_target)


def build_regret_lp(problem: InverseOTProblem, edges: EdgeSet, *,
                    box: float | None = None) -> _RegretSystem:
    """
    CN: 构造 §2 的稀疏 LP。`sigma_e = f_i + g_j - theta·phi_ij <= 0`，即
        `A_ub` 行为 `[+1 at f_i, +1 at g_j, -phi_ij at theta]`，右端 0。
        `box` 仅用于测试（默认 None）：`|f|,|g| <= box`。
    EN: Build §2's sparse LP. `sigma_e = f_i + g_j - theta·phi_ij <= 0`, i.e. each `A_ub` row is
        `[+1 at f_i, +1 at g_j, -phi_ij at theta]` with rhs 0. `box` is tests-only (default None).
    """
    if edges.size < 1:
        raise InverseOTError("regret master requires a non-empty edge set")
    if not isinstance(problem.theta_set, LinearThetaSet):
        raise InverseOTError(
            "the regret LP master requires a LinearThetaSet; "
            f"got {type(problem.theta_set).__name__}")
    key = (int(problem.n_features), int(problem.n_source), int(problem.n_target))
    n_theta, n_source, n_target = key
    size = n_theta + n_source + n_target

    constraint_rows = np.arange(edges.size, dtype=np.int64)
    features = problem.features.features(edges.rows, edges.cols)
    if features.shape != (edges.size, n_theta):
        raise InverseOTError("features() returned an unexpected number of feature columns")

    from scipy.sparse import coo_matrix

    # CN: 三种块分别给出 (行, 列) 坐标；统一用**绝对列号**，避免行主序与展平索引混用。
    # EN: three blocks, each with absolute (row, col) coordinates; avoids mixing row-major with flat indices.
    edge_idx = np.arange(edges.size, dtype=np.int64)
    theta_cols = np.tile(np.arange(n_theta, dtype=np.int64), edges.size)
    theta_rows = np.repeat(edge_idx, n_theta)
    f_rows = edge_idx
    f_cols = n_theta + edges.rows
    g_rows = edge_idx
    g_cols = n_theta + n_source + edges.cols
    a_ub = coo_matrix(
        (np.concatenate([-features.reshape(-1), np.ones(edges.size), np.ones(edges.size)]),
         (np.concatenate([theta_rows, f_rows, g_rows]),
          np.concatenate([theta_cols, f_cols, g_cols]))),
        shape=(edges.size, size),
    ).tocsr()
    b_ub = np.zeros(edges.size, dtype=np.float64)

    theta_constraints = problem.theta_set.linear_constraints()
    # CN: Theta 的等式只占 theta 列，需要右侧补零到总变量数，才能与规范行纵向拼接。
    # EN: Theta equalities live only in the theta columns; pad zeros to the full variable count so they
    #     can be vstacked with the gauge row.
    theta_rows_eq = np.asarray(theta_constraints.a_eq, dtype=np.float64)
    eq_wide = np.hstack([theta_rows_eq,
                         np.zeros((theta_rows_eq.shape[0], size - n_theta), dtype=np.float64)])
    gauge = np.zeros((1, size), dtype=np.float64)
    gauge[0, n_theta + n_source] = 1.0
    a_eq = sparse.csr_matrix(np.vstack([eq_wide, gauge]))
    b_eq = np.concatenate([np.asarray(theta_constraints.b_eq, dtype=np.float64),
                           np.zeros(1, dtype=np.float64)])

    bounds: list[tuple[float | None, float | None]] = []
    for index in range(n_theta):
        lower = float(theta_constraints.lower[index])
        upper = float(theta_constraints.upper[index])
        bounds.append((None if not np.isfinite(lower) else lower,
                       None if not np.isfinite(upper) else upper))
    dual_bound: tuple[float | None, float | None] = (
        (None, None) if box is None else (-float(box), float(box)))
    bounds.extend([dual_bound] * (n_source + n_target))
    return _RegretSystem(a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq, bounds=bounds,
                         n_features=n_theta, n_source=n_source, n_target=n_target)


def regret_objective(problem: InverseOTProblem) -> np.ndarray:
    """
    CN: `R` 的线性系数 `(phi_bar, -a, -b)`，其中 `phi_bar = sum_S pi_e phi_e`。
    EN: Linear coefficients `(phi_bar, -a, -b)` of `R`, with `phi_bar = sum_S pi_e phi_e`.
    """
    rows, cols, values = problem.support()
    features = problem.features.features(rows, cols)
    phi_bar = (np.asarray(values, dtype=np.float64)[:, None] * features).sum(axis=0)
    a = np.asarray(problem.resolved_source_mass, dtype=np.float64).reshape(-1)
    b = np.asarray(problem.resolved_target_mass, dtype=np.float64).reshape(-1)
    if abs(float(a.sum()) - float(b.sum())) > 1e-8 * max(1.0, float(a.sum())):
        raise InverseOTError("source and target masses must have equal total mass")
    return np.concatenate([phi_bar, -a, -b])


def support_duals(problem: InverseOTProblem, theta: np.ndarray, edges: EdgeSet) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 可行点构造：`f_i = min_{e=(i,j) in E} (theta·phi_e)`，`g = 0`。
    EN: A feasible point: `f_i = min_{e=(i,j) in E} (theta·phi_e)`, `g = 0`.
    """
    n_source = int(problem.n_source)
    n_target = int(problem.n_target)
    value = np.asarray(theta, dtype=np.float64)
    features = problem.features.features(edges.rows, edges.cols)
    cost = features @ value
    f = np.full(n_source, np.inf, dtype=np.float64)
    np.minimum.at(f, edges.rows, cost)
    f[~np.isfinite(f)] = 0.0
    return f, np.zeros(n_target, dtype=np.float64)


class RegretMasterSolver:
    """
    CN: §2 的 regret master，带 §4 的两段式选择规则（先最小化 R，再在 R <= R* + eta 上最大化 <r,w>）。
        **不带**扰动单次求解与 hysteresis（未覆盖，按任务要求暂不实现）。
    EN: §2's regret master with §4's two-stage rule (minimise R, then maximise `<r,w>` on `R <= R* + eta`).
        **No** perturbed one-solve rule and **no** hysteresis (uncovered; left out per the task).

    CN: 全程**故障即抛**：status 非 optimal 一律 `InverseOTError`，绝不给默认 theta。
    EN: **Fail loud**: any non-optimal status raises `InverseOTError`; a default theta is never returned.
    """

    def __init__(self, *, eta_scale: float = 1e-9, box: float | None = None,
                 box_check: bool = True, tie_break: bool = True) -> None:
        if float(eta_scale) < 0.0:
            raise ValueError("eta_scale must be nonnegative")
        self.eta_scale = float(eta_scale)
        self.box = box
        self.box_check = bool(box_check)
        self.tie_break = bool(tie_break)

    def solve(self, problem: InverseOTProblem, edges: EdgeSet) -> RegretSolution:
        """CN: 求解一次（含两段式）。EN: One solve (including the two-stage rule)."""
        key = problem.support_keys()
        if not np.all(np.isin(key, edges.keys)):
            raise InverseOTError("S must be a subset of the edge set E (B1)")

        system = build_regret_lp(problem, edges, box=self.box)
        objective = regret_objective(problem)
        if objective.size != system.size:
            raise InverseOTError("objective and variable layout disagree")

        primal, value, status = _run_lp(system, objective, method="highs")
        if primal is None or value is None:
            raise InverseOTError(
                "regret master (stage 1) is not optimal: "
                f"status={status!r}, |E|={edges.size}, m={problem.n_source}, n={problem.n_target}, "
                f"K={problem.n_features}")
        if not np.isfinite(value):
            raise InverseOTError(f"regret master returned a non-finite objective: {value!r}")

        box_active = self._box_active(system, primal)
        if self.box is not None and self.box_check and box_active:
            raise InverseOTError(
                "safety box is active; the reported regret would be box-dependent "
                f"(|E|={edges.size}, m={problem.n_source}, n={problem.n_target})")

        theta = np.asarray(primal[: system.n_features], dtype=np.float64)
        f = np.asarray(primal[system.n_features: system.n_features + system.n_source],
                       dtype=np.float64)
        g = np.asarray(primal[system.n_features + system.n_source:], dtype=np.float64)

        if self.tie_break:
            r = _generic_direction(system.size)
            extra_ub = [objective]
            extra_rhs = [float(value) + self.eta_scale * max(1.0, abs(float(value)))]
            chosen, _chosen_value, chosen_status = _run_lp(
                system, -r, extra_ub=extra_ub, extra_ub_rhs=extra_rhs)
            if chosen is None:
                raise InverseOTError(
                    f"regret master (stage 2) failed: status={chosen_status!r}, "
                    f"R*={value!r}, |E|={edges.size}")
            if self.box is not None and self.box_check and self._box_active(system, chosen):
                raise InverseOTError("safety box is active in stage 2")
            primal = chosen
            theta = np.asarray(primal[: system.n_features], dtype=np.float64)
            f = np.asarray(primal[system.n_features: system.n_features + system.n_source],
                           dtype=np.float64)
            g = np.asarray(primal[system.n_features + system.n_source:], dtype=np.float64)

        return RegretSolution(theta=theta, source_dual=f, target_dual=g, regret=float(value),
                              status=str(status), n_edges=int(edges.size),
                              box_bound_active=bool(box_active),
                              regret_at_solution=float(objective @ primal))

    def _box_active(self, system: _RegretSystem, primal: np.ndarray) -> bool:
        """CN: 检查 `f,g` 是否顶到 box 边界。EN: Whether `f,g` sit on the box boundary."""
        if self.box is None:
            return False
        limit = float(self.box)
        dual = primal[system.n_features:]
        return bool(np.any(np.abs(dual) >= limit - 1e-6 * max(1.0, limit)))


def _generic_direction(size: int, *, seed: int = 20261021) -> np.ndarray:
    """CN: 整轮固定的通用方向 `r ~ N(0, I)`。EN: Run-fixed generic direction `r ~ N(0, I)`."""
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal(int(size))


def _run_lp(system: _RegretSystem, objective: np.ndarray, *, method: str = "highs",
            extra_ub: list[np.ndarray] | None = None,
            extra_ub_rhs: list[float] | None = None,
            ) -> tuple[np.ndarray | None, float | None, str]:
    """
    CN: 调用 HiGHS；非最优一律返回 `(None, None, message)`，由调用方抛出（B5：无静默回退）。
    EN: Call HiGHS; any non-optimal status returns `(None, None, message)` for the caller to raise
        (B5: no silent fallback).
    """
    a_ub = system.a_ub
    b_ub = system.b_ub
    if extra_ub:
        a_ub = sparse.vstack([a_ub, sparse.csr_matrix(np.vstack(extra_ub))], format="csr")
        b_ub = np.concatenate([b_ub, np.asarray(extra_ub_rhs, dtype=np.float64)])
    result = linprog(
        np.asarray(objective, dtype=np.float64),
        A_ub=a_ub if a_ub.shape[0] else None,
        b_ub=b_ub if a_ub.shape[0] else None,
        A_eq=system.a_eq if system.a_eq.shape[0] else None,
        b_eq=system.b_eq if system.a_eq.shape[0] else None,
        bounds=system.bounds,
        method=method,
    )
    if not result.success or result.x is None:
        return None, None, str(result.message)
    return np.asarray(result.x, dtype=np.float64), float(result.fun), str(result.message)
