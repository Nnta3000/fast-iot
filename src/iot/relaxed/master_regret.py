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
from ..backends.lp_common import (
    LpLayout,
    assemble_lp,
    repair_point,
    cupdlpx_available,
    solve_lp_cupdlpx,
    solve_lp_highs,
)

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
    sigma_max_raw: float = 0.0
    scale: float = 1.0
    dual_bound: float | None = None
    regret_pre_repair: float = 0.0
    solver_iterations: int = 0
    solver_status: str = ""
    sigma_max_in_F: float = 0.0
    sigma_max_out_F: float = 0.0
    theta_pre: np.ndarray | None = None
    source_dual_pre: np.ndarray | None = None
    target_dual_pre: np.ndarray | None = None
    regret_pre: float = float("nan")
    s_pre: float = 0.0


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
    CN: 由共享装配器构造 regret master：S 边为**不等式**、无 delta 列、目标 `min R`。
        `box` 仅测试用（默认 None）。
    EN: Build the regret master through the shared assembler: S edges are **inequalities**, no delta
        column, objective `min R`. `box` is tests-only (default None).
    """
    if edges.size < 1:
        raise InverseOTError("regret master requires a non-empty edge set")
    objective = regret_objective(problem)
    shared = assemble_lp(problem, edges, has_delta=False, objective=objective,
                         support_as_equalities=False)
    lb, ub = shared.lb.copy(), shared.ub.copy()
    if box is not None:
        dual_bound = float(box)
        start = int(shared.layout.source_start)
        lb[start:] = -dual_bound
        ub[start:] = dual_bound
    system = _RegretSystem(a_ub=shared.a_ub, b_ub=shared.b_ub, a_eq=shared.a_eq, b_eq=shared.b_eq,
                           bounds=list(zip(lb.tolist(), ub.tolist())),
                           n_features=int(problem.n_features), n_source=int(problem.n_source),
                           n_target=int(problem.n_target))
    return system


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
                 box_check: bool = True, tie_break: bool = True,
                 backend: str = "cupdlpx", params: dict[str, object] | None = None) -> None:
        if float(eta_scale) < 0.0:
            raise ValueError("eta_scale must be nonnegative")
        if str(backend) not in {"cupdlpx", "highs"}:
            raise ValueError("backend must be 'cupdlpx' (GPU, production) or 'highs' (CPU, reference)")
        if str(backend) == "cupdlpx" and not cupdlpx_available():
            raise InverseOTError(
                "backend='cupdlpx' requested but CuPDLPx is unavailable; use backend='highs' for the "
                "CPU reference path.")
        self.eta_scale = float(eta_scale)
        self.box = box
        self.box_check = bool(box_check)
        self.tie_break = bool(tie_break)
        self.backend = str(backend)
        self.params = dict(params or {})

    def solve(self, problem: InverseOTProblem, edges: EdgeSet) -> RegretSolution:
        """CN: 求解一次（含两段式）。EN: One solve (including the two-stage rule)."""
        key = problem.support_keys()
        if not np.all(np.isin(key, edges.keys)):
            raise InverseOTError("S must be a subset of the edge set E (B1)")

        system = build_regret_lp(problem, edges, box=self.box)
        objective = regret_objective(problem)
        if objective.size != system.size:
            raise InverseOTError("objective and variable layout disagree")

        primal, value, status, result_info = self._solve_system(system, objective)
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

        # CN: 下游统一由 `repair_point` 从 `primal` 切片，这里不再重复切分（曾留下死赋值）。
        # EN: `repair_point` slices from `primal` downstream; no duplicate slicing here.
        if self.tie_break:
            r = _generic_direction(system.size)
            extra_ub = [objective]
            extra_rhs = [float(value) + self.eta_scale * max(1.0, abs(float(value)))]
            chosen, _chosen_value, chosen_status, _chosen_info = self._solve_system(
                system, -r, extra_ub=extra_ub, extra_ub_rhs=extra_rhs)
            if chosen is None:
                raise InverseOTError(
                    f"regret master (stage 2) failed: status={chosen_status!r}, "
                    f"R*={value!r}, |E|={edges.size}")
            if self.box is not None and self.box_check and self._box_active(system, chosen):
                raise InverseOTError("safety box is active in stage 2")
            # CN: 只用 `primal`；下游由 `repair_point` 负责切片，不必在这里再切一次。
            # EN: only `primal` is used; `repair_point` does the slicing downstream.
            primal = chosen

        # CN: 修复返回点（投影 theta、修规范、全图 sigma、f <- f - s），并只用修复后的向量报告。
        # EN: repair the returned point (project theta, fix gauge, full-grid sigma, f <- f - s) and report
        #     only quantities computed from the repaired vector.
        r, c = np.meshgrid(np.arange(system.n_source), np.arange(system.n_target), indexing="ij")
        all_rows, all_cols = r.reshape(-1), c.reshape(-1)
        sup_rows, sup_cols, sup_vals = problem.support()
        layout = LpLayout(has_delta=False, n_features=int(system.n_features),
                          n_source=int(system.n_source), n_target=int(system.n_target))
        repaired = repair_point(problem, primal, layout, all_rows=all_rows, all_cols=all_cols,
                                support_rows=sup_rows, support_cols=sup_cols,
                                support_values=sup_vals, objective=objective)
        dual_bound = None
        if isinstance(result_info, dict):
            for key in ("dual_objective", "lower_bound", "dual_obj", "obj_dual", "lb"):
                if result_info.get(key) is not None:
                    try:
                        dual_bound = float(result_info[key])
                    except (TypeError, ValueError):
                        dual_bound = None
                    break
        # CN: 修复前后两个 R，以及 F 内/外的最大违约（分开报告：F 内违约说明求解器违反了自己的约束）。
        # EN: R before and after repair, plus max sigma inside/outside F (a violation inside F means the
        #     solver breaks its own constraints).
        pre_repair = float(repaired.regret_pre)
        # CN: `edges` **就是 F 的全体**，因此"F 内" = 全部这些边；"F 外"必须用全图减去 F 来算。
        #     （此前把 `support_keys` 当作 F，于是"F 外"实际是 `F \\ S`，报出的是余量而非真实违约。）
        # EN: `edges` **is** all of F, so "inside F" means every one of these edges; "outside F" must be
        #     computed as the whole grid minus F. (Previously `support_keys` was used as F, so "outside F"
        #     was really `F \\ S` and reported a slack rather than a real violation.)
        sigma_edges = _sigma_pre(repaired, edges, problem)
        max_in = float(sigma_edges.max()) if sigma_edges.size else float("-inf")
        r_g, c_g = np.meshgrid(np.arange(int(problem.n_source)), np.arange(int(problem.n_target)),
                               indexing="ij")
        all_keys = (r_g * int(problem.n_target) + c_g).reshape(-1)
        out_keys = np.setdiff1d(all_keys, edges.keys, assume_unique=False)
        if out_keys.size:
            rows_o, cols_o = np.divmod(out_keys, int(problem.n_target))
            sig_o = (np.asarray(repaired.source_dual_pre, dtype=np.float64)[rows_o]
                     + np.asarray(repaired.target_dual_pre, dtype=np.float64)[cols_o]
                     - np.asarray(problem.features.costs(repaired.theta_pre, rows_o, cols_o),
                                  dtype=np.float64))
            max_out = float(sig_o.max())
        else:
            max_out = float("-inf")
        iters = int(result_info.get("iterations", 0)) if isinstance(result_info, dict) else 0
        return RegretSolution(theta=repaired.theta, source_dual=repaired.source_dual,
                              target_dual=repaired.target_dual, regret=repaired.regret,
                              status=str(status), n_edges=int(edges.size),
                              box_bound_active=bool(box_active),
                              regret_at_solution=repaired.regret,
                              sigma_max_raw=float(repaired.sigma_max_raw),
                              scale=float(repaired.scale), dual_bound=dual_bound,
                              regret_pre_repair=pre_repair, solver_iterations=iters,
                              solver_status=str(status),
                              sigma_max_in_F=max_in, sigma_max_out_F=max_out,
                              theta_pre=repaired.theta_pre,
                              source_dual_pre=repaired.source_dual_pre,
                              target_dual_pre=repaired.target_dual_pre,
                              regret_pre=float(repaired.regret_pre),
                              s_pre=float(repaired.sigma_max_raw))

    def _solve_system(self, system: "_RegretSystem", objective: np.ndarray, *,
                      extra_ub: list[np.ndarray] | None = None,
                      extra_ub_rhs: list[float] | None = None,
                      ) -> tuple[np.ndarray | None, float | None, str, dict]:
        """
        CN: 求解一个 regret LP。两条后端**都走共享装配/适配层**（`backends.lp_common`），
            本方法只负责把 regret 的 `_RegretSystem` 转成 `LpSystem`。
        EN: Solve one regret LP. Both backends go through the shared assembly/adapter layer
            (`backends.lp_common`); this method only converts the regret `_RegretSystem` into an
            `LpSystem`.
        """
        from ..backends.lp_common import LpLayout, LpSystem

        layout = LpLayout(has_delta=False, n_features=int(system.n_features),
                          n_source=int(system.n_source), n_target=int(system.n_target))
        lb = np.array([np.nan if lo is None else float(lo) for lo, _ in system.bounds])
        ub = np.array([np.nan if hi is None else float(hi) for _, hi in system.bounds])
        lb = np.where(np.isnan(lb), -np.inf, lb)
        ub = np.where(np.isnan(ub), np.inf, ub)
        a_ub, b_ub = system.a_ub, system.b_ub
        if extra_ub:
            a_ub = sparse.vstack([a_ub, sparse.csr_matrix(np.vstack(extra_ub))], format="csr")
            b_ub = np.concatenate([b_ub, np.asarray(extra_ub_rhs, dtype=np.float64)])
        lp = LpSystem(a_eq=system.a_eq, b_eq=system.b_eq, a_ub=a_ub, b_ub=b_ub,
                      c=np.asarray(objective, dtype=np.float64), lb=lb, ub=ub, layout=layout)
        try:
            if self.backend == "cupdlpx":
                res = solve_lp_cupdlpx(lp, params=self.params)
            else:
                res = solve_lp_highs(lp)
        except InverseOTError as error:
            return None, None, str(error), {}
        x = np.asarray(res.x, dtype=np.float64)
        return x, float(np.asarray(objective, dtype=np.float64) @ x), str(res.status), dict(res.info)

    def _box_active(self, system: _RegretSystem, primal: np.ndarray) -> bool:
        """CN: 检查 `f,g` 是否顶到 box 边界。EN: Whether `f,g` sit on the box boundary."""
        if self.box is None:
            return False
        limit = float(self.box)
        dual = primal[system.n_features:]
        return bool(np.any(np.abs(dual) >= limit - 1e-6 * max(1.0, limit)))


def _sigma_pre(repaired: object, edges: EdgeSet, problem: InverseOTProblem) -> np.ndarray:
    """
    CN: 在**平移前**的 `(theta, f, g)` 上计算 F 内每条边的 sigma。
    EN: per-edge sigma on the **pre-shift** `(theta, f, g)`.
    """
    theta = np.asarray(getattr(repaired, "theta_pre"), dtype=np.float64)
    f = np.asarray(getattr(repaired, "source_dual_pre"), dtype=np.float64)
    g = np.asarray(getattr(repaired, "target_dual_pre"), dtype=np.float64)
    cost = np.asarray(problem.features.costs(theta, edges.rows, edges.cols), dtype=np.float64)
    return f[edges.rows] + g[edges.cols] - cost


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
