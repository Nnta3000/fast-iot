"""
CN: **LP 装配与 CuPDLPx 适配的共享层** —— 两个逆 OT master 共用同一份代码：

      | | 精确（max-margin） | 松弛（min-regret） |
      |---|---|---|
      | 变量 | `delta, theta, f, g` | `theta, f, g` |
      | S 边 | **等式** | **不等式** |
      | 其它边 | `f+g-c+delta <= 0` | `f+g-c <= 0` |
      | 目标 | `max delta`（即 `c[delta] = -1`） | `min theta·phi_bar - a·f - b·g` |

    两者的变量布局、Theta 约束、规范 `g_0 = 0`、上下界、以及 `loadData` 调用**完全相同**。
    因此这里只写一次装配（`assemble_lp`），两个 master 只传三个参数：
    `has_delta`、`objective`、`support_as_equalities`。

    CN: 扫描（HELLO / torch）是**分离**阶段，不涉及 LP，因此不在本模块。
    CN: 复杂度两个 master 相同：非零元 `O(|E|·(K+2))`，变量 `O(K+m+n)`。松弛版少一列 delta、
        且 S 边不进等式块，所以略小。

EN: **Shared LP assembly and CuPDLPx adapter** for both inverse-OT masters. They differ in exactly
    three things (delta column, objective, S as equality or inequality); everything else -- layout,
    Theta rows, the `g_0 = 0` gauge, bounds and the `loadData` call -- is identical, so the LP is
    assembled once here. The separation scans (HELLO / torch) do not touch the LP and live elsewhere.

    Both masters have the same complexity: `O(|E|·(K+2))` nonzeros and `O(K+m+n)` variables. The
    relaxed one is slightly smaller (no delta column, S edges in the inequality block).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from ..core.edges import EdgeSet
from ..core.problem import InverseOTError, InverseOTProblem
from ..core.theta import LinearThetaSet, SimplexTheta

__all__ = [
    "RepairedPoint",
    "repair_point",
    "LpLayout",
    "LpSystem",
    "LpResult",
    "assemble_lp",
    "solve_lp_cupdlpx",
    "solve_lp_highs",
    "cupdlpx_available",
    "project_theta_if_simplex",
]


@dataclass(frozen=True)
class LpLayout:
    """
    CN: 变量布局 `x = [delta?] + theta + f + g`。无 delta 时 theta 从 0 开始，`delta` 为 -1。
    EN: Layout `x = [delta?] + theta + f + g`; without delta, theta starts at 0 and `delta` is -1.
    """

    has_delta: bool
    n_features: int
    n_source: int
    n_target: int

    @property
    def delta(self) -> int:
        return 0 if self.has_delta else -1

    @property
    def theta_start(self) -> int:
        return 1 if self.has_delta else 0

    @property
    def source_start(self) -> int:
        return self.theta_start + int(self.n_features)

    @property
    def target_start(self) -> int:
        return self.source_start + int(self.n_source)

    @property
    def size(self) -> int:
        return self.target_start + int(self.n_target)


@dataclass(frozen=True)
class LpSystem:
    """
    CN: 装配好的 LP：`a_eq x = b_eq`、`a_ub x <= b_ub`、`lb <= x <= ub`，最小化 `c·x`。
    EN: Assembled LP: `a_eq x = b_eq`, `a_ub x <= b_ub`, `lb <= x <= ub`, minimising `c·x`.
    """

    a_eq: sparse.csr_matrix
    b_eq: np.ndarray
    a_ub: sparse.csr_matrix
    b_ub: np.ndarray
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    layout: LpLayout

    @property
    def size(self) -> int:
        return int(self.c.size)

    def split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CN: 取出 (theta, f, g)。EN: split into (theta, f, g)."""
        lay = self.layout
        k, m = int(lay.n_features), int(lay.n_source)
        return (np.ascontiguousarray(x[lay.theta_start:lay.theta_start + k], dtype=np.float64),
                np.ascontiguousarray(x[lay.source_start:lay.source_start + m], dtype=np.float64),
                np.ascontiguousarray(x[lay.target_start:], dtype=np.float64))

    def delta_of(self, x: np.ndarray) -> float:
        """CN: delta 值（无 delta 列时为 0）。EN: delta value (0 when there is no delta column)."""
        return float(x[int(self.layout.delta)]) if self.layout.has_delta else 0.0


def _edge_block(features: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                layout: LpLayout, *, with_delta: bool) -> sparse.csr_matrix:
    """
    CN: 一组边的约束行：`theta` 上 `-phi_e`，`f_i` 上 `+1`，`g_j` 上 `+1`；可选 `delta` 上 `+1`。
        返回形状 `(n_edges, layout.size)`。
    EN: Constraint rows for a set of edges: `-phi_e` on theta, `+1` on `f_i`, `+1` on `g_j`, plus
        `+1` on delta when requested. Shape `(n_edges, layout.size)`.
    """
    n_edges, k = int(features.shape[0]), int(layout.n_features)
    block = sparse.lil_matrix((n_edges, layout.size), dtype=np.float64)
    e = np.arange(n_edges)
    block[e[:, None], layout.theta_start + np.arange(k)[None, :]] = -features
    block[e, layout.source_start + rows] = 1.0
    block[e, layout.target_start + cols] = 1.0
    if with_delta:
        block[:, int(layout.delta)] = 1.0
    return block.tocsr()


def assemble_lp(
    problem: InverseOTProblem,
    edges: EdgeSet,
    *,
    has_delta: bool,
    objective: np.ndarray,
    support_as_equalities: bool,
) -> LpSystem:
    """
    CN: **唯一的 LP 装配**。约束（每条边 e=(i,j)）：
          `f_i + g_j - theta·phi_e ( + delta if has_delta and e not in S )` 在 S 上 `= 0`（若
          `support_as_equalities`），否则 `<= 0`。再附加 Theta 的线性等式与 `g_0 = 0`。
    EN: **The single LP assembly.** Per edge e=(i,j): `f_i + g_j - theta·phi_e` (+ delta when
        `has_delta` and `e not in S`) equals 0 on S when `support_as_equalities`, otherwise <= 0. Theta's
        linear equalities and `g_0 = 0` are appended.
    """
    if edges.size < 1:
        raise InverseOTError("master requires a non-empty edge set")
    if not isinstance(problem.theta_set, LinearThetaSet):
        raise InverseOTError(
            "the LP master requires a LinearThetaSet; "
            f"got {type(problem.theta_set).__name__}. Pair a non-linear Theta with a dedicated master.")
    k, m, n = int(problem.n_features), int(problem.n_source), int(problem.n_target)
    layout = LpLayout(has_delta=bool(has_delta), n_features=k, n_source=m, n_target=n)
    if np.asarray(objective).size != layout.size:
        raise InverseOTError(
            f"objective has {np.asarray(objective).size} entries, layout has {layout.size}")

    features = problem.features.features(edges.rows, edges.cols)
    if features.shape != (edges.size, k):
        raise InverseOTError("features() returned an unexpected number of feature columns")
    in_support = np.isin(edges.keys, problem.support_keys())

    if support_as_equalities:
        eq_mask = in_support
    else:
        eq_mask = np.zeros(edges.size, dtype=bool)
    ub_mask = ~eq_mask

    a_eq_parts: list[sparse.csr_matrix] = []
    if np.any(eq_mask):
        a_eq_parts.append(_edge_block(features[eq_mask], edges.rows[eq_mask], edges.cols[eq_mask],
                                      layout, with_delta=False))
    theta_constraints = problem.theta_set.linear_constraints()
    theta_block = np.asarray(theta_constraints.a_eq, dtype=np.float64)
    extra = np.zeros((theta_block.shape[0] + 1, layout.size), dtype=np.float64)
    extra[:theta_block.shape[0], layout.theta_start:layout.theta_start + k] = theta_block
    extra[theta_block.shape[0], layout.target_start] = 1.0
    a_eq_parts.append(sparse.csr_matrix(extra))
    a_eq = sparse.vstack(a_eq_parts, format="csr")
    b_eq = np.concatenate([np.zeros(int(np.count_nonzero(eq_mask))),
                           np.asarray(theta_constraints.b_eq, dtype=np.float64),
                           np.zeros(1, dtype=np.float64)])

    if np.any(ub_mask):
        # CN: delta 只出现在非 S 边（S 边在 max-margin 里不加 delta）。
        # EN: delta appears only on non-S edges (S edges carry no delta in the max-margin form).
        a_ub = _edge_block(features[ub_mask], edges.rows[ub_mask], edges.cols[ub_mask],
                           layout, with_delta=bool(has_delta))
    else:
        a_ub = sparse.csr_matrix((0, layout.size), dtype=np.float64)
    b_ub = np.zeros(int(a_ub.shape[0]), dtype=np.float64)

    lb = np.full(layout.size, -np.inf, dtype=np.float64)
    ub = np.full(layout.size, np.inf, dtype=np.float64)
    if has_delta:
        lb[layout.delta] = 0.0
    for index in range(k):
        lower = float(theta_constraints.lower[index])
        upper = float(theta_constraints.upper[index])
        if np.isfinite(lower):
            lb[layout.theta_start + index] = lower
        if np.isfinite(upper):
            ub[layout.theta_start + index] = upper
    return LpSystem(a_eq=a_eq, b_eq=b_eq, a_ub=a_ub, b_ub=b_ub,
                    c=np.asarray(objective, dtype=np.float64), lb=lb, ub=ub, layout=layout)


def cupdlpx_available() -> bool:
    """CN: CuPDLPx 是否可用。EN: whether CuPDLPx is importable."""
    try:
        _require_pycupdlpx()
        return True
    except InverseOTError:
        return False


def _require_pycupdlpx() -> Any:
    """CN: 惰性导入 CuPDLPx（先导入 torch 以加载 libtorch）。EN: lazy import (torch first)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    try:
        from hello_ot._native import pycupdlpx  # type: ignore[import-not-found]

        return pycupdlpx
    except ImportError:
        pass
    try:
        import pycupdlpx  # type: ignore[import-not-found,no-redef]

        return pycupdlpx
    except ImportError as error:
        raise InverseOTError(
            "CuPDLPx (pycupdlpx) is not available. Build it with the documented stack "
            "(`bash scripts/create_hello_ot_env.sh`, then `HELLO_OT_BUILD_NATIVE=1 python -m pip "
            "install -e .`) or the local CUDA-12/13 helper. `backend='highs'` remains available."
        ) from error


@dataclass(frozen=True)
class RepairedPoint:
    """
    CN: 修复后的点：`theta` 精确投到单纯形、`g_0 = 0`（规范）、`f` 已减去 `s` 使全域可行。
        `regret` 由**修复后的向量**算出（绝不读求解器目标）；`sigma_max_raw` 是修复前的 `max sigma`。
    EN: The repaired point: `theta` exactly projected onto the simplex, `g_0 = 0` (gauge), and `f`
        shifted by `-s` so the whole grid is feasible. `regret` is computed from the **repaired vector**
        (never read from the solver objective); `sigma_max_raw` is the pre-repair maximum.
    """

    theta: np.ndarray
    source_dual: np.ndarray
    target_dual: np.ndarray
    regret: float
    sigma_max_raw: float
    scale: float
    theta_pre: np.ndarray | None = None
    source_dual_pre: np.ndarray | None = None
    target_dual_pre: np.ndarray | None = None
    regret_pre: float = float("nan")


def repair_point(
    problem: InverseOTProblem,
    x: np.ndarray,
    layout: LpLayout,
    *,
    all_rows: np.ndarray,
    all_cols: np.ndarray,
    support_rows: np.ndarray,
    support_cols: np.ndarray,
    support_values: np.ndarray,
    objective: np.ndarray,
    identity_rtol: float = 1e-9,
) -> RepairedPoint:
    """
    CN: **共享的返回点修复**（只在 relaxed 路径显式调用；exact 路径不受影响）。
        顺序严格如下，全部在同一个向量上进行：
          1. `theta` 精确投影到单纯形（仅当 Theta 是单纯形时）；
          2. 规范修复 `f <- f + g_0`，`g <- g - g_0`，于是 `g_0 = 0`；
          3. 用**修复后的 theta**在全图上重算 `sigma`，`s = max(0, max sigma)`；
          4. `f <- f - s`（此后全域可行）；
          5. 用修复后的向量算 `R = theta·phi_bar - a·f - b·g`，断言恒等式与 `R >= -rtol*scale`。
        `objective` 为 `(phi_bar, -a, -b)`，因此 `R = objective @ [theta, f, g]`。

    EN: **Shared returned-point repair** (called explicitly by the relaxed path only; exact paths are
        untouched). The order is exactly:
          1. project `theta` onto the simplex (only when Theta is a simplex);
          2. gauge repair `f <- f + g_0`, `g <- g - g_0`, so `g_0 = 0`;
          3. recompute `sigma` over the whole grid with the **repaired theta**, `s = max(0, max sigma)`;
          4. `f <- f - s` (the grid is now feasible);
          5. compute `R = theta·phi_bar - a·f - b·g` from the repaired vector, asserting the identity and
             `R >= -rtol*scale`.
    """
    k, m = int(layout.n_features), int(layout.n_source)
    theta = np.array(x[layout.theta_start:layout.theta_start + k], dtype=np.float64, copy=True)
    f = np.array(x[layout.source_start:layout.source_start + m], dtype=np.float64, copy=True)
    g = np.array(x[layout.target_start:layout.target_start + int(layout.n_target)],
                 dtype=np.float64, copy=True)

    # CN: 投到**实际的** Theta 上，而不只是单纯形：盒∩单纯形的情形单纯形投影会越界。
    # EN: project onto the **actual** Theta, not just the simplex: for box ∩ simplex a raw simplex
    #     projection leaves the box.
    theta = project_theta_if_simplex(theta, problem.theta_set)
    f += float(g[0])
    g -= float(g[0])

    cost_all = np.asarray(problem.features.costs(theta, all_rows, all_cols), dtype=np.float64)
    sigma_all = f[all_rows] + g[all_cols] - cost_all
    s = max(0.0, float(sigma_all.max()))
    # CN: 在平移前保存"投影 + 规范修复"后的点与其 R —— 调用方必须**扫描这个点**，
    #     否则扫描到的已是平移后（全域可行）的点，永远看不到违例。
    # EN: keep the projected + gauge-fixed point and its R **before** the shift; callers must scan this
    #     point, otherwise the scan sees the already-shifted (globally feasible) point and never finds a
    #     violation.
    theta_pre = np.array(theta, dtype=np.float64, copy=True)
    f_pre = np.array(f, dtype=np.float64, copy=True)
    g_pre = np.array(g, dtype=np.float64, copy=True)
    regret_pre = float(np.asarray(objective, dtype=np.float64) @ np.concatenate([theta_pre, f_pre, g_pre]))
    f -= s

    w = np.concatenate([theta, f, g])
    regret = float(np.asarray(objective, dtype=np.float64) @ w)
    phi_bar = float(np.asarray(objective[:k], dtype=np.float64) @ theta)
    a_f = float(np.asarray(problem.resolved_source_mass, dtype=np.float64) @ f)
    b_g = float(np.asarray(problem.resolved_target_mass, dtype=np.float64) @ g)
    scale = max(1.0, abs(phi_bar), abs(a_f), abs(b_g))

    cost_s = np.asarray(problem.features.costs(theta, support_rows, support_cols), dtype=np.float64)
    sig_s = f[support_rows] + g[support_cols] - cost_s
    identity = regret + float(np.dot(np.asarray(support_values, dtype=np.float64), sig_s))
    if abs(identity) > float(identity_rtol) * scale:
        raise InverseOTError(
            f"identity violated on the repaired array: R={regret!r}, -sum(pi*sigma)="
            f"{-float(np.dot(np.asarray(support_values, dtype=np.float64), sig_s))!r}, "
            f"rel={abs(identity) / scale:.2e}, tol={identity_rtol:.1e}x{scale:.2e}")
    if regret < -float(identity_rtol) * scale:
        raise InverseOTError(
            f"repaired regret is negative beyond tolerance: R={regret!r}, "
            f"tol={identity_rtol:.1e}x{scale:.2e}")
    if abs(float(theta.sum()) - 1.0) > 1e-9:
        raise InverseOTError(f"repaired theta does not sum to 1: {float(theta.sum())!r}")
    if float(theta.min()) < -1e-12:
        raise InverseOTError(f"repaired theta has a negative entry: {float(theta.min())!r}")
    if abs(float(g[0])) > 1e-12:
        raise InverseOTError(f"gauge not fixed: g_0={float(g[0])!r}")
    return RepairedPoint(theta=theta, source_dual=f, target_dual=g, regret=regret,
                         sigma_max_raw=s, scale=scale, theta_pre=theta_pre,
                         source_dual_pre=f_pre, target_dual_pre=g_pre, regret_pre=regret_pre)


@dataclass(frozen=True)
class LpResult:
    """CN: LP 求解结果。EN: LP solve result."""

    x: np.ndarray
    status: str
    backend: str
    info: dict[str, Any] = field(default_factory=dict)


# CN: 这些是 CuPDLPx **文档字符串** 里承诺的参数名。**实测**：本构建完全忽略 `params`——
#     用 `eps_opt=1e-1` 与默认值得到逐位相同的解、相同的迭代数（200）与相同的 `rel_obj_gap`。
#     因此这里对未知键**报错**，避免"调了参数却以为生效"。
#     本构建实际停在默认相对容差 ~1e-4（`rel_obj_gap` 实测 5.8e-5），且迭代数固定 200。
# EN: CuPDLPx's documented parameter names. **Measured:** this build ignores `params` entirely --
#     `eps_opt=1e-1` gives a bit-identical solution, the same iteration count (200) and the same
#     `rel_obj_gap` as the defaults. Unknown keys therefore raise, so "tuned" parameters cannot silently
#     do nothing. This build stops at its default relative tolerance (~1e-4; measured rel_obj_gap 5.8e-5)
#     with the iteration count pinned at 200.
CUPDLPX_KNOWN_PARAMS: frozenset[str] = frozenset({
    "eps_feas", "eps_opt", "eps_infeas_detect", "iter_limit", "time_sec_limit",
    "eps_feasible_absolute_primal", "eps_feasible_absolute_dual",
    "eps_feasible_relative_primal", "eps_feasible_relative_dual",
    "eps_optimal_absolute", "eps_optimal_relative",
    "dual_nnz_abs_tol", "power_tolerance", "power_max_iterations",
})


def solve_lp_cupdlpx(system: LpSystem, *, params: dict[str, Any] | None = None,
                     project: InverseOTProblem | None = None) -> LpResult:
    """
    CN: 用 CuPDLPx 解 `LpSystem`（等式行在前）。**两个 master 共用**。一阶求解只近似满足 Theta，
        因此给出 `project` 时把 theta 投回集合。
    EN: Solve `LpSystem` with CuPDLPx (equality rows first). **Shared by both masters.** A first-order
        solve satisfies Theta only approximately, so theta is projected back when `project` is given.
    """
    unknown = set(dict(params or {})) - CUPDLPX_KNOWN_PARAMS
    if unknown:
        raise InverseOTError(
            f"unknown CuPDLPx parameter(s) {sorted(unknown)}; known: {sorted(CUPDLPX_KNOWN_PARAMS)}")
    pycupdlpx = _require_pycupdlpx()
    a = sparse.vstack([system.a_eq, system.a_ub], format="csc")
    rhs = np.concatenate([system.b_eq, system.b_ub])
    solver = pycupdlpx.cupdlpx()
    solver.loadData(A=a, c=system.c, rhs=rhs, lb=system.lb, ub=system.ub,
                    nEqs=int(system.a_eq.shape[0]))
    out = solver.solve(dict(params or {}))
    primal = out.get("x") if isinstance(out, dict) else None
    if primal is None:
        raise InverseOTError(f"CuPDLPx returned no primal (out={out!r})")
    # CN: binding 的 out["x"] 是零拷贝视图，必须立刻复制。
    # EN: the binding's out["x"] is a zero-copy view; copy immediately.
    x = np.array(primal, dtype=np.float64, copy=True).reshape(-1)
    if x.size != system.size:
        raise InverseOTError(f"CuPDLPx primal has length {x.size}, expected {system.size}")
    if not np.all(np.isfinite(x)):
        reason = out.get("termination_reason") if isinstance(out, dict) else "n/a"
        raise InverseOTError(f"CuPDLPx returned a non-finite primal (termination_reason={reason})")
    if project is not None and isinstance(project.theta_set, SimplexTheta):
        lay = system.layout
        sl = slice(lay.theta_start, lay.theta_start + int(lay.n_features))
        x[sl] = project_theta_if_simplex(x[sl], project.theta_set)
    status = str(out["termination_reason"]) if isinstance(out, dict) and out.get(
        "termination_reason") is not None else "optimal"
    return LpResult(x=x, status=status, backend="cupdlpx",
                    info=dict(out) if isinstance(out, dict) else {})


def solve_lp_highs(system: LpSystem) -> LpResult:
    """
    CN: 用 HiGHS（CPU）解同一个 `LpSystem`，参考/回退用。**不是**生产路径。
    EN: Solve the same `LpSystem` with HiGHS (CPU) for reference/fallback. **Not** the production path.
    """
    from scipy.optimize import linprog

    res = linprog(system.c,
                  A_ub=system.a_ub if system.a_ub.shape[0] else None,
                  b_ub=system.b_ub if system.a_ub.shape[0] else None,
                  A_eq=system.a_eq if system.a_eq.shape[0] else None,
                  b_eq=system.b_eq if system.a_eq.shape[0] else None,
                  bounds=list(zip(system.lb.tolist(), system.ub.tolist())), method="highs")
    if not res.success or res.x is None:
        raise InverseOTError(f"HiGHS LP is not optimal: {res.message!r}")
    return LpResult(x=np.asarray(res.x, dtype=np.float64), status=str(res.message), backend="highs")


def project_theta_if_simplex(theta: np.ndarray, theta_set: Any) -> np.ndarray:
    """
    CN: 投到 `theta_set` 上。
        - 单纯形 `{Sum=1, theta>=0}`：精确欧氏投影（排序法）。
        - 带盒约束 `{Sum=1, lo<=theta<=hi}`：对偶问题 `theta_k = clip(v_k - tau, lo, hi)`，其中 `tau` 由
          `Sum(theta)=1` 唯一确定；`h(tau)=Sum(clip(v-tau,lo,hi))` 关于 tau 单调，用二分求解——**精确且保证**
          `Sum=1` 与盒约束同时成立。
    EN: Project onto `theta_set`.
        - Plain simplex `{Sum=1, theta>=0}`: exact Euclidean projection (sorting method).
        - With a box `{Sum=1, lo<=theta<=hi}`: the dual is `theta_k = clip(v_k - tau, lo, hi)` with `tau`
          fixed by `Sum(theta)=1`; `h(tau)=Sum(clip(v-tau,lo,hi))` is monotone in tau, so bisection gives an
          **exact** solution with both the sum and the box satisfied.
    """
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    try:
        constraints = theta_set.linear_constraints()
    except Exception:  # noqa: BLE001 - sets without a linear view keep the simplex projection
        return _project_simplex(values)
    lower = np.asarray(constraints.lower, dtype=np.float64)
    upper = np.asarray(constraints.upper, dtype=np.float64)
    if np.all(np.isneginf(lower)) and np.all(np.isposinf(upper)):
        return _project_simplex(values)
    lo = np.where(np.isfinite(lower), lower, 0.0)
    hi = np.where(np.isfinite(upper), upper, 1.0)
    # CN: 若盒本身不可行（Sum(lo) > 1 或 Sum(hi) < 1），退回单纯形投影并说明。
    # EN: if the box is itself infeasible (Sum(lo) > 1 or Sum(hi) < 1), fall back to the simplex projection.
    if float(lo.sum()) > 1.0 + 1e-12 or float(hi.sum()) < 1.0 - 1e-12:
        return _project_simplex(values)
    left = float(values.min() - hi.max() - 1.0)
    right = float(values.max() - lo.min() + 1.0)
    for _ in range(200):
        mid = 0.5 * (left + right)
        total = float(np.clip(values - mid, lo, hi).sum())
        if total > 1.0:
            left = mid
        else:
            right = mid
        if right - left <= 1e-15 * max(1.0, abs(mid)):
            break
    tau = 0.5 * (left + right)
    projected = np.clip(values - tau, lo, hi)
    # CN: 二分到机器精度后仍可能有 ~1e-15 的残差；把差量均摊到有夹紧余量的坐标上，保持盒约束。
    # EN: bisection may leave a ~1e-15 residual; spread it over coordinates with clipping headroom.
    residual = 1.0 - float(projected.sum())
    if abs(residual) > 0.0:
        headroom_hi = hi - projected
        headroom_lo = projected - lo
        room = headroom_hi + headroom_lo
        if float(room.sum()) > 0.0:
            projected = projected + residual * room / room.sum()
            projected = np.clip(projected, lo, hi)
    return projected


def _project_simplex(values: np.ndarray) -> np.ndarray:
    """CN: 单纯形欧氏投影（内部实现）。EN: Euclidean projection onto the simplex (internal)."""
    n = int(values.size)
    u = np.sort(values)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - css / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n)
    rho = int(ind[cond][-1])
    return np.maximum(values - css[rho - 1] / rho, 0.0)
