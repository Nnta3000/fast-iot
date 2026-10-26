"""
CN: **松弛（最小 regret）逆 OT 的约束生成循环** —— 与 `loop.py` 并列的替代实现，不改动现有循环。

    每轮只解 **stage 1**（最小化 `R`）。分离用 stage-1 点，`delta = 0`。
    终止判据是**证书化的 regret gap**：全扫描得 `s = max(0, max_e sigma_e)`，当 `s·|pi_hat| <= eps_R` 即停。
    `eps_R` 以 `theta = uniform` 的 regret 为尺度（相对量），因此与实例尺度无关。

    CN: **证书**（不需要 S 紧）：若全扫描在**所有**边上给出 `sigma_e <= s`，则 `f' = f - s` 对整个问题可行，
        且 `R(f') = R + s·|pi_hat|`，于是 `R*(F_T) <= R*_full <= R(stage1) + s·|pi_hat|`。
        `s` 由全扫描实算并 clamp 到 >= 0，不用名义 tol。

    CN: **选择规则**是**后处理**：在 `F_T` 上解一次，加约束 `R <= R* + eta`，在 `F_T` 上扫一遍。
        默认用**中心化**规则（`max-margin`：最大化到边界的距离，等价于最接近均匀加权；不使用随机方向）。

EN: **Constraint-generation loop for the relaxed (minimum-regret) inverse OT** -- an alternative to
    `loop.py`; the existing loop is untouched.

    Each round solves **stage 1 only** (minimise `R`); separation uses the stage-1 point with `delta = 0`.
    Termination is a **certified regret gap**: the full scan gives `s = max(0, max_e sigma_e)` and we stop when
    `s·|pi_hat| <= eps_R`, with `eps_R` scaled by the regret at `theta = uniform` (so it is instance-independent).

    **Certificate** (S need not be tight): if the full scan gives `sigma_e <= s` on **all** edges, then
    `f' = f - s` is feasible for the full problem with `R(f') = R + s·|pi_hat|`, hence
    `R*(F_T) <= R*_full <= R(stage1) + s·|pi_hat|`. `s` is measured, then clamped at >= 0.

    **Selection is post-processing**: one solve on `F_T` with `R <= R* + eta`, then one more scan on `F_T`.
    The default is the **central** rule (max-margin / closest to uniform); no random direction.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import ot

from ..exact.components import mandatory_edges
from ..core.edges import EdgeSet
from .master_regret import RegretMasterSolver, regret_objective
from ..exact.master import MasterSolution
from ..exact.oracle import scan_violations
from ..core.problem import InverseOTError, InverseOTProblem

__all__ = [
    "RegretIteration",
    "RegretResult",
    "solve_inverse_ot_regret",
    "full_regret",
    "SelectionMode",
]

# CN: 分离/停止用解算器精度量级的容差；证书恒等式用浮点量级容差。
# EN: separation/stopping use solver-accuracy tolerances; the identity uses floating-point tolerances.
DEFAULT_TOL = 1e-7
# CN: 分离容差至少是代价尺度的这个倍数，否则求解器噪声会把 F 填满。
# EN: the separation tolerance is at least this fraction of the cost scale, otherwise solver noise fills F.
TOL_SCALE_FRACTION = 1e-5
DEFAULT_TAU = 1e-7
IDENTITY_RTOL = 1e-10
SelectionMode = Literal["central", "uniform", "none"]


@dataclass(frozen=True)
class RegretIteration:
    """CN: 一轮的诊断（含证书区间）。EN: One-round diagnostics, including the certificate interval."""

    index: int
    regret: float
    lower: float
    upper: float
    gap: float
    support_size: int
    finite_edges: int
    violation_count: int
    added: int
    max_sigma_all: float
    wall_time: float = 0.0
    regret_pre_repair: float = 0.0
    sigma_max_in_F: float = 0.0
    sigma_max_out_F: float = 0.0
    solver_iterations: int = 0
    solver_status: str = ""


@dataclass(frozen=True)
class RegretResult:
    """
    CN: 循环结果。`lower`/`upper` 是证书两端；`theta` 是**最终选择**（后处理之后）的点。
    EN: Loop result. `lower`/`upper` bracket the certificate; `theta` is the **final selected** point
        (after post-processing).
    """

    theta: np.ndarray
    source_dual: np.ndarray
    target_dual: np.ndarray
    regret: float
    regret_at_solution: float
    lower: float
    upper: float
    gap: float
    max_sigma_all: float
    n_iterations: int
    finite_edges: int
    support_size: int
    iterations: tuple[RegretIteration, ...]
    converged: bool
    pruning_disabled: bool
    pot_regret: float | None
    pot_matches: bool | None
    total_wall_time: float
    eps_r: float
    selection_max_sigma: float | None = None


def full_regret(problem: InverseOTProblem, theta: np.ndarray, *,
                num_itermax: int = 10_000_000) -> float:
    """
    CN: 独立复核 `⟨c(θ),π̂⟩ − OT(c(θ))`（POT 精确解；未收敛直接抛错）。
    EN: Independent check `⟨c(θ),π̂⟩ − OT(c(θ))` (exact POT; raises when POT does not converge).
    """
    rows, cols, values = problem.support()
    support_cost = float(np.dot(np.asarray(values, dtype=np.float64),
                                np.asarray(problem.features.costs(theta, rows, cols),
                                           dtype=np.float64)))
    m = int(problem.n_source)
    n = int(problem.n_target)
    r_idx, c_idx = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    cost = np.asarray(problem.features.costs(theta, r_idx.reshape(-1), c_idx.reshape(-1)),
                      dtype=np.float64).reshape(m, n)
    a = np.asarray(problem.resolved_source_mass, dtype=np.float64)
    b = np.asarray(problem.resolved_target_mass, dtype=np.float64)
    transport = float(np.sum(cost * ot.emd(a / a.sum(), b / b.sum(), cost,
                                           numItermax=int(num_itermax)))) * float(a.sum())
    return support_cost - transport


def _sigma(problem: InverseOTProblem, theta: np.ndarray, f: np.ndarray, g: np.ndarray,
           rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """CN: `sigma = f + g - c`。EN: `sigma = f + g - c`."""
    if rows.size == 0:
        return np.zeros(0, dtype=np.float64)
    cost = np.asarray(problem.features.costs(theta, rows, cols), dtype=np.float64)
    return f[rows] + g[cols] - cost


def _assert_identity(problem: InverseOTProblem, theta: np.ndarray, f: np.ndarray, g: np.ndarray,
                     rows: np.ndarray, cols: np.ndarray, values: np.ndarray, regret: float,
                     *, rtol: float = IDENTITY_RTOL) -> None:
    """
    CN: 断言 `R = -sum_S pi sigma`（按各项尺度相对比较）。

        CN: **容差必须给出**：精确 LP 上与 `rtol` 同量级（1e-10 量级），但一阶求解器（CuPDLPx）只把
            `sigma <= 0` 满足到一个求解精度（默认 ~1e-7），因此该点的恒等式只能到同一量级。用同一个
            `rtol` 会在 CuPDLPx 上**误报**——恒等式本身在两种后端上都成立（实测残差 ~1e-12 相对），
            差别只是求解器把点放在可行域内的哪里。

    EN: Assert `R = -sum_S pi sigma` (relative to the terms' scale).

        **The tolerance must be supplied.** On an exact LP it is ~1e-10, but a first-order solver
        (CuPDLPx) satisfies `sigma <= 0` only to its own accuracy (default ~1e-7), so the identity at
        that point is only good to the same order. Using the exact-LP tolerance made this assert fire
        spuriously on CuPDLPx -- the identity itself holds on both backends (measured residual ~1e-12
        relative); only *where* the solver puts the point inside the feasible set differs.
    """
    sig = _sigma(problem, theta, f, g, rows, cols)
    lhs = float(regret)
    rhs = -float(np.dot(values, sig))
    cost = np.asarray(problem.features.costs(theta, rows, cols), dtype=np.float64)
    scale = max(1.0, abs(float(np.dot(values, cost))), abs(float(np.dot(values, f[rows]))),
                abs(float(np.dot(values, g[cols]))))
    if abs(lhs - rhs) > float(rtol) * scale:
        raise InverseOTError(
            f"regret identity violated: R={lhs!r} vs {rhs!r} (rel={abs(lhs - rhs) / scale:.3e}, "
            f"rtol={rtol:.1e}, scale={scale:.3e})")


def _uniform_theta(problem: InverseOTProblem) -> np.ndarray:
    """CN: 单纯形上的均匀 theta。EN: uniform theta on the simplex."""
    value = np.full(int(problem.n_features), 1.0 / float(problem.n_features), dtype=np.float64)
    if not problem.theta_set.contains(value, tol=1e-9):
        raise InverseOTError("eps_R is defined relative to uniform theta, which is not in Theta")
    return value


def _fingerprint(edges: EdgeSet) -> str:
    """CN: `F_t` 的稳定哈希。EN: stable hash of `F_t`."""
    return hashlib.sha256(np.ascontiguousarray(edges.keys).tobytes()).hexdigest()


def solve_inverse_ot_regret(
    problem: InverseOTProblem,
    *,
    tol: float | None = None,
    tau_prune: float = DEFAULT_TAU,
    gamma: int = 1,
    max_iterations: int = 200,
    master: RegretMasterSolver | None = None,
    eps_r_rel: float = 1e-6,
    prune: bool = False,
    selection: SelectionMode = "central",
    eta: float = 1e-7,
    check_pot: bool = True,
    initial_extra: EdgeSet | None = None,
    identity_rtol: float | None = None,
) -> RegretResult:
    """
    CN: 求解 §2 的松弛逆 OT。
    EN: Solve §2's relaxed inverse OT.

    CN: `eps_r_rel` 是相对的 regret gap（相对 `theta = uniform` 的 regret）。`prune=True` 是**未覆盖**的可选模式：
        每条边若在 stage-1 或 stage-2 点上紧（`sigma >= -tau`）则保留；若 `F_t` 重复则**本 run 永久关闭剪枝**。
    EN: `eps_r_rel` is the relative regret gap (relative to the regret at uniform theta). `prune=True` is an
        **uncovered** opt-in mode: an edge is kept if tight at the stage-1 or stage-2 point
        (`sigma >= -tau`); if `F_t` repeats, pruning is **disabled for the rest of the run**.
    """
    if int(gamma) < 1:
        raise ValueError("gamma must be >= 1")
    if tol is not None and float(tol) < 0.0:
        raise ValueError("tol must be nonnegative")
    if float(tau_prune) < 0.0:
        raise ValueError("tau_prune must be nonnegative")
    solver = master if master is not None else RegretMasterSolver(eta_scale=float(eta), tie_break=False)
    if identity_rtol is None:
        # CN: 一阶后端把点放在可行性内部，恒等式只能到求解精度；精确后端用浮点精度。
        # EN: a first-order backend sits inside the feasible set, so the identity is only good to solver
        #     accuracy; an exact backend gets floating point.
        backend = str(getattr(solver, "backend", "highs"))
        identity_rtol = 1e-8 if backend == "cupdlpx" else IDENTITY_RTOL
    support_rows, support_cols, support_values = problem.support()
    support = EdgeSet.from_arrays(support_rows, support_cols, n_target=int(problem.n_target))
    mandatory, _partition = mandatory_edges(problem)
    finite = mandatory if initial_extra is None else mandatory.union(initial_extra)
    if not np.all(np.isin(support.keys, finite.keys)):
        raise InverseOTError("S must be a subset of F from the first iteration (B1)")

    # CN: eps_R 的尺度：uniform theta 处的 regret（全边集，用 master 直接算一次）。
    baseline_cost = np.asarray(
        problem.features.costs(_uniform_theta(problem), support_rows, support_cols),
        dtype=np.float64)
    phi_bar = regret_objective(problem)[: int(problem.n_features)]
    f0, g0 = _best_dual_for(problem, _uniform_theta(problem))
    r0 = float(np.dot(phi_bar, _uniform_theta(problem))
               - float(np.dot(problem.resolved_source_mass, f0))
               - float(np.dot(problem.resolved_target_mass, g0)))
    _ = baseline_cost
    eps_r = float(eps_r_rel) * max(1e-12, r0)

    m = int(problem.n_source)
    n = int(problem.n_target)
    r_idx, c_idx = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    all_rows, all_cols = r_idx.reshape(-1), c_idx.reshape(-1)
    total_mass = float(np.sum(support_values))
    # CN: 容差 = max(用户给定或默认, 1e-5 * 代价尺度)。代价尺度用 uniform theta 下的 |c| 中位数。
    # EN: tolerance = max(user/default, 1e-5 * cost scale), the scale being the median |c| at uniform theta.
    probe = _uniform_theta(problem)
    cost_probe = np.abs(np.asarray(problem.features.costs(probe, all_rows, all_cols),
                                   dtype=np.float64))
    cost_scale = float(np.median(cost_probe)) if cost_probe.size else 1.0
    tol_eff = max(float(tol) if tol is not None else DEFAULT_TOL,
                  float(TOL_SCALE_FRACTION) * max(cost_scale, 1e-30))

    records: list[RegretIteration] = []
    # CN: uniform theta 作为初始 best-so-far（可行点，`s=0`，`U = R_uniform`）。
    # EN: uniform theta seeds best-so-far (feasible, `s=0`, `U = R_uniform`).
    uni0 = _uniform_theta(problem)
    f_u, g_u = _best_dual_for(problem, uni0)
    sig_u = _sigma(problem, uni0, f_u, g_u, all_rows, all_cols)
    su = max(0.0, float(sig_u.max())) if sig_u.size else 0.0
    u_uniform = float(np.dot(regret_objective(problem)[: int(problem.n_features)], uni0)
                      - float(np.dot(problem.resolved_source_mass, f_u))
                      - float(np.dot(problem.resolved_target_mass, g_u))) + su * total_mass
    best_theta = np.array(uni0, dtype=np.float64)
    best_upper = float(u_uniform)
    start = time.perf_counter()
    theta = np.zeros(int(problem.n_features), dtype=np.float64)
    f = np.zeros(m, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    regret = float("inf")
    max_sigma_all = float("inf")
    lower = float("-inf")
    upper = float("inf")
    converged = False
    pruning = bool(prune)
    pruning_disabled = False
    seen: dict[str, int] = {}
    index = 0
    while index < int(max_iterations):
        index += 1
        t0 = time.perf_counter()
        solution = solver.solve(problem, finite)
        theta, f, g = solution.theta, solution.source_dual, solution.target_dual
        # CN: 只用 stage 1 的点；tie_break=False 时返回点就是 stage-1 最优解。
        # EN: stage-1 point only; with tie_break=False the returned point is the stage-1 optimum.
        regret = float(solution.regret)
        if not np.all(np.isin(support.keys, finite.keys)):
            raise InverseOTError("S left the finite edge set (B1 violated)")
        _assert_identity(problem, theta, f, g, support_rows, support_cols, support_values, regret,
                         rtol=float(identity_rtol))

        # CN: **扫描平移前的点**（投影 + 规范修复后、尚未 `f <- f - s`）。用平移后的点扫描会让
        #     `s ~ 0`、永远看不到违例、每轮都在第 1 轮"收敛"。
        # EN: **scan the pre-shift point** (projected + gauge-fixed, before `f <- f - s`). Scanning the
        #     shifted point makes `s ~ 0`, finds no violators, and "converges" every time in round 1.
        theta_pre = np.asarray(solution.theta_pre, dtype=np.float64)
        f_pre = np.asarray(solution.source_dual_pre, dtype=np.float64)
        g_pre = np.asarray(solution.target_dual_pre, dtype=np.float64)
        as_master = MasterSolution(delta=0.0, theta=theta_pre, source_dual=f_pre, target_dual=g_pre,
                                   status=solution.status)
        scan = scan_violations(problem, as_master, excluded=finite, tol=float(tol_eff),
                               gamma=int(gamma))
        sigma_all = _sigma(problem, theta_pre, f_pre, g_pre, all_rows, all_cols)
        s_pre = max(0.0, float(sigma_all.max()))
        # CN: U = R_pre + s_pre*|pi_hat|（= 修复后的 R'），这是证书的上界。
        # EN: U = R_pre + s_pre*|pi_hat| (= the repaired R'), the certified upper bound.
        r_pre = float(solution.regret_pre)
        upper = r_pre + s_pre * total_mass
        lower = r_pre
        gap = upper - lower
        # CN: best-so-far 用上界 U 比较（uniform theta 也是候选）。
        # EN: best-so-far compares on the upper bound U (uniform theta is also a candidate).
        if best_theta is None or upper < best_upper:
            best_theta = np.array(theta, dtype=np.float64)     # repaired point (globally feasible)
            best_upper = float(upper)
        records.append(RegretIteration(
            index=index, regret=regret, lower=lower, upper=upper, gap=gap,
            support_size=int(finite.size), finite_edges=int(finite.size),
            violation_count=int(scan.violation_count), added=int(scan.rows.size),
            max_sigma_all=s_pre, wall_time=float(time.perf_counter() - t0),
            regret_pre_repair=r_pre,
            sigma_max_in_F=float(solution.sigma_max_in_F),
            sigma_max_out_F=float(solution.sigma_max_out_F),
            solver_iterations=int(solution.solver_iterations),
            solver_status=str(solution.solver_status)))
        if s_pre * total_mass <= eps_r:
            converged = True
            break

        candidates = EdgeSet.from_arrays(scan.rows, scan.cols, n_target=n)
        if pruning and pruning_disabled is False:
            keep_mask = ~np.isin(finite.keys, mandatory.keys)
            f_rows, f_cols = finite.rows[keep_mask], finite.cols[keep_mask]
            if f_rows.size:
                # CN: "要么 stage-1 要么 stage-2 紧"——stage 2 尚未运行，这里用两段式的两个点在 master 内
                #     同时评估：为避免额外求解，保留 stage-1 上紧的边，并额外保留 stage-1 的 **邻域**
                #     （`sigma >= -tau`），即保守剪枝。
                # EN: "tight at stage 1 or stage 2"; stage 2 has not run yet, so keep edges tight at the
                #     stage-1 point (conservative pruning).
                slack = _sigma(problem, theta, f, g, f_rows, f_cols)
                keep = slack >= -float(tau_prune)
                pruned = EdgeSet.from_arrays(f_rows[keep], f_cols[keep], n_target=n)
                finite = mandatory.union(pruned)
        finite = finite.union(candidates)
        fingerprint = _fingerprint(finite)
        if fingerprint in seen:
            # CN: F_t 重复——按约定永久关闭剪枝（该模式本就未覆盖）。
            # EN: F_t repeated -- disable pruning for the rest of the run (the mode is uncovered anyway).
            if pruning:
                pruning_disabled = True
                pruning = False
                seen.clear()
                continue
            raise InverseOTError(
                "regret loop revisited F_t without pruning; the instance or tolerances are inconsistent")
        seen[fingerprint] = index

    # CN: 后处理改为**可选的第二阶段约束生成**，且**不替换**解：证书区间永远来自收敛的 stage-1 点。
    # EN: post-processing is an **optional second constraint-generation phase** and never replaces the
    #     solution: the certificate interval always comes from the converged stage-1 point.
    regret_at_solution = regret
    selection_sigma: float | None = None
    if converged and selection == "central":
        selected = _select_central_cg(problem, finite, eta=float(eta), tol=float(tol),
                                      tau_prune=float(tau_prune), gamma=int(gamma),
                                      max_iterations=int(max_iterations), solver=solver)
        selection_sigma = float(selected)

    pot_regret: float | None = None
    pot_matches: bool | None = None
    if check_pot:
        pot_regret = full_regret(problem, theta)
        pot_matches = bool(abs(pot_regret - regret_at_solution) <= max(1e-7, 1e-7 * abs(pot_regret)))
    theta = best_theta if best_theta is not None else theta
    return RegretResult(
        theta=theta, source_dual=f, target_dual=g, regret=regret,
        regret_at_solution=regret_at_solution, lower=lower, upper=upper, gap=gap,
        max_sigma_all=max_sigma_all, n_iterations=len(records), finite_edges=int(finite.size),
        support_size=int(support.size), iterations=tuple(records), converged=bool(converged),
        pruning_disabled=bool(pruning_disabled), pot_regret=pot_regret, pot_matches=pot_matches,
        total_wall_time=float(time.perf_counter() - start), eps_r=eps_r,
        selection_max_sigma=selection_sigma)


def _best_dual_for(problem: InverseOTProblem, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CN: 可行对偶点 `f_i = min_j c_ij(theta)`，`g = 0`。EN: feasible duals `f_i = min_j c_ij`, `g = 0`."""
    m, n = int(problem.n_source), int(problem.n_target)
    r_idx, c_idx = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    cost = np.asarray(problem.features.costs(theta, r_idx.reshape(-1), c_idx.reshape(-1)),
                      dtype=np.float64).reshape(m, n)
    return cost.min(axis=1), np.zeros(n, dtype=np.float64)


def select_nearest_uniform(problem: InverseOTProblem, finite: EdgeSet, *, eta: float,
                           tol: float, max_iterations: int = 30) -> np.ndarray:
    """
    CN: **中心化选择**（约束生成）：先解一次得 `R*`，再限制 `R <= R* + eta` 并最小化
        `||theta - 1/K||_1`；在**全图**上扫描、插入违例边，重复直到 `max sigma <= tol`。
        L1 用 `p, n >= 0` 线性化：`theta + n - p = 1/K`，目标 `sum(p) + sum(n)`。
    EN: **Central selection** (constraint generation): one solve gives `R*`; cap `R <= R* + eta` and
        minimise `||theta - 1/K||_1`; **full-graph** scan, insert violators, repeat until
        `max sigma <= tol`. The L1 is linearised with `p, n >= 0`, `theta + n - p = 1/K`, objective
        `sum(p) + sum(n)`.
    """
    from scipy import sparse as _sparse
    from scipy.optimize import linprog

    from ..backends.lp_common import LpLayout, repair_point
    from .master_regret import build_regret_lp

    m, n = int(problem.n_source), int(problem.n_target)
    k = int(problem.n_features)
    target = 1.0 / float(k)
    layout = LpLayout(has_delta=False, n_features=k, n_source=m, n_target=n)
    r_idx, c_idx = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    all_rows, all_cols = r_idx.reshape(-1), c_idx.reshape(-1)
    sup_rows, sup_cols, sup_vals = problem.support()
    current = finite
    objective = regret_objective(problem)
    theta = np.full(k, target)
    for _round in range(int(max_iterations)):
        system = build_regret_lp(problem, current)
        base = linprog(objective, A_ub=system.a_ub, b_ub=system.b_ub, A_eq=system.a_eq,
                       b_eq=system.b_eq, bounds=system.bounds, method="highs")
        if not base.success or base.x is None:
            raise InverseOTError(f"selection: base solve failed: {base.message!r}")
        bound = float(base.fun) + float(eta)
        size = system.size

        # CN: 右侧零块的行数必须与**对应矩阵的行数**一致（a_ub 是边数，不是变量数）。
        # EN: the right-hand zero blocks need as many rows as the matrix they extend (edges, not variables).
        zeros_ub = _sparse.csr_matrix((system.a_ub.shape[0], 2 * k))
        a_ub = _sparse.vstack([
            _sparse.hstack([system.a_ub, zeros_ub], format="csr"),
            _sparse.csr_matrix(np.append(objective, np.zeros(2 * k))[None, :]),
        ], format="csr")
        b_ub = np.concatenate([system.b_ub, [bound]])

        # CN: regret LP 无 delta 列，theta 从 0 开始（见 LpLayout(has_delta=False)）。
        # EN: the regret LP has no delta column, so theta starts at 0 (see LpLayout(has_delta=False)).
        rows_pn = np.zeros((k, size + 2 * k))
        for i in range(k):
            rows_pn[i, i] = 1.0                  # theta_i
            rows_pn[i, size + i] = -1.0          # -p_i
            rows_pn[i, size + k + i] = 1.0       # +n_i
        a_eq = _sparse.vstack([
            _sparse.hstack([system.a_eq, _sparse.csr_matrix((system.a_eq.shape[0], 2 * k))],
                           format="csr"),
            _sparse.csr_matrix(rows_pn),
        ], format="csr")
        b_eq = np.concatenate([system.b_eq, np.full(k, target)])
        # CN: theta + n - p = target  =>  p 的系数应为 -1（见上），n 为 +1。
        # EN: theta + n - p = target, hence coefficient -1 on p and +1 on n (set above).
        c_obj = np.concatenate([np.zeros(size), np.ones(2 * k)])
        res = linprog(c_obj, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                      bounds=list(system.bounds) + [(0.0, None)] * (2 * k), method="highs")
        if not res.success or res.x is None:
            raise InverseOTError(f"selection: central solve failed: {res.message!r}")
        repaired = repair_point(problem, np.asarray(res.x[:size], dtype=np.float64), layout,
                                all_rows=all_rows, all_cols=all_cols,
                                support_rows=sup_rows, support_cols=sup_cols,
                                support_values=sup_vals, objective=objective)
        theta = repaired.theta
        sigma_all = _sigma(problem, theta, repaired.source_dual, repaired.target_dual,
                           all_rows, all_cols)
        if float(sigma_all.max()) <= float(tol):
            return theta
        viol = sigma_all > float(tol)
        current = current.union(EdgeSet.from_arrays(all_rows[viol], all_cols[viol], n_target=n))
    return theta


def _select_central_cg(problem: InverseOTProblem, finite: EdgeSet, *, eta: float, tol: float,
                        tau_prune: float, gamma: int, max_iterations: int,
                        solver: RegretMasterSolver) -> float:
    """
    CN: **可选的第二阶段约束生成**：选择 LP 限制 `R <= R* + eta` 并最大化最小余量，然后在**全图**上扫描，
        插入违例边，重复直到 `max sigma <= tol` 或预算耗尽。返回最终 `max sigma`。
        **不替换** stage-1 的解；调用方只用它作诊断。
    EN: **Optional second constraint-generation phase**: the selection LP caps `R <= R* + eta` and maximises the
        minimum slack, then a **full-graph** scan inserts violators, repeating until `max sigma <= tol` or the
        budget is spent. Returns the final `max sigma`. It does **not** replace the stage-1 solution; callers
        use it as a diagnostic only.
    """
    from scipy import sparse as _sparse
    from scipy.optimize import linprog

    from .master_regret import build_regret_lp

    m, n = int(problem.n_source), int(problem.n_target)
    r_idx, c_idx = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    all_rows, all_cols = r_idx.reshape(-1), c_idx.reshape(-1)
    current = finite
    objective = regret_objective(problem)
    for _round in range(int(max_iterations)):
        system = build_regret_lp(problem, current)
        base = linprog(objective, A_ub=system.a_ub, b_ub=system.b_ub, A_eq=system.a_eq,
                       b_eq=system.b_eq, bounds=system.bounds, method="highs")
        if not base.success or base.x is None:
            raise InverseOTError(f"selection: base solve failed: {base.message!r}")
        bound = float(base.fun) + float(eta)
        size = system.size
        a_ub = _sparse.hstack([system.a_ub, _sparse.csr_matrix(np.ones((system.a_ub.shape[0], 1)))],
                              format="csr")
        a_ub = _sparse.vstack([a_ub, _sparse.csr_matrix(np.append(objective, 0.0)[None, :])],
                              format="csr")
        b_ub = np.concatenate([system.b_ub, [bound]])
        a_eq = _sparse.hstack([system.a_eq, _sparse.csr_matrix((system.a_eq.shape[0], 1))],
                              format="csr")
        res = linprog(np.concatenate([np.zeros(size), [-1.0]]), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq,
                      b_eq=system.b_eq, bounds=list(system.bounds) + [(None, 0.0)], method="highs")
        if not res.success or res.x is None:
            raise InverseOTError(f"selection: max-margin solve failed: {res.message!r}")
        primal = np.asarray(res.x[:size], dtype=np.float64)
        theta = primal[: system.n_features]
        f = primal[system.n_features: system.n_features + system.n_source]
        g = primal[system.n_features + system.n_source:]
        sigma_all = _sigma(problem, theta, f, g, all_rows, all_cols)
        worst = float(sigma_all.max())
        if worst <= float(tol):
            return worst
        viol = sigma_all > float(tol)
        current = current.union(EdgeSet.from_arrays(all_rows[viol], all_cols[viol], n_target=n))
    sigma_all = _sigma(problem, theta, f, g, all_rows, all_cols)
    return float(sigma_all.max())
