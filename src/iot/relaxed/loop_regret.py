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

import dataclasses
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import ot
from jaxtyping import Float, Int

from ..core.edges import EdgeSet
from ..core.edges_advance import advance_edges, keep_by_tightness, slack_toward_cost
from ..core.parameters import RelaxedObjective, SolverParameters
from ..core.problem import InverseOTProblem
from ..core.pruning import HysteresisPruner
from ..core.scan import maximum_slack_grid, scan_for_violations
from ..exact.components import mandatory_edges
from ..exact.master import MasterSolution
from .diagnostics import (IDENTITY_RTOL, assert_regret_identity, fingerprint,
                          median_cost_blockwise, uniform_theta)
from .master_regret import RegretMasterSolver, regret_objective
from .selection import select_central_cg

DEFAULT_TOL: float = 1e-7
TOL_SCALE_FRACTION: float = 1e-5
SOLVER_FLOOR_FRACTION: float = 1e-4

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
# CN: **求解器地板**。CuPDLPx 的相对精度约 1e-4，因此扫描容差再紧也不会更准；`tol_eff` 必须至少是
#     `SOLVER_FLOOR_FRACTION * cost_scale`，否则循环会把求解器噪声当成违例、反复重加同一条边、`F_t` 重复。
#     实测：K=64 的对角代价尺度约 `1/k = 0.0156`，`tol_eff` 落在 1e-7 而求解器精度是 1e-4 ⇒ 循环报
#     "revisited F_t"。这一项就是此前已知的 scan-tolerance 修正。
# EN: **solver floor**. CuPDLPx's relative accuracy is about 1e-4, so a tighter scan tolerance buys nothing;
#     `tol_eff` must be at least `SOLVER_FLOOR_FRACTION * cost_scale`, otherwise the loop treats solver noise
#     as violations, re-adds the same edge and `F_t` repeats. Measured: at K=64 the diagonal cost scale is about
#     `1/k = 0.0156` while `tol_eff` sat at 1e-7 against 1e-4 solver accuracy => "revisited F_t". This is the
#     scan-tolerance fix that was pending.
SOLVER_FLOOR_FRACTION = 1e-4
DEFAULT_TAU = 1e-7
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
    # CN: 只用于**测量**：把一轮的时间拆成"master 求解"与"分离扫描（含修复）"，否则无法判断
    #     层级是否有杠杆（若瓶颈是扫描，减少轮数不是唯一手段）。
    # EN: **measurement only**: split one round into "master solve" and "separation scan (incl. repair)",
    #     without which we cannot tell whether the hierarchy has any lever.
    master_time: float = 0.0
    scan_time: float = 0.0


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
    # CN: **求解器地板**：F 内不可消除的 `s_pre` 会给出一个非零的 floor；当 `eps_r` 低于它时，
    #     继续迭代只是在追求解器噪声，因此标记为 `solver_floor` 而不是"已收敛"或反复迭代。
    # EN: **solver floor**: the irreducible `s_pre` inside F yields a nonzero floor; when `eps_r` is below it,
    #     further rounds chase solver noise, so the run is marked `solver_floor` rather than "converged" or
    #     left to spin.
    solver_floor: float = float("nan")
    stall_status: str = "none"


def full_regret(
    problem: InverseOTProblem, theta: np.ndarray, *, num_itermax: int = 10_000_000
) -> float:
    """
    CN: 独立复核 `⟨c(θ),π̂⟩ − OT(c(θ))`（POT 精确解；未收敛直接抛错）。
    EN: Independent check `⟨c(θ),π̂⟩ − OT(c(θ))` (exact POT; raises when POT does not converge).
    """
    rows, cols, values = problem.support()
    support_cost = float(
        np.dot(
            np.asarray(values, dtype=np.float64),
            np.asarray(problem.features.costs(theta, rows, cols), dtype=np.float64),
        )
    )
    m = int(problem.n_source)
    n = int(problem.n_target)
    r_idx, c_idx = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    cost = np.asarray(
        problem.features.costs(theta, r_idx.reshape(-1), c_idx.reshape(-1)),
        dtype=np.float64,
    ).reshape(m, n)
    a = np.asarray(problem.resolved_source_mass, dtype=np.float64)
    b = np.asarray(problem.resolved_target_mass, dtype=np.float64)
    transport = float(
        np.sum(
            cost * ot.emd(a / a.sum(), b / b.sum(), cost, numItermax=int(num_itermax))
        )
    ) * float(a.sum())
    return support_cost - transport


def _make_solver(objective: RelaxedObjective, master: RegretMasterSolver | None) -> RegretMasterSolver:
    """CN: `master` 缺省时按 `objective.eta` 构造。EN: build the default master from `objective.eta`."""
    if master is not None:
        return master
    return RegretMasterSolver(eta_scale=objective.eta, tie_break=False)


def _identity_rtol(objective: RelaxedObjective, solver: RegretMasterSolver) -> float:
    """
    CN: `identity_rtol=None` 时按后端取：一阶后端只能到求解精度，精确后端用浮点精度。
    EN: with `identity_rtol=None`, pick per backend: a first-order backend is good only to solver accuracy,
        an exact one to floating-point precision.
    """
    if objective.identity_rtol is not None:
        return objective.identity_rtol
    backend = str(getattr(solver, "backend", "highs"))
    return 1e-8 if backend == "cupdlpx" else IDENTITY_RTOL


@dataclasses.dataclass(frozen=True)
class _Baseline:
    """CN: 循环开始前的常量：均匀 θ 的证书、容差、索引。EN: constants fixed before the loop starts."""

    uniform_theta: Float[np.ndarray, "k"]
    uniform_upper: float
    eps_r: float
    tol_eff: float
    total_mass: float
    identity_rtol: float


@dataclasses.dataclass(frozen=True)
class _Round:
    """CN: 一轮的产物。EN: one round's products."""

    theta: Float[np.ndarray, "k"]
    source_dual: Float[np.ndarray, "m"]
    target_dual: Float[np.ndarray, "n"]
    regret: float
    lower: float
    upper: float
    record: RegretIteration
    candidates: EdgeSet
    finite: EdgeSet
    fingerprint: str
    stop: str | None


def _compute_baseline(problem: InverseOTProblem, params: SolverParameters,
                      objective: RelaxedObjective, solver: RegretMasterSolver,
                      support_rows: Int[np.ndarray, "support"],
                      support_cols: Int[np.ndarray, "support"],
                      support_values: Float[np.ndarray, "support"]) -> _Baseline:
    """
    CN: 计算 `eps_R` 的尺度（uniform θ 处的 regret）与扫描容差 `tol_eff`，并准备分块扫描的索引。
        `tol_eff` 里的**关键**是"中位数只需样本、不需要全部 `m·n` 条边"——全部求值在 m=5000 时
        约 13 GB（实测 OOM）。
    EN: compute the `eps_R` scale (the regret at uniform theta) and the scan tolerance `tol_eff`, and
        prepare the blockwise-scan indices. The **key** point in `tol_eff` is that a median needs a
        **sample**, not all `m*n` edges: evaluating everything needed ~13 GB at m=5000 (measured OOM).
    """
    m, n = int(problem.n_source), int(problem.n_target)
    phi_bar: Float[np.ndarray, "k"] = regret_objective(problem)[: int(problem.n_features)]
    uni = uniform_theta(problem)
    f0, g0 = _best_dual_for(problem, uni)
    r0 = float(np.dot(phi_bar, uni)
               - float(np.dot(problem.resolved_source_mass, f0))
               - float(np.dot(problem.resolved_target_mass, g0)))
    eps_r = float(objective.eps_r_rel) * max(1e-12, r0)

    cost_scale = median_cost_blockwise(problem, uni, m=m, n=n)
    tol_eff = max(float(DEFAULT_TOL) if params.tol is None else float(params.tol),
                  TOL_SCALE_FRACTION * max(cost_scale, 1e-30),
                  SOLVER_FLOOR_FRACTION * max(cost_scale, 1e-30))

    # CN: 原来在这里 `meshgrid` 出全部 m*n 条边的索引（2*8 字节/边，n=6890 时 760 MB，且整个求解期间保留）。
    #     现在 `maximum_slack_grid` 按行块**生成**索引，只访问同一组边，取值不变。
    # EN: this used to `meshgrid` the indices of all m*n edges (2*8 bytes/edge -- 760 MB at n=6890 -- kept for
    #     the whole solve). `maximum_slack_grid` now **generates** them per row block; same edges, same value.
    total_mass = float(np.sum(support_values))
    f_u, g_u = _best_dual_for(problem, uni)
    su = max(0.0, maximum_slack_grid(problem, uni, f_u, g_u))
    # CN: uniform θ 也是一个候选点：`U = R_uniform + s*|pi_hat|`。
    # EN: uniform theta is also a candidate: `U = R_uniform + s*|pi_hat|`.
    uniform_upper: float = r0 + su * total_mass
    return _Baseline(uniform_theta=uni, uniform_upper=uniform_upper, eps_r=eps_r,
                     tol_eff=float(tol_eff), total_mass=total_mass,
                     identity_rtol=_identity_rtol(objective, solver))


def _run_round(problem: InverseOTProblem, finite: EdgeSet, mandatory: EdgeSet, *,
               index: int, params: SolverParameters, objective: RelaxedObjective,
               solver: RegretMasterSolver, baseline: _Baseline, pruner: HysteresisPruner,
               seen: dict[str, int], pruning: bool, pruning_disabled: bool,
               warm_x: Float[np.ndarray, "k m n"] | None,
               previous_gaps: tuple[float, ...],
               support_rows: Int[np.ndarray, "support"],
               support_cols: Int[np.ndarray, "support"],
               support_values: Float[np.ndarray, "support"]
               ) -> tuple[_Round, HysteresisPruner, bool, bool, Float[np.ndarray, "k m n"] | None]:
    """
    CN: **一轮**：解 master → 检查恒等式 → 扫描平移前的点 → 算证书区间 → 剪枝/推进边集 → 判定停止。
        返回 `(本轮产物, 新的 pruner, pruning, pruning_disabled, 新的热启动向量)`。
    EN: **one round**: solve the master, check the identity, scan the pre-shift point, form the certificate
        interval, prune/advance the edge set, decide whether to stop. Returns
        `(round, pruner, pruning, pruning_disabled, warm-start vector)`.
    """
    import time as _time  # noqa: PLC0415

    n = int(problem.n_target)
    t0 = _time.perf_counter()
    if objective.warm_start and hasattr(solver, "set_warm_start"):
        solver.set_warm_start(warm_x)
    solution = solver.solve(problem, finite)
    master_time = float(_time.perf_counter() - t0)

    theta: Float[np.ndarray, "k"] = np.asarray(solution.theta, dtype=np.float64)
    f: Float[np.ndarray, "m"] = np.asarray(solution.source_dual, dtype=np.float64)
    g: Float[np.ndarray, "n"] = np.asarray(solution.target_dual, dtype=np.float64)
    regret = float(solution.regret)
    if not np.all(np.isin(problem.support_keys(), finite.keys)):
        raise RuntimeError("S left the finite edge set (B1 violated)")
    assert_regret_identity(problem, theta, f, g, support_rows, support_cols, support_values, regret,
                           rtol=baseline.identity_rtol)

    # CN: **扫描平移前的点**（投影 + 规范修复后、尚未 `f <- f - s`）。扫描平移后的点会让 `s ~ 0`、
    #     永远看不到违例、每轮都在第 1 轮"收敛"。
    # EN: **scan the pre-shift point** (projected and gauge-fixed, before `f <- f - s`); scanning the shifted
    #     point makes `s ~ 0`, finds no violators, and "converges" in round 1 every time.
    theta_pre: Float[np.ndarray, "k"] = np.asarray(solution.theta_pre, dtype=np.float64)
    f_pre: Float[np.ndarray, "m"] = np.asarray(solution.source_dual_pre, dtype=np.float64)
    g_pre: Float[np.ndarray, "n"] = np.asarray(solution.target_dual_pre, dtype=np.float64)
    as_master = MasterSolution(delta=0.0, theta=theta_pre, source_dual=f_pre, target_dual=g_pre,
                               status=solution.status)
    scan = scan_for_violations(problem, as_master, excluded=finite, tol=baseline.tol_eff,
                               gamma=params.gamma, scan_backend=params.scan.backend,
                               scan_device=params.scan.device,
                               scan_max_tile_bytes=params.scan.max_tile_bytes)
    s_pre = max(0.0, maximum_slack_grid(problem, theta_pre, f_pre, g_pre))
    r_pre = float(solution.regret_pre)
    upper = r_pre + s_pre * baseline.total_mass
    lower = r_pre

    new_warm: Float[np.ndarray, "k m n"] | None = warm_x
    if objective.warm_start:
        new_warm = np.concatenate([theta, f, g])

    record = RegretIteration(
        index=index, regret=regret, lower=lower, upper=upper, gap=upper - lower,
        support_size=int(finite.size), finite_edges=int(finite.size),
        violation_count=int(scan.violation_count), added=int(scan.rows.size),
        max_sigma_all=s_pre, wall_time=float(_time.perf_counter() - t0),
        regret_pre_repair=r_pre, sigma_max_in_F=float(solution.sigma_max_in_F),
        sigma_max_out_F=float(solution.sigma_max_out_F),
        solver_iterations=int(solution.solver_iterations), solver_status=str(solution.solver_status),
        master_time=master_time, scan_time=float(_time.perf_counter() - t0) - master_time)

    # CN: **地板检测**：`s_pre * |pi_hat|` 在 F 内无法被消除（来自求解器精度，不是真正的违例）。
    #     与原实现一致：只有**前两轮**的 gap 与当前接近 5% 以内才判为地板（需要至少三轮回看）。
    # EN: **floor detection**: `s_pre * |pi_hat|` cannot be eliminated inside F (solver accuracy, not a
    #     genuine violation). As in the original: only when **both previous** gaps are within 5% of the
    #     current one, which requires at least three rounds of history.
    gap_now = s_pre * baseline.total_mass
    stop: str | None = None
    if gap_now <= baseline.eps_r:
        stop = "converged"
    elif len(previous_gaps) == 2 and all(
            abs(gp - gap_now) <= max(1e-12, 0.05 * gap_now) for gp in previous_gaps
    ):
        # CN: 必须**恰好**有前两轮的 gap（原实现 `len(records) >= 3`）；少一个就会过早判为地板。
        # EN: **exactly** the two previous gaps are required (the original's `len(records) >= 3`); with one
        #     fewer it declares the floor too early.
        stop = "floor"

    # CN: 收敛/地板时**提前返回**：原实现 `break` 在推进边集之前，所以 `finite` 保持本轮的值——
    #     报告里的 `finite_edges` 是"收敛时所在的边集"，不是下一轮的边集。实测差 52 vs 42。
    # EN: return early on converge/floor: the original `break`s before advancing the edge set, so `finite`
    #     keeps this round's value -- the reported `finite_edges` is "the set the run converged on", not the
    #     next round's. Measured 52 vs 42.
    if stop is not None:
        return (_Round(theta=theta, source_dual=f, target_dual=g, regret=regret, lower=lower,
                       upper=upper, record=record, candidates=EdgeSet.from_arrays(
                           scan.rows, scan.cols, n_target=n), finite=finite,
                       fingerprint=fingerprint(finite), stop=stop),
                pruner, pruning, pruning_disabled, new_warm)
    candidates = EdgeSet.from_arrays(scan.rows, scan.cols, n_target=n)
    if pruning and not pruning_disabled:
        keep_mask = ~np.isin(finite.keys, mandatory.keys)
        f_rows, f_cols = finite.rows[keep_mask], finite.cols[keep_mask]
        if f_rows.size:
            slack = slack_toward_cost(problem, theta, f, g, f_rows, f_cols)
            block = EdgeSet.from_arrays(f_rows, f_cols, n_target=n)
            if params.pruning.hysteresis:
                pruned, pruner = pruner.step(block, slack)
            else:
                pruned = keep_by_tightness(block, slack, tau_prune=params.pruning.tau_prune)
        else:
            pruned = EdgeSet.from_arrays(np.zeros(0, dtype=np.int64),
                                         np.zeros(0, dtype=np.int64), n_target=n)
        new_finite, _ = advance_edges(mandatory, pruned, candidates)
    else:
        new_finite = finite.union(candidates)

    new_print = fingerprint(new_finite)
    repeat = new_print in seen
    if repeat:
        if pruning:
            pruning_disabled = True
            pruning = False
            seen.clear()
            # CN: 原实现 `continue`：**丢弃本轮推进结果**，`finite` 保持不变。
            # EN: the original `continue`s: this round's advance is **discarded**, `finite` is unchanged.
            new_finite = finite
        elif stop is None:
            # CN: **不覆盖已判定的停止原因**：原实现先判收敛/地板再判 `F_t` 重复，
            #     因此第 6 轮两个条件同时成立时报 `converged`。
            # EN: **do not overwrite an already-decided reason**: the original tests convergence and floor
            #     before the repeated-`F_t` case, so when both hold it reports `converged`.
            stop = "no_new_edges"
    seen[new_print] = index
    return (_Round(theta=theta, source_dual=f, target_dual=g, regret=regret, lower=lower,
                   upper=upper, record=record, candidates=candidates, finite=new_finite,
                   fingerprint=new_print, stop=stop),
            pruner, pruning, pruning_disabled, new_warm)


def solve_inverse_ot_regret(problem: InverseOTProblem, *,
                            objective: RelaxedObjective | None = None,
                            params: SolverParameters | None = None,
                            master: RegretMasterSolver | None = None) -> RegretResult:
    """
    CN: 求解 §2 的松弛逆 OT（regret 最小化 + 约束生成）。

        CN: 结构：`_compute_baseline` 固定均匀 θ 的尺度与容差；`_run_round` 是**一轮**的全部工作；
            本函数只是"准备 → 循环 → 收尾"。
    EN: Solve §2's relaxed inverse OT (regret minimisation plus constraint generation).

        Structure: `_compute_baseline` fixes the uniform-theta scale and tolerances, `_run_round` does
        **one round**, and this function is only prepare -> loop -> report.
    """
    objective = RelaxedObjective() if objective is None else objective
    params = SolverParameters() if params is None else params

    if params.hierarchical is not None:
        from dataclasses import replace  # noqa: PLC0415

        from ..hierarchical import solve_hierarchical  # noqa: PLC0415

        flat_params: SolverParameters = replace(params, hierarchical=None)
        run = solve_hierarchical(
            problem,
            lambda sub_problem, initial: solve_inverse_ot_regret(
                sub_problem, objective=replace(objective, initial_extra=initial),
                params=flat_params, master=master),
            params.hierarchical,
            mandatory_of=lambda sub_problem: mandatory_edges(sub_problem)[0])
        if not isinstance(run.fine, RegretResult):
            raise RuntimeError("hierarchical fine level did not return a RegretResult")
        return run.fine

    support_rows, support_cols, support_values = problem.support()
    support = EdgeSet.from_arrays(support_rows, support_cols, n_target=int(problem.n_target))
    mandatory, _partition = mandatory_edges(problem)
    finite = mandatory if objective.initial_extra is None else mandatory.union(objective.initial_extra)
    if not np.all(np.isin(support.keys, finite.keys)):
        raise RuntimeError("S must be a subset of F from the first iteration (B1)")

    solver = _make_solver(objective, master)
    baseline = _compute_baseline(problem, params, objective, solver, support_rows, support_cols,
                                 support_values)
    pruner = HysteresisPruner.fresh(tau_prune=params.pruning.tau_prune)
    seen: dict[str, int] = {}
    gap_history: list[float] = []
    pruning = params.pruning.enabled
    pruning_disabled = False
    warm_x: Float[np.ndarray, "k m n"] | None = None
    best_theta: Float[np.ndarray, "k"] = baseline.uniform_theta
    best_upper: float = baseline.uniform_upper
    records: list[RegretIteration] = []
    last: _Round | None = None
    stop: str | None = None
    floor_value = float("nan")

    start = time.perf_counter()
    for index in range(1, params.max_iterations + 1):
        last, pruner, pruning, pruning_disabled, warm_x = _run_round(
            problem, finite, mandatory, index=index, params=params, objective=objective,
            solver=solver, baseline=baseline, pruner=pruner, seen=seen, pruning=pruning,
            pruning_disabled=pruning_disabled, warm_x=warm_x,
            previous_gaps=tuple(gap_history[-2:]), support_rows=support_rows,
            support_cols=support_cols, support_values=support_values)
        records.append(last.record)
        gap_history.append(last.record.max_sigma_all * baseline.total_mass)
        finite = last.finite
        if last.upper < best_upper:
            best_theta, best_upper = np.array(last.theta, dtype=np.float64), float(last.upper)
        if last.stop is not None:
            stop = last.stop
            if stop in ("converged", "floor"):
                floor_value = last.record.max_sigma_all * baseline.total_mass
            break

    if last is None:
        raise RuntimeError("the relaxed loop produced no round")
    converged = stop == "converged"
    stall_status = "none"
    if stop == "converged":
        stall_status = "converged"
    elif stop == "no_new_edges":
        stall_status = "solver_floor_no_new_edges"
    elif stop == "floor":
        stall_status = "solver_floor"
    else:
        stall_status = "max_iterations"

    selection_sigma: float | None = None
    if converged and objective.selection == "central":
        selected = select_central_cg(problem, finite, eta=objective.eta, tol=baseline.tol_eff,
                                     tau_prune=params.pruning.tau_prune, gamma=params.gamma,
                                     max_iterations=params.max_iterations, solver=solver)
        selection_sigma = float(selected)

    pot_regret: float | None = None
    pot_matches: bool | None = None
    if objective.check_pot:
        pot_regret = full_regret(problem, last.theta)
        pot_matches = bool(abs(pot_regret - last.regret) <= max(1e-7, 1e-7 * abs(pot_regret)))

    return RegretResult(
        theta=best_theta, source_dual=last.source_dual, target_dual=last.target_dual,
        regret=last.regret, regret_at_solution=last.regret, lower=last.lower, upper=last.upper,
        gap=last.upper - last.lower, max_sigma_all=last.record.max_sigma_all,
        n_iterations=len(records), finite_edges=int(finite.size), support_size=int(support.size),
        iterations=tuple(records), converged=converged, pruning_disabled=pruning_disabled,
        pot_regret=pot_regret, pot_matches=pot_matches,
        total_wall_time=float(time.perf_counter() - start), eps_r=baseline.eps_r,
        selection_max_sigma=selection_sigma, solver_floor=float(floor_value),
        stall_status=stall_status)


def _best_dual_for(
    problem: InverseOTProblem, theta: np.ndarray, *, block: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 可行对偶点 `f_i = min_j c_ij(theta)`，`g = 0`。**按行分块**计算，从不落地整个 `m×n` 代价矩阵：
        此前一次性算全部边会在 m=5000 时需要约 13 GB（实测 OOM）。
    EN: feasible duals `f_i = min_j c_ij(theta)`, `g = 0`, computed **in row blocks** so the full `m x n` cost
        matrix never materialises; computing all edges at once needed ~13 GB at m=5000 (measured OOM).
    """
    m, n = int(problem.n_source), int(problem.n_target)
    f = np.empty(m, dtype=np.float64)
    cols = np.arange(n)
    for r0 in range(0, m, int(block)):
        r1 = min(r0 + int(block), m)
        rows = np.repeat(np.arange(r0, r1), n)
        cols_tiled = np.tile(cols, r1 - r0)
        block_cost = np.asarray(
            problem.features.costs(theta, rows, cols_tiled), dtype=np.float64
        ).reshape(r1 - r0, n)
        f[r0:r1] = block_cost.min(axis=1)
    return f, np.zeros(n, dtype=np.float64)
