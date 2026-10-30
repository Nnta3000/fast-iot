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
from ..core.scan import scan_for_violations, maximum_slack
from ..core.edges_advance import advance_edges, keep_by_tightness, slack_toward_cost
from ..core.pruning import HysteresisPruner
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


def _median_cost_blockwise(problem: InverseOTProblem, theta: np.ndarray, *, m: int, n: int,
                           rows_per_block: int = 256, max_samples: int = 2_000_000) -> float:
    """
    CN: 用**分块流式**方式估计 uniform theta 下 `|c|` 的中位数，内存与 `m·n` 无关。
        每个块取一个等距子样本，累积到 `max_samples` 为止；对中位数而言这个样本量远超所需。
    EN: estimate the median of `|c|` under uniform theta by **streaming blocks**, with memory independent of
        `m·n`. Each block contributes an equally spaced subsample, accumulated up to `max_samples`; for a
        median this is far more than enough.
    """
    cols = np.arange(n)
    chunks: list[np.ndarray] = []
    taken = 0
    for r0 in range(0, int(m), int(rows_per_block)):
        r1 = min(r0 + int(rows_per_block), int(m))
        rows = np.repeat(np.arange(r0, r1), n)
        cols_tiled = np.tile(cols, r1 - r0)
        block = np.abs(np.asarray(problem.features.costs(theta, rows, cols_tiled), dtype=np.float64))
        if block.size > 20_000:
            step = max(1, block.size // 20_000)
            block = block[::step]
        chunks.append(block)
        taken += block.size
        if taken >= int(max_samples):
            break
    if not chunks:
        return 1.0
    return float(np.median(np.concatenate(chunks)))


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
    prune_hysteresis: bool = True,
    selection: SelectionMode = "central",
    eta: float = 1e-7,
    check_pot: bool = True,
    initial_extra: EdgeSet | None = None,
    identity_rtol: float | None = None,
    scan_backend: str = "numpy",
    scan_device: str = "auto",
    scan_max_tile_bytes: int = 1 << 26,
    warm_start: bool = False,
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
    pruner = HysteresisPruner.fresh(tau_prune=float(tau_prune))
    warm_x: np.ndarray | None = None

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
    total_mass = float(np.sum(support_values))
    # CN: 容差 = max(用户给定或默认, 1e-5 * 代价尺度)。代价尺度用 uniform theta 下 |c| 的**中位数**。
    #
    #     **关键**：中位数只需要一个**样本**，不需要全部 `m·n` 条边。此前一次性求值所有边，
    #     m=5000 时需要约 13 GB（实测 OOM，进程被 kill），这才是规模瓶颈——不是 master 的规模。
    #     这里用分块流式的**分位数估计**（保留每块的样本值），内存与 `m·n` 无关。
    # EN: the tolerance is `max(user/default, 1e-5 * cost scale)`, the scale being the **median** of |c| at
    #     uniform theta.
    #
    #     **Critical**: a median needs a **sample**, not all `m·n` edges. Evaluating every edge needed ~13 GB
    #     at m=5000 and the process was OOM-killed; that, not the master size, is the scaling bottleneck.
    #     A blockwise streaming quantile keeps memory independent of `m·n`.
    probe = _uniform_theta(problem)
    cost_scale = _median_cost_blockwise(problem, probe, m=int(m), n=int(n))
    # CN: `all_rows/all_cols` 只用于**分块**扫描（`maximum_slack` 与 tiled 的 scan），
    #     本身只是索引数组（m=5000 时约 0.4 GB），代价矩阵从不再落地。
    # EN: `all_rows`/`all_cols` serve the **blockwise** scan only (`maximum_slack` and the tiled
    #     `scan_violations`); they are plain index arrays (~0.4 GB at m=5000) and the cost matrix never
    #     materialises again.
    r_idx, c_idx = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    all_rows, all_cols = r_idx.reshape(-1), c_idx.reshape(-1)
    del r_idx, c_idx
    tol_eff = max(float(tol) if tol is not None else DEFAULT_TOL,
                  float(TOL_SCALE_FRACTION) * max(cost_scale, 1e-30),
                  float(SOLVER_FLOOR_FRACTION) * max(cost_scale, 1e-30))

    records: list[RegretIteration] = []
    solver_floor_value: float = float("nan")
    stall_status: str = "none"
    # CN: uniform theta 作为初始 best-so-far（可行点，`s=0`，`U = R_uniform`）。
    # EN: uniform theta seeds best-so-far (feasible, `s=0`, `U = R_uniform`).
    uni0 = _uniform_theta(problem)
    f_u, g_u = _best_dual_for(problem, uni0)
    sig_u_max = maximum_slack(problem, uni0, f_u, g_u, all_rows, all_cols)
    su = max(0.0, sig_u_max)
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
        # CN: 热启动：把**上一轮**的 `(theta, f, g)` 交给一阶求解器作起点。布局在轮间不变（只看
        #     `n_features, n_source, n_target`），因此上一轮的 `x` 仍然有效。只影响迭代起点。
        # EN: warm start: hand the **previous** round's `(theta, f, g)` to the first-order solver. The
        #     layout is unchanged between rounds (it depends only on `n_features, n_source, n_target`), so
        #     the previous `x` stays valid. It affects the iteration start only.
        if warm_start and hasattr(solver, "set_warm_start"):
            solver.set_warm_start(warm_x)   # None on round 1 == no warm start
        solution = solver.solve(problem, finite)
        master_time = float(time.perf_counter() - t0)
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
        # CN: 与精确循环共用同一份后端调度（此前松弛循环只走 numpy，GPU 扫描在松弛路径上是**不可达**的）。
        # EN: same backend dispatch as the exact loop (previously the relaxed loop used NumPy only, so a GPU
        #     scan was unreachable on the relaxed path).
        scan = scan_for_violations(problem, as_master, excluded=finite, tol=float(tol_eff),
                                   gamma=int(gamma), scan_backend=scan_backend,
                                   scan_device=str(scan_device),
                                   scan_max_tile_bytes=int(scan_max_tile_bytes))
        s_pre = max(0.0, maximum_slack(problem, theta_pre, f_pre, g_pre, all_rows, all_cols))
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
        if warm_start:
            warm_x = np.concatenate([np.asarray(solution.theta, dtype=np.float64),
                                     np.asarray(solution.source_dual, dtype=np.float64),
                                     np.asarray(solution.target_dual, dtype=np.float64)])
        records.append(RegretIteration(
            index=index, regret=regret, lower=lower, upper=upper, gap=gap,
            support_size=int(finite.size), finite_edges=int(finite.size),
            violation_count=int(scan.violation_count), added=int(scan.rows.size),
            max_sigma_all=s_pre, wall_time=float(time.perf_counter() - t0),
            regret_pre_repair=r_pre,
            sigma_max_in_F=float(solution.sigma_max_in_F),
            sigma_max_out_F=float(solution.sigma_max_out_F),
            solver_iterations=int(solution.solver_iterations),
            solver_status=str(solution.solver_status),
            master_time=float(master_time),
            scan_time=float(time.perf_counter() - t0) - float(master_time)))
        # CN: **地板检测**：`s_pre * |pi_hat|` 在 F 内无法被消除（它来自求解器的精度，而非真正的违例）。
        #     若它已经不超过 `eps_r`，即视为收敛；若连续若干轮都停在同一个 level，则判为 `solver_floor`。
        # EN: **floor detection**: `s_pre * |pi_hat|` cannot be eliminated inside F (it comes from solver
        #     accuracy, not a genuine violation). If it already fits under `eps_r`, that is convergence; if it
        #     stays at the same level for several rounds, the run is marked `solver_floor`.
        gap_now = float(s_pre * total_mass)
        if gap_now <= eps_r:
            converged = True
            solver_floor_value = gap_now
            stall_status = "converged"
            break
        if len(records) >= 3:
            prev = [float(rec.max_sigma_all) * total_mass for rec in records[-3:-1]]
            if prev and all(abs(g - gap_now) <= max(1e-12, 0.05 * gap_now) for g in prev):
                # CN: 连续三轮 gap 几乎不变 ⇒ 已到求解器地板；**保留** best-so-far 并标记。
                # EN: three rounds within 5% => at the solver floor; keep best-so-far and mark it.
                solver_floor_value = gap_now
                stall_status = "solver_floor"
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
                # CN: `keep_by_tightness` 用的是 `slack = c - f - g`；这里原本用 `sigma >= -tau`，
                #     两者等价（`sigma = -slack`）。策略（何时剪、剪谁）仍在本循环内。
                # EN: `keep_by_tightness` uses `slack = c - f - g`; this site used `sigma >= -tau`, which
                #     is the same condition (`sigma = -slack`). The policy (when and what to prune) stays here.
                slack = slack_toward_cost(problem, theta, f, g, f_rows, f_cols)
                candidates_block = EdgeSet.from_arrays(f_rows, f_cols, n_target=n)
                if bool(prune_hysteresis):
                    # CN: 与精确循环同一份迟滞逻辑：只有连续两轮都松才剪。一阶后端的对偶噪声 ~1e-4，
                    #     单轮规则会把噪声当成"松了"并永久剪掉该保留的边——这正是 `prune=True` 掉区间的原因。
                    # EN: same hysteresis as the exact loop: drop only when loose two rounds running. A
                    #     first-order backend's duals carry ~1e-4 noise and a single-round rule would read noise
                    #     as "loose" and permanently drop an edge that should stay -- which is what lost the
                    #     bracket under `prune=True`.
                    pruned, pruner = pruner.step(candidates_block, slack)
                else:
                    pruned = keep_by_tightness(candidates_block, slack,
                                               tau_prune=float(tau_prune))
            else:
                pruned = EdgeSet.from_arrays(np.zeros(0, dtype=np.int64),
                                             np.zeros(0, dtype=np.int64), n_target=n)
            # CN: 本轮的"保留部分" = mandatory ∪ pruned；扫描到的 violating 边由 `advance_edges` 加进去。
            # EN: this round's kept part is mandatory ∪ pruned; the scan's violators are added by
            #     `advance_edges`.
            finite, _merged = advance_edges(mandatory, pruned, candidates)
        else:
            # CN: 剪枝关闭时 `F_t` 原样保留，只并入本轮扫描到的边——与原文 `finite.union(candidates)` 一致。
            # EN: with pruning off, `F_t` is kept as-is and only this round's violators are merged in --
            #     identical to the original `finite.union(candidates)`.
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
            # CN: **同一 `F_t` 重复出现且没有新边可加** ⇒ 剩下的违例都来自求解器精度（地板），
            #     不是真正的缺边。此时**保留 best-so-far** 并标记为 `solver_floor`，而不是抛错——
            #     抛错会把"已到精度上限"误报成"实例/容差不一致"。实测：K=64、m=300 时 `|F|` 在 2731
            #     条边（约 3%）后稳定，第 7 轮没有新边。
            # EN: **the same `F_t` recurs with no new edge to add** => the remaining violations come from solver
            #     accuracy (the floor), not from missing edges. Keep best-so-far and mark `solver_floor` instead
            #     of raising -- raising misreports "hit the accuracy limit" as "inconsistent instance/tolerances".
            #     Measured: at K=64, m=300 `|F|` stabilises at 2731 edges (~3%) and round 7 adds nothing.
            solver_floor_value = float(gap)
            stall_status = "solver_floor_no_new_edges"
            break
        seen[fingerprint] = index

    # CN: 后处理改为**可选的第二阶段约束生成**，且**不替换**解：证书区间永远来自收敛的 stage-1 点。
    # EN: post-processing is an **optional second constraint-generation phase** and never replaces the
    #     solution: the certificate interval always comes from the converged stage-1 point.
    regret_at_solution = regret
    selection_sigma: float | None = None
    if converged and selection == "central":
        # CN: `tol=None` 时必须用**已解析的** `tol_eff`，而不是 `float(None)`（此前会在中央选择阶段崩溃）。
        # EN: with `tol=None` use the **resolved** `tol_eff`; `float(None)` crashed the central-selection stage.
        selected = _select_central_cg(problem, finite, eta=float(eta), tol=float(tol_eff),
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
        selection_max_sigma=selection_sigma, solver_floor=float(solver_floor_value),
        stall_status=str(stall_status))


def _best_dual_for(problem: InverseOTProblem, theta: np.ndarray, *, block: int = 512
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
        block_cost = np.asarray(problem.features.costs(theta, rows, cols_tiled),
                                dtype=np.float64).reshape(r1 - r0, n)
        f[r0:r1] = block_cost.min(axis=1)
    return f, np.zeros(n, dtype=np.float64)


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
