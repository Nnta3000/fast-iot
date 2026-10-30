"""
CN: 逆 OT 的约束生成主循环（spec 第 3 节）。
EN: Constraint-generation main loop for inverse OT (spec section 3).

CN: F 的更新为 F = B ∪ T ∪ V：B 永不被剪枝，T 只保留 slack 不超过 tau_prune 的边，
    V 是逐行/逐列 top-gamma 的违例边。
EN: The edge set update is F = B ∪ T ∪ V: B is never pruned, T keeps only edges with
    slack at most tau_prune, and V holds the per-row/per-column top-gamma violations.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .components import mandatory_edges
from ..core.edges import EdgeSet
from .master import (
    LinearMasterSolver,
    MasterSolution,
    MasterSolver,
    TieBreak,
    draw_generic_direction,
)
from ..core.scan import scan_for_violations
from ..core.edges_advance import advance_edges, slack_toward_cost
from ..core.pruning import HysteresisPruner
from ..core.problem import InverseOTError, InverseOTProblem
from ..core.result import InverseOTResult, IterationRecord
from ..eval.validation import is_optimal_plan

_LOGGER = logging.getLogger(__name__)


def _peak_memory_mb() -> float:
    """CN: 当前进程峰值 RSS(MB)，Linux 下 ru_maxrss 为 KB。EN: Peak RSS (MB); ru_maxrss is KB on Linux."""
    try:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:  # noqa: BLE001 - platform without resource
        return 0.0


def _prune_by_slack(
    problem: InverseOTProblem,
    master_theta: np.ndarray,
    source_dual: np.ndarray,
    target_dual: np.ndarray,
    delta: float,
    non_support: EdgeSet,
    *,
    tau_prune: float,
) -> EdgeSet:
    """
    CN: 只保留 slack = c_ij - f_i - g_j - delta <= tau_prune 的候选边。
    EN: Keep only edges with slack = c_ij - f_i - g_j - delta <= tau_prune.

    CN: 保留多余边永远安全；只有剪掉严格正 slack 的边才是被证明安全的。
    EN: Keeping extra edges is always safe; only pruning strictly positive slack is
        proved safe.
    """
    if non_support.size == 0:
        return non_support
    costs = problem.features.costs(master_theta, non_support.rows, non_support.cols)
    slack = costs - source_dual[non_support.rows] - target_dual[non_support.cols] - float(delta)
    return non_support.select_mask(slack <= float(tau_prune))


def solve_inverse_ot(
    problem: InverseOTProblem,
    *,
    _flat_only: bool = False,
    tol: float = 1e-9,
    tau_prune: float = 1e-12,
    gamma: int = 1,
    tie_break: TieBreak = "lex",
    max_iterations: int = 100,
    solver: MasterSolver | None = None,
    random_seed: int = 0,
    validate: bool = True,
    validate_max_elements: int = 50_000_000,
    max_tile_elements: int = 1 << 20,
    # CN: **剪枝默认关闭**：剪枝是启发式，只有"剪掉严格正 slack 的边"被证明端到端安全；默认保留全部边
    #     以保持**精确性**（保留多余边永远安全，只是更大更慢）。要剪枝请显式 `prune=True`。
    # EN: **pruning is off by default**: pruning is a heuristic, and only dropping strictly-positive-slack
    #     edges is proved safe end to end; the default keeps every edge so the result stays **exact**
    #     (keeping extra edges is always safe, just bigger and slower). Pass `prune=True` explicitly.
    prune: bool = False,
    prune_hysteresis: bool = True,
    scan_backend: str = "numpy",
    scan_device: str = "auto",
    scan_max_tile_bytes: int = 64 * 1024 * 1024,
    hierarchical: "HierarchyParameters | None" = None,
) -> InverseOTResult:
    """
    CN: 求解逆 OT：返回使 pi_hat 最优的 max-margin theta（约束生成）。
    EN: Solve inverse OT: return a max-margin theta making pi_hat optimal via
        constraint generation.

    CN: `scan_backend="hello"` 用 HELLO-OT 的 torch/GPU 分块扫描替代 NumPy 扫描（仅对
        `DiagonalWeightFeatures` 生效，其余自动回退）；`scan_backend="torch"` 用 `src/iot` 自带的
        通用 torch 扫描，对**任意**线性于 theta 的代价都可用（有双线性嵌入时走 `O(m n d)` matmul，
        否则 `O(m n K)`），且**不需要** native 扩展。`scan_device` 透传（"auto"/"cpu"/"cuda"）。
    EN: `scan_backend="hello"` uses HELLO-OT's torch/GPU blockwise scan instead of the NumPy scan
        (only for `DiagonalWeightFeatures`; otherwise it falls back); `scan_backend="torch"` uses the
        generic torch scan shipped in `src/iot`, valid for **any** cost linear in theta (the `O(m n d)`
        matmul path when a bilinear embedding exists, else `O(m n K)`), and needs **no** native
        extension. `scan_device` is forwarded ("auto"/"cpu"/"cuda").
    """
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be >= 1")
    if int(gamma) < 1:
        raise ValueError("gamma must be >= 1")
    if float(tol) < 0.0 or float(tau_prune) < 0.0:
        raise ValueError("tol and tau_prune must be nonnegative")
    if str(scan_backend) not in ("numpy", "hello", "torch"):
        raise ValueError("scan_backend must be 'numpy', 'hello' or 'torch'")
    # CN: **层级入口**。`hierarchical=None`（默认）时完全走原来的扁平路径，结果逐位不变。
    #     给了参数则粗到细求解，细层仍调用**本函数**（`_flat_only=True` 防止递归）。
    # EN: **hierarchical entry**. With `hierarchical=None` (default) the original flat path runs unchanged,
    #     bit for bit. With parameters, solve coarse to fine, and the fine level calls **this function**
    #     again (`_flat_only=True` prevents recursion).
    if hierarchical is not None and not bool(_flat_only):
        from ..hierarchical import solve_hierarchical  # noqa: PLC0415

        run = solve_hierarchical(
            problem,
            lambda p, initial: solve_inverse_ot(
                p, tol=tol, tau_prune=tau_prune, gamma=gamma, tie_break=tie_break,
                max_iterations=max_iterations, solver=solver, random_seed=random_seed,
                validate=validate, validate_max_elements=validate_max_elements,
                max_tile_elements=max_tile_elements, prune=prune,
                prune_hysteresis=prune_hysteresis, scan_backend=scan_backend,
                scan_device=scan_device, scan_max_tile_bytes=scan_max_tile_bytes,
                hierarchical=None, _flat_only=True),
            hierarchical,
            mandatory_of=lambda p: mandatory_edges(p)[0])
        result = run.fine
        if not isinstance(result, InverseOTResult):
            raise InverseOTError("hierarchical fine level did not return an InverseOTResult")
        return result
    if solver is not None:
        master_solver: MasterSolver = solver
    elif tie_break == "lex_generic":
        # CN: 整轮只抽一次固定的通用方向 r，并记录 seed 与指纹（证明第 6 节的 w_r^*）。
        # EN: Draw the run-fixed generic direction r exactly once and log the seed and a
        #     fingerprint (proof section 6's w_r^*).
        direction = draw_generic_direction(
            random_seed=int(random_seed),
            n_features=int(problem.n_features),
            n_source=int(problem.n_source),
            n_target=int(problem.n_target),
        )
        _LOGGER.info(
            "lex_generic tie-break: r drawn once over (theta,f,g), seed=%d, |r|_2=%.6f, "
            "r[:4]=%s",
            int(random_seed),
            float(np.linalg.norm(direction)),
            np.array2string(direction[:4], precision=5),
        )
        master_solver = LinearMasterSolver(
            random_seed=int(random_seed),
            generic_direction=tuple(float(value) for value in direction),
        )
    else:
        master_solver = LinearMasterSolver(random_seed=int(random_seed))
    support = EdgeSet.from_arrays(*problem.support()[:2], n_target=int(problem.n_target))
    mandatory, _partition = mandatory_edges(problem)
    edges = mandatory

    records: list[IterationRecord] = []
    converged = False
    started = time.perf_counter()
    final_max_violation = 0.0
    pruner = HysteresisPruner.fresh(tau_prune=float(tau_prune))
    master: MasterSolution | None = None
    for iteration in range(1, int(max_iterations) + 1):
        iteration_started = time.perf_counter()
        master_started = time.perf_counter()
        master = master_solver.solve(problem, edges, tie_break=tie_break)
        master_time = time.perf_counter() - master_started
        oracle_started = time.perf_counter()
        # CN: 后端调度已抽到 `core/scan.py`，与松弛循环共用同一份实现（语义逐字不变）。
        # EN: the backend dispatch lives in `core/scan.py` now, shared with the relaxed loop (verbatim).
        scan = scan_for_violations(
            problem,
            master,
            excluded=edges,
            tol=float(tol),
            gamma=int(gamma),
            scan_backend=scan_backend,
            scan_device=str(scan_device),
            scan_max_tile_bytes=int(scan_max_tile_bytes),
            max_tile_elements=int(max_tile_elements),
        )
        oracle_time = time.perf_counter() - oracle_started
        peak_mb = _peak_memory_mb()
        final_max_violation = float(scan.max_violation)
        if not scan.has_violations:
            records.append(
                IterationRecord(
                    index=iteration,
                    delta=float(master.delta),
                    support_size=int(edges.size),
                    violation_count=0,
                    added=0,
                    max_violation=float(scan.max_violation),
                    wall_time=float(time.perf_counter() - iteration_started),
                    master_time=float(master_time),
                    oracle_time=float(oracle_time),
                    prune_time=0.0,
                    assembly_time=float(master_time + oracle_time),
                    peak_memory_mb=peak_mb,
                )
            )
            converged = True
            break

        candidates = EdgeSet.from_arrays(scan.rows, scan.cols, n_target=int(problem.n_target))
        non_support = edges.difference(support)
        prune_started = time.perf_counter()
        if not bool(prune):
            # CN: 关闭剪枝时保留全部候选边，F 单调增长（保留多余边永远安全）。
            # EN: With pruning off, keep every candidate edge so F grows monotonically
            #     (keeping extra edges is always safe).
            pruned = non_support
        elif bool(prune_hysteresis) and non_support.size:
            # CN: 迟滞剪枝已抽成 `core/pruning.HysteresisPruner`，与松弛循环共用同一份逻辑。
            #     slack 由共用函数计算（`c - f - g - delta`，与原来的内联式子逐字等价）。
            # EN: hysteresis now lives in `core/pruning.HysteresisPruner`, shared with the relaxed loop.
            #     slack comes from the shared helper (`c - f - g - delta`, verbatim the old inline form).
            slack = slack_toward_cost(problem, master.theta, master.source_dual,
                                      master.target_dual, non_support.rows, non_support.cols,
                                      delta=float(master.delta))
            pruned, pruner = pruner.step(non_support, slack)
        else:
            pruned = _prune_by_slack(
                problem,
                master.theta,
                master.source_dual,
                master.target_dual,
                master.delta,
                non_support,
                tau_prune=float(tau_prune),
            )
        prune_time = time.perf_counter() - prune_started
        assembly_started = time.perf_counter()
        # CN: 共用的边集推进（与松弛循环同一份代数）。
        # EN: the shared edge-set advance (same algebra as the relaxed loop).
        new_edges, _merged_size = advance_edges(mandatory, pruned, candidates)
        added = int(_merged_size) - int(edges.size)
        assembly_time = time.perf_counter() - assembly_started
        records.append(
            IterationRecord(
                index=iteration,
                delta=float(master.delta),
                support_size=int(edges.size),
                violation_count=int(scan.violation_count),
                added=added,
                max_violation=float(scan.max_violation),
                wall_time=float(time.perf_counter() - iteration_started),
                master_time=float(master_time),
                oracle_time=float(oracle_time),
                prune_time=float(prune_time),
                assembly_time=float(assembly_time),
                peak_memory_mb=_peak_memory_mb(),
            )
        )
        edges = new_edges

    is_optimal: bool | None = None
    relative_gap: float | None = None
    if master is None:
        raise InverseOTError("inverse-OT loop produced no master solution")
    total_edges = int(problem.n_source) * int(problem.n_target)
    if bool(validate) and total_edges <= int(validate_max_elements):
        check = is_optimal_plan(problem, master.theta, tol=1e-7, max_elements=int(validate_max_elements))
        is_optimal = bool(check.is_optimal)
        relative_gap = float(check.relative_gap)
    return InverseOTResult(
        theta=master.theta,
        delta=float(master.delta),
        source_dual=master.source_dual,
        target_dual=master.target_dual,
        n_iterations=len(records),
        support_size=int(edges.size),
        iterations=tuple(records),
        converged=bool(converged),
        is_optimal=is_optimal,
        relative_gap=relative_gap,
        total_wall_time=float(time.perf_counter() - started),
        final_max_violation=float(final_max_violation),
    )


__all__ = [
    "solve_inverse_ot",
]
