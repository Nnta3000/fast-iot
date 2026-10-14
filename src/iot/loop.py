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
from .edges import EdgeSet
from .master import (
    LinearMasterSolver,
    MasterSolution,
    MasterSolver,
    TieBreak,
    draw_generic_direction,
)
from .oracle import scan_violations
from .problem import InverseOTError, InverseOTProblem
from .result import InverseOTResult, IterationRecord
from .validation import is_optimal_plan

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
    prune: bool = True,
    prune_hysteresis: bool = True,
    scan_backend: str = "numpy",
    scan_device: str = "auto",
    scan_max_tile_bytes: int = 64 * 1024 * 1024,
) -> InverseOTResult:
    """
    CN: 求解逆 OT：返回使 pi_hat 最优的 max-margin theta（约束生成）。
    EN: Solve inverse OT: return a max-margin theta making pi_hat optimal via
        constraint generation.

    CN: `scan_backend="hello"` 用 HELLO-OT 的 torch/GPU 分块扫描替代 NumPy 扫描（仅对
        `DiagonalWeightFeatures` 生效，其余自动回退）；`scan_device` 透传给它（"auto"/"cpu"/"cuda"）。
    EN: `scan_backend="hello"` uses HELLO-OT's torch/GPU blockwise scan instead of the NumPy
        scan (only for `DiagonalWeightFeatures`; otherwise it falls back); `scan_device` is
        forwarded ("auto"/"cpu"/"cuda").
    """
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be >= 1")
    if int(gamma) < 1:
        raise ValueError("gamma must be >= 1")
    if float(tol) < 0.0 or float(tau_prune) < 0.0:
        raise ValueError("tol and tau_prune must be nonnegative")
    if str(scan_backend) not in ("numpy", "hello"):
        raise ValueError("scan_backend must be 'numpy' or 'hello'")
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
    previous_positive_keys = np.zeros(0, dtype=np.int64)
    master: MasterSolution | None = None
    for iteration in range(1, int(max_iterations) + 1):
        iteration_started = time.perf_counter()
        master_started = time.perf_counter()
        master = master_solver.solve(problem, edges, tie_break=tie_break)
        master_time = time.perf_counter() - master_started
        oracle_started = time.perf_counter()
        if str(scan_backend) == "hello":
            from .hello_scan import scan_violations_fast

            scan = scan_violations_fast(
                problem,
                master,
                excluded=edges,
                tol=float(tol),
                gamma=int(gamma),
                device=str(scan_device),
                max_tile_bytes=int(scan_max_tile_bytes),
            )
        else:
            scan = scan_violations(
                problem,
                master,
                excluded=edges,
                tol=float(tol),
                gamma=int(gamma),
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
            # CN: 迟滞剪枝：只有当 slack 连续两轮 > tau 时才剪掉，抑制循环（Bland 风格）。
            # EN: Hysteresis pruning: drop an edge only when its slack exceeded tau in two
            #     consecutive iterations, suppressing cycling (Bland's-rule flavor).
            slack = (
                problem.features.costs(master.theta, non_support.rows, non_support.cols)
                - master.source_dual[non_support.rows]
                - master.target_dual[non_support.cols]
                - float(master.delta)
            )
            positive = slack > float(tau_prune)
            if previous_positive_keys.size:
                prune_mask = positive & np.isin(non_support.keys, previous_positive_keys)
            else:
                prune_mask = np.zeros(non_support.size, dtype=bool)
            pruned = non_support.select_mask(~prune_mask)
            previous_positive_keys = np.ascontiguousarray(non_support.keys[positive], dtype=np.int64)
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
        new_edges = mandatory.union(pruned).union(candidates)
        added = int(new_edges.size) - int(edges.size)
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
