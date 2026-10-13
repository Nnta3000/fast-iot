"""
CN: 统一的单次运行入口：`run_spec(spec) -> RunRecord`。三个轴 + solver 配置 → 一条记录。
EN: The single unified run entry point: `run_spec(spec) -> RunRecord`. Three axes + solver
    config -> one record.

CN: 求解路径与 shipped scale path 一致（`RegularizedMasterSolver(epsilon=1e-4)`、
    tie_break="none"、hysteresis 剪枝）；bilinear-PSD 用 `SdpMasterSolver`。
EN: The solve follows the shipped scale path (`RegularizedMasterSolver(epsilon=1e-4)`,
    tie_break="none", hysteresis pruning); bilinear-PSD uses `SdpMasterSolver`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import ot
from scipy import sparse

from iot import (
    CuPdlpxMasterSolver,
    InverseOTError,
    InverseOTProblem,
    RegularizedMasterSolver,
    SdpMasterSolver,
    dense_cost_matrix,
    dual_optimality_certificate,
    is_optimal_plan,
    mandatory_edges,
    solve_inverse_ot,
    support_components,
)

from .families import FamilySpec, build_family
from .marginals import marginal_spec
from .records import RunRecord, RunSpec, write_csv, write_json

EPSILON: float = 1e-4
LP_TOL: float = 1e-9
SDP_TOL: float = 1e-8
TAU_PRUNE: float = 1e-12
GAMMA: int = 1


def _mean(values: list[float]) -> float:
    """CN: 标量均值（空列表返回 0）。EN: Mean of a list (0.0 when empty)."""
    return float(np.mean(values)) if values else 0.0


def run_spec(spec: RunSpec) -> RunRecord:
    """
    CN: 运行一个 `RunSpec`：建实例（POT 真值）→ 求解 → 认证/验证 → 记录。
    EN: Run one `RunSpec`: build the instance (POT ground truth) -> solve -> certify/validate
        -> record.
    """
    rng = np.random.default_rng(int(spec.seed))
    family_spec = FamilySpec(
        spec.family, point_dimension=int(spec.point_dimension), lowrank_k=int(spec.lowrank_k)
    )
    features, theta_set, theta_star = build_family(family_spec, rng, int(spec.m), int(spec.n))

    source_mass = marginal_spec(spec.marginal).weights(int(spec.m))
    target_mass = marginal_spec(spec.marginal).weights(int(spec.n))
    cost = dense_cost_matrix(features, theta_star)

    scaffold_started = time.perf_counter()
    plan = np.asarray(
        ot.emd(source_mass, target_mass, cost, numItermax=int(spec.num_itermax)), dtype=np.float64
    )
    scaffold_seconds = float(time.perf_counter() - scaffold_started)

    problem = InverseOTProblem(
        features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan)
    )
    certificate = dual_optimality_certificate(
        problem, theta_star, num_itermax=int(spec.num_itermax)
    )
    partition = support_components(problem)
    mandatory, _ = mandatory_edges(problem)
    positive = problem.plan.data[problem.plan.data > 0.0]
    min_positive = float(np.min(positive)) if positive.size else 0.0

    solver_name = spec.solver
    if solver_name == "auto":
        solver_name = "sdp" if family_spec.is_sdp else "regularized"
    if solver_name == "sdp":
        solver = SdpMasterSolver()
    elif solver_name == "cupdlpx":
        solver = CuPdlpxMasterSolver()
    else:
        solver = RegularizedMasterSolver(epsilon=EPSILON, random_seed=int(spec.seed))
    tol = SDP_TOL if solver_name == "sdp" else LP_TOL

    solve_started = time.perf_counter()
    outcome = "converged"
    error: str | None = None
    is_optimal: bool | None = None
    relative_gap: float | None = None
    iterations = 0
    converged = False
    max_support = int(problem.support_size)
    delta = float("nan")
    final_max_violation = float("nan")
    master_seconds: list[float] = []
    oracle_seconds: list[float] = []
    theta: np.ndarray | None = None
    try:
        result = solve_inverse_ot(
            problem,
            tie_break="none",
            solver=solver,
            max_iterations=int(spec.max_iterations),
            random_seed=int(spec.seed),
            tol=tol,
            tau_prune=TAU_PRUNE,
            gamma=GAMMA,
            validate=False,
            prune=bool(spec.prune),
            prune_hysteresis=bool(spec.prune),
            scan_backend=str(spec.scan_backend),
        )
        iterations = int(result.n_iterations)
        converged = bool(result.converged)
        records = result.iterations
        max_support = int(max((rec.support_size for rec in records), default=int(problem.support_size)))
        delta = float(result.delta)
        final_max_violation = float(result.final_max_violation)
        master_seconds = [rec.master_time for rec in records]
        oracle_seconds = [rec.oracle_time for rec in records]
        theta = np.asarray(result.theta, dtype=np.float64)
        outcome = "converged" if converged else "max_iterations"
        if bool(spec.validate) and theta is not None:
            try:
                check = is_optimal_plan(
                    problem, theta, tol=1e-7, num_itermax=int(spec.num_itermax)
                )
                is_optimal = bool(check.is_optimal)
                relative_gap = float(check.relative_gap)
            except InverseOTError as exc:
                error = f"validate failed: {exc}"
    except InverseOTError as exc:
        outcome = "infeasible"
        error = str(exc)
    solve_seconds = float(time.perf_counter() - solve_started)

    return RunRecord(
        spec=spec,
        n_features=int(problem.n_features),
        support_size=int(problem.support_size),
        n_components=int(partition.n_components),
        iterations=int(iterations),
        converged=bool(converged),
        delta=float(delta),
        final_max_violation=float(final_max_violation),
        max_support_over_m_plus_n=float(max_support) / float(problem.n_source + problem.n_target),
        is_optimal=is_optimal,
        relative_gap=relative_gap,
        certified=bool(certificate.certified),
        cert_support_residual=float(certificate.support_residual),
        min_positive_plan_entry=min_positive,
        outcome=str(outcome),
        error=error,
        scaffold_seconds=float(scaffold_seconds),
        solve_seconds=solve_seconds,
        mean_master_seconds=_mean(master_seconds),
        mean_oracle_seconds=_mean(oracle_seconds),
    )


def run_grid(
    specs: Sequence[RunSpec],
    *,
    out: Path,
    meta: dict[str, object] | None = None,
    verbose: bool = True,
) -> list[RunRecord]:
    """
    CN: 顺序运行一组 RunSpec，每完成一条就原子写出 JSON+CSV（预算被砍也保留轨迹）。
    EN: Run a sequence of RunSpecs, flushing JSON+CSV after every record (a budget kill still
        leaves the trajectory).
    """
    out = Path(out)
    csv_out = out.with_suffix(".csv")
    records: list[RunRecord] = []
    header = (
        f"{'run_id':>44} {'K':>4} {'|S|':>7} {'p':>6} {'it':>4} {'conv':>5} {'cert':>5} "
        f"{'master_s/iter':>13} {'oracle_s/iter':>13} {'solve_s':>8} {'outcome':>12}"
    )
    if verbose:
        print(header, flush=True)
    for spec in specs:
        record = run_spec(spec)
        records.append(record)
        if verbose:
            print(
                f"{record.run_id:>44} {record.n_features:>4} {record.support_size:>7} "
                f"{record.n_components:>6} {record.iterations:>4} {str(record.converged):>5} "
                f"{str(record.certified):>5} {record.mean_master_seconds:>13.3f} "
                f"{record.mean_oracle_seconds:>13.3f} {record.solve_seconds:>8.2f} "
                f"{record.outcome:>12}",
                flush=True,
            )
        write_json(out, records, meta=meta)
        write_csv(csv_out, records)
    return records


__all__ = ["EPSILON", "GAMMA", "LP_TOL", "SDP_TOL", "TAU_PRUNE", "run_grid", "run_spec"]
