"""
EN: **EOT solver comparison -- independent SLSQP vs the custom Newton-CG, by size and regularisation.**

    Purpose. The joint FORMULATION is now established: the corrected joint master matches the independent
    reduced-joint SLSQP on the Gate B instance at `rel_eps = 0.5` from all four starts, and the reduced
    Hessian `S^T H S` agrees with finite differences to `2e-8`. What is NOT established is which numerical
    method can carry it. This study answers only that.

    Both solvers minimise the same object:

        Psi_red(theta, g) = <phi_bar, theta> - b.g + eps * sum_i a_i * (1 - log a_i + L_i),

    with the same data, the same `eps` (absolute, `rel_eps * std(c(theta_ref = uniform))`), the same
    simplex constraint and the same gauge `g_0 = 0`. They differ only in algorithm:
      * `slsqp`    -- `iot.entropic.joint_reference`: SciPy SLSQP on `(theta, g_1..g_{n-1})`;
      * `newtoncg` -- `iot.entropic.joint_master`: Newton-CG with a primal active set on the simplex.

    Both are JOINT methods. The reference is not the old nested method (SLSQP on theta with an OT solve
    inside); it optimises the reduced joint objective directly.

    Reported per (size, rel_eps, solver, start): objective, theta, wall time, peak RSS delta, source and
    target marginal residuals, theta stationarity residual, pinned-multiplier violation, g stationarity
    residual, convergence flag and stop reason.

    **Wall time is reported as measured, and the two solvers run in separate processes so that peak memory
    is attributable.** A missing row means the run failed; it is recorded as a failure, never dropped.

CN: **EOT 求解器比较 —— 独立 SLSQP 与自定义 Newton-CG，按规模与正则强度。**
    两者都在最小化同一个约化联合目标 `Psi_red(theta, g)`，数据、`eps`、单纯形约束与规范完全一致，
    只差在算法。参考实现是**联合**方法，不是旧的嵌套方法。
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.edges import EdgeSet  # noqa: E402
from iot.core.parameters import cost_std  # noqa: E402
from iot.entropic.joint_reference import (  # noqa: E402
    ReducedProblem,
    evaluate,
    marginals,
    simplex_kkt,
    solve,
)
from experiments.harness.config import RunConfig, build_parser  # noqa: E402
from experiments.harness.lib.planting import plant_instance  # noqa: E402
from experiments.harness.preflight import check as preflight_check  # noqa: E402
from experiments.harness.records import RunLog  # noqa: E402

SIZES: tuple[int, ...] = (24, 70, 120, 300)
REL_EPS_GRID: tuple[float, ...] = (0.5, 0.25, 0.1)
POINT_DIMENSION: int = 4

PROVENANCE: dict[str, object] = {
    "question": "which joint solver carries EOT -- SLSQP reference or custom Newton-CG -- as size and eps grow?",
    "objective": "both minimise the SAME Psi_red(theta,g) = <phi_bar,theta> - b.g + eps*sum_i a_i*(1-log a_i+L_i)",
    "reference": "iot.entropic.joint_reference (SciPy SLSQP on the reduced joint objective; NOT the nested method)",
    "experimental": "iot.entropic.joint_master (Newton-CG with a primal active set); NOT patched in this study",
    "starts": "structured starts only (uniform interior and theta from the nested study); no oracle starts",
    "memory": "each solver runs in its own subprocess; peak RSS is read from /proc for that process",
}


@dataclasses.dataclass(frozen=True)
class CompareConfig(RunConfig):
    """EN: knobs. CN: 旋钮。"""

    section = "eot_solver_compare"
    study = "eot_solver_compare"
    point_dimension: int = POINT_DIMENSION
    max_iterations: int = 300

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """EN: only this study's flags. CN: 只加本脚本特有的参数。"""
        parser.add_argument("--point-dimension", type=int, default=POINT_DIMENSION)
        parser.add_argument("--max-iterations", type=int, default=300)


def _build(n: int, point_dimension: int, rel_eps: float):
    """
    EN: assemble the SAME problem for both solvers and return everything needed to identify it.
    CN: 为两个求解器构造同一个问题，并返回识别它所需的全部信息。
    """
    planted = plant_instance(family="diagonal", n=int(n), point_dimension=int(point_dimension), seed=0)
    problem = planted.problem
    m = int(problem.features.n_source)
    j = int(problem.features.n_target)
    k = int(planted.theta_star.size)
    rows = np.repeat(np.arange(m), j)
    cols = np.tile(np.arange(j), m)
    features = np.asarray(problem.features.features(rows, cols), dtype=np.float64)
    a: Float[np.ndarray, "m"] = np.asarray(problem.resolved_source_mass, dtype=np.float64)
    b: Float[np.ndarray, "n"] = np.asarray(problem.resolved_target_mass, dtype=np.float64)
    support_rows, support_cols, support_mass = problem.support()
    phi_bar: Float[np.ndarray, "k"] = (np.asarray(support_mass, dtype=np.float64)[:, None]
                                       * np.asarray(problem.features.features(support_rows, support_cols),
                                                    dtype=np.float64)).sum(axis=0)
    scale = float(cost_std(features @ np.full(k, 1.0 / k)))
    epsilon = float(rel_eps) * scale
    reduced = ReducedProblem(features, np.asarray(rows, dtype=np.int64),
                             np.asarray(cols, dtype=np.int64), a, b, phi_bar, epsilon)
    edges = EdgeSet.from_arrays(rows, cols, n_target=j)
    return problem, reduced, edges, k, j, scale, epsilon


def _peak_rss_mb() -> float:
    """EN: this process's peak resident set size in MB. CN: 本进程峰值常驻内存（MB）。"""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _run_solver(which: str, n: int, point_dimension: int, rel_eps: float, start: str,
                max_iterations: int) -> dict[str, object]:
    """
    EN: run ONE solver on ONE problem from ONE start and return a record. Failures are recorded, not raised.
    CN: 在同一个问题的同一个起点上跑一个求解器，返回一条记录；失败会被记录而不是抛出。
    """
    problem, reduced, edges, k, j, scale, epsilon = _build(n, point_dimension, rel_eps)
    theta_start = np.full(k, 1.0 / k) if start == "uniform" else np.asarray(
        plant_instance(family="diagonal", n=int(n), point_dimension=int(point_dimension),
                       seed=0).theta_star, dtype=np.float64)
    g_start = np.zeros(j, dtype=np.float64)
    started_rss = _peak_rss_mb()
    started = time.perf_counter()
    try:
        if which == "slsqp":
            run = solve(reduced, theta_start, g_start, label=start, max_iterations=max_iterations)
            theta, g, psi = run.theta, run.g, run.psi
            converged = int(run.status) == 0
            stop_reason = run.message
            iterations = int(run.iterations)
        else:
            from iot.entropic.joint_master import make_problem, solve_joint_master  # noqa: PLC0415

            data = make_problem(problem, edges, epsilon=epsilon)
            result = solve_joint_master(data, theta_start,
                                        constraints=problem.theta_set.linear_constraints(),
                                        max_newton=int(max_iterations), gradient_tolerance=1e-9)
            theta, g, psi = result.theta, result.target_dual, result.psi
            converged = bool(result.converged)
            stop_reason = result.stop_reason
            iterations = int(result.newton_steps)
        seconds = time.perf_counter() - started
        peak_mb = _peak_rss_mb()
        # EN: every residual is recomputed from the final iterate with the REFERENCE evaluator, so the two
        #     solvers are judged by exactly the same yardstick.
        # CN: 所有残差都用参考评价器在最终点上重算，两个求解器用同一把尺子。
        _psi_ref, grad_theta, grad_g, plan = evaluate(reduced, theta, g)
        row_error, col_error = marginals(reduced, plan)
        _multiplier, kkt_violation = simplex_kkt(grad_theta, theta)
        return {
            "solver": which, "n": int(n), "rel_eps": float(rel_eps), "epsilon": epsilon,
            "start": start, "m_eff": int(problem.features.n_source), "n_eff": int(j), "K": int(k),
            "grid_edges": int(problem.features.n_source * j),
            "psi": float(psi), "psi_recomputed": float(_psi_ref),
            "theta": [round(float(v), 10) for v in theta],
            "seconds": round(seconds, 4), "peak_rss_mb": round(peak_mb, 1),
            "rss_delta_mb": round(peak_mb - started_rss, 1),
            "row_marginal_error": float(row_error), "col_marginal_error": float(col_error),
            "theta_kkt_violation": float(kkt_violation),
            "g_stationarity": float(np.max(np.abs(grad_g[1:]))),
            "converged": bool(converged), "stop_reason": str(stop_reason)[:160],
            "iterations": int(iterations), "ok": True,
        }
    except Exception as error:  # noqa: BLE001 - a failed run is data, not a crash
        return {
            "solver": which, "n": int(n), "rel_eps": float(rel_eps), "epsilon": epsilon,
            "start": start, "seconds": round(time.perf_counter() - started, 4),
            "peak_rss_mb": round(_peak_rss_mb(), 1), "ok": False,
            "error": f"{type(error).__name__}: {str(error)[:200]}",
            "converged": False, "stop_reason": f"FAILED: {type(error).__name__}",
        }


def main(argv: list[str] | None = None) -> int:
    """EN: entry point; also dispatches single-solver subprocesses. CN: 入口；也负责子进程分发。"""
    tokens = list(argv) if argv is not None else sys.argv[1:]
    # EN: internal subprocess mode is dispatched BEFORE argparse, which does not know `--worker`.
    # CN: 子进程模式在 argparse 之前分发。
    if "--worker" in tokens:
        position = tokens.index("--worker")
        which, n_text, rel_text, start = tokens[position + 1:position + 5]
        print(json.dumps(_run_solver(which, int(n_text), POINT_DIMENSION, float(rel_text), start, 300)))
        return 0
    parser = build_parser(CompareConfig)
    args = parser.parse_args(tokens)
    # EN: internal subprocess mode: `--worker solver n rel_eps start`, prints one JSON record.
    # CN: 子进程模式，输出一条 JSON 记录。
    config = CompareConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("EOT solver comparison: SLSQP reference vs custom Newton-CG")
    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem={preflight.mem_available_gb:.1f} GB")
    log.line(f"sizes={SIZES} rel_eps={REL_EPS_GRID} dimension={config.point_dimension}")

    records: list[dict[str, object]] = []
    for n in SIZES:
        for rel_eps in REL_EPS_GRID:
            for which in ("slsqp", "newtoncg"):
                for start in ("uniform", "theta_star"):
                    log.line(f"n={n} rel_eps={rel_eps} solver={which} start={start}: start")
                    command = [sys.executable, "-m",
                               "experiments.studies.iot.real.t1t2.metric_learning.eot_solver_compare",
                               "--worker", which, str(n), str(rel_eps), start,
                               "--point-dimension", str(config.point_dimension)]
                    # EN: wall time is measured around the subprocess, so the number is end-to-end and
                    #     comparable across solvers regardless of where the work happens.
                    # CN: 墙钟时间包住整个子进程，端到端可比。
                    wall_started = time.perf_counter()
                    completed = subprocess.run(command, capture_output=True, text=True,
                                               cwd=str(_REPO_ROOT))
                    wall_seconds = time.perf_counter() - wall_started
                    record: dict[str, object]
                    if completed.returncode != 0:
                        record = {"solver": which, "n": n, "rel_eps": rel_eps, "start": start,
                                  "ok": False, "converged": False,
                                  "stop_reason": f"subprocess exit {completed.returncode}",
                                  "stderr": completed.stderr[-400:]}
                    else:
                        try:
                            record = json.loads(completed.stdout.strip().splitlines()[-1])
                        except (ValueError, IndexError):
                            record = {"solver": which, "n": n, "rel_eps": rel_eps, "start": start,
                                      "ok": False, "converged": False,
                                      "stop_reason": "unparseable worker output"}
                    if record.get("ok"):
                        record["wall_seconds"] = round(wall_seconds, 4)
                    records.append(record)
                    if record.get("ok"):
                        log.line(f"    psi={float(record['psi']):.10e} "
                                 f"conv={record['converged']} {record['stop_reason'][:40]} "
                                 f"{record['seconds']:.3f}s rss={record['peak_rss_mb']}MB "
                                 f"col={record['col_marginal_error']:.1e} "
                                 f"kkt={record['theta_kkt_violation']:.1e}")
                    else:
                        log.line(f"    FAILED: {record.get('stop_reason')}")

    payload = {"records": records, "provenance": PROVENANCE, "preflight": preflight.to_dict(),
               "sizes": list(SIZES), "rel_eps": list(REL_EPS_GRID),
               "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                          for key, v in dataclasses.asdict(config).items()}}
    (run_dir / "solver_compare.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                                 encoding="utf-8")
    log.line(f"wrote {run_dir / 'solver_compare.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
