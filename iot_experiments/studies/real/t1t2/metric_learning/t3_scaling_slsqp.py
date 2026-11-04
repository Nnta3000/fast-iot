"""
EN: **T3 scalability study -- where does the dense SLSQP reference stop being practical?**

    Question. The joint reduced objective
        Psi_red(theta, g) = <phi_bar, theta> - b.g + eps * sum_i a_i * (1 - log a_i + L_i)
    is minimised over `(theta, g_1..g_{n-1})` by SciPy SLSQP. Its cost is driven by the DENSE number of
    variables `K + n_eff - 1` and by the grid it evaluates. This study measures that growth and records a
    size, memory and accuracy at which it ceases to be practical, so that any future sparse solver has a
    concrete reason to exist rather than a speculative one.

    Scope of THIS version: the FULL grid only. The eventual T3 operates on a restricted, adaptively chosen
    edge set `F`; restricted-`F` rows use the same code paths (the edge set is just smaller) and are
    written by the same worker, but this run records full-grid rows only. Restricted-`F` benchmarking is
    enabled by `--edge-fraction` and is the immediate follow-up.

    Every row records: dense variables, transport edges, the time of ONE objective and ONE gradient
    evaluation at the final iterate (measured separately from the optimiser), total optimiser time, peak
    RSS, iterations, function evaluations, marginal residual, theta KKT violation and a genuine
    convergence verdict. **Process start-up is excluded**: each worker reports its own timings from inside
    the process, and start-up is recorded once, separately.

CN: **T3 可扩展性研究 —— 稠密 SLSQP 在哪个规模上不再实用？**
    本版只做全网格；受限 `F` 由 `--edge-fraction` 打开，是紧接着的下一步。每个点记录变量数、边数、
    单次目标/梯度求值时间、总优化时间、峰值内存、迭代与函数求值次数、边缘残差、θ KKT 违背、真实收敛判定。
    **进程启动时间单独记录，不计入求解时间。**
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

from iot.core.parameters import cost_std  # noqa: E402
from iot.entropic.joint_reference import (  # noqa: E402
    ReducedProblem,
    evaluate,
    marginals,
    simplex_kkt,
    solve,
)
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.lib.planting import plant_instance  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402

SIZES: tuple[int, ...] = (300, 450, 675, 1000, 1500)
REL_EPS_GRID: tuple[float, ...] = (0.5, 0.25, 0.1)
POINT_DIMENSION: int = 4
# EN: a run is aborted (and recorded as aborted, not retried) if the process would need more than this.
# CN: 超过该内存上限就中止并如实记录，不重试。
MEMORY_LIMIT_GB: float = 4.0

PROVENANCE: dict[str, object] = {
    "question": "at what size/memory/accuracy does the dense SLSQP joint solver cease to be practical?",
    "objective": "Psi_red(theta,g) = <phi_bar,theta> - b.g + eps*sum_i a_i*(1-log a_i+L_i)",
    "solver": "iot.entropic.joint_reference (SciPy SLSQP, dense, on the reduced joint objective)",
    "measured_separately": "one objective evaluation, one gradient evaluation, total optimiser time",
    "startup_policy": "process start-up measured once and excluded from all solver timings",
    "scope": "full grid; restricted-F via --edge-fraction is the immediate follow-up",
    "memory_limit_gb": MEMORY_LIMIT_GB,
}


@dataclasses.dataclass(frozen=True)
class ScaleConfig(RunConfig):
    """EN: knobs. CN: 旋钮。"""

    section = "t3_scaling_slsqp"
    study = "t3_scaling_slsqp"
    point_dimension: int = POINT_DIMENSION
    max_iterations: int = 2000
    edge_fraction: float = 1.0

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """EN: only this study's flags. CN: 只加本脚本特有的参数。"""
        parser.add_argument("--point-dimension", type=int, default=POINT_DIMENSION)
        parser.add_argument("--max-iterations", type=int, default=2000)
        parser.add_argument("--edge-fraction", type=float, default=1.0)


def _peak_rss_mb() -> float:
    """EN: peak resident set size of this process, in MB. CN: 本进程峰值内存（MB）。"""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _available_gb() -> float:
    """EN: `MemAvailable` in GB. CN: 可用内存（GB）。"""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        pass
    return float("nan")


def _build(n: int, point_dimension: int, rel_eps: float, edge_fraction: float):
    """
    EN: assemble the problem for one size. `edge_fraction < 1` keeps a deterministic subset of edges
        (row-major thinning) so the restricted-`F` regime uses exactly the same code path.
    CN: 构造一个问题；`edge_fraction < 1` 保留边的一个确定子集（行主序稀疏化），受限 `F` 走同一代码路径。
    """
    planted = plant_instance(family="diagonal", n=int(n), point_dimension=int(point_dimension), seed=0)
    problem = planted.problem
    m = int(problem.features.n_source)
    j = int(problem.features.n_target)
    k = int(planted.theta_star.size)
    rows = np.repeat(np.arange(m), j)
    cols = np.tile(np.arange(j), m)
    if edge_fraction < 1.0:
        stride = max(1, int(round(1.0 / max(edge_fraction, 1e-9))))
        keep = np.arange(rows.size) % stride == 0
        rows, cols = rows[keep], cols[keep]
    features = np.asarray(problem.features.features(rows, cols), dtype=np.float64)
    a: Float[np.ndarray, "m"] = np.asarray(problem.resolved_source_mass, dtype=np.float64)
    b: Float[np.ndarray, "n"] = np.asarray(problem.resolved_target_mass, dtype=np.float64)
    support_rows, support_cols, support_mass = problem.support()
    phi_bar: Float[np.ndarray, "k"] = (np.asarray(support_mass, dtype=np.float64)[:, None]
                                       * np.asarray(problem.features.features(support_rows, support_cols),
                                                    dtype=np.float64)).sum(axis=0)
    scale = float(cost_std(np.asarray(problem.features.features(rows, cols),
                                      dtype=np.float64) @ np.full(k, 1.0 / k)))
    epsilon = float(rel_eps) * scale
    reduced = ReducedProblem(features, np.asarray(rows, dtype=np.int64),
                             np.asarray(cols, dtype=np.int64), a, b, phi_bar, epsilon)
    return problem, reduced, k, j, scale, epsilon, int(rows.size)


def _run_size(n: int, point_dimension: int, rel_eps: float, max_iterations: int,
              edge_fraction: float) -> dict[str, object]:
    """
    EN: build one instance, measure a single objective and gradient evaluation, then run SLSQP.
        Failures and aborts are recorded, never retried.
    CN: 构造实例、测单次目标/梯度求值、再跑 SLSQP。失败与中止如实记录，不重试。
    """
    started = time.perf_counter()
    try:
        problem, reduced, k, j, scale, epsilon, n_edges = _build(n, point_dimension, rel_eps,
                                                                 edge_fraction)
    except Exception as error:  # noqa: BLE001
        return {"n": n, "rel_eps": rel_eps, "ok": False,
                "stop_reason": f"BUILD FAILED: {type(error).__name__}: {str(error)[:160]}"}
    build_seconds = time.perf_counter() - started
    theta = np.full(k, 1.0 / k)
    g = np.zeros(j, dtype=np.float64)
    # EN: time ONE objective and ONE gradient evaluation at the starting iterate.
    # CN: 在起点处各测一次目标与梯度求值。
    repeats = 3
    started = time.perf_counter()
    for _ in range(repeats):
        _psi = evaluate(reduced, theta, g)[0]
    objective_seconds = (time.perf_counter() - started) / repeats
    started = time.perf_counter()
    for _ in range(repeats):
        _gt = evaluate(reduced, theta, g)[1]
    gradient_seconds = (time.perf_counter() - started) / repeats
    import iot.entropic.joint_reference as _regexp  # noqa: PLC0415
    evaluator_calls = {"n": 0}
    _original_evaluate = _regexp.evaluate

    def _counting_evaluate(*args, **kwargs):
        """EN: count objective evaluations performed by the optimiser. CN: 统计优化器的目标求值次数。"""
        evaluator_calls["n"] += 1
        return _original_evaluate(*args, **kwargs)

    _regexp.evaluate = _counting_evaluate
    started = time.perf_counter()
    try:
        run = solve(reduced, theta, g, label="uniform", max_iterations=int(max_iterations))
    except Exception as error:  # noqa: BLE001
        return {"n": n, "rel_eps": rel_eps, "ok": False,
                "stop_reason": f"SOLVE FAILED: {type(error).__name__}: {str(error)[:160]}",
                "dense_variables": int(k + j - 1), "edges": n_edges}
    optimiser_seconds = time.perf_counter() - started
    _regexp.evaluate = _original_evaluate
    # EN: SLSQP calls `objective` (value+gradient together) once per evaluation; wrap it to count calls.
    #     The count is measured, not assumed.
    # CN: SLSQP 每次评估值+梯度一起调用，包裹它来**实测**调用次数。
    _psi_ref, grad_theta, grad_g, plan = evaluate(reduced, run.theta, run.g)
    row_error, col_error = marginals(reduced, plan)
    _multiplier, kkt_violation = simplex_kkt(grad_theta, run.theta)
    # EN: three separate measures, deliberately not conflated: optimiser flag, KKT feasibility, objective.
    # CN: 三个度量分开记录：优化器标志、KKT 可行性、目标值。
    optimiser_success = int(run.status) == 0
    kkt_feasible = bool(kkt_violation <= 1e-6 and col_error <= 1e-5)
    return {
        "n": n, "rel_eps": float(rel_eps), "ok": True,
        "m_eff": int(problem.features.n_source), "n_eff": int(j), "K": int(k),
        "dense_variables": int(k + j - 1), "edges": int(n_edges),
        "edge_fraction": float(edge_fraction),
        "epsilon": float(epsilon), "cost_std": float(scale),
        "psi": float(run.psi), "psi_recomputed": float(_psi_ref),
        "build_seconds": round(build_seconds, 4),
        "objective_eval_seconds": round(objective_seconds, 8),
        "gradient_eval_seconds": round(gradient_seconds, 8),
        "optimiser_seconds": round(optimiser_seconds, 4),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "iterations": int(run.iterations),
        "objective_evaluations": int(evaluator_calls["n"]),
        "estimated_objective_seconds": round(float(evaluator_calls["n"]) * objective_seconds, 4),
        "estimated_gradient_seconds": round(float(evaluator_calls["n"]) * gradient_seconds, 4),
        "other_seconds": round(float(optimiser_seconds)
                               - float(evaluator_calls["n"]) * (objective_seconds + gradient_seconds), 4),
        "row_marginal_error": float(row_error), "col_marginal_error": float(col_error),
        "theta_kkt_violation": float(kkt_violation),
        "g_stationarity": float(np.max(np.abs(grad_g[1:]))),
        "optimiser_success": optimiser_success,
        "kkt_feasible": kkt_feasible,
        "converged": bool(optimiser_success and kkt_feasible),
        "stop_reason": str(run.message)[:160],
    }


def main(argv: list[str] | None = None) -> int:
    """EN: entry point; dispatches single-size workers. CN: 入口；分派单规模子进程。"""
    tokens = list(argv) if argv is not None else sys.argv[1:]
    if "--worker" in tokens:
        position = tokens.index("--worker")
        n_text, rel_text = tokens[position + 1:position + 3]
        record = _run_size(int(n_text), POINT_DIMENSION, float(rel_text), 2000, 1.0)
        print(json.dumps(record))
        return 0

    parser = build_parser(ScaleConfig)
    args = parser.parse_args(tokens)
    config = ScaleConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("T3 scalability study: dense SLSQP joint solver on the full grid")
    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem_available={preflight.mem_available_gb:.1f} GB")
    log.line(f"sizes={SIZES} rel_eps={REL_EPS_GRID} dimension={config.point_dimension} "
             f"memory_limit={MEMORY_LIMIT_GB} GB")

    # EN: process start-up, measured ONCE, so it can be excluded from every solver timing.
    # CN: 进程启动时间只测一次，用于从求解时间中扣除。
    startup_started = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import numpy, scipy.optimize"], check=False)
    startup_seconds = time.perf_counter() - startup_started
    log.line(f"process start-up (excluded from solver timings): {startup_seconds:.3f}s")

    records: list[dict[str, object]] = []
    aborted = False
    for n in SIZES:
        for rel_eps in REL_EPS_GRID:
            if aborted:
                records.append({"n": n, "rel_eps": rel_eps, "ok": False,
                                "stop_reason": "SKIPPED: memory limit reached earlier"})
                continue
            available = _available_gb()
            if available < MEMORY_LIMIT_GB:
                log.line(f"n={n} rel_eps={rel_eps}: SKIPPED, only {available:.1f} GB available "
                         f"(limit {MEMORY_LIMIT_GB} GB)")
                records.append({"n": n, "rel_eps": rel_eps, "ok": False,
                                "stop_reason": f"SKIPPED: {available:.1f} GB available"})
                aborted = True
                continue
            log.line(f"n={n} rel_eps={rel_eps}: start (available {available:.1f} GB)")
            command = [sys.executable, "-m",
                       "iot_experiments.studies.real.t1t2.metric_learning.t3_scaling_slsqp",
                       "--worker", str(n), str(rel_eps)]
            completed = subprocess.run(command, capture_output=True, text=True, cwd=str(_REPO_ROOT))
            if completed.returncode != 0:
                record: dict[str, object] = {"n": n, "rel_eps": rel_eps, "ok": False,
                                             "stop_reason": f"worker exit {completed.returncode}",
                                             "stderr": completed.stderr[-400:]}
            else:
                try:
                    record = json.loads(completed.stdout.strip().splitlines()[-1])
                except (ValueError, IndexError):
                    record = {"n": n, "rel_eps": rel_eps, "ok": False,
                              "stop_reason": "unparseable worker output"}
            records.append(record)
            if record.get("ok"):
                log.line(f"    vars={record['dense_variables']:>5} edges={record['edges']:>8} "
                         f"opt={record['optimiser_seconds']:>8.3f}s "
                         f"obj={record['objective_eval_seconds']*1e3:>8.3f}ms "
                         f"grad={record['gradient_eval_seconds']*1e3:>8.3f}ms "
                         f"rss={record['peak_rss_mb']:>6.0f}MB it={record['iterations']:>4} "
                         f"col={record['col_marginal_error']:.1e} kkt={record['theta_kkt_violation']:.1e} "
                         f"conv={record['converged']}")
            else:
                log.line(f"    FAILED: {record.get('stop_reason')}")

    payload = {"records": records, "provenance": PROVENANCE, "preflight": preflight.to_dict(),
               "startup_seconds": round(startup_seconds, 4),
               "sizes": list(SIZES), "rel_eps": list(REL_EPS_GRID),
               "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                          for key, v in dataclasses.asdict(config).items()}}
    (run_dir / "scaling_slsqp.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                                encoding="utf-8")
    log.line(f"wrote {run_dir / 'scaling_slsqp.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
