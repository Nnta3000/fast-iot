"""
CN: 精确 OT 次梯度法（projected subgradient on L0）的实验 runner。内层精确 OT 可选用 POT
    （稠密代价，CPU）或 HELLO-OT（点云重缩放后的平方欧氏 OT，O(m+n)，支持 CPU/GPU）。
EN: Experiment runner for the exact-OT subgradient method (projected subgradient on L0). The
    inner exact OT is either POT (dense cost, CPU) or HELLO-OT (squared-Euclidean OT on rescaled
    points, O(m+n), CPU/GPU).

CN: 该脚本不修改 `src/iot`；它复用 `iot.subgradient` 的学习器、共享 checkpoint schema 与
    离线 G/R 评估。评估在计时区之外进行。
EN: This script does not modify `src/iot`; it reuses the `iot.subgradient` learner, the shared
    checkpoint schema, and the offline G/R evaluator. Evaluation runs outside the timed section.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.alternatives.subgradient import (  # noqa: E402
    ExactPlanSolver,
    SubgradientConfig,
    fit_theta_subgradient,
)
from iot_experiments.benchmarks.checkpoint import (  # noqa: E402
    RunCheckpoint,
    ThetaCheckpoint,
    write_run,
)
from iot_experiments.benchmarks.evaluate import reference_scores, score_theta  # noqa: E402
from iot_experiments.benchmarks.instances import (  # noqa: E402
    DEFAULT_K,
    DEFAULT_NUM_ITERMAX,
    BenchmarkInstance,
    build_instance,
)
from iot_experiments.benchmarks.runner import params_from_instance  # noqa: E402
from iot.alternatives.sinkhorn_theta import project_theta  # noqa: E402

PINNED_THREADS: tuple[int, int, int] = (8, 8, 8)
LOCK_PATH = _REPO_ROOT / "iot_experiments" / "benchmark" / ".cpu_lock"
STEP_RULES: tuple[str, ...] = ("constant", "inv_sqrt", "polyak")


def _peak_rss_mb() -> float:
    """CN: 真实进程峰值 RSS（KB → MB）。EN: Real process peak RSS (KB -> MB)."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _threads() -> dict[str, int]:
    """CN: 记录固定线程数。EN: Record the pinned thread counts."""
    return {
        "OMP_NUM_THREADS": int(os.environ.get("OMP_NUM_THREADS", PINNED_THREADS[0])),
        "MKL_NUM_THREADS": int(os.environ.get("MKL_NUM_THREADS", PINNED_THREADS[1])),
        "OPENBLAS_NUM_THREADS": int(os.environ.get("OPENBLAS_NUM_THREADS", PINNED_THREADS[2])),
    }


def make_pot_solver(instance: BenchmarkInstance) -> ExactPlanSolver:
    """CN: POT 精确 OT 内层求解器（稠密代价）。EN: POT exact-OT inner solver (dense cost)."""
    from iot.alternatives.subgradient import make_pot_solver as _make

    return _make(instance.observed_problem(), num_itermax=DEFAULT_NUM_ITERMAX)


def make_hello_solver(
    instance: BenchmarkInstance,
    *,
    device: str = "cpu",
    max_iterations: int = 100,
) -> ExactPlanSolver:
    """
    CN: HELLO-OT 内层求解器：`c_ij(theta) = ||sqrt(theta)*x_i - sqrt(theta)*y_j||^2`，
        通过对点云做 sqrt(theta) 重缩放到平方欧氏 OT，O(m+n) 内存，无 m*n 代价矩阵。
    EN: HELLO-OT inner solver: `c_ij(theta) = ||sqrt(theta)*x_i - sqrt(theta)*y_j||^2` by
        rescaling points by sqrt(theta) into a squared-Euclidean OT, O(m+n) memory and no
        m*n cost matrix.
    """
    import hello_ot
    from hello_ot.config import SolverOptions

    source = np.ascontiguousarray(
        np.asarray(instance.features.source_points, dtype=np.float64)
    )
    target = np.ascontiguousarray(
        np.asarray(instance.features.target_points, dtype=np.float64)
    )
    source_mass = np.ascontiguousarray(instance.observed_source_mass, dtype=np.float64)
    target_mass = np.ascontiguousarray(instance.observed_target_mass, dtype=np.float64)
    options = SolverOptions(backend="torch", torch_device=str(device), verbose="off")

    def solve(theta: np.ndarray) -> sparse.spmatrix:
        root = np.sqrt(np.clip(np.asarray(theta, dtype=np.float64).reshape(-1), 0.0, None))
        scaled_source = np.ascontiguousarray(source * root[None, :], dtype=np.float32)
        scaled_target = np.ascontiguousarray(target * root[None, :], dtype=np.float32)
        result = hello_ot.solve(
            scaled_source,
            scaled_target,
            source_mass=source_mass,
            target_mass=target_mass,
            max_iterations=int(max_iterations),
            options=options,
        )
        solution = result.solution
        return sparse.coo_matrix(
            (solution.values, (solution.rows, solution.cols)), shape=solution.shape
        ).tocsr()

    return solve


def _checkpoint_indices(total: int, count: int) -> list[int]:
    """CN: 几何间隔的 checkpoint 迭代（含末次）。EN: Geometrically spaced checkpoint indices."""
    if total <= int(count):
        return list(range(1, total + 1))
    indices = sorted({int(round(float(value))) for value in np.geomspace(1, total, int(count))})
    indices = [min(max(1, index), total) for index in indices]
    unique = sorted(set(indices))
    if unique[-1] != total:
        unique.append(total)
    return unique


@dataclass(frozen=True)
class SubgradientOutcome:
    """
    CN: 一次次梯度 run 的评估结果（计时之外计算）。
    EN: Evaluation outcome of one subgradient run (computed outside the timed section).
    """

    n: int
    seed: int
    noise_multiplier: int | None
    solver: str
    device: str
    step_rule: str
    learning_rate: float
    max_iterations: int
    tol: float
    iterations: int
    converged: bool
    stopped_reason: str
    final_gap: float
    best_gap: float
    wall_time_seconds: float
    peak_rss_mb: float
    terminal_G: float
    terminal_R: float
    terminal_G_raw: float
    terminal_R_raw: float
    best_G: float
    best_R: float
    uniform_G: float
    uniform_R: float
    theta_star_G: float
    theta_star_R: float
    is_optimal_terminal: bool
    threads: dict[str, int]


def _build_checkpoint(
    instance: BenchmarkInstance,
    *,
    solver_label: str,
    device: str,
    step_rule: str,
    learning_rate: float,
    max_iterations: int,
    result,
    peak_rss_mb: float,
    stopped_reason: str,
) -> RunCheckpoint:
    """CN: 把次梯度轨迹转成共享 checkpoint。EN: Convert the trace into the shared checkpoint."""
    chosen = set(_checkpoint_indices(len(result.iterations), 12))
    checkpoints = tuple(
        ThetaCheckpoint(
            iteration=int(record.index),
            elapsed_seconds=float(record.wall_time),
            theta=tuple(float(value) for value in record.theta),
            epsilon=None,
            sinkhorn_marginal_error=None,
            entropic_gap=float(record.gap),
        )
        for record in result.iterations
        if int(record.index) in chosen
    )
    return RunCheckpoint(
        params=params_from_instance(instance),
        method=f"exact-ot-subgradient[{solver_label}]",
        theta_parametrisation=f"direct-constrained-projected/{step_rule}",
        optimiser="subgradient",
        weight_decay=0.0,
        learning_rate=float(learning_rate),
        epsilon_start=0.0,
        epsilon_decay=0.0,
        epsilon_floor=0.0,
        num_outer=int(len(result.iterations)),
        sinkhorn_iterations=0,
        hardware=str(device),
        threads=_threads(),
        converged=bool(result.converged),
        peak_rss_mb=float(peak_rss_mb),
        wall_time_seconds=float(result.wall_time),
        final_sinkhorn_marginal_error=float("nan"),
        epsilon_floor_reached=False,
        terminal_theta=tuple(float(value) for value in result.theta),
        checkpoints=checkpoints,
        stopped_reason=str(stopped_reason),
    )


def _worker(argv: argparse.Namespace) -> int:
    """CN: 独立子进程中运行一次并写出 checkpoint + meta。EN: Run one in a subprocess."""
    instance = build_instance(
        n=int(argv.n), k=int(argv.k), seed=int(argv.seed),
        num_itermax=int(argv.num_itermax),
        noise_multiplier=(None if argv.noise is None else int(argv.noise)),
    )
    if argv.solver == "hello":
        solver = make_hello_solver(
            instance, device=str(argv.device), max_iterations=int(argv.hello_max_iterations)
        )
    else:
        solver = make_pot_solver(instance)
    initial = project_theta(
        np.full(int(instance.n_features), 1.0 / float(instance.n_features)), instance.theta_set
    )
    if int(argv.warmup) > 0:
        solver(initial)  # untimed warmup (torch compile / library init)
    config = SubgradientConfig(
        step_rule=argv.step_rule,  # type: ignore[arg-type]
        learning_rate=float(argv.learning_rate),
        max_iterations=int(argv.max_iterations),
        tol=float(argv.tol),
    )
    started = time.perf_counter()
    result = fit_theta_subgradient(
        instance.observed_problem(), config=config, initial_theta=initial,
        exact_plan_solver=solver,
    )
    wall = float(time.perf_counter() - started)
    rss = _peak_rss_mb()
    stopped_reason = "converged" if result.converged else "max_iterations"
    checkpoint = _build_checkpoint(
        instance,
        solver_label=str(argv.solver),
        device=str(argv.device),
        step_rule=str(argv.step_rule),
        learning_rate=float(argv.learning_rate),
        max_iterations=int(argv.max_iterations),
        result=result,
        peak_rss_mb=rss,
        stopped_reason=stopped_reason,
    )
    write_run(Path(argv.out), checkpoint)

    terminal = score_theta(instance, result.theta, check_optimality=True)
    best = score_theta(instance, result.best_theta)
    references = reference_scores(instance)
    outcome = SubgradientOutcome(
        n=int(instance.n_source),
        seed=int(instance.seed),
        noise_multiplier=instance.noise_multiplier,
        solver=str(argv.solver),
        device=str(argv.device),
        step_rule=str(argv.step_rule),
        learning_rate=float(argv.learning_rate),
        max_iterations=int(argv.max_iterations),
        tol=float(argv.tol),
        iterations=int(len(result.iterations)),
        converged=bool(result.converged),
        stopped_reason=stopped_reason,
        final_gap=float(result.final_gap),
        best_gap=float(result.best_gap),
        wall_time_seconds=float(wall),
        peak_rss_mb=float(rss),
        terminal_G=float(terminal.G),
        terminal_R=float(terminal.R),
        terminal_G_raw=float(terminal.G_raw),
        terminal_R_raw=float(terminal.R_raw),
        best_G=float(best.G),
        best_R=float(best.R),
        uniform_G=float(references.uniform.G),
        uniform_R=float(references.uniform.R),
        theta_star_G=float(references.theta_star.G),
        theta_star_R=float(references.theta_star.R),
        is_optimal_terminal=bool(terminal.is_optimal),
        threads=_threads(),
    )
    if argv.meta is not None:
        Path(argv.meta).parent.mkdir(parents=True, exist_ok=True)
        Path(argv.meta).write_text(json.dumps(asdict(outcome), indent=2) + "\n", encoding="utf-8")
    print(
        f"n={outcome.n} seed={outcome.seed} noise={outcome.noise_multiplier} "
        f"solver={outcome.solver}/{outcome.device} rule={outcome.step_rule} "
        f"it={outcome.iterations} conv={outcome.converged} gap={outcome.final_gap:.2e} "
        f"G={outcome.terminal_G:.2e} R={outcome.terminal_R:.2e} "
        f"unifR={outcome.uniform_R:.2e} wall={outcome.wall_time_seconds:.2f}s "
        f"rss={outcome.peak_rss_mb:.0f}MB",
        flush=True,
    )
    return 0


def _acquire_lock(timeout_seconds: float) -> None:
    """CN: 获取 CPU 锁。EN: Acquire the CPU lock."""
    deadline = time.perf_counter() + float(timeout_seconds)
    while True:
        try:
            LOCK_PATH.mkdir()
            (LOCK_PATH / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "started": time.time()}) + "\n", encoding="utf-8"
            )
            return
        except FileExistsError:
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"CPU lock {LOCK_PATH} still held after {timeout_seconds}s")
            time.sleep(5.0)


def _release_lock() -> None:
    """CN: 释放 CPU 锁。EN: Release the CPU lock."""
    owner = LOCK_PATH / "owner.json"
    if owner.exists():
        owner.unlink()
    if LOCK_PATH.exists():
        LOCK_PATH.rmdir()


def _sweep(argv: argparse.Namespace) -> int:
    """CN: 顺序为每个 (n, seed, noise, rule) 启动子进程。EN: Spawn one subprocess per run."""
    out_dir = Path(argv.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(PINNED_THREADS[0])
    env["MKL_NUM_THREADS"] = str(PINNED_THREADS[1])
    env["OPENBLAS_NUM_THREADS"] = str(PINNED_THREADS[2])
    _acquire_lock(float(argv.lock_timeout))
    try:
        for n in [int(value) for value in argv.sizes]:
            for noise in [None, *[int(value) for value in argv.noise]]:
                for seed in [int(value) for value in argv.seeds]:
                    for rule in [str(value) for value in argv.step_rules]:
                        tag = "clean" if noise is None else f"noise{noise}"
                        stem = f"subgrad__{argv.solver}__n{n}__{tag}__seed{seed}__{rule}"
                        command = [
                            sys.executable, str(Path(__file__).resolve()), "worker",
                            "--n", str(n), "--k", str(argv.k), "--seed", str(seed),
                            "--num-itermax", str(argv.num_itermax),
                            "--solver", str(argv.solver), "--device", str(argv.device),
                            "--step-rule", rule, "--learning-rate", str(argv.learning_rate),
                            "--max-iterations", str(argv.max_iterations), "--tol", str(argv.tol),
                            "--warmup", str(argv.warmup),
                            "--hello-max-iterations", str(argv.hello_max_iterations),
                            "--out", str(out_dir / f"{stem}.json"),
                            "--meta", str(out_dir / f"{stem}.meta.json"),
                        ]
                        if noise is not None:
                            command += ["--noise", str(noise)]
                        started = time.perf_counter()
                        outcome = "done"
                        try:
                            subprocess.run(command, env=env, check=False, timeout=float(argv.budget))
                        except subprocess.TimeoutExpired:
                            outcome = "budget"
                        print(f"[{stem}] driver={outcome} in {time.perf_counter()-started:.1f}s",
                              flush=True)
    finally:
        _release_lock()
    return 0


def _report(argv: argparse.Namespace) -> int:
    """CN: 汇总一个目录下的 meta，输出每个 (solver, rule, n, noise) 的中位数与范围。EN: Aggregate metas."""
    rows: list[dict] = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(argv.meta_dir).glob("*.meta.json"))
    ]
    if not rows:
        print("no meta files")
        return 1
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["solver"], row["step_rule"], row["n"], row["noise_multiplier"])
        groups.setdefault(key, []).append(row)
    header = (
        f"{'solver':>6} {'rule':>8} {'n':>5} {'noise':>6} {'k':>2} "
        f"{'medG':>9} {'medR':>9} {'rangeR':>17} {'medWall':>8} {'medIt':>6} {'conv':>4}"
    )
    print(header)
    for key in sorted(groups, key=lambda item: (str(item[0]), str(item[1]), int(item[2]),
                                                -1 if item[3] is None else int(item[3]))):
        solver, rule, n, noise = key
        bucket = groups[key]
        gaps = np.array([float(row["terminal_G"]) for row in bucket])
        regrets = np.array([float(row["terminal_R"]) for row in bucket])
        walls = np.array([float(row["wall_time_seconds"]) for row in bucket])
        iters = np.array([float(row["iterations"]) for row in bucket])
        conv = np.mean([bool(row["converged"]) for row in bucket])
        print(
            f"{solver:>6} {rule:>8} {int(n):>5} {str(noise):>6} {len(bucket):>2} "
            f"{np.median(gaps):>9.2e} {np.median(regrets):>9.2e} "
            f"[{regrets.min():.2e},{regrets.max():.2e}] "
            f"{np.median(walls):>8.2f} {np.median(iters):>6.0f} {conv:>4.2f}"
        )
    uniform = np.median([float(row["uniform_R"]) for row in rows])
    print(f"reference: median uniform R = {uniform:.2e}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    worker = sub.add_parser("worker")
    worker.add_argument("--n", type=int, required=True)
    worker.add_argument("--k", type=int, default=DEFAULT_K)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--noise", type=int, default=None)
    worker.add_argument("--num-itermax", type=int, default=DEFAULT_NUM_ITERMAX)
    worker.add_argument("--solver", choices=("pot", "hello"), default="pot")
    worker.add_argument("--device", type=str, default="cpu")
    worker.add_argument("--step-rule", choices=STEP_RULES, default="constant")
    worker.add_argument("--learning-rate", type=float, default=0.05)
    worker.add_argument("--max-iterations", type=int, default=200)
    worker.add_argument("--tol", type=float, default=1e-7)
    worker.add_argument("--warmup", type=int, default=0)
    worker.add_argument("--hello-max-iterations", type=int, default=100)
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--meta", type=Path, default=None)
    worker.set_defaults(func=_worker)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--out-dir", type=Path, required=True)
    sweep.add_argument("--sizes", type=int, nargs="+", required=True)
    sweep.add_argument("--seeds", type=int, nargs="+", required=True)
    sweep.add_argument("--noise", type=int, nargs="*", default=[])
    sweep.add_argument("--step-rules", nargs="+", choices=STEP_RULES, default=list(STEP_RULES))
    sweep.add_argument("--solver", choices=("pot", "hello"), default="pot")
    sweep.add_argument("--device", type=str, default="cpu")
    sweep.add_argument("--k", type=int, default=DEFAULT_K)
    sweep.add_argument("--learning-rate", type=float, default=0.05)
    sweep.add_argument("--max-iterations", type=int, default=200)
    sweep.add_argument("--tol", type=float, default=1e-7)
    sweep.add_argument("--num-itermax", type=int, default=DEFAULT_NUM_ITERMAX)
    sweep.add_argument("--warmup", type=int, default=0)
    sweep.add_argument("--hello-max-iterations", type=int, default=100)
    sweep.add_argument("--budget", type=float, default=600.0)
    sweep.add_argument("--lock-timeout", type=float, default=7200.0)
    sweep.set_defaults(func=_sweep)

    report = sub.add_parser("report")
    report.add_argument("--meta-dir", type=Path, required=True)
    report.set_defaults(func=_report)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: Entry point."""
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
