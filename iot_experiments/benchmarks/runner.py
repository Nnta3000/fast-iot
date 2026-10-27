"""
CN: 梯度（entropic-θ）runner 与共享基准的 worker/driver。梯度方法在 Θ 上直接最小化熵 OT 间隙
    `L_ε(θ) = <C_θ, pi_hat> - <C_θ, P_ε(θ)>`，用 AdamW（weight_decay=0）投影梯度、ε 退火到
    地板 1e-3，并在外层步之间热启动 Sinkhorn 势。
EN: Gradient (entropic-theta) runner and the shared benchmark worker/driver. The gradient
    method minimises the entropic OT gap `L_eps(theta) = <C_theta, pi_hat> - <C_theta,
    P_eps(theta)>` over Theta with projected AdamW (weight_decay=0), annealing epsilon to
    the 1e-3 floor and warm-starting the Sinkhorn potentials across outer steps.

CN: 该 runner 从不评估 G/R（POT 昂贵），只保存 theta checkpoint；评估由
    `iot_experiments/benchmarks/evaluate.py` 离线完成。每个 run 在自己的子进程中执行，峰值
    内存取真实进程 RSS（`resource.getrusage`）。
EN: The runner never evaluates G/R (POT is expensive); it only saves theta checkpoints, and
    `iot_experiments/benchmarks/evaluate.py` scores them offline. Each run executes in its own
    subprocess and peak memory is the real process RSS (`resource.getrusage`).
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import (  # noqa: E402
    RegularizedMasterSolver,
    dense_cost_matrix,
    solve_inverse_ot,
)
from iot.alternatives.entropic import SinkhornResult, sinkhorn_plan_with_potentials  # noqa: E402
from iot.alternatives.sinkhorn_theta import project_theta  # noqa: E402

from iot_experiments.benchmarks.checkpoint import (  # noqa: E402
    InstanceParams,
    RunCheckpoint,
    ThetaCheckpoint,
    write_run,
)
from iot_experiments.benchmarks.instances import (  # noqa: E402
    DEFAULT_K,
    DEFAULT_NUM_ITERMAX,
    BenchmarkInstance,
    build_instance,
)

PINNED_THREADS: tuple[int, int, int] = (8, 8, 8)
LOCK_PATH = _REPO_ROOT / "iot_experiments" / "benchmark" / ".cpu_lock"


@dataclass(frozen=True)
class GradientConfig:
    """
    CN: 冻结的超参数（在 tuning seeds 1000-1004 上选定一次）。`weight_decay=0` 使 AdamW 与
        历史 Adam 代数等价。
    EN: Frozen hyperparameters (chosen once on tuning seeds 1000-1004). `weight_decay=0`
        makes AdamW algebraically identical to the historical Adam.
    """

    learning_rate: float = 0.05
    epsilon_start_multiplier: float = 0.1
    epsilon_decay: float = 0.95
    epsilon_floor: float = 1e-3
    num_outer: int = 100
    sinkhorn_iterations: int = 500
    sinkhorn_tolerance: float = 1e-9
    beta1: float = 0.9
    beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    checkpoint_count: int = 12
    max_wall_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """CN: 转字典。EN: Convert to a dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GradientConfig":
        """CN: 从字典恢复。EN: Rebuild from a dict."""
        return cls(**{str(key): payload[key] for key in asdict(cls()).keys()})  # type: ignore[arg-type]


@dataclass(frozen=True)
class ExactConfig:
    """
    CN: exact 方法的调用配置（仅调用他方 API，不修改其实现）。默认取他方 REPORT 中收敛的
        safeguarded 设置。
    EN: Invocation config for the exact method (calls the other agent's API only; never edits
        it). Defaults follow the safeguarded configuration their REPORT found convergent.
    """

    tie_break: str = "none"
    solver_kind: str = "regularized"
    epsilon: float = 1e-4
    max_iterations: int = 300
    tol: float = 1e-9
    tau_prune: float = 1e-12
    gamma: int = 1
    prune_hysteresis: bool = True
    validate: bool = False

    def to_dict(self) -> dict[str, object]:
        """CN: 转字典。EN: Convert to a dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ExactConfig":
        """CN: 从字典恢复。EN: Rebuild from a dict."""
        return cls(**{str(key): payload[key] for key in asdict(cls()).keys()})  # type: ignore[arg-type]


def _threads() -> dict[str, int]:
    """CN: 记录固定的线程数。EN: Record the pinned thread counts."""
    return {
        "OMP_NUM_THREADS": int(os.environ.get("OMP_NUM_THREADS", PINNED_THREADS[0])),
        "MKL_NUM_THREADS": int(os.environ.get("MKL_NUM_THREADS", PINNED_THREADS[1])),
        "OPENBLAS_NUM_THREADS": int(os.environ.get("OPENBLAS_NUM_THREADS", PINNED_THREADS[2])),
    }


def _peak_rss_mb() -> float:
    """
    CN: 真实进程峰值 RSS（Linux 下 ru_maxrss 单位为 KB）。
    EN: Real process peak RSS (`ru_maxrss` is in KB on Linux).
    """
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _feature_moment(
    features: object,
    plan: np.ndarray,
    *,
    max_elements: int = 1 << 20,
) -> np.ndarray:
    """
    CN: 分块计算 `Phi^T vec(plan)`，避免在 n 很大时materialize (mn, K) 特征矩阵。
    EN: Blockwise `Phi^T vec(plan)`, avoiding materialising an (mn, K) feature matrix at
        large n.
    """
    source_size = int(getattr(features, "n_source"))
    target_size = int(getattr(features, "n_target"))
    dimension = int(getattr(features, "n_features"))
    plan_matrix = np.asarray(plan, dtype=np.float64)
    accumulator = np.zeros(dimension, dtype=np.float64)
    rows_per_block = max(1, int(max_elements) // max(1, target_size))
    for start in range(0, source_size, rows_per_block):
        stop = min(source_size, start + rows_per_block)
        rows = np.repeat(np.arange(start, stop, dtype=np.int64), target_size)
        cols = np.tile(np.arange(target_size, dtype=np.int64), stop - start)
        phi = np.asarray(features.features(rows, cols), dtype=np.float64)
        accumulator += phi.T @ plan_matrix[start:stop].reshape(-1)
    return np.ascontiguousarray(accumulator, dtype=np.float64)


def _steps_to_floor(epsilon_start: float, decay: float, floor: float) -> int:
    """
    CN: 从 epsilon_start 几何退火到 floor 所需的最小外层步数。
    EN: Minimal number of outer steps to anneal geometrically from epsilon_start to floor.
    """
    if float(epsilon_start) <= float(floor):
        return 1
    return 1 + int(np.ceil(np.log(float(floor) / float(epsilon_start)) / np.log(float(decay))))


def _checkpoint_iterations(outer: int, count: int) -> list[int]:
    """
    CN: 选择的快照迭代（几何间隔，含首末，≥`count` 个）。
    EN: Chosen checkpoint iterations (geometric spacing, includes first and last, >= count).
    """
    if outer <= int(count):
        return list(range(1, outer + 1))
    indices = sorted({int(round(float(value))) for value in np.geomspace(1, outer, int(count))})
    indices = [min(max(1, index), outer) for index in indices]
    unique = sorted(set(indices))
    if len(unique) < int(count):
        unique = sorted({int(round(value)) for value in np.linspace(1, outer, int(count))})
    return unique


def fit_theta_adamw(
    instance: BenchmarkInstance,
    config: GradientConfig,
    *,
    hardware: str = "cpu",
) -> RunCheckpoint:
    """
    CN: 运行一次梯度方法并返回可序列化的 run checkpoint（不做 G/R 评估）。
    EN: Run the gradient method once and return a serialisable run checkpoint (no G/R
        evaluation).
    """
    if config.weight_decay != 0.0:
        raise ValueError("the frozen optimiser requires weight_decay == 0")
    started = time.perf_counter()
    features = instance.features
    theta_set = instance.theta_set
    dimension = int(instance.n_features)
    theta = project_theta(
        np.full(dimension, 1.0 / float(dimension), dtype=np.float64), theta_set
    )
    observed_moment = _feature_moment(features, instance.observed_plan)
    source_mass = instance.observed_source_mass
    target_mass = instance.observed_target_mass

    initial_cost = dense_cost_matrix(features, theta)
    epsilon_start = float(config.epsilon_start_multiplier) * float(np.std(initial_cost))
    if not np.isfinite(epsilon_start) or epsilon_start <= 0.0:
        epsilon_start = 1e-1
    required_outer = _steps_to_floor(epsilon_start, config.epsilon_decay, config.epsilon_floor)
    outer = max(int(config.num_outer), int(required_outer))
    save_iterations = set(_checkpoint_iterations(outer, int(config.checkpoint_count)))

    first_moment = np.zeros(dimension, dtype=np.float64)
    second_moment = np.zeros(dimension, dtype=np.float64)
    potential_source: np.ndarray | None = None
    potential_target: np.ndarray | None = None
    checkpoints: list[ThetaCheckpoint] = []
    epsilon = float(epsilon_start)
    last_marginal = float("nan")
    last_sinkhorn_converged = False
    last_epsilon_used = float(epsilon_start)
    stopped_reason = "completed"
    completed_outer = 0

    for step in range(outer):
        if (
            float(config.max_wall_seconds) > 0.0
            and (time.perf_counter() - started) > float(config.max_wall_seconds)
        ):
            stopped_reason = "wall_budget"
            break
        cost = dense_cost_matrix(features, theta)
        sinkhorn = _sinkhorn(
            cost, source_mass, target_mass, epsilon=epsilon, config=config,
            potential_source=potential_source, potential_target=potential_target,
        )
        potential_source = sinkhorn.log_source_potential
        potential_target = sinkhorn.log_target_potential
        fitted_moment = _feature_moment(features, sinkhorn.plan)
        gradient = observed_moment - fitted_moment
        entropic_gap = float(theta @ gradient)

        first_moment = config.beta1 * first_moment + (1.0 - config.beta1) * gradient
        second_moment = config.beta2 * second_moment + (1.0 - config.beta2) * gradient * gradient
        corrected_first = first_moment / (1.0 - config.beta1 ** (step + 1))
        corrected_second = second_moment / (1.0 - config.beta2 ** (step + 1))
        # AdamW decoupled weight decay; with weight_decay=0 this is exactly Adam.
        update = corrected_first / (np.sqrt(corrected_second) + config.adam_epsilon)
        update = update + config.weight_decay * theta
        theta = project_theta(theta - config.learning_rate * update, theta_set)

        last_marginal = float(sinkhorn.marginal_error)
        last_sinkhorn_converged = bool(sinkhorn.converged)
        last_epsilon_used = float(epsilon)
        completed_outer = step + 1
        if (step + 1) in save_iterations:
            checkpoints.append(
                ThetaCheckpoint(
                    iteration=step + 1,
                    elapsed_seconds=float(time.perf_counter() - started),
                    theta=tuple(float(value) for value in theta),
                    epsilon=float(epsilon),
                    sinkhorn_marginal_error=float(sinkhorn.marginal_error),
                    entropic_gap=entropic_gap,
                )
            )
        epsilon = max(float(epsilon) * float(config.epsilon_decay), float(config.epsilon_floor))

    # Final Sinkhorn pass at the floor from the warm-started potentials (only if the run
    # completed its anneal; a wall-budget stop keeps the last measured marginal error).
    if stopped_reason == "completed":
        final_cost = dense_cost_matrix(features, theta)
        final_sinkhorn = _sinkhorn(
            final_cost, source_mass, target_mass, epsilon=float(config.epsilon_floor),
            config=config, potential_source=potential_source, potential_target=potential_target,
        )
        final_marginal = float(final_sinkhorn.marginal_error)
        final_converged = bool(final_sinkhorn.converged)
    else:
        final_marginal = float(last_marginal)
        final_converged = bool(last_sinkhorn_converged)
    wall_time = float(time.perf_counter() - started)
    if stopped_reason == "completed" and (not checkpoints or checkpoints[-1].iteration != outer):
        checkpoints.append(
            ThetaCheckpoint(
                iteration=int(outer),
                elapsed_seconds=wall_time,
                theta=tuple(float(value) for value in theta),
                epsilon=float(config.epsilon_floor),
                sinkhorn_marginal_error=float(final_marginal),
                entropic_gap=float(
                    theta @ (observed_moment - _feature_moment(features, final_sinkhorn.plan))
                ),
            )
        )
    floor_reached = bool(
        stopped_reason == "completed"
        and last_epsilon_used <= float(config.epsilon_floor) * (1.0 + 1e-12)
    )

    return RunCheckpoint(
        params=params_from_instance(instance),
        method="gradient-entropic-theta",
        theta_parametrisation="direct-constrained-projected",
        optimiser="AdamW",
        weight_decay=float(config.weight_decay),
        learning_rate=float(config.learning_rate),
        epsilon_start=float(epsilon_start),
        epsilon_decay=float(config.epsilon_decay),
        epsilon_floor=float(config.epsilon_floor),
        num_outer=int(outer),
        sinkhorn_iterations=int(config.sinkhorn_iterations),
        hardware=str(hardware),
        threads=_threads(),
        converged=bool(floor_reached and final_converged),
        peak_rss_mb=_peak_rss_mb(),
        wall_time_seconds=wall_time,
        final_sinkhorn_marginal_error=float(final_marginal),
        epsilon_floor_reached=floor_reached,
        stopped_reason=str(stopped_reason),
        terminal_theta=tuple(float(value) for value in theta),
        checkpoints=tuple(checkpoints),
    )


def _sinkhorn(
    cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    *,
    epsilon: float,
    config: GradientConfig,
    potential_source: np.ndarray | None,
    potential_target: np.ndarray | None,
) -> SinkhornResult:
    """CN: 带热启动的 Sinkhorn 封装。EN: Warm-started Sinkhorn wrapper."""
    return sinkhorn_plan_with_potentials(
        cost,
        source_mass,
        target_mass,
        epsilon=float(epsilon),
        max_iterations=int(config.sinkhorn_iterations),
        tolerance=float(config.sinkhorn_tolerance),
        initial_log_source_potential=potential_source,
        initial_log_target_potential=potential_target,
    )


def run_exact(instance: BenchmarkInstance, config: ExactConfig) -> RunCheckpoint:
    """
    CN: 调用他方 exact 方法一次，转成同一 checkpoint schema（单快照：终态 theta）。
    EN: Call the other agent's exact method once and convert to the same checkpoint schema
        (a single snapshot: the terminal theta).
    """
    solver = (
        RegularizedMasterSolver(epsilon=float(config.epsilon), random_seed=int(instance.seed))
        if config.solver_kind == "regularized"
        else None
    )
    started = time.perf_counter()
    result = solve_inverse_ot(
        instance.observed_problem(),
        tie_break=config.tie_break,  # type: ignore[arg-type]
        solver=solver,
        max_iterations=int(config.max_iterations),
        random_seed=int(instance.seed),
        tol=float(config.tol),
        tau_prune=float(config.tau_prune),
        gamma=int(config.gamma),
        validate=bool(config.validate),
        prune_hysteresis=bool(config.prune_hysteresis),
        prune=True,
    )
    wall_time = float(time.perf_counter() - started)
    checkpoint = ThetaCheckpoint(
        iteration=int(result.n_iterations),
        elapsed_seconds=wall_time,
        theta=tuple(float(value) for value in result.theta),
        epsilon=None,
        sinkhorn_marginal_error=None,
        entropic_gap=None,
    )
    return RunCheckpoint(
        params=params_from_instance(instance),
        method="exact-constraint-generation",
        theta_parametrisation="direct-constrained",
        optimiser="",
        weight_decay=0.0,
        learning_rate=0.0,
        epsilon_start=0.0,
        epsilon_decay=0.0,
        epsilon_floor=0.0,
        num_outer=int(result.n_iterations),
        sinkhorn_iterations=0,
        hardware="cpu",
        threads=_threads(),
        converged=bool(result.converged),
        peak_rss_mb=_peak_rss_mb(),
        wall_time_seconds=wall_time,
        final_sinkhorn_marginal_error=float("nan"),
        epsilon_floor_reached=False,
        terminal_theta=tuple(float(value) for value in result.theta),
        checkpoints=(checkpoint,),
    )


def params_from_instance(instance: BenchmarkInstance) -> InstanceParams:
    """
    CN: 从实例构造 checkpoint 的实例参数（供离线重建同一实例）。
    EN: Build the checkpoint's instance params from an instance (for offline rebuild).
    """
    return InstanceParams(
        instance_id=instance.instance_id,
        seed=int(instance.seed),
        n=int(instance.source_indices.size),
        n_features=int(instance.n_features),
        noise_multiplier=instance.noise_multiplier,
        noise_pairs=instance.noise_pairs,
        mixing_strength=float(instance.mixing_strength),
        num_itermax=int(instance.num_itermax),
        n_source=int(instance.n_source),
        n_target=int(instance.n_target),
        zero_rows_dropped=int(instance.zero_rows_dropped),
        zero_cols_dropped=int(instance.zero_cols_dropped),
        source_mass_total=float(instance.observed_source_mass.sum()),
        target_mass_total=float(instance.observed_target_mass.sum()),
    )


def _load_config(path: Path | None, kind: str) -> GradientConfig | ExactConfig:
    """CN: 读取冻结配置（缺省使用默认值）。EN: Load the frozen config (defaults if absent)."""
    if path is None:
        return GradientConfig() if kind == "gradient" else ExactConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if kind == "gradient":
        return GradientConfig.from_dict(payload)
    return ExactConfig.from_dict(payload)


def _worker(argv: argparse.Namespace) -> int:
    """CN: 在独立子进程中运行一个 run 并写出 checkpoint。EN: Run one run in its own subprocess."""
    if argv.kind == "gradient":
        config: GradientConfig | ExactConfig = _load_config(argv.config, "gradient")
        instance = build_instance(
            n=int(argv.n), k=int(argv.k), seed=int(argv.seed),
            mixing_strength=float(argv.mixing), num_itermax=int(argv.num_itermax),
            noise_multiplier=(None if argv.noise is None else int(argv.noise)),
        )
        assert isinstance(config, GradientConfig)
        run = fit_theta_adamw(instance, config, hardware=str(argv.hardware))
    else:
        config = _load_config(argv.config, "exact")
        instance = build_instance(
            n=int(argv.n), k=int(argv.k), seed=int(argv.seed),
            mixing_strength=float(argv.mixing), num_itermax=int(argv.num_itermax),
            noise_multiplier=(None if argv.noise is None else int(argv.noise)),
        )
        assert isinstance(config, ExactConfig)
        run = run_exact(instance, config)
    write_run(Path(argv.out), run)
    print(
        f"wrote {argv.out} method={run.method} n={run.params.n_source} seed={run.params.seed} "
        f"noise={run.params.noise_multiplier} conv={run.converged} "
        f"rss={run.peak_rss_mb:.1f}MB wall={run.wall_time_seconds:.2f}s",
        flush=True,
    )
    return 0


def _acquire_lock(timeout_seconds: float) -> None:
    """CN: 原子获取 CPU 锁（目录），必要时等待。EN: Atomically acquire the CPU lock (directory)."""
    deadline = time.perf_counter() + float(timeout_seconds)
    while True:
        try:
            LOCK_PATH.mkdir()
            (LOCK_PATH / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "started": time.time()}) + "\n",
                encoding="utf-8",
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
    """CN: 顺序为每个 (n, seed, noise) 启动一个子进程。EN: Spawn one subprocess per run."""
    out_dir = Path(argv.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(PINNED_THREADS[0])
    env["MKL_NUM_THREADS"] = str(PINNED_THREADS[1])
    env["OPENBLAS_NUM_THREADS"] = str(PINNED_THREADS[2])
    configs = {
        "gradient": argv.config if argv.config is not None else None,
        "exact": argv.exact_config,
    }
    _acquire_lock(float(argv.lock_timeout))
    try:
        for method in argv.methods:
            for n in [int(value) for value in argv.sizes]:
                for seed in [int(value) for value in argv.seeds]:
                    for noise in [None, *[int(value) for value in argv.noise]]:
                        if noise is None and not argv.include_clean:
                            continue
                        noise_tag = "clean" if noise is None else f"noise{noise}"
                        config_path = configs[method]
                        command = [
                            sys.executable, str(Path(__file__).resolve()),
                            "worker",
                            "--kind", method,
                            "--n", str(n), "--k", str(argv.k), "--seed", str(seed),
                            "--mixing", str(argv.mixing), "--num-itermax", str(argv.num_itermax),
                            "--hardware", str(argv.hardware),
                            "--out", str(out_dir / f"{method}__n{n}__{noise_tag}__seed{seed}.json"),
                        ]
                        if config_path is not None:
                            command += ["--config", str(config_path)]
                        if noise is not None:
                            command += ["--noise", str(noise)]
                        completed = subprocess.run(command, env=env, check=False)
                        if completed.returncode != 0:
                            print(f"FAILED: {' '.join(command)}", file=sys.stderr, flush=True)
    finally:
        _release_lock()
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    worker = sub.add_parser("worker")
    worker.add_argument("--kind", choices=("gradient", "exact"), required=True)
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--n", type=int, required=True)
    worker.add_argument("--k", type=int, default=DEFAULT_K)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--noise", type=int, default=None)
    worker.add_argument("--mixing", type=float, default=0.0)
    worker.add_argument("--num-itermax", type=int, default=DEFAULT_NUM_ITERMAX)
    worker.add_argument("--config", type=Path, default=None)
    worker.add_argument("--hardware", type=str, default="cpu")
    worker.set_defaults(func=_worker)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--out-dir", type=Path, required=True)
    sweep.add_argument("--sizes", type=int, nargs="+", required=True)
    sweep.add_argument("--seeds", type=int, nargs="+", required=True)
    sweep.add_argument("--noise", type=int, nargs="*", default=[])
    sweep.add_argument("--include-clean", action="store_true")
    sweep.add_argument("--methods", nargs="+", choices=("gradient", "exact"), default=["gradient"])
    sweep.add_argument("--config", type=Path, default=None)
    sweep.add_argument("--exact-config", type=Path, default=None)
    sweep.add_argument("--k", type=int, default=DEFAULT_K)
    sweep.add_argument("--mixing", type=float, default=0.0)
    sweep.add_argument("--num-itermax", type=int, default=DEFAULT_NUM_ITERMAX)
    sweep.add_argument("--hardware", type=str, default="cpu")
    sweep.add_argument("--lock-timeout", type=float, default=3600.0)
    sweep.set_defaults(func=_sweep)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: Entry point."""
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
