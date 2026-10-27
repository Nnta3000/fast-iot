"""
CN: 切割平面（cutting-plane）逆 OT 的小基准：对同一批共享基准实例（`benchmark.instances`）比较
    三种方法到 ``L0 <= tol`` 的成本——切割平面、精确约束生成、精确 OT 次梯度。切割平面用
    POT 作分离 oracle（每个 ``(theta_k, P_k)`` 给出一条割 ``g_k = Phi^T(pi_hat - P_k)``）；
    这里只测「达到可容许集合需要多少次 OT 求解 / 多少墙钟」，不重跑 G/R（那由
    ``benchmark/evaluate.py`` 离线完成）。

    EN: Small benchmark for cutting-plane inverse OT: on the shared benchmark instances
    (`benchmark.instances`), compare the cost to reach ``L0 <= tol`` for three methods —
    cutting plane, exact constraint generation, and exact-OT subgradient. The cutting plane
    uses POT as the separation oracle (each ``(theta_k, P_k)`` gives one cut
    ``g_k = Phi^T(pi_hat - P_k)``); only the number of OT solves / wall time to reach the
    admissible set is measured here, not G/R (that is the offline evaluator's job).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import (  # noqa: E402
    CuttingPlaneConfig,
    RegularizedMasterSolver,
    is_optimal_plan,
    solve_cutting_plane,
    solve_inverse_ot,
)
from iot.alternatives.subgradient import SubgradientConfig, fit_theta_subgradient  # noqa: E402
from iot_experiments.benchmarks.instances import (  # noqa: E402
    DEFAULT_NUM_ITERMAX,
    BenchmarkInstance,
    build_instance,
)

METHODS: tuple[str, ...] = ("cutting-plane", "exact", "subgradient")
VALIDATE_MAX_ELEMENTS: int = 50_000_000


@dataclass(frozen=True)
class MethodRow:
    """
    CN: 一次方法运行的一行结果。``oracle_calls`` 是内层精确 OT 求解次数；``iterations`` 是
        外层次数（对切割平面与 exact 相同，对次梯度是步数）。
    EN: One method run. ``oracle_calls`` is the number of inner exact-OT solves;
        ``iterations`` is the outer-iteration count (same as oracle_calls for the cutting
        plane and exact, the number of steps for subgradient).
    """

    n: int
    k: int
    seed: int
    method: str
    config_label: str
    converged: bool
    oracle_calls: int
    iterations: int
    final_gap: float
    wall_time: float
    per_oracle_time: float
    is_optimal: bool | None


def _validate(instance: BenchmarkInstance, theta: np.ndarray) -> bool | None:
    """CN: 规模允许时调用权威检查 ``is_optimal_plan``；否则返回 None。EN: Authoritative check when affordable."""
    problem = instance.observed_problem()
    if int(problem.n_source) * int(problem.n_target) > VALIDATE_MAX_ELEMENTS:
        return None
    return bool(is_optimal_plan(problem, theta).is_optimal)


def run_cutting_plane(
    instance: BenchmarkInstance,
    *,
    master_mode: str,
    normalize_cuts: bool,
    max_iterations: int,
    tol: float,
) -> MethodRow:
    """CN: 用 POT oracle 运行经典割平面。EN: Run classic cutting planes with the POT oracle."""
    problem = instance.observed_problem()
    config = CuttingPlaneConfig(
        master_mode=master_mode,  # type: ignore[arg-type]
        max_iterations=int(max_iterations),
        tol=float(tol),
        normalize_cuts=bool(normalize_cuts),
    )
    started = time.perf_counter()
    result = solve_cutting_plane(problem, config=config)
    wall = time.perf_counter() - started
    return MethodRow(
        n=int(instance.n_source),
        k=int(instance.n_features),
        seed=int(instance.seed),
        method="cutting-plane",
        config_label=f"master={master_mode},normalize={normalize_cuts}",
        converged=bool(result.converged),
        oracle_calls=int(result.n_oracle_calls),
        iterations=int(result.n_oracle_calls),
        final_gap=float(result.final_gap),
        wall_time=float(wall),
        per_oracle_time=float(wall) / max(1, int(result.n_oracle_calls)),
        is_optimal=_validate(instance, result.theta),
    )


def run_exact(instance: BenchmarkInstance, *, max_iterations: int) -> MethodRow:
    """CN: 运行随仓库发布的精确约束生成配置（正则化 master，迟滞剪枝）。EN: Shipped exact config."""
    problem = instance.observed_problem()
    started = time.perf_counter()
    result = solve_inverse_ot(
        problem,
        solver=RegularizedMasterSolver(epsilon=1e-4),
        max_iterations=int(max_iterations),
    )
    wall = time.perf_counter() - started
    return MethodRow(
        n=int(instance.n_source),
        k=int(instance.n_features),
        seed=int(instance.seed),
        method="exact",
        config_label="regularized-eps1e-4",
        converged=bool(result.converged),
        oracle_calls=int(result.n_iterations),
        iterations=int(result.n_iterations),
        final_gap=float(result.final_max_violation),
        wall_time=float(wall),
        per_oracle_time=float(wall) / max(1, int(result.n_iterations)),
        is_optimal=(bool(result.is_optimal) if result.is_optimal is not None else None),
    )


def run_subgradient(
    instance: BenchmarkInstance,
    *,
    step_rule: str,
    learning_rate: float,
    max_iterations: int,
    tol: float,
) -> MethodRow:
    """CN: 运行投影次梯度（POT oracle）。EN: Run projected subgradient (POT oracle)."""
    problem = instance.observed_problem()
    config = SubgradientConfig(
        step_rule=step_rule,  # type: ignore[arg-type]
        learning_rate=float(learning_rate),
        max_iterations=int(max_iterations),
        tol=float(tol),
    )
    started = time.perf_counter()
    result = fit_theta_subgradient(problem, config=config)
    wall = time.perf_counter() - started
    iterations = int(len(result.iterations))
    return MethodRow(
        n=int(instance.n_source),
        k=int(instance.n_features),
        seed=int(instance.seed),
        method="subgradient",
        config_label=f"rule={step_rule},lr={learning_rate}",
        converged=bool(result.converged),
        oracle_calls=iterations,
        iterations=iterations,
        final_gap=float(result.final_gap),
        wall_time=float(wall),
        per_oracle_time=float(wall) / max(1, iterations),
        is_optimal=_validate(instance, result.theta),
    )


def _print_rows(rows: list[MethodRow]) -> None:
    header = (
        f"{'method':>16} {'n':>5} {'K':>3} {'seed':>4} {'conv':>5} {'calls':>6} "
        f"{'gap':>10} {'wall_s':>8} {'s/solve':>8} {'is_opt':>6} config"
    )
    print(header)
    for row in rows:
        print(
            f"{row.method:>16} {row.n:>5} {row.k:>3} {row.seed:>4} {str(row.converged):>5} "
            f"{row.oracle_calls:>6} {row.final_gap:>10.2e} {row.wall_time:>8.2f} "
            f"{row.per_oracle_time:>8.4f} {str(row.is_optimal):>6} {row.config_label}"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cutting-plane inverse-OT benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 800])
    parser.add_argument("--features", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["cutting-plane"])
    parser.add_argument("--master-mode", choices=("max_margin", "kelley"), default="max_margin")
    parser.add_argument("--normalize-cuts", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--subgradient-step-rule", choices=("constant", "inv_sqrt", "polyak"), default="polyak")
    parser.add_argument("--subgradient-learning-rate", type=float, default=1.0)
    parser.add_argument("--subgradient-max-iterations", type=int, default=2000)
    parser.add_argument("--num-itermax", type=int, default=DEFAULT_NUM_ITERMAX)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows: list[MethodRow] = []
    for n in [int(value) for value in args.sizes]:
        for k in [int(value) for value in args.features]:
            for seed in [int(value) for value in args.seeds]:
                instance = build_instance(
                    n=n, k=k, seed=seed, num_itermax=int(args.num_itermax)
                )
                for method in [str(value) for value in args.methods]:
                    if method == "cutting-plane":
                        rows.append(
                            run_cutting_plane(
                                instance,
                                master_mode=str(args.master_mode),
                                normalize_cuts=bool(int(args.normalize_cuts)),
                                max_iterations=int(args.max_iterations),
                                tol=float(args.tol),
                            )
                        )
                    elif method == "exact":
                        rows.append(run_exact(instance, max_iterations=int(args.max_iterations)))
                    else:
                        rows.append(
                            run_subgradient(
                                instance,
                                step_rule=str(args.subgradient_step_rule),
                                learning_rate=float(args.subgradient_learning_rate),
                                max_iterations=int(args.subgradient_max_iterations),
                                tol=float(args.tol),
                            )
                        )
                    print(
                        f"[done] n={n} K={k} seed={seed} method={method}", flush=True
                    )
    _print_rows(rows)
    payload = {
        "experiment": "cutting_plane_benchmark",
        "config": {
            "master_mode": str(args.master_mode),
            "normalize_cuts": bool(int(args.normalize_cuts)),
            "max_iterations": int(args.max_iterations),
            "tol": float(args.tol),
            "subgradient_step_rule": str(args.subgradient_step_rule),
            "subgradient_learning_rate": float(args.subgradient_learning_rate),
            "subgradient_max_iterations": int(args.subgradient_max_iterations),
            "methods": [str(value) for value in args.methods],
        },
        "rows": [asdict(row) for row in rows],
    }
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    cutting = [row for row in rows if row.method == "cutting-plane"]
    return 0 if all(row.converged for row in cutting) else 1


if __name__ == "__main__":
    raise SystemExit(main())
