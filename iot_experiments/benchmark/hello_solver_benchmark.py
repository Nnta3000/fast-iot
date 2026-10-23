"""
CN: 次梯度法的内层精确 OT 求解器对比：POT（稠密 CPU）vs HELLO-OT（O(m+n)，CPU/GPU）。
    只测时间与最终间隙；两种内层求解器产生相同的投影次梯度（同一最优方案代价）。
EN: Compare the subgradient method's exact-OT inner solver: POT (dense CPU) vs HELLO-OT
    (O(m+n), CPU/GPU). Only time and final gap are measured; both inner solvers yield the same
    projected subgradient (same optimal plan cost).

CN: 第一次 HELLO 调用有一次性 import/CUDA 开销，脚本先热身。
EN: The first HELLO call pays a one-time import/CUDA cost; the script warms up first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import make_hello_solver  # noqa: E402
from iot.alternatives.subgradient import SubgradientConfig, fit_theta_subgradient, make_pot_solver  # noqa: E402
from iot_experiments.benchmark.instances import build_instance  # noqa: E402


@dataclass(frozen=True)
class SolverRow:
    """CN: 一次内层求解器运行的结果行。EN: One inner-solver run."""

    n: int
    k: int
    inner_solver: str
    device: str
    converged: bool
    iterations: int
    wall_time: float
    per_solve: float
    best_gap: float


def run_one(*, n: int, k: int, seed: int, solver: str, device: str,
            max_iterations: int, learning_rate: float) -> SolverRow:
    instance = build_instance(n=int(n), k=int(k), seed=int(seed))
    problem = instance.observed_problem()
    if solver == "hello":
        inner = make_hello_solver(problem, device=str(device))
    else:
        inner = make_pot_solver(problem)
    config = SubgradientConfig(
        step_rule="inv_sqrt", learning_rate=float(learning_rate),
        max_iterations=int(max_iterations), tol=1e-7,
    )
    started = time.perf_counter()
    result = fit_theta_subgradient(problem, config=config, exact_plan_solver=inner)
    wall = time.perf_counter() - started
    iterations = len(result.iterations)
    return SolverRow(
        n=int(n), k=int(k), inner_solver=str(solver), device=str(device),
        converged=bool(result.converged), iterations=int(iterations),
        wall_time=float(wall), per_solve=float(wall) / max(1, iterations),
        best_gap=float(result.best_gap),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="subgradient inner-solver benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 800, 1600])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # warmup import/CUDA (untimed)
    warm = build_instance(n=40, k=4, seed=0)
    make_hello_solver(warm.observed_problem(), device=str(args.device))(
        __import__("numpy").full(4, 0.25)
    )
    rows: list[SolverRow] = []
    for n in [int(v) for v in args.sizes]:
        for seed in [int(v) for v in args.seeds]:
            rows.append(run_one(n=n, k=int(args.k), seed=seed, solver="pot", device="cpu",
                                max_iterations=int(args.max_iterations),
                                learning_rate=float(args.learning_rate)))
            rows.append(run_one(n=n, k=int(args.k), seed=seed, solver="hello",
                                device=str(args.device),
                                max_iterations=int(args.max_iterations),
                                learning_rate=float(args.learning_rate)))
    print(f"{'n':>5} {'solver':>6} {'conv':>5} {'it':>5} {'wall_s':>8} {'s/solve':>8} {'best_gap':>10}")
    for row in rows:
        print(f"{row.n:>5} {row.inner_solver:>6} {str(row.converged):>5} {row.iterations:>5} "
              f"{row.wall_time:>8.2f} {row.per_solve:>8.4f} {row.best_gap:>10.2e}")
    if args.out is not None:
        payload = {"experiment": "hello_solver_benchmark",
                   "rows": [asdict(r) for r in rows]}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
