"""
CN: 比较两种 master 后端在约束生成循环里的表现：默认 HiGHS（`RegularizedMasterSolver`）与
    CuPDLPx（`CuPdlpxMasterSolver`，需要 native 扩展）。度量 delta、迭代数、峰值 |F|、每解时间、
    墙钟与 `is_optimal_plan`；CuPDLPx 是一阶求解器，精度由容差控制，正确性只看独立证书。
EN: Compare the two master backends inside the constraint-generation loop: the default HiGHS
    (`RegularizedMasterSolver`) and CuPDLPx (`CuPdlpxMasterSolver`, needs the native extension).
    Report delta, iterations, peak |F|, per-solve time, wall time and `is_optimal_plan`; CuPDLPx is
    first-order, so its accuracy is tolerance-bound and correctness is read off the independent
    certificate.

CN: 在官方 native 环境（torch 2.7.1+cu118）或 CUDA 13.4 兼容构建里运行；扩展缺失时脚本以
    清晰提示退出。CuPDLPx 的 dual/slack 精度不足以支撑出厂循环的 `tol=1e-9` 证书与迟滞剪枝，
    因此这里同时报告 `prune=True`（matched `tau_prune`）与 `prune=False` 两种配置。
EN: Run inside the documented native environment (torch 2.7.1+cu118) or a CUDA 13.4-compatible
    build; a clear message is printed when the extension is missing. CuPDLPx's dual/slack accuracy
    cannot support the shipped loop's `tol=1e-9` certificate or hysteresis pruning, so this script
    reports both `prune=True` (matched `tau_prune`) and `prune=False`.
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

from iot import (  # noqa: E402
    CuPdlpxMasterSolver,
    InverseOTError,
    RegularizedMasterSolver,
    is_optimal_plan,
    solve_inverse_ot,
)
from iot_experiments.benchmarks.instances import build_instance  # noqa: E402


@dataclass(frozen=True)
class MasterRow:
    """CN: 一个 master 后端的结果行。EN: One master-backend result row."""

    master: str
    n: int
    k: int
    seed: int
    converged: bool
    delta: float
    iterations: int
    max_support: int
    wall_time: float
    per_solve: float
    final_max_violation: float
    is_optimal: bool | None


def run_master(
    label: str,
    solver,
    *,
    n: int,
    k: int,
    seed: int,
    max_iterations: int,
    prune: bool,
    tau_prune: float,
) -> MasterRow:
    """CN: 用给定 master 跑一次完整循环。EN: Run the whole loop with the given master."""
    instance = build_instance(n=int(n), k=int(k), seed=int(seed))
    problem = instance.observed_problem()
    started = time.perf_counter()
    result = solve_inverse_ot(
        problem,
        solver=solver,
        tie_break="none",
        max_iterations=int(max_iterations),
        validate=False,
        prune=bool(prune),
        tau_prune=float(tau_prune),
    )
    wall = time.perf_counter() - started
    iterations = int(result.n_iterations)
    max_support = max((record.support_size for record in result.iterations), default=0)
    check = is_optimal_plan(problem, result.theta)
    return MasterRow(
        master=label,
        n=int(n),
        k=int(k),
        seed=int(seed),
        converged=bool(result.converged),
        delta=float(result.delta),
        iterations=iterations,
        max_support=int(max_support),
        wall_time=float(wall),
        per_solve=float(wall) / max(1, iterations),
        final_max_violation=float(result.final_max_violation),
        is_optimal=bool(check.is_optimal),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CuPDLPx master vs HiGHS master")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 800])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument(
        "--tau-prune",
        type=float,
        default=1e-3,
        help="slack threshold for the matched-pruning arms (default 1e-3 ~ PDHG residual)",
    )
    parser.add_argument("--skip-cupdlpx", action="store_true", help="HiGHS arms only")
    parser.add_argument(
        "--cupdlpx-modes",
        nargs="+",
        choices=("matched", "prune-off"),
        default=["matched", "prune-off"],
        help="which CuPDLPx pruning configurations to run",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def _cupdlpx_arms(args: argparse.Namespace, n: int, seed: int) -> list[MasterRow]:
    """CN: CuPDLPx 的 matched-pruning 与 prune-off 两种配置。EN: CuPDLPx matched-prune and prune-off."""
    rows: list[MasterRow] = []
    modes = {str(value) for value in args.cupdlpx_modes}
    configurations: list[tuple[str, bool]] = []
    if "matched" in modes:
        configurations.append(("cupdlpx-matched", True))
    if "prune-off" in modes:
        configurations.append(("cupdlpx-prune-off", False))
    for label, prune in configurations:
        try:
            rows.append(
                run_master(
                    label,
                    CuPdlpxMasterSolver(),
                    n=n,
                    k=int(args.k),
                    seed=seed,
                    max_iterations=int(args.max_iterations),
                    prune=prune,
                    tau_prune=float(args.tau_prune),
                )
            )
        except InverseOTError as error:
            print(f"[cupdlpx] n={n} seed={seed} {label}: {error}", file=sys.stderr)
            print(
                "[cupdlpx] build the native extension first; "
                "see iot_experiments/reports/components/CUPDLPX_MASTER_REPORT.md.",
                file=sys.stderr,
            )
            raise
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows: list[MasterRow] = []
    for n in [int(value) for value in args.sizes]:
        for seed in [int(value) for value in args.seeds]:
            rows.append(
                run_master(
                    "highs-shipped",
                    RegularizedMasterSolver(epsilon=1e-4),
                    n=n,
                    k=int(args.k),
                    seed=seed,
                    max_iterations=int(args.max_iterations),
                    prune=True,
                    tau_prune=1e-12,
                )
            )
            rows.append(
                run_master(
                    "highs-matched",
                    RegularizedMasterSolver(epsilon=1e-4),
                    n=n,
                    k=int(args.k),
                    seed=seed,
                    max_iterations=int(args.max_iterations),
                    prune=True,
                    tau_prune=float(args.tau_prune),
                )
            )
            if not args.skip_cupdlpx:
                try:
                    rows.extend(_cupdlpx_arms(args, n, seed))
                except InverseOTError:
                    return 2
    header = (
        f"{'master':>18} {'n':>5} {'K':>3} {'seed':>4} {'conv':>5} {'delta':>11} "
        f"{'it':>4} {'max|F|':>7} {'wall':>8} {'s/solve':>8} {'maxviol':>9} {'opt':>5}"
    )
    print(header)
    for row in rows:
        print(
            f"{row.master:>18} {row.n:>5} {row.k:>3} {row.seed:>4} {str(row.converged):>5} "
            f"{row.delta:>11.3e} {row.iterations:>4} {row.max_support:>7} {row.wall_time:>8.2f} "
            f"{row.per_solve:>8.3f} {row.final_max_violation:>9.2e} {str(row.is_optimal):>5}"
        )
    if args.out is not None:
        payload = {
            "experiment": "cupdlpx_master_benchmark",
            "config": {
                "tau_prune": float(args.tau_prune),
                "max_iterations": int(args.max_iterations),
            },
            "rows": [asdict(row) for row in rows],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
