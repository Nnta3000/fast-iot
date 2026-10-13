"""
CN: HELLO 快速 oracle 扫描的基准：比较 NumPy 扫描（CPU）与 HELLO torch 扫描（GPU）在
    (a) 单次全边扫描和 (b) 完整约束生成循环里的耗时。两者给出逐位相同的轨迹（见
    `tests/iot_hello_scan_test.py`），所以这里只测时间。
EN: Benchmark for the HELLO fast oracle scan: compare the NumPy scan (CPU) with the HELLO torch
    scan (GPU) for (a) one full-edge scan and (b) the whole constraint-generation loop. Both give
    bit-identical trajectories (see `tests/iot_hello_scan_test.py`), so only time is measured.

CN: 第一次 HELLO 调用会付出一次性的 import / CUDA 上下文开销，脚本在计时前先热身。
EN: The first HELLO call pays a one-time import / CUDA-context cost; the script warms up first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import (  # noqa: E402
    LinearMasterSolver,
    RegularizedMasterSolver,
    mandatory_edges,
    scan_violations,
    scan_violations_fast,
    solve_inverse_ot,
)
from iot_experiments.benchmark.instances import build_instance  # noqa: E402


@dataclass(frozen=True)
class ScanRow:
    """CN: 单次全边扫描的时间。EN: Timing of one full-edge scan."""

    n: int
    k: int
    backend: str
    device: str
    seconds: float
    max_violation: float
    n_candidates: int


@dataclass(frozen=True)
class LoopRow:
    """CN: 完整循环的时间分解。EN: Timing breakdown of the whole loop."""

    n: int
    k: int
    backend: str
    device: str
    converged: bool
    iterations: int
    wall_seconds: float
    oracle_seconds: float
    master_seconds: float
    final_max_violation: float


def _best(fn, *, repeats: int) -> float:
    best = float("inf")
    for _ in range(int(repeats)):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return float(best)


def run_scan_only(*, sizes: list[int], k: int, device: str, repeats: int) -> list[ScanRow]:
    """CN: 固定 master 上的单次扫描时间。EN: One-scan timing on a fixed master."""
    rows: list[ScanRow] = []
    for n in sizes:
        instance = build_instance(n=int(n), k=int(k), seed=0)
        problem = instance.observed_problem()
        edges, _ = mandatory_edges(problem)
        master = LinearMasterSolver().solve(problem, edges, tie_break="none")
        cpu = _best(
            lambda: scan_violations(problem, master, excluded=edges, tol=1e-9, gamma=1),
            repeats=int(repeats),
        )
        gpu = _best(
            lambda: scan_violations_fast(
                problem, master, excluded=edges, tol=1e-9, gamma=1, device=device
            ),
            repeats=int(repeats),
        )
        ref = scan_violations(problem, master, excluded=edges, tol=1e-9, gamma=1)
        rows.append(ScanRow(int(n), int(k), "numpy", "cpu", cpu, ref.max_violation, ref.rows.size))
        rows.append(
            ScanRow(int(n), int(k), "hello", device, gpu, ref.max_violation, ref.rows.size)
        )
        print(f"[scan] n={n} numpy={cpu*1e3:.1f}ms hello/{device}={gpu*1e3:.1f}ms", flush=True)
    return rows


def run_loop(*, sizes: list[int], k: int, device: str, max_iterations: int) -> list[LoopRow]:
    """CN: 完整约束生成循环，分别用两种扫描后端。EN: Whole loop under both scan backends."""
    rows: list[LoopRow] = []
    for n in sizes:
        instance = build_instance(n=int(n), k=int(k), seed=0)
        for backend, backend_device in (("numpy", "cpu"), ("hello", device)):
            problem = instance.observed_problem()
            started = time.perf_counter()
            result = solve_inverse_ot(
                problem,
                solver=RegularizedMasterSolver(epsilon=1e-4),
                scan_backend=backend,
                scan_device=backend_device,
                max_iterations=int(max_iterations),
                validate=False,
            )
            wall = time.perf_counter() - started
            rows.append(
                LoopRow(
                    n=int(n), k=int(k), backend=backend, device=backend_device,
                    converged=bool(result.converged), iterations=int(result.n_iterations),
                    wall_seconds=float(wall),
                    oracle_seconds=float(sum(r.oracle_time for r in result.iterations)),
                    master_seconds=float(sum(r.master_time for r in result.iterations)),
                    final_max_violation=float(result.final_max_violation),
                )
            )
            print(
                f"[loop] n={n} {backend}/{backend_device}: conv={result.converged} "
                f"it={result.n_iterations} wall={wall:.2f}s",
                flush=True,
            )
    return rows


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HELLO fast oracle-scan benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 800, 1600, 3200])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--loop", action="store_true", help="also run the whole loop")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # one-time import / CUDA warmup, untimed
    warm = build_instance(n=60, k=4, seed=0)
    solve_inverse_ot(
        warm.observed_problem(), solver=RegularizedMasterSolver(epsilon=1e-4),
        scan_backend="hello", scan_device=str(args.device), max_iterations=3, validate=False,
    )
    scan_rows = run_scan_only(
        sizes=[int(v) for v in args.sizes], k=int(args.k), device=str(args.device),
        repeats=int(args.repeats),
    )
    loop_rows: list[LoopRow] = []
    if bool(args.loop):
        loop_rows = run_loop(
            sizes=[int(v) for v in args.sizes], k=int(args.k), device=str(args.device),
            max_iterations=int(args.max_iterations),
        )
    if args.out is not None:
        payload = {
            "experiment": "hello_scan_benchmark",
            "config": {"k": int(args.k), "device": str(args.device),
                       "repeats": int(args.repeats), "max_iterations": int(args.max_iterations)},
            "scan_rows": [asdict(r) for r in scan_rows],
            "loop_rows": [asdict(r) for r in loop_rows],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
