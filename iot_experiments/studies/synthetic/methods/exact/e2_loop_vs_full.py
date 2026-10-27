"""
CN: e2_loop_vs_full —— 比较约束生成循环与全边 master 的结果，并记录 |F| 随迭代变化。
EN: e2_loop_vs_full -- compare the constraint-generation loop against the full-edge
    master and record |F| over iterations.

CN: 期望：两者 delta 一致，且 |F| 保持在 m + n 附近，远小于 m * n。
EN: Expectation: the two agree on delta and |F| stays near m + n, far below m * n.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from iot import (
    EdgeSet,
    InverseOTProblem,
    LinearMasterSolver,
    is_optimal_plan,
    make_diagonal_grounded_instance,
    solve_inverse_ot,
)


@dataclass(frozen=True)
class E2Row:
    """
    CN: e2 在单个规模上的对比结果。
    EN: e2 comparison result at a single problem size.
    """

    n_source: int
    n_target: int
    n_features: int
    delta_loop: float
    delta_full: float
    delta_difference: float
    theta_distance: float
    dual_distance: float
    support_loop: int
    support_full: int
    iterations: int
    support_trace: list[int]
    converged: bool
    is_optimal: bool


def _all_edges(problem: InverseOTProblem) -> EdgeSet:
    m, n = int(problem.n_source), int(problem.n_target)
    rows = np.repeat(np.arange(m, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), m)
    return EdgeSet.from_arrays(rows, cols, n_target=n)


def run_e2(*, seed: int, sizes: list[int], n_features: int, mixing_strength: float) -> list[E2Row]:
    """
    CN: 对每个规模运行循环与全边 master 并比较。
    EN: For each size, run the loop and the full-edge master and compare.
    """
    rows: list[E2Row] = []
    for size in sizes:
        rng = np.random.default_rng(int(seed) + int(size))
        instance = make_diagonal_grounded_instance(
            n_source=int(size),
            n_target=int(size),
            n_features=int(n_features),
            rng=rng,
            mixing_strength=float(mixing_strength),
        )
        loop = solve_inverse_ot(instance.problem, tie_break="lex", max_iterations=200)
        full = LinearMasterSolver().solve(instance.problem, _all_edges(instance.problem), tie_break="lex")
        loop_dual = np.concatenate([loop.source_dual, loop.target_dual])
        full_dual = np.concatenate([full.source_dual, full.target_dual])
        rows.append(
            E2Row(
                n_source=int(instance.problem.n_source),
                n_target=int(instance.problem.n_target),
                n_features=int(instance.problem.n_features),
                delta_loop=float(loop.delta),
                delta_full=float(full.delta),
                delta_difference=abs(float(loop.delta) - float(full.delta)),
                theta_distance=float(np.linalg.norm(loop.theta - full.theta)),
                dual_distance=float(np.linalg.norm(loop_dual - full_dual)),
                support_loop=int(loop.support_size),
                support_full=int(instance.problem.n_source) * int(instance.problem.n_target),
                iterations=int(loop.n_iterations),
                support_trace=[int(record.support_size) for record in loop.iterations],
                converged=bool(loop.converged),
                is_optimal=bool(loop.is_optimal) and bool(is_optimal_plan(instance.problem, full.theta).is_optimal),
            )
        )
    return rows


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="e2: loop vs full-master inverse OT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12, 16])
    parser.add_argument("--features", type=int, default=3, help="number of theta coordinates")
    parser.add_argument("--mixing", type=float, default=0.8, help="coordinate correlation strength")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = run_e2(
        seed=int(args.seed),
        sizes=[int(size) for size in args.sizes],
        n_features=int(args.features),
        mixing_strength=float(args.mixing),
    )
    print(
        f"{'m=n':>5} {'delta_loop':>11} {'delta_full':>11} {'|ddelta|':>9} "
        f"{'|dtheta|':>9} {'|ddual|':>9} {'|F|':>6} {'iters':>6} {'optimal':>8}"
    )
    for row in rows:
        print(
            f"{row.n_source:>5} {row.delta_loop:>11.6f} {row.delta_full:>11.6f} "
            f"{row.delta_difference:>9.2e} {row.theta_distance:>9.2e} {row.dual_distance:>9.2e} "
            f"{row.support_loop:>6} {row.iterations:>6} {str(row.is_optimal):>8}"
        )
    payload = {"experiment": "e2_loop_vs_full", "rows": [asdict(row) for row in rows]}
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row.is_optimal and row.delta_difference < 1e-5 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
