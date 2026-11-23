"""
CN: e1_full_master —— 在全部边上求 master 的精确解，报告 delta*、theta 可辨识范围与
    最大间隔 theta 到 theta_star 的距离（随 m = n 增大，跨多个种子平均）。
EN: e1_full_master -- solve the master exactly on all edges and report delta*, the theta
    identifiability range, and the distance from the max-margin theta to theta_star as
    m = n grows, averaged over seeds.

CN: 数据默认使用相关坐标（mixing_strength>0），否则乘积分布下 theta 对任意值都最优，
    宽度不会随 n 收缩。
EN: Data uses correlated coordinates by default (mixing_strength>0); under product
    marginals theta is optimal for every value and the widths never shrink with n.
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
    optimal_theta_ranges,
)


@dataclass(frozen=True)
class E1Row:
    """
    CN: e1 在单个规模上跨多个种子的平均结果。
    EN: e1 result at a single size, averaged over seeds.
    """

    n_source: int
    n_target: int
    n_features: int
    repeats: int
    mean_delta: float
    std_delta: float
    mean_range_width_max: float
    std_range_width_max: float
    mean_distance_to_theta_star: float
    std_distance_to_theta_star: float
    all_optimal: bool


def _all_edges(problem: InverseOTProblem) -> EdgeSet:
    m, n = int(problem.n_source), int(problem.n_target)
    rows = np.repeat(np.arange(m, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), m)
    return EdgeSet.from_arrays(rows, cols, n_target=n)


def run_e1(
    *,
    seed: int,
    sizes: list[int],
    n_features: int,
    mixing_strength: float,
    repeats: int,
) -> list[E1Row]:
    """
    CN: 对每个规模重复多个种子，求解全边 master 与 theta 范围并取平均。
    EN: For each size, repeat over seeds, solve the full-edge master and theta range, and
        average.
    """
    rows: list[E1Row] = []
    master = LinearMasterSolver()
    for size in sizes:
        deltas: list[float] = []
        widths: list[float] = []
        distances: list[float] = []
        optimal: list[bool] = []
        for repeat in range(int(repeats)):
            rng = np.random.default_rng(int(seed) + 1000 * int(repeat) + int(size))
            instance = make_diagonal_grounded_instance(
                n_source=int(size),
                n_target=int(size),
                n_features=int(n_features),
                rng=rng,
                mixing_strength=float(mixing_strength),
            )
            solution = master.solve(instance.problem, _all_edges(instance.problem), tie_break="lex")
            theta_range = optimal_theta_ranges(instance.problem)
            check = is_optimal_plan(instance.problem, solution.theta)
            deltas.append(float(solution.delta))
            widths.append(float(theta_range.width().max()))
            distances.append(float(np.linalg.norm(solution.theta - instance.theta_star)))
            optimal.append(bool(check.is_optimal))
        rows.append(
            E1Row(
                n_source=int(size),
                n_target=int(size),
                n_features=int(n_features),
                repeats=int(repeats),
                mean_delta=float(np.mean(deltas)),
                std_delta=float(np.std(deltas)),
                mean_range_width_max=float(np.mean(widths)),
                std_range_width_max=float(np.std(widths)),
                mean_distance_to_theta_star=float(np.mean(distances)),
                std_distance_to_theta_star=float(np.std(distances)),
                all_optimal=bool(all(optimal)),
            )
        )
    return rows


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="e1: full-master inverse OT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 10, 12, 16, 20])
    parser.add_argument("--features", type=int, default=4, help="number of theta coordinates")
    parser.add_argument("--mixing", type=float, default=0.8, help="coordinate correlation strength")
    parser.add_argument("--repeats", type=int, default=3, help="seeds averaged per size")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = run_e1(
        seed=int(args.seed),
        sizes=[int(size) for size in args.sizes],
        n_features=int(args.features),
        mixing_strength=float(args.mixing),
        repeats=int(args.repeats),
    )
    print(f"{'m=n':>5} {'delta*':>10} {'range_width':>12} {'dist_to_star':>13} {'optimal':>8}")
    for row in rows:
        print(
            f"{row.n_source:>5} {row.mean_delta:>10.6f} {row.mean_range_width_max:>12.6f} "
            f"{row.mean_distance_to_theta_star:>13.6f} {str(row.all_optimal):>8}"
        )
    payload = {"experiment": "e1_full_master", "rows": [asdict(row) for row in rows]}
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row.all_optimal for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
