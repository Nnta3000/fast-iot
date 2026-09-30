"""
CN: e3_scaling —— 随 m = n 增大测量约束生成循环的每轮时间、|F| 与迭代数（K <= 20）。
EN: e3_scaling -- measure per-iteration time, |F|, and iteration count as m = n grows
    (K <= 20).

CN: 用 POT 生成真值以隔离逆求解器；同时报告 |B|、每轮 LP 与打分耗时，以及无需重解 OT 的
    delta 证书（final_max_violation <= tol）。
EN: POT generates ground truth to isolate the inverse solver; we also report |B|, the LP
    and scoring cost per iteration, and the OT-free delta certificate
    (final_max_violation <= tol).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import ot
from scipy import sparse

from iot import (
    DiagonalWeightFeatures,
    InverseOTProblem,
    SimplexTheta,
    dense_cost_matrix,
    mandatory_edges,
    solve_inverse_ot,
)


@dataclass(frozen=True)
class E3Row:
    """
    CN: e3 在单个规模上跨种子平均的结果。
    EN: e3 result at a single size, averaged over seeds.
    """

    n_source: int
    n_target: int
    n_features: int
    repeats: int
    mandatory_size: int
    final_support: float
    max_support: float
    max_support_over_m_plus_n: float
    iterations: float
    converged_fraction: float
    total_wall_time: float
    mean_iteration_time: float
    mean_master_time: float
    mean_oracle_time: float
    mean_final_max_violation: float
    certificate_fraction: float
    is_optimal: bool | None


def run_e3(
    *,
    seed: int,
    sizes: list[int],
    n_features: int,
    max_iterations: int,
    repeats: int,
) -> list[E3Row]:
    """
    CN: 对每个规模重复生成真值并运行逆求解器，记录规模与耗时统计。
    EN: For each size, repeat ground-truth generation and the inverse solve, recording
        scaling and timing statistics.
    """
    rows: list[E3Row] = []
    for size in sizes:
        final_supports: list[float] = []
        max_supports: list[float] = []
        iteration_counts: list[float] = []
        converged_flags: list[bool] = []
        total_times: list[float] = []
        iteration_times: list[float] = []
        master_times: list[float] = []
        oracle_times: list[float] = []
        final_violations: list[float] = []
        certificates: list[bool] = []
        optimal_flags: list[bool] = []
        mandatory_size = 0
        for repeat in range(int(repeats)):
            rng = np.random.default_rng(int(seed) + 1000 * int(repeat) + int(size))
            theta_set = SimplexTheta(int(n_features))
            theta_star = theta_set.sample(rng)
            source = np.ascontiguousarray(rng.normal(size=(int(size), int(n_features))), dtype=np.float64)
            target = np.ascontiguousarray(rng.normal(size=(int(size), int(n_features))), dtype=np.float64)
            features = DiagonalWeightFeatures(source, target)
            cost = dense_cost_matrix(features, theta_star)
            mass = np.full(int(size), 1.0 / float(size), dtype=np.float64)
            plan = ot.emd(mass, mass, cost, numItermax=2_000_000)
            problem = InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))
            mandatory, _partition = mandatory_edges(problem)
            mandatory_size = int(mandatory.size)
            result = solve_inverse_ot(
                problem,
                tie_break="random",
                max_iterations=int(max_iterations),
                validate=(int(size) <= 200),
                random_seed=int(seed),
            )
            supports = [record.support_size for record in result.iterations]
            final_supports.append(float(result.support_size))
            max_supports.append(float(max(supports)) if supports else 0.0)
            iteration_counts.append(float(result.n_iterations))
            converged_flags.append(bool(result.converged))
            total_times.append(float(result.total_wall_time))
            iteration_times.extend(record.wall_time for record in result.iterations)
            master_times.extend(record.master_time for record in result.iterations)
            oracle_times.extend(record.oracle_time for record in result.iterations)
            final_violations.append(float(result.final_max_violation))
            certificates.append(bool(result.converged and result.final_max_violation <= 1e-6))
            if result.is_optimal is not None:
                optimal_flags.append(bool(result.is_optimal))
        rows.append(
            E3Row(
                n_source=int(size),
                n_target=int(size),
                n_features=int(n_features),
                repeats=int(repeats),
                mandatory_size=mandatory_size,
                final_support=float(np.mean(final_supports)),
                max_support=float(np.mean(max_supports)),
                max_support_over_m_plus_n=float(np.mean(max_supports) / (2.0 * float(size))),
                iterations=float(np.mean(iteration_counts)),
                converged_fraction=float(np.mean(converged_flags)),
                total_wall_time=float(np.mean(total_times)),
                mean_iteration_time=float(np.mean(iteration_times)) if iteration_times else 0.0,
                mean_master_time=float(np.mean(master_times)) if master_times else 0.0,
                mean_oracle_time=float(np.mean(oracle_times)) if oracle_times else 0.0,
                mean_final_max_violation=float(np.mean(final_violations)),
                certificate_fraction=float(np.mean(certificates)),
                is_optimal=(bool(all(optimal_flags)) if optimal_flags else None),
            )
        )
    return rows


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="e3: inverse-OT scaling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--features", type=int, default=8, help="number of theta coordinates (K <= 20)")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1, help="seeds averaged per size")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = run_e3(
        seed=int(args.seed),
        sizes=[int(size) for size in args.sizes],
        n_features=int(args.features),
        max_iterations=int(args.max_iterations),
        repeats=int(args.repeats),
    )
    print(
        f"{'m=n':>6} {'K':>3} {'|B|':>7} {'max|F|':>8} {'max/(m+n)':>10} {'iters':>6} "
        f"{'cert':>5} {'ms/iter':>8} {'ms/lp':>8} {'ms/oracle':>10}"
    )
    for row in rows:
        print(
            f"{row.n_source:>6} {row.n_features:>3} {row.mandatory_size:>7} {row.max_support:>8.0f} "
            f"{row.max_support_over_m_plus_n:>10.2f} {row.iterations:>6.1f} "
            f"{row.certificate_fraction:>5.2f} {1000.0 * row.mean_iteration_time:>8.2f} "
            f"{1000.0 * row.mean_master_time:>8.2f} {1000.0 * row.mean_oracle_time:>10.2f}"
        )
    payload = {"experiment": "e3_scaling", "rows": [asdict(row) for row in rows]}
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row.certificate_fraction == 1.0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
