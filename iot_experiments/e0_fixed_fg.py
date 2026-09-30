"""
CN: e0_fixed_fg —— 固定 OT 对偶势 (f*, g*)，只求 theta 的可行范围。
EN: e0_fixed_fg -- fix the OT dual pair (f*, g*) and report the feasible theta range.

CN: 这是最简实验，也是主管要求的 sanity check：它不引入未知的 (f, g)，只用线性约束
    检查 theta 的可辨识性，并确认 theta_star 落在可行域内。
EN: This is the simplest experiment and the sanity check requested by the supervisor: it
    introduces no unknown (f, g), only checks theta identifiability under linear
    constraints and confirms that theta_star lies in the feasible region.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from iot import (
    fixed_dual_theta_ranges,
    is_optimal_plan,
    make_diagonal_grounded_instance,
)


@dataclass(frozen=True)
class E0Report:
    """
    CN: e0 的可序列化报告。
    EN: Serializable report for e0.
    """

    seed: int
    n_source: int
    n_target: int
    n_features: int
    support_size: int
    theta_star: list[float]
    lower: list[float]
    upper: list[float]
    width: list[float]
    theta_star_in_range: bool
    is_optimal_plan: bool
    relative_gap: float
    wall_time_seconds: float


def run_e0(
    *,
    seed: int,
    n_source: int,
    n_target: int,
    n_features: int,
    equality_tol: float = 1e-6,
    feasibility_tol: float = 1e-6,
) -> E0Report:
    """
    CN: 生成带真值实例，求固定对偶势下的 theta 范围，并做最优性验证。
    EN: Generate a grounded instance, compute the fixed-dual theta range, and validate.
    """
    started = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    instance = make_diagonal_grounded_instance(
        n_source=int(n_source),
        n_target=int(n_target),
        n_features=int(n_features),
        rng=rng,
        solver_seed=int(seed),
    )
    theta_range = fixed_dual_theta_ranges(
        instance.problem,
        instance.source_dual,
        instance.target_dual,
        equality_tol=float(equality_tol),
        feasibility_tol=float(feasibility_tol),
    )
    check = is_optimal_plan(instance.problem, instance.theta_star)
    elapsed = time.perf_counter() - started
    return E0Report(
        seed=int(seed),
        n_source=int(instance.problem.n_source),
        n_target=int(instance.problem.n_target),
        n_features=int(instance.problem.n_features),
        support_size=int(instance.problem.support_size),
        theta_star=np.asarray(instance.theta_star, dtype=np.float64).tolist(),
        lower=theta_range.lower.tolist(),
        upper=theta_range.upper.tolist(),
        width=theta_range.width().tolist(),
        theta_star_in_range=bool(theta_range.contains(instance.theta_star)),
        is_optimal_plan=bool(check.is_optimal),
        relative_gap=float(check.relative_gap),
        wall_time_seconds=float(elapsed),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="e0: fixed-dual theta range for inverse OT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--m", type=int, default=16, help="number of source points")
    parser.add_argument("--n", type=int, default=16, help="number of target points")
    parser.add_argument("--features", type=int, default=4, help="number of theta coordinates")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_e0(
        seed=int(args.seed),
        n_source=int(args.m),
        n_target=int(args.n),
        n_features=int(args.features),
    )
    text = json.dumps(asdict(report), indent=2)
    print(text)
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if (report.theta_star_in_range and report.is_optimal_plan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
