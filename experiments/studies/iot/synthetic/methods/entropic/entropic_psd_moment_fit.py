"""
CN: entropic_psd_moment_fit —— 在 bilinear-PSD 度量上比较精确 OT 次梯度与熵 OT 平滑梯度。
EN: entropic_psd_moment_fit -- compare the exact-OT subgradient against the
    entropic-OT smoothed gradient for bilinear-PSD metric fitting.

CN: 精确 OT 的 plan 对 A 分段常值，`fitted - observed` 只是次梯度，1/sqrt(k) 次梯度下降会震荡。
    本实验改用 `fit_psd_moment_entropic`（log-domain Sinkhorn + Adam + Polyak 平均 +
    epsilon 退火），并报告真实的 L0 间隙与 `is_optimal_plan`，而不是只看梯度范数。
EN: The exact-OT plan is piecewise constant in A, so `fitted - observed` is only a
    subgradient and a 1/sqrt(k) rule zig-zags. This experiment uses
    `fit_psd_moment_entropic` (log-domain Sinkhorn + Adam + Polyak averaging + epsilon
    annealing) and reports the true L0 gap and `is_optimal_plan`, not just gradient norms.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from iot import (
    fit_psd_moment_entropic,
    is_optimal_plan,
    make_bilinear_psd_grounded_instance,
    moment_from_plan,
    scale_invariant_error,
)
from iot.eval.evaluation import l0_gap


@dataclass(frozen=True)
class FitScenarioReport:
    """
    CN: 单个场景（精确 plan / 采样 plan）的拟合结果。
    EN: Fitting result for one scenario (exact plan / sampled plan).
    """

    scenario: str
    n_source: int
    n_target: int
    point_dimension: int
    scale_error: float
    moment_error_initial: float
    moment_error_final: float
    entropic_objective_first: float
    entropic_objective_last: float
    exact_gap_initial: float
    exact_gap: float
    is_optimal: bool
    wall_time: float


def _sample_moment(
    plan: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    CN: 从 plan 采样 `count` 个 (x, y) 对，返回经验交叉矩。
    EN: Sample `count` (x, y) pairs from the plan and return the empirical cross moment.
    """
    probabilities = np.asarray(plan, dtype=np.float64).reshape(-1)
    probabilities = probabilities / probabilities.sum()
    flat_choices = rng.choice(probabilities.size, size=int(count), p=probabilities)
    rows, cols = np.divmod(flat_choices, plan.shape[1])
    moment = source_points[rows].T @ target_points[cols]
    return np.ascontiguousarray(moment / float(count), dtype=np.float64)


def run(
    *,
    seed: int,
    n_source: int,
    n_target: int,
    point_dimension: int,
    num_outer: int,
    samples: int,
    learning_rate: float,
    epsilon_end: float,
) -> list[FitScenarioReport]:
    """
    CN: 生成真值实例，分别用真值矩与采样矩拟合 A，并报告多项诊断。
    EN: Generate a grounded instance, fit A from the true moment and from sampled
        observations, and report several diagnostics.
    """
    rng = np.random.default_rng(int(seed))
    instance = make_bilinear_psd_grounded_instance(
        n_source=int(n_source),
        n_target=int(n_target),
        point_dimension=int(point_dimension),
        rng=rng,
    )
    problem = instance.problem
    source = problem.features.source_points
    target = problem.features.target_points
    source_mass = problem.resolved_source_mass
    target_mass = problem.resolved_target_mass
    truth_plan = problem.plan.toarray()
    true_matrix = instance.theta_star.reshape(int(point_dimension), int(point_dimension))
    true_moment = moment_from_plan(truth_plan, source, target)
    initial_matrix = np.eye(int(point_dimension)) / float(point_dimension)
    initial_gap = l0_gap(problem, initial_matrix.reshape(-1))

    scenarios = [("exact", true_moment), ("sampled", _sample_moment(truth_plan, source, target, int(samples), rng))]
    reports: list[FitScenarioReport] = []
    for name, observed_moment in scenarios:
        started = time.perf_counter()
        result = fit_psd_moment_entropic(
            source_points=source,
            target_points=target,
            source_mass=source_mass,
            target_mass=target_mass,
            observed_moment=observed_moment,
            initial_matrix=initial_matrix,
            num_outer=int(num_outer),
            epsilon_end=float(epsilon_end),
            learning_rate=float(learning_rate),
        )
        elapsed = time.perf_counter() - started
        fitted_matrix = result.matrix
        reports.append(
            FitScenarioReport(
                scenario=name,
                n_source=int(n_source),
                n_target=int(n_target),
                point_dimension=int(point_dimension),
                scale_error=float(scale_invariant_error(true_matrix, fitted_matrix)),
                moment_error_initial=float(result.initial_moment_error),
                moment_error_final=float(result.final_moment_error),
                entropic_objective_first=float(result.iterations[0].objective),
                entropic_objective_last=float(result.iterations[-1].objective),
                exact_gap_initial=float(initial_gap),
                exact_gap=l0_gap(problem, fitted_matrix.reshape(-1)),
                is_optimal=bool(is_optimal_plan(problem, fitted_matrix.reshape(-1)).is_optimal),
                wall_time=float(elapsed),
            )
        )
    return reports


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="entropic PSD moment fitting for inverse OT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=120, help="source and target count")
    parser.add_argument("--dim", type=int, default=2, help="point dimension")
    parser.add_argument("--num-outer", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epsilon-end", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reports = run(
        seed=int(args.seed),
        n_source=int(args.n),
        n_target=int(args.n),
        point_dimension=int(args.dim),
        num_outer=int(args.num_outer),
        samples=int(args.samples),
        learning_rate=float(args.learning_rate),
        epsilon_end=float(args.epsilon_end),
    )
    print(
        f"{'scenario':>8} {'A_err':>8} {'mom_first':>10} {'mom_last':>9} "
        f"{'L0_init':>10} {'L0_final':>10} {'optimal':>8} {'time':>7}"
    )
    for report in reports:
        print(
            f"{report.scenario:>8} {report.scale_error:>8.4f} {report.moment_error_initial:>10.4f} "
            f"{report.moment_error_final:>9.4f} {report.exact_gap_initial:>10.2e} "
            f"{report.exact_gap:>10.2e} {str(report.is_optimal):>8} {report.wall_time:>7.2f}"
        )
    payload = {"experiment": "entropic_psd_moment_fit", "rows": [asdict(report) for report in reports]}
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    exact_reports = [report for report in reports if report.scenario == "exact"]
    improved = all(
        report.exact_gap <= report.exact_gap_initial + 1e-9 and report.scale_error < 0.5
        for report in exact_reports
    )
    return 0 if improved else 1


if __name__ == "__main__":
    raise SystemExit(main())
