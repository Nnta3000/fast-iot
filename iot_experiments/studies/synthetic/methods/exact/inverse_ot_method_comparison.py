"""
CN: inverse_ot_method_comparison —— 在同一问题上比较约束生成 loop、θ-空间 Sinkhorn 基线与
    PSD 矩拟合器，并输出三张表、ε 调度与 confounds（写入 scratch/COMPARISON_REPORT.md）。
EN: inverse_ot_method_comparison -- compare the constraint-generation loop, the theta-space
    Sinkhorn baseline, and the PSD moment fitter on identical problems, emitting three
    tables, the epsilon schedule, and confounds to scratch/COMPARISON_REPORT.md.

CN: 两个求解器都用子模块导入（`iot.loop` / `iot.sinkhorn_theta`），避免其中一方经过
    `__init__` 的额外封装而引入不可追踪的差异。
EN: Both solvers are imported from their submodules (`iot.loop` / `iot.sinkhorn_theta`) so
    neither goes through extra `__init__` sugar that could create untraceable differences.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import ot
from scipy import sparse

from iot.alternatives.entropic import fit_psd_moment_entropic
from iot.eval.evaluation import evaluate_inverse_ot
from iot.core.features import BilinearPSDFeatures, DiagonalWeightFeatures, dense_cost_matrix
from iot.exact.loop import solve_inverse_ot
from iot.core.problem import InverseOTError, InverseOTProblem
from iot.exact.sdp import SdpMasterSolver
from iot.alternatives.sinkhorn_theta import fit_theta_sinkhorn
from iot.core.theta import BilinearPSDTheta, SimplexTheta

DEFAULT_REPORT = Path("iot_experiments/scratch/COMPARISON_REPORT.md")


@dataclass(frozen=True)
class ComparisonRow:
    """
    CN: 一张表中的一个 (comparison, method, size) 行，跨 seed 聚合。
    EN: One (comparison, method, size) row of a table, aggregated over seeds.
    """

    comparison: str
    method: str
    n_source: int
    n_target: int
    n_features: int
    seeds: int
    failures: int
    is_optimal_fraction: float | None
    converged_fraction: float | None
    l0_gap_mean: float | None
    l0_gap_std: float | None
    moment_error_mean: float | None
    dist_to_theta_star_mean: float | None
    wall_time_mean: float | None
    peak_memory_mb_mean: float | None
    iterations_mean: float | None
    final_gradient_norm_mean: float | None
    final_epsilon_mean: float | None


@dataclass
class _SeedSample:
    is_optimal: bool
    l0_gap: float
    moment_error: float
    dist_to_theta_star: float
    wall_time: float
    peak_memory_mb: float
    iterations: float
    converged: bool | None
    final_gradient_norm: float | None
    final_epsilon: float | None


def build_diagonal_problem(*, size: int, n_features: int, seed: int, mixing_strength: float) -> tuple[InverseOTProblem, np.ndarray]:
    """
    CN: 用 POT 构造 simplex 真值实例（不依赖 HELLO-OT，保证两方法看到同一问题）。
    EN: Build a simplex grounded instance with POT (no HELLO-OT dependency; both methods
        see the identical problem).
    """
    rng = np.random.default_rng(int(seed))
    source = rng.normal(size=(int(size), int(n_features)))
    target = rng.normal(size=(int(size), int(n_features)))
    dim = int(n_features)
    if float(mixing_strength) > 0.0:
        source = source @ (np.eye(dim) + float(mixing_strength) * np.tril(rng.normal(size=(dim, dim)), -1))
        target = target @ (np.eye(dim) + float(mixing_strength) * np.tril(rng.normal(size=(dim, dim)), -1))
    theta_set = SimplexTheta(int(n_features))
    theta_star = theta_set.sample(rng)
    features = DiagonalWeightFeatures(source, target)
    cost = dense_cost_matrix(features, theta_star)
    mass = np.full(int(size), 1.0 / float(size))
    plan = ot.emd(mass, mass, cost, numItermax=2_000_000)
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))
    return problem, theta_star


def build_psd_problem(*, size: int, point_dimension: int, seed: int) -> tuple[InverseOTProblem, np.ndarray]:
    """
    CN: 用 POT 构造 bilinear-PSD 真值实例。
    EN: Build a bilinear-PSD grounded instance with POT.
    """
    rng = np.random.default_rng(int(seed))
    source = rng.normal(size=(int(size), int(point_dimension)))
    target = rng.normal(size=(int(size), int(point_dimension)))
    theta_set = BilinearPSDTheta(int(point_dimension))
    theta_star = theta_set.sample(rng)
    features = BilinearPSDFeatures(source, target)
    cost = dense_cost_matrix(features, theta_star)
    mass = np.full(int(size), 1.0 / float(size))
    plan = ot.emd(mass, mass, cost, numItermax=2_000_000)
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))
    return problem, theta_star


def _measure(callable_):
    """
    CN: 用 tracemalloc 近似峰值内存（不捕获 numpy 原生分配），并计时。
    EN: Approximate peak memory with tracemalloc (does not capture NumPy native
        allocations) and measure wall time.
    """
    tracemalloc.start()
    started = time.perf_counter()
    result = callable_()
    wall_time = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, wall_time, float(peak) / 1_000_000.0


def _sample(metrics, *, iterations: float, converged: bool | None, final_gradient_norm: float | None, final_epsilon: float | None) -> _SeedSample:
    return _SeedSample(
        is_optimal=bool(metrics.is_optimal),
        l0_gap=float(metrics.l0_gap),
        moment_error=float(metrics.moment_error),
        dist_to_theta_star=float(metrics.dist_to_theta_star),
        wall_time=float(metrics.wall_time),
        peak_memory_mb=float(metrics.peak_memory_mb),
        iterations=float(iterations),
        converged=converged,
        final_gradient_norm=final_gradient_norm,
        final_epsilon=final_epsilon,
    )


def _run_loop(problem: InverseOTProblem, *, tie_break: str, solver, max_iterations: int):
    def call():
        return solve_inverse_ot(
            problem,
            tie_break=tie_break,
            solver=solver,
            max_iterations=int(max_iterations),
            validate=False,
            prune=(solver is None),
        )

    return _measure(call)


def _run_sinkhorn(problem: InverseOTProblem, *, step_rule: str, num_outer: int, sinkhorn_iterations: int):
    return _measure(
        lambda: fit_theta_sinkhorn(
            problem,
            num_outer=int(num_outer),
            sinkhorn_iterations=int(sinkhorn_iterations),
            step_rule=step_rule,
        )
    )


def _run_moment_fitter(problem: InverseOTProblem, *, num_outer: int, sinkhorn_iterations: int):
    from iot.alternatives.entropic import moment_from_plan

    source = problem.features.source_points
    target = problem.features.target_points
    observed = moment_from_plan(problem.plan.toarray(), source, target)
    dim = int(problem.n_features)
    initial = (np.eye(int(round(np.sqrt(dim)))) / int(round(np.sqrt(dim)))).reshape(-1)
    return _measure(
        lambda: fit_psd_moment_entropic(
            source_points=source,
            target_points=target,
            source_mass=problem.resolved_source_mass,
            target_mass=problem.resolved_target_mass,
            observed_moment=observed,
            initial_matrix=initial.reshape(int(round(np.sqrt(dim))), int(round(np.sqrt(dim)))),
            num_outer=int(num_outer),
            sinkhorn_iterations=int(sinkhorn_iterations),
        )
    )


def _aggregate(comparison: str, method: str, size: int, n_features: int, samples: list[_SeedSample | None]) -> ComparisonRow:
    valid = [sample for sample in samples if sample is not None]
    failures = len(samples) - len(valid)

    def mean_of(name: str) -> float | None:
        values = [float(getattr(sample, name)) for sample in valid if getattr(sample, name) is not None]
        return float(np.mean(values)) if values else None

    converged_values = [sample.converged for sample in valid if sample.converged is not None]
    l0_values = [sample.l0_gap for sample in valid]
    return ComparisonRow(
        comparison=comparison,
        method=method,
        n_source=int(size),
        n_target=int(size),
        n_features=int(n_features),
        seeds=len(samples),
        failures=int(failures),
        is_optimal_fraction=(float(np.mean([sample.is_optimal for sample in valid])) if valid else None),
        converged_fraction=(float(np.mean(converged_values)) if converged_values else None),
        l0_gap_mean=(float(np.mean(l0_values)) if l0_values else None),
        l0_gap_std=(float(np.std(l0_values)) if l0_values else None),
        moment_error_mean=mean_of("moment_error"),
        dist_to_theta_star_mean=mean_of("dist_to_theta_star"),
        wall_time_mean=mean_of("wall_time"),
        peak_memory_mb_mean=mean_of("peak_memory_mb"),
        iterations_mean=mean_of("iterations"),
        final_gradient_norm_mean=mean_of("final_gradient_norm"),
        final_epsilon_mean=mean_of("final_epsilon"),
    )


def run_comparison_a(*, sizes, seeds, n_features, num_outer, sinkhorn_iterations, max_iterations) -> list[ComparisonRow]:
    methods = ["constraint-generation", "sinkhorn-theta-adam", "sinkhorn-theta-plain"]
    rows: list[ComparisonRow] = []
    for size in sizes:
        buckets: dict[str, list[_SeedSample | None]] = {method: [] for method in methods}
        for seed in seeds:
            problem, theta_star = build_diagonal_problem(
                size=int(size), n_features=int(n_features), seed=int(seed), mixing_strength=0.8
            )
            try:
                result, wall, peak = _run_loop(problem, tie_break="random", solver=None, max_iterations=max_iterations)
                metrics = evaluate_inverse_ot(
                    problem, result.theta, theta_star=theta_star, wall_time=wall,
                    peak_memory_mb=peak, iterations=result.n_iterations,
                )
                buckets["constraint-generation"].append(
                    _sample(metrics, iterations=result.n_iterations, converged=result.converged,
                            final_gradient_norm=None, final_epsilon=None)
                )
            except Exception:
                buckets["constraint-generation"].append(None)
            for rule, name in (("adam", "sinkhorn-theta-adam"), ("plain", "sinkhorn-theta-plain")):
                try:
                    result, wall, peak = _run_sinkhorn(
                        problem, step_rule=rule, num_outer=num_outer, sinkhorn_iterations=sinkhorn_iterations
                    )
                    metrics = evaluate_inverse_ot(
                        problem, result.theta, theta_star=theta_star, wall_time=wall,
                        peak_memory_mb=peak, iterations=len(result.iterations),
                    )
                    buckets[name].append(
                        _sample(metrics, iterations=len(result.iterations), converged=None,
                                final_gradient_norm=result.final_gradient_norm, final_epsilon=result.final_epsilon)
                    )
                except Exception:
                    buckets[name].append(None)
        order = {"constraint-generation": 0, "sinkhorn-theta-adam": 1, "sinkhorn-theta-plain": 2}
        for method in sorted(methods, key=lambda m: order[m]):
            rows.append(_aggregate("a_simplex", method, int(size), int(n_features), buckets[method]))
    return rows


def run_comparison_b(*, sizes, seeds, point_dimension, num_outer, sinkhorn_iterations, max_iterations) -> list[ComparisonRow]:
    methods = ["constraint-generation-lex", "constraint-generation-none", "sinkhorn-theta-adam", "sinkhorn-theta-plain"]
    rows: list[ComparisonRow] = []
    for size in sizes:
        buckets: dict[str, list[_SeedSample | None]] = {method: [] for method in methods}
        for seed in seeds:
            problem, theta_star = build_psd_problem(
                size=int(size), point_dimension=int(point_dimension), seed=int(seed)
            )
            for tie, name, solver in (
                ("lex", "constraint-generation-lex", SdpMasterSolver()),
                ("none", "constraint-generation-none", SdpMasterSolver()),
            ):
                try:
                    result, wall, peak = _run_loop(problem, tie_break=tie, solver=solver, max_iterations=max_iterations)
                    metrics = evaluate_inverse_ot(
                        problem, result.theta, theta_star=theta_star, wall_time=wall,
                        peak_memory_mb=peak, iterations=result.n_iterations,
                    )
                    buckets[name].append(
                        _sample(metrics, iterations=result.n_iterations, converged=result.converged,
                                final_gradient_norm=None, final_epsilon=None)
                    )
                except InverseOTError:
                    buckets[name].append(None)
            for rule, name in (("adam", "sinkhorn-theta-adam"), ("plain", "sinkhorn-theta-plain")):
                try:
                    result, wall, peak = _run_sinkhorn(
                        problem, step_rule=rule, num_outer=num_outer, sinkhorn_iterations=sinkhorn_iterations
                    )
                    metrics = evaluate_inverse_ot(
                        problem, result.theta, theta_star=theta_star, wall_time=wall,
                        peak_memory_mb=peak, iterations=len(result.iterations),
                    )
                    buckets[name].append(
                        _sample(metrics, iterations=len(result.iterations), converged=None,
                                final_gradient_norm=result.final_gradient_norm, final_epsilon=result.final_epsilon)
                    )
                except Exception:
                    buckets[name].append(None)
        for method in methods:
            rows.append(_aggregate("b_psd", method, int(size), int(point_dimension) ** 2, buckets[method]))
    return rows


def run_comparison_c(*, sizes, seeds, point_dimension, num_outer, sinkhorn_iterations) -> list[ComparisonRow]:
    methods = ["sinkhorn-theta-adam", "psd-moment-fitter"]
    rows: list[ComparisonRow] = []
    for size in sizes:
        buckets: dict[str, list[_SeedSample | None]] = {method: [] for method in methods}
        for seed in seeds:
            problem, theta_star = build_psd_problem(
                size=int(size), point_dimension=int(point_dimension), seed=int(seed)
            )
            result, wall, peak = _run_sinkhorn(
                problem, step_rule="adam", num_outer=num_outer, sinkhorn_iterations=sinkhorn_iterations
            )
            metrics = evaluate_inverse_ot(
                problem, result.theta, theta_star=theta_star, wall_time=wall,
                peak_memory_mb=peak, iterations=len(result.iterations),
            )
            buckets["sinkhorn-theta-adam"].append(
                _sample(metrics, iterations=len(result.iterations), converged=None,
                        final_gradient_norm=result.final_gradient_norm, final_epsilon=result.final_epsilon)
            )
            try:
                fitted, wall2, peak2 = _run_moment_fitter(
                    problem, num_outer=num_outer, sinkhorn_iterations=sinkhorn_iterations
                )
                metrics2 = evaluate_inverse_ot(
                    problem, fitted.matrix.reshape(-1), theta_star=theta_star, wall_time=wall2,
                    peak_memory_mb=peak2, iterations=len(fitted.iterations),
                )
                buckets["psd-moment-fitter"].append(
                    _sample(metrics2, iterations=len(fitted.iterations), converged=None,
                            final_gradient_norm=None, final_epsilon=fitted.iterations[-1].epsilon)
                )
            except Exception:
                buckets["psd-moment-fitter"].append(None)
        for method in methods:
            rows.append(_aggregate("c_psd_information", method, int(size), int(point_dimension) ** 2, buckets[method]))
    return rows


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}g}"


def _gap(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.3e}"
    return f"{mean:.3e}±{std:.1e}"


def _table(title: str, rows: list[ComparisonRow]) -> str:
    header = (
        "| method | n | opt frac | L0 gap (mean±std) | moment err | dist θ* | wall s | peak MB "
        "| iters | conv frac | final ‖grad‖ | final ε |"
    )
    separator = "|" + "---|" * 12
    lines = [f"### {title}", "", header, separator]
    for row in rows:
        lines.append(
            "| {method} | {n} | {opt} | {gap} | {moment} | {dist} | {wall} | {mem} | {iters} "
            "| {conv} | {grad} | {eps} |".format(
                method=row.method,
                n=row.n_source,
                opt=_fmt(row.is_optimal_fraction, 3) if row.failures == 0 else f"{_fmt(row.is_optimal_fraction, 3)} ({row.failures}/{row.seeds} fail)",
                gap=_gap(row.l0_gap_mean, row.l0_gap_std),
                moment=_fmt(row.moment_error_mean),
                dist=_fmt(row.dist_to_theta_star_mean),
                wall=_fmt(row.wall_time_mean),
                mem=_fmt(row.peak_memory_mb_mean),
                iters=_fmt(row.iterations_mean, 4),
                conv=_fmt(row.converged_fraction, 3),
                grad=_fmt(row.final_gradient_norm_mean),
                eps=_fmt(row.final_epsilon_mean, 3),
            )
        )
    return "\n".join(lines)


def _markdown(rows_a: list[ComparisonRow], rows_b: list[ComparisonRow], rows_c: list[ComparisonRow], *, num_outer: int, sinkhorn_iterations: int, features_a: int, dim_b: int) -> str:
    epsilon_start_note = "0.1 * std(C_theta0)"
    schedule = (
        f"- epsilon_start: `{epsilon_start_note}`（每个实例自适应）\n"
        f"- epsilon_decay: `0.95`（几何退火）\n"
        f"- epsilon_end (floor): `1e-3`\n"
        f"- Sinkhorn: `max_iterations={sinkhorn_iterations}`, `tolerance=1e-9`（log-domain）\n"
        f"- outer iterations (budget): `{num_outer}`（Sinkhorn 与 moment fitter 相同预算）"
    )
    confounds = (
        "## Confounds\n\n"
        "- **Optimality tolerance.** `evaluate_inverse_ot` uses `optimality_tol=1e-7` everywhere, "
        "matching `loop.py`'s own `is_optimal_plan(..., tol=1e-7)`. We do not use 1e-9 because at "
        "these LP/POT sizes it classifies numerical noise as infeasibility.\n"
        "- **Loop tie-break.** Comparison (a) runs the loop with `tie_break=\"random\"` (the practical "
        "choice; `lex` is O(N) re-solves per iteration and infeasible here) and comparison (b) with "
        "`lex` where it is defined. `random`/`none` are outside the termination proof.\n"
        "- **SDP (b) brittleness.** `SdpMasterSolver` with the proof-covered `lex` tie-break fails "
        "with `infeasible_inaccurate` for `n >= 24` here; the `constraint-generation-none` rows are "
        "not proof-covered and are shown only to keep a same-size reference.\n"
        "- **Budget vs certificate.** Loop rows run until the δ certificate or a cap; Sinkhorn rows "
        "always run to `num_outer` and carry no certificate. Every Sinkhorn row reports "
        "`final ‖grad‖` and `final ε` next to `wall s`; a fast Sinkhorn row that did not anneal is "
        "not evidence of a better method.\n"
        "- **Memory.** `peak_memory_mb` is `tracemalloc` peak, which does **not** capture NumPy's "
        "native allocations (only the Python allocator); treat it as a lower bound, not true RSS.\n"
        "- **Information asymmetry in (c).** The PSD moment fitter sees only the observed "
        "cross-moment `X^T pi_hat Y`; Sinkhorn-theta sees the full plan `pi_hat`. A Sinkhorn row "
        "winning on `dist_to_theta_star` there is *expected*, not a fair \"ours is better\" claim.\n"
        "- **`dist_to_theta_star` is a diagnostic only.** On discrete data theta is generically "
        "non-identifiable; the authoritative check is `is_optimal_plan`, and `L0 gap` is the "
        "raw quantity to read.\n"
    )
    return "\n\n".join(
        [
            "# Inverse-OT method comparison",
            (
                f"Generated by `iot_experiments/methods/exact/inverse_ot_method_comparison.py`. "
                f"Simplex: K={features_a}; PSD: d={dim_b}. Budgets: num_outer={num_outer}, "
                f"sinkhorn_iterations={sinkhorn_iterations}. Ground truth built with POT; both "
                f"methods see the identical problem per seed."
            ),
            "## Epsilon schedule",
            schedule,
            _table("(a) Simplex: constraint-generation vs Sinkhorn-θ", rows_a),
            _table("(b) PSD+trace1: constraint-generation vs Sinkhorn-θ", rows_b),
            _table("(c) PSD: Sinkhorn-θ (full plan) vs PSD moment fitter (moment only)", rows_c),
            "## Reading the tables\n\n"
            "`opt frac` = fraction of seeds where `is_optimal_plan` passes; `L0 gap` = "
            "`<C,π̂> - min_P <C,P>` (raw, mean±std over seeds; lower is better, 0 means π̂ optimal); "
            "`conv frac` = fraction of loop runs that reached the δ certificate; `final ‖grad‖` and "
            "`final ε` apply to Sinkhorn rows only.",
            confounds,
        ]
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="compare inverse-OT methods")
    parser.add_argument("--sizes-a", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    parser.add_argument("--sizes-b", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--sizes-c", type=int, nargs="+", default=[8, 16, 32, 40])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--features-a", type=int, default=4)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--num-outer", type=int, default=25)
    parser.add_argument("--sinkhorn-iterations", type=int, default=80)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    seeds = [int(seed) for seed in args.seeds]
    rows_a = run_comparison_a(
        sizes=[int(s) for s in args.sizes_a], seeds=seeds, n_features=int(args.features_a),
        num_outer=int(args.num_outer), sinkhorn_iterations=int(args.sinkhorn_iterations),
        max_iterations=int(args.max_iterations),
    )
    rows_b = run_comparison_b(
        sizes=[int(s) for s in args.sizes_b], seeds=seeds, point_dimension=int(args.dim),
        num_outer=int(args.num_outer), sinkhorn_iterations=int(args.sinkhorn_iterations),
        max_iterations=int(args.max_iterations),
    )
    rows_c = run_comparison_c(
        sizes=[int(s) for s in args.sizes_c], seeds=seeds, point_dimension=int(args.dim),
        num_outer=int(args.num_outer), sinkhorn_iterations=int(args.sinkhorn_iterations),
    )
    markdown = _markdown(
        rows_a, rows_b, rows_c,
        num_outer=int(args.num_outer), sinkhorn_iterations=int(args.sinkhorn_iterations),
        features_a=int(args.features_a), dim_b=int(args.dim),
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(markdown + "\n", encoding="utf-8")
    if args.json is not None:
        payload = {
            "a_simplex": [asdict(row) for row in rows_a],
            "b_psd": [asdict(row) for row in rows_b],
            "c_psd_information": [asdict(row) for row in rows_c],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


