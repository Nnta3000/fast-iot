"""
CN: 「更难的代价族」基准的 runner。三条臂：
    1. `families` — 在非可分对角族（quadratic / bilinear-signed / lowrank）上比较
       exact、cutting-plane、subgradient（entropic 可选），报告每次求解的 OT/LP 工作量与
       L0/G/R；
    2. `psd` — bilinear-PSD（谱面体）上比较 subgradient、theta-Sinkhorn、PSD 矩拟合器，以及
       小 n 的 exact SDP 参考；
    3. `noisy` — 抽样方案（PROTOCOL R1）上比较一阶方法，报告 R 相对 uniform 参考。
EN: Runner for the "harder cost families" benchmark. Three arms:
    1. `families` -- on non-separable-diagonal families (quadratic / bilinear-signed / lowrank)
       compare exact, cutting-plane and subgradient (entropic optional), reporting the OT/LP
       work per solve plus L0/G/R;
    2. `psd` -- bilinear-PSD (spectrahedron): subgradient, theta-Sinkhorn, PSD moment fitter,
       plus a small-n exact-SDP reference;
    3. `noisy` -- sampled plans (PROTOCOL R1): first-order methods, reporting R vs the uniform
       reference.

CN: 这不是共享 PROTOCOL 的一部分；它是对共享基准的一层补充，回答「对角代价之外会怎样」。
EN: This is not part of the shared PROTOCOL; it is a supplement that answers "what happens
    beyond diagonal costs".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import (  # noqa: E402
    CuPdlpxMasterSolver,
    CuttingPlaneConfig,
    InverseOTError,
    InverseOTProblem,
    RegularizedMasterSolver,
    SdpMasterSolver,
    fit_psd_moment_entropic,
    moment_from_plan,
    solve_cutting_plane,
    solve_inverse_ot,
)
from iot.alternatives.sinkhorn_theta import fit_theta_sinkhorn  # noqa: E402
from iot.alternatives.subgradient import SubgradientConfig, fit_theta_subgradient  # noqa: E402
from iot_experiments.benchmark.hard_instances import (  # noqa: E402
    LP_FAMILIES,
    HardInstance,
    build_hard_instance,
    score_hard,
    uniform_theta,
)

ARM_CHOICES: tuple[str, ...] = ("families", "psd", "noisy")
FAMILY_METHODS: tuple[str, ...] = ("exact", "cutting-plane", "subgradient", "entropic", "cupdlpx")
PSD_METHODS: tuple[str, ...] = ("exact-sdp", "subgradient", "sinkhorn", "moment-fitter")
NOISY_METHODS: tuple[str, ...] = ("exact", "cutting-plane", "subgradient", "sinkhorn")
VALIDATE_MAX_ELEMENTS: int = 40_000
DEFAULT_L0_TOL: float = 1e-9


@dataclass(frozen=True)
class MethodRow:
    """
    CN: 一次运行的一行。`unit` 说明 `iterations` 是什么单位（不同方法不可直接比较）：
        exact 是主 LP 行生成轮数；cutting-plane 是精确 OT 求解次数；subgradient 是梯度步数；
        sinkhorn 是外层退火步数。
    EN: One run. `unit` says what `iterations` counts (not comparable across methods):
        exact = master LP row-generation rounds; cutting-plane = exact-OT solves;
        subgradient = gradient steps; sinkhorn = outer annealing steps.
    """

    arm: str
    family: str
    n: int
    d: int
    k: int
    seed: int
    method: str
    config: str
    unit: str
    converged: bool
    iterations: int
    wall_time: float
    per_iteration: float
    l0: float | None
    l0_initial: float | None
    G: float | None
    R: float | None
    is_optimal: bool | None
    error: str | None = None
    extra: dict = field(default_factory=dict)


def _row(
    *, arm: str, instance: HardInstance, method: str, config: str, unit: str,
    converged: bool, iterations: int, wall: float, theta: np.ndarray | None,
    l0_initial: float, error: str | None = None, extra: dict | None = None,
    observed_plan: np.ndarray | None = None, true_plan: np.ndarray | None = None,
) -> MethodRow:
    """CN: 统一记录一行并离线评分。EN: Record one row and score it offline."""
    l0: float | None = None
    G: float | None = None
    R: float | None = None
    is_optimal: bool | None = None
    if theta is not None:
        try:
            score = score_hard(
                instance, theta,
                observed_plan=observed_plan, true_plan=true_plan,
            )
            l0 = float(score["l0"])
            G = float(score["G"])
            R = float(score["R"])
            is_optimal = score["is_optimal"]
        except Exception as exc:  # pragma: no cover - evaluator failure is reported
            error = f"{error}; score failed: {exc}" if error else f"score failed: {exc}"
    return MethodRow(
        arm=arm,
        family=instance.family,
        n=int(instance.n),
        d=int(instance.point_dimension),
        k=int(instance.k),
        seed=int(instance.seed),
        method=method,
        config=config,
        unit=unit,
        converged=bool(converged),
        iterations=int(iterations),
        wall_time=float(wall),
        per_iteration=float(wall) / max(1, int(iterations)),
        l0=l0,
        l0_initial=float(l0_initial),
        G=G,
        R=R,
        is_optimal=is_optimal,
        error=error,
        extra=extra or {},
    )


def run_family_method(
    instance: HardInstance,
    method: str,
    *,
    l0_initial: float,
    max_iterations: int,
    subgradient_lr: float,
    subgradient_rule: str,
    entropic_outer: int,
) -> MethodRow:
    """CN: 在 LP 族实例上运行一个方法。EN: Run one method on an LP-family instance."""
    problem = instance.problem
    validate = int(instance.n) * int(instance.n) <= VALIDATE_MAX_ELEMENTS
    if method == "exact":
        started = time.perf_counter()
        result = solve_inverse_ot(
            problem,
            solver=RegularizedMasterSolver(epsilon=1e-4),
            max_iterations=int(max_iterations),
            validate=validate,
        )
        return _row(
            arm="families", instance=instance, method="exact",
            config="regularized-eps1e-4", unit="master rounds",
            converged=bool(result.converged), iterations=int(result.n_iterations),
            wall=time.perf_counter() - started, theta=result.theta,
            l0_initial=l0_initial,
        )
    if method == "cutting-plane":
        started = time.perf_counter()
        result = solve_cutting_plane(
            problem,
            config=CuttingPlaneConfig(
                master_mode="max_margin", normalize_cuts=True,
                tol=DEFAULT_L0_TOL, max_iterations=int(max_iterations),
            ),
        )
        return _row(
            arm="families", instance=instance, method="cutting-plane",
            config="max_margin,normalized", unit="OT solves",
            converged=bool(result.converged), iterations=int(result.n_oracle_calls),
            wall=time.perf_counter() - started, theta=result.theta,
            l0_initial=l0_initial,
        )
    if method == "subgradient":
        started = time.perf_counter()
        result = fit_theta_subgradient(
            problem,
            config=SubgradientConfig(
                step_rule=subgradient_rule,  # type: ignore[arg-type]
                learning_rate=float(subgradient_lr),
                max_iterations=int(max_iterations), tol=DEFAULT_L0_TOL,
            ),
        )
        return _row(
            arm="families", instance=instance, method="subgradient",
            config=f"{subgradient_rule},lr={subgradient_lr}", unit="grad steps",
            converged=bool(result.converged), iterations=len(result.iterations),
            wall=time.perf_counter() - started, theta=result.theta,
            l0_initial=l0_initial,
        )
    if method == "entropic":
        started = time.perf_counter()
        result = fit_theta_sinkhorn(
            problem, num_outer=int(entropic_outer), epsilon_decay=0.9,
            epsilon_end=1e-3, learning_rate=0.05, step_rule="adam",
        )
        return _row(
            arm="families", instance=instance, method="entropic",
            config=f"adam,outer={entropic_outer}", unit="outer steps",
            converged=False, iterations=len(result.iterations),
            wall=time.perf_counter() - started, theta=result.theta,
            l0_initial=l0_initial, error="no convergence criterion (smooth surrogate)",
        )
    if method == "cupdlpx":
        # CN: 一阶 master；出厂迟滞剪枝不适用（见 CUPDLPX_MASTER_REPORT.md），故 prune=False。
        # EN: First-order master; the shipped hysteresis pruning does not apply (see
        #     CUPDLPX_MASTER_REPORT.md), so prune=False.
        started = time.perf_counter()
        result = solve_inverse_ot(
            problem,
            solver=CuPdlpxMasterSolver(),
            tie_break="none",
            max_iterations=int(max_iterations),
            validate=validate,
            prune=False,
        )
        return _row(
            arm="families", instance=instance, method="cupdlpx",
            config="cupdlpx,prune=off", unit="master rounds",
            converged=bool(result.converged), iterations=int(result.n_iterations),
            wall=time.perf_counter() - started, theta=result.theta,
            l0_initial=l0_initial,
        )
    raise ValueError(f"unknown method {method!r}")


def run_families_arm(
    *, families: list[str], sizes: list[int], seeds: list[int], point_dimension: int,
    lowrank_k: int, methods: list[str], max_iterations: int, subgradient_lr: float,
    subgradient_rule: str, entropic_outer: int,
) -> list[MethodRow]:
    """CN: 遍历族 x n x seed x 方法。EN: Loop over family x n x seed x method."""
    rows: list[MethodRow] = []
    for family in families:
        if family not in LP_FAMILIES:
            raise ValueError(f"family {family!r} is not an LP family; use the psd arm")
        for n in sizes:
            for seed in seeds:
                instance = build_hard_instance(
                    int(n), family=family, seed=int(seed),
                    point_dimension=int(point_dimension), lowrank_k=int(lowrank_k),
                )
                l0_initial = float(score_hard(instance, uniform_theta(instance), check_optimality=False)["l0"])
                for method in methods:
                    try:
                        row = run_family_method(
                            instance, method, l0_initial=l0_initial,
                            max_iterations=int(max_iterations),
                            subgradient_lr=float(subgradient_lr),
                            subgradient_rule=str(subgradient_rule),
                            entropic_outer=int(entropic_outer),
                        )
                    except InverseOTError as error:
                        row = _row(
                            arm="families", instance=instance, method=method,
                            config="error", unit="-", converged=False, iterations=0,
                            wall=0.0, theta=None, l0_initial=l0_initial, error=str(error),
                        )
                    rows.append(row)
                    print(
                        f"[families] {family} n={n} seed={seed} {method}: "
                        f"conv={row.converged} it={row.iterations} l0={row.l0}",
                        flush=True,
                    )
    return rows


def run_psd_arm(
    *, sizes: list[int], seeds: list[int], point_dimension: int, methods: list[str],
    max_iterations: int, subgradient_lr: float, subgradient_rule: str,
    entropic_outer: int,
) -> list[MethodRow]:
    """CN: bilinear-PSD 臂。EN: bilinear-PSD arm."""
    rows: list[MethodRow] = []
    for n in sizes:
        for seed in seeds:
            instance = build_hard_instance(
                int(n), family="bilinear-psd", seed=int(seed),
                point_dimension=int(point_dimension),
            )
            l0_initial = float(score_hard(instance, uniform_theta(instance), check_optimality=False)["l0"])
            for method in methods:
                if method == "exact-sdp" and int(n) > 16:
                    row = _row(
                        arm="psd", instance=instance, method="exact-sdp",
                        config="skipped", unit="-", converged=False,
                        iterations=0, wall=0.0, theta=None, l0_initial=l0_initial,
                        error="skipped: SDP master is numerically brittle for n>16",
                    )
                    rows.append(row)
                    print(f"[psd] n={n} seed={seed} exact-sdp: SKIPPED (brittle)", flush=True)
                    continue
                started = time.perf_counter()
                try:
                    if method == "exact-sdp":
                        result = solve_inverse_ot(
                            instance.problem, solver=SdpMasterSolver(),
                            tie_break="none", max_iterations=int(max_iterations),
                            validate=False,
                        )
                        row = _row(
                            arm="psd", instance=instance, method="exact-sdp",
                            config="SDP tie=none", unit="master rounds",
                            converged=bool(result.converged),
                            iterations=int(result.n_iterations),
                            wall=time.perf_counter() - started, theta=result.theta,
                            l0_initial=l0_initial,
                        )
                    elif method == "subgradient":
                        result = fit_theta_subgradient(
                            instance.problem,
                            config=SubgradientConfig(
                                step_rule=subgradient_rule,  # type: ignore[arg-type]
                                learning_rate=float(subgradient_lr),
                                max_iterations=int(max_iterations), tol=DEFAULT_L0_TOL,
                            ),
                        )
                        row = _row(
                            arm="psd", instance=instance, method="subgradient",
                            config=f"{subgradient_rule},lr={subgradient_lr}", unit="grad steps",
                            converged=bool(result.converged), iterations=len(result.iterations),
                            wall=time.perf_counter() - started, theta=result.theta,
                            l0_initial=l0_initial,
                        )
                    elif method == "sinkhorn":
                        result = fit_theta_sinkhorn(
                            instance.problem, num_outer=int(entropic_outer),
                            epsilon_decay=0.9, epsilon_end=1e-3, learning_rate=0.05,
                            step_rule="adam",
                        )
                        row = _row(
                            arm="psd", instance=instance, method="sinkhorn",
                            config=f"adam,outer={entropic_outer}", unit="outer steps",
                            converged=False, iterations=len(result.iterations),
                            wall=time.perf_counter() - started, theta=result.theta,
                            l0_initial=l0_initial, error="no certificate",
                        )
                    elif method == "moment-fitter":
                        source = instance.features.source_points
                        target = instance.features.target_points
                        observed = moment_from_plan(instance.plan, source, target)
                        d = int(instance.point_dimension)
                        initial = np.eye(d, dtype=np.float64) / float(d)
                        result = fit_psd_moment_entropic(
                            source_points=source, target_points=target,
                            source_mass=instance.source_mass, target_mass=instance.target_mass,
                            observed_moment=observed, initial_matrix=initial,
                            num_outer=int(entropic_outer), sinkhorn_iterations=200,
                        )
                        theta = result.matrix.reshape(-1)
                        row = _row(
                            arm="psd", instance=instance, method="moment-fitter",
                            config=f"polyak-avg,outer={entropic_outer}", unit="outer steps",
                            converged=False, iterations=len(result.iterations),
                            wall=time.perf_counter() - started, theta=theta,
                            l0_initial=l0_initial,
                            error="moment-only; no certificate",
                        )
                    else:
                        raise ValueError(f"unknown PSD method {method!r}")
                except (InverseOTError, ValueError) as error:
                    row = _row(
                        arm="psd", instance=instance, method=method, config="error",
                        unit="-", converged=False, iterations=0, wall=0.0,
                        theta=None, l0_initial=l0_initial, error=str(error),
                    )
                rows.append(row)
                print(
                    f"[psd] n={n} seed={seed} {method}: conv={row.converged} "
                    f"it={row.iterations} l0={row.l0} err={row.error}",
                    flush=True,
                )
    return rows


def run_noisy_arm(
    *, n: int, k: int, seeds: list[int], mix_values: list[float],
    methods: list[str], max_iterations: int, subgradient_lr: float,
    subgradient_rule: str, entropic_outer: int,
) -> list[MethodRow]:
    """
    CN: 离支撑噪声臂：`pi_hat = (1-mix) pi* + mix * (a b^T)`，边际仍为均匀，但支撑变稠密，
        因此没有任何 theta 能让 pi_hat 最优 —— exact / cutting-plane 不可行，只有一阶方法能
        最小化正的 L0。这与 PROTOCOL 的抽样模型（支撑 = 置换支撑，近似平凡）形成对照。
    EN: Off-support noisy arm: `pi_hat = (1-mix) pi* + mix * (a b^T)`. Marginals stay uniform
        but the support is dense, so no theta makes pi_hat optimal: exact / cutting-plane are
        infeasible and only first-order methods minimise a positive L0. Contrast with the
        PROTOCOL sampler (support = permutation support, near-trivial).
    """
    rows: list[MethodRow] = []
    for mix in mix_values:
        for seed in seeds:
            instance = build_hard_instance(
                int(n), family="diagonal", seed=int(seed), point_dimension=int(k)
            )
            pi_star = instance.plan
            uniform = np.outer(instance.source_mass, instance.target_mass)
            pi_hat = (1.0 - float(mix)) * pi_star + float(mix) * uniform
            problem = InverseOTProblem(
                instance.features, instance.theta_set, sparse.csr_matrix(pi_hat)
            )
            uniform_row = score_hard(
                instance, uniform_theta(instance),
                observed_plan=pi_hat, true_plan=pi_star, check_optimality=False,
            )
            l0_initial = float(uniform_row["l0"])
            uniform_r = float(uniform_row["R"])
            for method in methods:
                started = time.perf_counter()
                theta = None
                converged = False
                iterations = 0
                unit = "?"
                config = ""
                error: str | None = None
                try:
                    if method == "exact":
                        result = solve_inverse_ot(
                            problem, solver=RegularizedMasterSolver(epsilon=1e-4),
                            max_iterations=int(max_iterations), validate=False,
                        )
                        theta, converged, iterations, unit = result.theta, bool(result.converged), int(result.n_iterations), "master rounds"
                        config = "regularized-eps1e-4"
                    elif method == "cutting-plane":
                        result = solve_cutting_plane(
                            problem,
                            config=CuttingPlaneConfig(
                                master_mode="max_margin", normalize_cuts=True,
                                tol=DEFAULT_L0_TOL, max_iterations=int(max_iterations),
                            ),
                        )
                        theta, converged, iterations, unit = result.theta, bool(result.converged), int(result.n_oracle_calls), "OT solves"
                        config = "max_margin,normalized"
                    elif method == "subgradient":
                        result = fit_theta_subgradient(
                            problem,
                            config=SubgradientConfig(
                                step_rule=subgradient_rule,  # type: ignore[arg-type]
                                learning_rate=float(subgradient_lr),
                                max_iterations=int(max_iterations),
                                tol=1e-7, optimal_value=0.0,
                            ),
                        )
                        theta, converged, iterations, unit = result.best_theta, bool(result.converged), len(result.iterations), "grad steps"
                        config = f"{subgradient_rule},lr={subgradient_lr}"
                    elif method == "sinkhorn":
                        result = fit_theta_sinkhorn(
                            problem, num_outer=int(entropic_outer), epsilon_decay=0.9,
                            epsilon_end=1e-3, learning_rate=0.05, step_rule="adam",
                        )
                        theta, converged, iterations, unit = result.theta, False, len(result.iterations), "outer steps"
                        config = f"adam,outer={entropic_outer}"
                        error = "no certificate"
                    else:
                        raise ValueError(f"unknown noisy method {method!r}")
                except InverseOTError as exc:
                    error = str(exc)
                wall = time.perf_counter() - started
                row = _row(
                    arm="noisy", instance=instance, method=method, config=config,
                    unit=unit, converged=bool(converged), iterations=int(iterations),
                    wall=float(wall), theta=theta, l0_initial=l0_initial, error=error,
                    extra={"mix": float(mix), "uniform_R": uniform_r},
                    observed_plan=pi_hat, true_plan=pi_star,
                )
                rows.append(row)
                print(
                    f"[noisy] mix={mix} n={instance.n} seed={seed} {method}: "
                    f"R={row.R} uniform_R={uniform_r} err={row.error}",
                    flush=True,
                )
    return rows


def _print_rows(rows: list[MethodRow]) -> None:
    header = (
        f"{'arm':>9} {'family':>16} {'n':>5} {'K':>4} {'seed':>4} {'method':>14} "
        f"{'conv':>5} {'unit':>14} {'it':>6} {'wall':>8} {'L0':>10} {'G':>10} {'R':>10} {'opt':>5}"
    )
    print(header)
    for row in rows:
        print(
            f"{row.arm:>9} {row.family:>16} {row.n:>5} {row.k:>4} {row.seed:>4} {row.method:>14} "
            f"{str(row.converged):>5} {row.unit:>14} {row.iterations:>6} {row.wall_time:>8.2f} "
            f"{('-' if row.l0 is None else f'{row.l0:.2e}'):>10} "
            f"{('-' if row.G is None else f'{row.G:.2e}'):>10} "
            f"{('-' if row.R is None else f'{row.R:.2e}'):>10} {str(row.is_optimal):>5}"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="harder-family inverse-OT benchmark")
    parser.add_argument("--arm", choices=ARM_CHOICES, default="families")
    parser.add_argument("--families", nargs="+", default=["diagonal", "quadratic", "bilinear-signed", "lowrank"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[60, 120])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--point-dimension", type=int, default=3)
    parser.add_argument("--lowrank-k", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--mix", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--subgradient-rule", choices=("constant", "inv_sqrt", "polyak"), default="inv_sqrt")
    parser.add_argument("--subgradient-learning-rate", type=float, default=0.5)
    parser.add_argument("--entropic-outer", type=int, default=40)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sizes = [int(value) for value in args.sizes]
    seeds = [int(value) for value in args.seeds]
    if args.arm == "families":
        methods = [str(v) for v in (args.methods or ["exact", "cutting-plane", "subgradient"])]
        rows = run_families_arm(
            families=[str(v) for v in args.families], sizes=sizes, seeds=seeds,
            point_dimension=int(args.point_dimension), lowrank_k=int(args.lowrank_k),
            methods=methods, max_iterations=int(args.max_iterations),
            subgradient_lr=float(args.subgradient_learning_rate),
            subgradient_rule=str(args.subgradient_rule),
            entropic_outer=int(args.entropic_outer),
        )
    elif args.arm == "psd":
        methods = [str(v) for v in (args.methods or ["subgradient", "sinkhorn", "moment-fitter"])]
        rows = run_psd_arm(
            sizes=sizes, seeds=seeds, point_dimension=int(args.point_dimension),
            methods=methods, max_iterations=int(args.max_iterations),
            subgradient_lr=float(args.subgradient_learning_rate),
            subgradient_rule=str(args.subgradient_rule),
            entropic_outer=int(args.entropic_outer),
        )
    else:
        methods = [str(v) for v in (args.methods or ["subgradient", "sinkhorn", "exact", "cutting-plane"])]
        rows = run_noisy_arm(
            n=int(args.sizes[0]), k=int(args.k), seeds=seeds,
            mix_values=[float(v) for v in args.mix], methods=methods,
            max_iterations=int(args.max_iterations),
            subgradient_lr=float(args.subgradient_learning_rate),
            subgradient_rule=str(args.subgradient_rule),
            entropic_outer=int(args.entropic_outer),
        )
    _print_rows(rows)
    payload = {
        "experiment": "harder_benchmark",
        "arm": str(args.arm),
        "config": {
            "sizes": sizes, "seeds": seeds,
            "point_dimension": int(args.point_dimension),
            "families": [str(v) for v in args.families],
            "methods": [str(v) for v in (args.methods or [])],
            "mix": [float(v) for v in args.mix],
            "max_iterations": int(args.max_iterations),
            "subgradient_rule": str(args.subgradient_rule),
            "subgradient_learning_rate": float(args.subgradient_learning_rate),
            "entropic_outer": int(args.entropic_outer),
        },
        "rows": [asdict(row) for row in rows],
    }
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
