"""
EN: **Gate B reference study -- an independent answer to "does the joint formulation have the right optimum?"**

    Runs the reduced joint objective (`iot.entropic.joint_reference`, written from the boxed formulas with
    SciPy's `logsumexp` + SLSQP, sharing nothing with the custom Newton-CG) from several starting points,
    then compares against the nested reference and against the frozen custom solver, and enumerates all 15
    faces of the 4-simplex independently.

    Every configuration value is written to the JSON record, because two scripts using the same experiment
    "name" is not evidence that they solve the same problem.

CN: **Gate B 参考研究 —— 独立回答"联合形式的最优点对不对"。**

    用 `iot.entropic.joint_reference`（直接按公式实现，SciPy SLSQP，与自定义牛顿-CG 无共享）从多个起点
    求解，再与嵌套参考解、冻结的自定义解对比，并独立枚举 4-单纯形的全部 15 个面。
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.parameters import cost_std  # noqa: E402
from iot.entropic.joint_reference import (  # noqa: E402
    ReducedProblem,
    evaluate,
    face_enumeration,
    marginals,
    solve,
)
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.lib.planting import plant_instance  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402

REL_EPS: float = 0.5
POINT_DIMENSION: int = 4
INSTANCE_SIZE: int = 24

PROVENANCE: dict[str, object] = {
    "question": "does the joint reduced objective have the nested reference's optimum, independently of Newton-CG?",
    "objective": "Psi_red(theta,g) = <phi_bar,theta> - b.g + eps*sum_i a_i*(1 - log a_i + L_i)",
    "theta_constraint": "equality sum(theta)=1 with bounds theta>=0, real constraint in SLSQP (no elimination)",
    "g_gauge": "g_0 = 0 by parameterising only g_1..g_{n-1}; d/dg_0 is NOT subtracted from the block",
    "independence": "no import of iot.entropic.joint_master; separate objective, gradients, optimiser, enumeration",
    "eps": "absolute, frozen: rel_eps * std(c(theta_ref = uniform))",
}


@dataclasses.dataclass(frozen=True)
class ReferenceConfig(RunConfig):
    """EN: knobs. CN: 旋钮。"""

    section = "eot_joint_reference"
    study = "eot_joint_reference"
    point_dimension: int = POINT_DIMENSION
    instance_size: int = INSTANCE_SIZE
    rel_eps: float = REL_EPS

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """EN: only this study's flags. CN: 只加本脚本特有的参数。"""
        parser.add_argument("--point-dimension", type=int, default=POINT_DIMENSION)
        parser.add_argument("--instance-size", type=int, default=INSTANCE_SIZE)
        parser.add_argument("--rel-eps", type=float, default=REL_EPS)


def _git_sha() -> str:
    """EN: the commit this record belongs to. CN: 本记录对应的提交。"""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=str(_REPO_ROOT)).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = build_parser(ReferenceConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = ReferenceConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("Gate B independent reference (no Newton-CG)")
    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem={preflight.mem_available_gb:.1f} GB git={_git_sha()}")

    planted = plant_instance(family="diagonal", n=int(config.instance_size),
                             point_dimension=int(config.point_dimension), seed=0)
    problem = planted.problem
    m = int(problem.features.n_source)
    n = int(problem.features.n_target)
    k = int(planted.theta_star.size)
    rows = np.repeat(np.arange(m), n)
    cols = np.tile(np.arange(n), m)
    features = np.asarray(problem.features.features(rows, cols), dtype=np.float64)
    a: Float[np.ndarray, "m"] = np.asarray(problem.resolved_source_mass, dtype=np.float64)
    b: Float[np.ndarray, "n"] = np.asarray(problem.resolved_target_mass, dtype=np.float64)
    support_rows, support_cols, support_mass = problem.support()
    phi_bar: Float[np.ndarray, "k"] = (np.asarray(support_mass, dtype=np.float64)[:, None]
                                       * np.asarray(problem.features.features(support_rows, support_cols),
                                                    dtype=np.float64)).sum(axis=0)
    scale = float(cost_std(features @ np.full(k, 1.0 / k)))
    epsilon = float(config.rel_eps) * scale
    reduced = ReducedProblem(features, np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64),
                             a, b, phi_bar, epsilon)
    configuration = {
        "instance": {"family": "diagonal", "size": int(config.instance_size),
                     "point_dimension": int(config.point_dimension), "seed": 0},
        "n_source": m, "n_target": n, "n_features": k, "grid_edges": int(m * n),
        "sum_source_mass": float(a.sum()), "sum_target_mass": float(b.sum()),
        "source_mass_min": float(a.min()), "target_mass_min": float(b.min()),
        "theta_star": [round(float(v), 10) for v in planted.theta_star],
        "cost_std_uniform_theta": scale, "rel_eps": float(config.rel_eps), "epsilon": epsilon,
        "theta_constraint": "sum(theta)=1, theta>=0", "g_gauge": "g_0 = 0",
        "edge_set": "full grid",
    }
    log.line(f"m={m} n={n} K={k} grid={m * n} eps={epsilon:.10e} (rel {config.rel_eps} of {scale:.6e})")

    # EN: starting points -- the nested answer, the custom solver's stalled answer, an interior point,
    #     and a different feasible point. None of them comes from the custom solver's initialisation.
    # CN: 起点 —— 嵌套解、自定义解所处点、内点、另一个可行点。
    nested_theta = np.array([0.48097171, 0.49665787, 0.02237042, 0.0])
    nested_theta = nested_theta / nested_theta.sum()
    stalled_theta = np.array([0.493804, 0.506196, 0.0, 0.0])
    stalled_theta = stalled_theta / stalled_theta.sum()
    interior_theta = np.full(k, 1.0 / k)
    other_theta = np.array([0.70, 0.10, 0.15, 0.05], dtype=np.float64)
    starts = [("nested_reference", nested_theta), ("custom_stalled", stalled_theta),
              ("interior_uniform", interior_theta), ("other_feasible", other_theta)]

    runs: list[dict[str, object]] = []
    for label, theta_start in starts:
        log.line(f"  start={label}: theta={np.round(theta_start, 6).tolist()}")
        started = time.perf_counter()
        run = solve(reduced, theta_start, np.zeros(n), label=label)
        seconds = time.perf_counter() - started
        runs.append({
            "start_label": label,
            "start_theta": [round(float(v), 10) for v in theta_start],
            "status": int(run.status), "message": run.message, "iterations": int(run.iterations),
            "psi": float(run.psi), "theta": [round(float(v), 10) for v in run.theta],
            "theta_positive": [int(v) for v in np.flatnonzero(run.theta > 1e-7)],
            "grad_theta": [round(float(v), 8) for v in run.grad_theta],
            "simplex_multiplier": float(run.simplex_multiplier),
            "simplex_kkt_violation": float(run.simplex_kkt_violation),
            "row_marginal_error": float(run.row_marginal_error),
            "col_marginal_error": float(run.col_marginal_error),
            "grad_g_max_abs": float(np.max(np.abs(run.grad_g))),
            "theta_min": float(run.theta.min()), "theta_sum": float(run.theta.sum()),
            "seconds": round(seconds, 3),
        })
        log.line(f"    psi={run.psi:.12e} kkt={run.simplex_kkt_violation:.2e} "
                 f"colres={run.col_marginal_error:.2e} status={run.status} {seconds:.2f}s")
        log.line(f"    theta={np.round(run.theta, 8).tolist()}")

    log.line("  enumerating all faces of the simplex ...")
    faces = face_enumeration(reduced, theta_start=interior_theta)
    for record in faces:
        log.line(f"    face={record['face']} psi={record['psi']:.10e} "
                 f"excl_mult={record['excluded_multipliers']} ok={record['excluded_kkt_ok']}")

    best = faces[0]
    payload = {
        "git_sha": _git_sha(), "provenance": PROVENANCE, "configuration": configuration,
        "runs": runs, "face_enumeration": faces,
        "best_kkt_face": best,
        "preflight": preflight.to_dict(),
        "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                   for key, v in dataclasses.asdict(config).items()},
    }
    (run_dir / "joint_reference.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                                  encoding="utf-8")
    log.line(f"wrote {run_dir / 'joint_reference.json'}")
    log.line(f"BEST psi={best['psi']:.12e} theta={best['theta']}")
    log.close()
    _ = evaluate, marginals
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
