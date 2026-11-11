"""
CN: **EOT 循环在 planted 合成实例上** —— 有真值 `θ*` 与一个**精确最优**的观察方案。

    CN: 与之前那些"随机点"探针的区别：这里 `π̂` 是 `c(θ*)` 的 OT 最优方案（互补松弛精确成立，见
        `harness/lib/planting.py`），所以实例有真答案：`Ψ_F` 在该点达到其最优，且 `π` 应当能识别它。

    CN: 规模按用户的明确要求：`m ∈ {500, 1000}`，不是 20。之前 m=20 的结论（`|F|`→98%）是在**无结构**
        的随机点上测的，不能推广。

    CN: `ε` **绝对且冻结**：`ε = rel_eps * std(c(θ_ref))`，`θ_ref` 为均匀 θ。

EN: **EOT's loop on a planted synthetic instance** -- with a ground truth `theta*` and an **exactly optimal**
    observed plan.

    Unlike the earlier "random points" probes, here `pi_hat` is the OT-optimal plan for `c(theta*)`
    (complementary slackness exact, see `harness/lib/planting.py`), so the instance has a known answer.

    Sizes per the explicit requirement: `m in {500, 1000}`, not 20. The earlier m=20 result (`|F|` -> 98 %) was
    measured on **structureless** random points and does not generalise.

    `epsilon` is **absolute and frozen**: `eps = rel_eps * std(c(theta_ref))` with `theta_ref` uniform.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.parameters import cost_std  # noqa: E402
from iot.entropic.soft_loop import solve_entropic_soft  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.lib.planting import plant_instance  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402

SIZES: tuple[int, ...] = (500, 1000)
REL_EPS_GRID: tuple[float, ...] = (0.25, 0.1)
POINT_DIMENSION: int = 8
ETA_MASS: float = 1e-3
"""CN: 停止判据：逐点超出质量分数 `max_i M_out[i]/a_i <= eta_mass`。EN: per-node excess mass fraction."""

PROVENANCE: dict[str, object] = {
    "purpose": "EOT's loop on PLANTED instances: what final |F| does the loop produce, and is theta recovered?",
    "instance": "harness/lib/planting.plant_instance, family=diagonal, Monge, exactly-optimal pi_hat under theta*",
    "sizes": list(SIZES),
    "eps": "absolute and frozen: rel_eps * std(c(theta_ref=uniform))",
    "stop": "per-node excess mass fraction <= eta_mass",
    "note": "the loop's final |F| is the quantity that decides whether a restricted edge set is worth having",
}


@dataclasses.dataclass(frozen=True)
class PlantedSoftConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    section = "eot_planted"
    study: str = "eot_planted"
    point_dimension: int = POINT_DIMENSION
    max_rounds: int = 25
    max_outer: int = 60
    gamma: int = 1

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--point-dimension", type=int, default=POINT_DIMENSION)
        parser.add_argument("--max-rounds", type=int, default=25)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(PlantedSoftConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = PlantedSoftConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("EOT loop on planted instances (Monge, exactly-optimal pi_hat)")
    log.line(f"sizes={SIZES} rel_eps={REL_EPS_GRID} dimension={config.point_dimension} "
             f"eta_mass={ETA_MASS:g} gamma={config.gamma}")

    rows: list[dict[str, object]] = []
    for size in SIZES:
        planted = plant_instance(family="diagonal", n=int(size),
                                 point_dimension=int(config.point_dimension), seed=0)
        problem = planted.problem
        # CN: **注意** planting 剪掉空的目标列，所以 `m_eff` 与 `n_eff` 不同（m_eff=70, n_eff=120 在 n=120 时）。
        #     `features` 的真实形状是 `(m_eff, n_eff)` —— 之前把两者搞混导致越界。
        # EN: planting prunes empty target columns, so `m_eff != n_eff` (70 vs 120 at n=120). The real feature
        #     shape is `(m_eff, n_eff)`; mixing the two caused an out-of-bounds.
        m_eff = int(planted.problem.features.n_source)
        n_eff = int(planted.problem.features.n_target)
        theta_star: Float[np.ndarray, "k"] = np.asarray(planted.theta_star, dtype=np.float64)
        k = int(theta_star.size)
        # CN: 代价尺度在**均匀 θ** 处取一次（冻结 ε）。
        # EN: the cost scale is taken once at **uniform theta** (frozen epsilon).
        theta_ref = np.full(k, 1.0 / k, dtype=np.float64)
        rows_all = np.repeat(np.arange(m_eff), n_eff)
        cols_all = np.tile(np.arange(n_eff), m_eff)
        cost_ref = np.asarray(problem.features.costs(theta_ref, rows_all, cols_all),
                              dtype=np.float64)
        scale = cost_std(cost_ref)
        grid = int(m_eff) * int(n_eff)
        log.line(f"n={size}: m_eff={m_eff} n_eff={n_eff} grid={grid} K={k} support={planted.support_size} "
                 f"max_dual_violation={planted.max_dual_violation:.2e} cost_std={scale:.4e}")
        for rel_eps in REL_EPS_GRID:
            eps = float(rel_eps) * scale
            started = time.perf_counter()
            try:
                result = solve_entropic_soft(problem, epsilon_absolute=eps, gamma=int(config.gamma),
                                             eta_mass=ETA_MASS, max_rounds=int(config.max_rounds),
                                             max_outer=int(config.max_outer))
            except Exception as error:  # noqa: BLE001 - reported, never substituted
                rows.append({"n": int(size), "m_eff": m_eff, "n_eff": n_eff, "rel_eps": float(rel_eps),
                             "status": f"FAILED: {type(error).__name__}: {str(error)[:120]}"})
                log.line(f"  rel_eps={rel_eps}: FAILED {type(error).__name__}: {str(error)[:100]}")
                continue
            elapsed = time.perf_counter() - started
            final = int(result.final_edges.size)
            distance = float(np.abs(np.asarray(result.theta, dtype=np.float64) - theta_star).sum())
            row = {
                "n": int(size), "m_eff": m_eff, "n_eff": n_eff, "K": k, "rel_eps": float(rel_eps),
                "eps_absolute": eps, "status": result.stall_status, "converged": bool(result.converged),
                "rounds": int(len(result.rounds)),
                "final_F": final, "grid": grid, "final_F_fraction": round(final / grid, 5),
                "final_edges_per_node": round(final / float(m_eff + n_eff), 3),
                "exact_loop_edges_per_node": 5.57,
                "theta_l1_to_star": round(distance, 5),
                "seconds": round(elapsed, 1),
                "inner_sweeps_total": int(sum(r.inner_sweeps for r in result.rounds)),
                "per_round": [{"round": r.index, "F": r.f_size, "psi": r.psi,
                               "gap": r.frank_wolfe_gap, "frac": r.max_excess_fraction,
                               "sweeps": r.inner_sweeps, "s": round(r.seconds, 2)}
                              for r in result.rounds],
            }
            rows.append(row)
            log.line(f"  rel_eps={rel_eps}: {result.stall_status} rounds={row['rounds']} "
                     f"|F|={final} ({100 * final / grid:.2f}% of grid, "
                     f"{row['final_edges_per_node']:.2f}/node) "
                     f"L1(theta,theta*)={distance:.4f} sweeps={row['inner_sweeps_total']} {elapsed:.1f}s")
            if result.rounds:
                r0, rl = result.rounds[0], result.rounds[-1]
                log.line(f"     round1 |F|={r0.f_size} frac={r0.max_excess_fraction:.2e} "
                         f"-> round{len(result.rounds)} |F|={rl.f_size} frac={rl.max_excess_fraction:.2e}")

    payload = {"rows": rows, "provenance": PROVENANCE, "eta_mass": ETA_MASS,
               "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                          for key, v in dataclasses.asdict(config).items()}}
    (run_dir / "planted.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                          encoding="utf-8")
    log.line(f"wrote {run_dir / 'planted.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
