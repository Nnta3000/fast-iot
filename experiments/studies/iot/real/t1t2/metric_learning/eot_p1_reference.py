"""
CN: **P1 —— 全网格参考解（F = 100 %）**。

    CN: 在 planted 实例上，用**全边集**解两件事，并对每个求解器分别记录：

        1. **固定 θ 的传输问题**：`min_g` 的熵 plan。两个求解器**各自独立**：
           - `newton`：`inner_f.solve_inner_on_f`（半对偶牛顿 + CG）；
           - `sinkhorn`：`inner_f.sinkhorn_sweeps_on_f`（交替缩放）。
        2. **完整 θ master**：`min_{θ∈Θ} Ψ_U(θ,f,g)`，梯度 `φ̄ − Σπφ`，停止于 Frank–Wolfe gap
           `<= 1e-4·ε`。也在两个求解器上跑，因为内层精度影响外层收敛。

    CN: 记录：收敛、边缘残差、目标值、牛顿步数、CG 迭代数、内外层扫描总数、墙钟时间。
        结果写 JSON，供 P2 / P3 作为基准引用。

    CN: **ε 绝对且冻结**：`ε = rel_eps * std(c(θ_ref))`，`θ_ref` 为均匀 θ。

EN: **P1 -- full-grid reference solution (F = 100 %).**

    On planted instances, solve two things on the full edge set, recording each solver separately:

      1. the **fixed-theta transport problem**: the entropic plan, with two **independent** solvers:
         `newton` (`inner_f.solve_inner_on_f`, semi-dual Newton + CG) and `sinkhorn`
         (`inner_f.sinkhorn_sweeps_on_f`, alternating scaling);
      2. the **full theta master**: `min_{theta in Theta} Psi_U(theta,f,g)`, gradient `phi_bar - sum pi phi`,
         stopping on the Frank-Wolfe gap `<= 1e-4 * eps`. Run with both inner solvers, because inner accuracy
         drives outer convergence.

    Records convergence, marginal residuals, objective value, Newton steps, CG iterations, inner sweep totals
    and wall time. Written to JSON as the P2/P3 reference.

    `eps` is **absolute and frozen**: `eps = rel_eps * std(c(theta_ref))` with `theta_ref` uniform.
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

from iot.core.edges import EdgeSet  # noqa: E402
from iot.core.parameters import cost_std  # noqa: E402
from iot.entropic.inner_f import sinkhorn_sweeps_on_f, solve_inner_on_f  # noqa: E402
from iot.entropic.soft_master import soft_master  # noqa: E402
from experiments.harness.config import RunConfig, build_parser  # noqa: E402
from experiments.harness.lib.planting import plant_instance  # noqa: E402
from experiments.harness.records import RunLog  # noqa: E402

SIZES: tuple[int, ...] = (120, 300)
REL_EPS_GRID: tuple[float, ...] = (0.25, 0.1, 0.05)
POINT_DIMENSION: int = 8

PROVENANCE: dict[str, object] = {
    "priority": "P1 — full-grid reference (F = 100%)",
    "instance": "planting.plant_instance family=diagonal (Monge, exactly-optimal pi_hat under theta*)",
    "solvers": ["newton (solve_inner_on_f)", "sinkhorn (sinkhorn_sweeps_on_f)"],
    "eps": "absolute, frozen: rel_eps * std(c(theta_ref=uniform))",
    "parts": ["fixed-theta transport at theta*", "full theta master (soft_master)"],
    "note": "this is the reference that P2 (how small can F be) and P3 (adaptive vs oracle) compare against",
}


@dataclasses.dataclass(frozen=True)
class P1Config(RunConfig):
    """CN: 旋钮。EN: knobs."""

    section = "eot_p1"
    study = "eot_p1"
    point_dimension: int = POINT_DIMENSION
    max_outer: int = 60

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--point-dimension", type=int, default=POINT_DIMENSION)
        parser.add_argument("--max-outer", type=int, default=60)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(P1Config)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = P1Config.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("P1: full-grid reference (F = 100%) on planted instances")
    log.line(f"sizes={SIZES} rel_eps={REL_EPS_GRID} dimension={config.point_dimension}")

    rows: list[dict[str, object]] = []
    for size in SIZES:
        planted = plant_instance(family="diagonal", n=int(size),
                                 point_dimension=int(config.point_dimension), seed=0)
        problem = planted.problem
        m_eff = int(problem.features.n_source)
        n_eff = int(problem.features.n_target)
        k = int(planted.theta_star.size)
        theta_star: Float[np.ndarray, "k"] = np.asarray(planted.theta_star, dtype=np.float64)
        a: Float[np.ndarray, "m"] = np.asarray(problem.resolved_source_mass, dtype=np.float64)
        b: Float[np.ndarray, "n"] = np.asarray(problem.resolved_target_mass, dtype=np.float64)
        rows_all = np.repeat(np.arange(m_eff), n_eff)
        cols_all = np.tile(np.arange(n_eff), m_eff)
        full = EdgeSet.from_arrays(rows_all, cols_all, n_target=n_eff)
        grid = int(m_eff) * int(n_eff)
        theta_ref = np.full(k, 1.0 / k, dtype=np.float64)
        scale = cost_std(np.asarray(problem.features.costs(theta_ref, rows_all, cols_all),
                                    dtype=np.float64))
        log.line(f"--- n={size}: m_eff={m_eff} n_eff={n_eff} grid={grid} K={k} "
                 f"support={planted.support_size} cost_std={scale:.4e} ---")

        for rel_eps in REL_EPS_GRID:
            eps = float(rel_eps) * scale
            cost_star = np.asarray(problem.features.costs(theta_star, rows_all, cols_all),
                                   dtype=np.float64)

            # ---- part 1: fixed-theta transport, two independent solvers ----
            transport: dict[str, object] = {}
            started = time.perf_counter()
            try:
                newton = solve_inner_on_f(full, cost_star, a, b, epsilon=eps,
                                          marginal_tolerance=1e-10, max_newton=60,
                                          cg_iterations=200)
                transport["newton"] = {
                    "status": "ok" if newton.converged else "not converged",
                    "converged": bool(newton.converged),
                    "marginal_error": float(newton.marginal_error),
                    "objective": float(newton.objective),
                    "newton_steps": int(newton.newton_steps),
                    "cg_iterations": int(newton.cg_iterations),
                    "hvps": int(newton.hessian_vector_products),
                    "seconds": round(time.perf_counter() - started, 2),
                }
            except Exception as error:  # noqa: BLE001
                transport["newton"] = {"status": f"FAILED: {type(error).__name__}: {str(error)[:100]}",
                                       "seconds": round(time.perf_counter() - started, 2)}
            started = time.perf_counter()
            try:
                sink = sinkhorn_sweeps_on_f(full, cost_star, a, b, epsilon=eps,
                                            marginal_tolerance=1e-10, max_sweeps=400000)
                transport["sinkhorn"] = {
                    "status": "ok" if sink.converged else "not converged",
                    "converged": bool(sink.converged),
                    "marginal_error": float(sink.marginal_error),
                    "objective": float(sink.objective),
                    "sweeps": int(sink.newton_steps),
                    "seconds": round(time.perf_counter() - started, 2),
                }
            except Exception as error:  # noqa: BLE001
                transport["sinkhorn"] = {"status": f"FAILED: {type(error).__name__}: {str(error)[:100]}",
                                         "seconds": round(time.perf_counter() - started, 2)}
            # CN: 两个求解器应给出同一个 plan；记录差异以证明独立性不是借口。
            # EN: the two solvers should agree on the plan; record the gap so independence is not an excuse.
            if "newton" in transport and "sinkhorn" in transport:
                ok_n = bool(transport["newton"].get("converged"))
                ok_s = bool(transport["sinkhorn"].get("converged"))
                if ok_n and ok_s:
                    d_newton = np.abs(np.asarray(newton.plan_on_f) - np.asarray(sink.plan_on_f)).max()
                    d_obj = abs(float(newton.objective) - float(sink.objective))
                    transport["newton_vs_sinkhorn"] = {"max_abs_plan_diff": float(d_newton),
                                                       "abs_objective_diff": float(d_obj)}

            # ---- part 2: full theta master ----
            started = time.perf_counter()
            try:
                master = soft_master(problem, full, epsilon=eps, max_outer=int(config.max_outer),
                                     inner_tolerance=1e-8)
                theta_err = float(np.abs(np.asarray(master.theta, dtype=np.float64) - theta_star).sum())
                master_entry = {
                    "status": "ok" if master.converged else "not converged",
                    "converged": bool(master.converged),
                    "psi": float(master.psi),
                    "frank_wolfe_gap": float(master.frank_wolfe_gap),
                    "fw_threshold": float(1e-4 * eps),
                    "outer_iterations": int(master.outer_iterations),
                    "inner_sweeps": int(master.inner_sweeps),
                    "inner_marginal_error": float(master.inner_marginal_error),
                    "theta_l1_to_star": round(theta_err, 6),
                    "theta": [round(float(v), 8) for v in master.theta],
                    "seconds": round(time.perf_counter() - started, 2),
                }
            except Exception as error:  # noqa: BLE001
                master_entry = {"status": f"FAILED: {type(error).__name__}: {str(error)[:100]}",
                                "seconds": round(time.perf_counter() - started, 2)}

            row = {"n": int(size), "m_eff": m_eff, "n_eff": n_eff, "grid": grid, "K": k,
                   "rel_eps": float(rel_eps), "eps_absolute": eps,
                   "transport_at_theta_star": transport, "full_theta_master": master_entry}
            rows.append(row)
            tr_n = transport.get("newton", {})
            tr_s = transport.get("sinkhorn", {})
            log.line(f"  rel_eps={rel_eps:<5} transport: newton "
                     f"{('conv' if tr_n.get('converged') else 'NO')} err={tr_n.get('marginal_error', float('nan')):.1e} "
                     f"steps={tr_n.get('newton_steps')} cg={tr_n.get('cg_iterations')} {tr_n.get('seconds')}s | "
                     f"sinkhorn {('conv' if tr_s.get('converged') else 'NO')} "
                     f"sw={tr_s.get('sweeps')} {tr_s.get('seconds')}s")
            log.line(f"      master: {master_entry['status']} psi={master_entry.get('psi', float('nan')):.6e} "
                     f"gap={master_entry.get('frank_wolfe_gap', float('nan')):.1e} "
                     f"L1(theta,theta*)={master_entry.get('theta_l1_to_star')} "
                     f"outer={master_entry.get('outer_iterations')} {master_entry.get('seconds')}s")

    payload = {"rows": rows, "provenance": PROVENANCE,
               "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                          for key, v in dataclasses.asdict(config).items()}}
    (run_dir / "p1_reference.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                              encoding="utf-8")
    log.line(f"wrote {run_dir / 'p1_reference.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
