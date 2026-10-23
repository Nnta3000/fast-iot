"""
CN: 逐迭代 trace 一个逆 OT 求解：每轮 master 的 (delta, |F|)，Separate 的违例数/最大违反，
    Select 选入 F 的边（数量、sigma 分布、是否在 S 上、每行/每列命中数），以及剪枝去掉的边。
    目的：看清楚「一个问题是怎么被解掉的、哪些边进了 F、需要多少轮」。

EN: Per-iteration trace of one inverse-OT solve: the master's (delta, |F|), the separator's
    violation count / max violation, the edges Select puts into F (count, sigma spread, on/off
    support, per-row/col hits), and what pruning drops. Goal: see how one problem is solved, which
    edges enter F, and how many rounds it takes.

用法 / EN: usage
    python -m iot_experiments.core.trace_solve --family quadratic --n 1000 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import EdgeSet, InverseOTProblem, solve_inverse_ot  # noqa: E402
from iot.exact.components import mandatory_edges  # noqa: E402
from iot.exact.master import MasterSolver  # noqa: E402
from iot.exact.oracle import scan_violations  # noqa: E402


class TracingSolver:
    """
    CN: 包住任意 MasterSolver，每轮记录 (|F|, delta)；并在 master 解出后立即复算 Separate+Select，
        记录本轮会被选入 F 的边（数量、sigma 分布、在/离支撑的比例）。
    EN: Wrap a MasterSolver; per round record (|F|, delta) and, right after the master solves, recompute
        Separate+Select to record which edges would enter F (count, sigma spread, on/off-support split).
    """

    def __init__(self, inner: MasterSolver, problem: InverseOTProblem, *, tau_sep: float, gamma: int) -> None:
        self.inner = inner
        self.problem = problem
        self.tau_sep = float(tau_sep)
        self.gamma = int(gamma)
        self.log: list[dict[str, object]] = []

    def solve(self, problem: InverseOTProblem, edges: EdgeSet, *, tie_break: str = "none"):  # noqa: ANN201
        solution = self.inner.solve(problem, edges, tie_break=tie_break)  # type: ignore[arg-type]
        scan = scan_violations(problem, solution, excluded=edges, tol=self.tau_sep, gamma=self.gamma)
        rows = np.asarray(scan.rows, dtype=np.int64)
        cols = np.asarray(scan.cols, dtype=np.int64)
        sigma = np.zeros(0, dtype=np.float64)
        if rows.size:
            cost = np.asarray(problem.features.costs(solution.theta, rows, cols), dtype=np.float64)
            sigma = solution.source_dual[rows] + solution.target_dual[cols] - cost
        sup_keys = np.sort(problem.support_rows * problem.n_target + problem.support_cols)
        on_sup = (
            np.isin(np.sort(rows * problem.n_target + cols), sup_keys) if rows.size else np.zeros(0, dtype=bool)
        )
        self.log.append({
            "F_in": int(edges.size),
            "delta": float(solution.delta),
            "violated": int(scan.violation_count),
            "maxviol": float(scan.max_violation),
            "cand": int(rows.size),
            "cand_rows": int(len(set(rows.tolist()))) if rows.size else 0,
            "cand_cols": int(len(set(cols.tolist()))) if rows.size else 0,
            "on_sup": int(on_sup.sum()) if rows.size else 0,
            "sigma_lo": float(sigma.min()) if sigma.size else 0.0,
            "sigma_hi": float(sigma.max()) if sigma.size else 0.0,
            "sigma_med": float(np.median(sigma)) if sigma.size else 0.0,
            "top": [(int(rows[k]), int(cols[k]), float(sigma[k])) for k in np.argsort(-sigma)[:5]] if sigma.size else [],
        })
        return solution


def trace_solve(*, family: str, n: int, point_dimension: int, seed: int, marginal: str,
                scan: str, solver: str, prune: str, tau_prune: float, tau_sep: float, gamma: int,
                master_time_limit: float, max_iterations: int, use_planting: bool) -> None:
    from iot.backends.cupdlpx_master import CuPdlpxMasterSolver
    from iot.exact.master import LinearMasterSolver

    if use_planting:
        from iot_experiments.core.planting import plant_instance
        planted = plant_instance(family=family, n=int(n), point_dimension=int(point_dimension), seed=int(seed))
        problem = planted.problem
        print(f"[instance] planted Monge: n_eff={planted.n_eff} m_eff={planted.m_eff} |S|={planted.support_size} "
              f"cert={planted.max_dual_violation:.1e} gap50={planted.gap_quantiles['0.50']:.2e}")
    else:
        from iot_experiments.core.families import FamilySpec, build_family
        from iot_experiments.core.marginals import marginal_spec
        from iot_experiments.core.runner import _hello_plan
        rng = np.random.default_rng(int(seed))
        features, theta_set, theta_star = build_family(
            FamilySpec(family, point_dimension=int(point_dimension)), rng, int(n), int(n))
        a = marginal_spec(marginal).weights(int(n))
        b = marginal_spec(marginal).weights(int(n))
        plan, _, _ = _hello_plan(features, theta_star, a, b, device="cpu", max_iterations=200)
        from scipy import sparse
        problem = InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))
        print(f"[instance] {family}/{marginal}: n_eff={problem.n_source} m_eff={problem.n_target} "
              f"|S|={problem.support_size}")

    if solver == "cupdlpx":
        inner = CuPdlpxMasterSolver(params={"time_sec_limit": float(master_time_limit)})
    else:
        inner = LinearMasterSolver(random_seed=int(seed))
    tracer = TracingSolver(inner, problem, tau_sep=float(tau_sep), gamma=int(gamma))

    result = solve_inverse_ot(
        problem, solver=tracer, tie_break="none", max_iterations=int(max_iterations),
        scan_backend=str(scan), validate=False, prune=(prune != "none"),
        prune_hysteresis=(prune == "hysteresis"), tau_prune=float(tau_prune),
        tol=float(tau_sep), gamma=int(gamma), random_seed=int(seed),
    )

    print(f"\n[solve] conv={result.converged} iters={result.n_iterations} delta={result.delta:.4e} "
          f"maxviol={result.final_max_violation:+.3e} |S|={problem.support_size} n={problem.n_source}")
    print(f"\n{'it':>3} {'|F|in':>7} {'violated':>12} {'cand':>8} {'onS':>6} {'offS':>6} {'rows':>7} "
          f"{'cols':>7} {'sig_med':>11} {'sig_hi':>11} {'delta':>11} {'maxviol':>11}")
    for i, rec in enumerate(tracer.log, start=1):
        print(f"{i:>3} {rec['F_in']:>7} {rec['violated']:>12} {rec['cand']:>8} {rec['on_sup']:>6} "
              f"{rec['cand']-rec['on_sup']:>6} {rec['cand_rows']:>7} {rec['cand_cols']:>7} "
              f"{rec['sigma_med']:>11.3e} {rec['sigma_hi']:>11.3e} {rec['delta']:>11.3e} {rec['maxviol']:>11.3e}")
    if tracer.log:
        print("\n[last round, top-5 selected edges by sigma] (i, j, sigma):")
        for (ri, ci, si) in tracer.log[-1]["top"]:
            print(f"    ({ri:>7}, {ci:>7})  sigma={si:+.6e}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="per-iteration trace of one inverse-OT solve")
    p.add_argument("--family", default="quadratic")
    p.add_argument("--marginal", default="uniform")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--point-dimension", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scan", default="hello", choices=("hello", "torch", "numpy"))
    p.add_argument("--solver", default="cupdlpx", choices=("cupdlpx", "highs"))
    p.add_argument("--prune", default="none", choices=("none", "tolerance", "hysteresis"))
    p.add_argument("--tau-prune", type=float, default=1e-3)
    p.add_argument("--tau-sep", type=float, default=1e-9)
    p.add_argument("--gamma", type=int, default=1)
    p.add_argument("--master-time-limit", type=float, default=60.0)
    p.add_argument("--max-iterations", type=int, default=60)
    p.add_argument("--planting", action="store_true", help="use the planted Monge generator")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    trace_solve(family=str(args.family), n=int(args.n), point_dimension=int(args.point_dimension),
                seed=int(args.seed), marginal=str(args.marginal), scan=str(args.scan),
                solver=str(args.solver), prune=str(args.prune), tau_prune=float(args.tau_prune),
                tau_sep=float(args.tau_sep), gamma=int(args.gamma),
                master_time_limit=float(args.master_time_limit), max_iterations=int(args.max_iterations),
                use_planting=bool(args.planting))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
