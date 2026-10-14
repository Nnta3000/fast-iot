"""
CN: **无 HiGHS 的证书**：GPU 循环结束后，对最终 F 做一次紧的 CuPDLPx FP64 求解，然后在全网格上
    用 FP64 计算 ε-最优界。HiGHS **只**用于离线验证，绝不进入生产路径。
EN: **HiGHS-free certificate**: after the GPU loop, take one tight CuPDLPx FP64 solve on the final
    F, then compute an epsilon-optimality bound over the whole grid in FP64. HiGHS is used *only*
    offline for validation, never in production.

CN: 界的形式（总质量 = 1）：给定 (θ, f, g)，令
      v = max_ij (f_i + g_j - c_ij(θ))       (全网格最大对偶违反, FP64)
      r = max_{(i,j) ∈ supp(π̂)} |f_i + g_j - c_ij(θ)|   (支撑上的最大残差, FP64)
    则 ⟨c(θ), π̂⟩ - OPT(θ) <= v + r（弱对偶 + 支撑分解）。这就是“把 shifted-δ 修复读成一个界”。
EN: Bound form (total mass 1): with (theta,f,g),
      v = max_ij (f_i+g_j-c_ij(theta)),  r = max_{supp(pi_hat)} |f_i+g_j-c_ij(theta)|,
    then <c(theta),pi_hat> - OPT(theta) <= v + r (weak duality + support split). This is the
    "shifted-delta repair" read as a bound.

用法 / EN: usage:
    python -m iot_experiments.gpu_bench.certify --n 3200 --family diagonal --eps 1e-6 1e-7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import CuPdlpxMasterSolver  # noqa: E402
from iot_experiments.gpu_bench.snapshots import build_hello_problem  # noqa: E402
from iot_experiments.gpu_bench.loop_diag import RecordingSolver, run_loop  # noqa: E402

_OUT = _REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "certify"


def epsilon_bound(problem, theta, source_dual, target_dual, *, block: int = 256) -> dict[str, float]:
    """CN: FP64 全网格 ε-界：v (最大对偶违反) + r (支撑最大残差)。EN: FP64 grid epsilon-bound v + r."""
    m, n = int(problem.n_source), int(problem.n_target)
    f = np.asarray(source_dual, dtype=np.float64)
    g = np.asarray(target_dual, dtype=np.float64)
    worst = -np.inf
    for start in range(0, m, block):
        stop = min(m, start + block)
        rr = np.repeat(np.arange(start, stop, dtype=np.int64), n)
        cc = np.tile(np.arange(n, dtype=np.int64), stop - start)
        cost = np.asarray(problem.features.costs(np.asarray(theta), rr, cc), dtype=np.float64)
        worst = max(worst, float(np.max(f[rr] + g[cc] - cost)))
    rows, cols = problem.support_rows, problem.support_cols
    cost_supp = np.asarray(problem.features.costs(np.asarray(theta), rows, cols), dtype=np.float64)
    residual = float(np.max(np.abs(f[rows] + g[cols] - cost_supp))) if rows.size else 0.0
    return {"v_max_dual_violation": float(worst), "r_support_residual": residual,
            "epsilon_bound": float(worst + residual)}


def certify_one(*, family: str, point_dimension: int, marginal: str, n: int, seed: int,
                eps_list: list[float], loop_cap: float, device: str) -> dict[str, object]:
    """CN: 跑 GPU 循环（不剪枝）→ 取最终 F → 每个 eps 做一次紧解 + ε-界 + 时间拆解。
    EN: Run the GPU loop (no prune) -> final F -> per eps: one tight solve + epsilon-bound + time split."""
    problem = build_hello_problem(family=family, point_dimension=int(point_dimension),
                                  marginal=marginal, n=int(n), seed=int(seed), device=str(device))
    recorder = RecordingSolver(CuPdlpxMasterSolver(params={"time_sec_limit": float(loop_cap)}))
    result = run_loop(problem, inner=recorder, prune=False, hysteresis=False,
                      max_iterations=60, scan_backend="numpy")[0]
    recorder_edges = recorder.edge_sets[-1]
    loop_delta = float(result.delta)
    loop_master = float(np.sum(recorder.seconds))
    out: dict[str, object] = {
        "family": family, "n": int(n), "seed": int(seed), "K": int(problem.n_features),
        "loop_final_F": int(recorder_edges.size), "loop_delta": loop_delta,
        "loop_master_seconds_total": loop_master, "eps": {},
    }
    for eps in eps_list:
        solver = CuPdlpxMasterSolver(params={"eps_optimal_relative": eps, "eps_feasible_relative": eps,
                                             "time_sec_limit": float(loop_cap)})
        t0 = time.perf_counter()
        sol = solver.solve(problem, recorder_edges, tie_break="none")
        final_solve_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        bound = epsilon_bound(problem, np.asarray(sol.theta), sol.source_dual, sol.target_dual)
        bound_s = time.perf_counter() - t0
        d_over_eps = float(sol.delta) / eps
        out["eps"][str(eps)] = {
            "final_solve_seconds": final_solve_s, "bound_seconds": bound_s, "delta": float(sol.delta),
            "delta_over_eps": d_over_eps, "v_max_dual_violation": bound["v_max_dual_violation"],
            "r_support_residual": bound["r_support_residual"], "epsilon_bound": bound["epsilon_bound"],
            "bound_below_loop_delta": bool(bound["epsilon_bound"] < loop_delta),
        }
        e = out["eps"][str(eps)]  # type: ignore[index]
        print(f"[certify] n={n} {family} eps={eps:.0e}: solve={final_solve_s:6.2f}s "
              f"bound={e['epsilon_bound']:.2e} (v={e['v_max_dual_violation']:.2e} r={e['r_support_residual']:.2e}) "
              f"loop_delta={loop_delta:.2e} bound<delta={e['bound_below_loop_delta']}", flush=True)
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HiGHS-free epsilon-certificate")
    p.add_argument("--n", type=int, nargs="+", default=[3200])
    p.add_argument("--families", nargs="+", default=["diagonal", "quadratic", "lowrank"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--point-dimension", type=int, default=8)
    p.add_argument("--marginal", default="gmm_floored")
    p.add_argument("--eps", type=float, nargs="+", default=[1e-6, 1e-7])
    p.add_argument("--loop-cap", type=float, default=60.0)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = [
        certify_one(family=f, point_dimension=int(args.point_dimension), marginal=str(args.marginal),
                    n=n, seed=s, eps_list=[float(e) for e in args.eps],
                    loop_cap=float(args.loop_cap), device=str(args.device))
        for f in args.families for n in args.n for s in args.seeds
    ]
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "certify.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[certify] wrote {_OUT / 'certify.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
