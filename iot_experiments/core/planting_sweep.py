"""
CN: planting（Monge 问题）的**参数扫描**驱动：扫 (family, d, n, g 分布, theta* 位置)，
    记录证书、n_eff/m_eff、gap 分位数，以及循环是否收敛（分别用 prune=False 与默认设置）。
EN: **Parameter sweep** for planting (Monge problems): vary (family, d, n, g profile, theta* location),
    recording the certificate, n_eff/m_eff, gap quantiles, and whether the loop converges (both with
    prune=False and with the default settings).

用法 / EN: usage
    python -m iot_experiments.core.planting_sweep --out iot_experiments/core/out/planting
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import LinearMasterSolver, solve_inverse_ot  # noqa: E402
from iot.backends.cupdlpx_master import CuPdlpxMasterSolver  # noqa: E402
from iot_experiments.core.planting import plant_instance  # noqa: E402


def _solve(planted, *, strategy: str, scan: str, solver: str, tie_break: str,
           epsilon: float, tau_prune: float, tol_sep: float, gamma: int,
           master_time_limit: float, max_iterations: int) -> dict[str, object]:
    prune = strategy in ("tolerance", "hysteresis")
    if solver == "cupdlpx":
        master = CuPdlpxMasterSolver(params={"time_sec_limit": float(master_time_limit)},
                                     epsilon=float(epsilon))
    else:
        master = LinearMasterSolver(random_seed=0)
    t0 = time.perf_counter()
    result = solve_inverse_ot(
        planted.problem, solver=master, tie_break=str(tie_break),
        max_iterations=int(max_iterations), scan_backend=str(scan), validate=True,
        prune=bool(prune), prune_hysteresis=(strategy == "hysteresis"),
        tau_prune=float(tau_prune), tol=float(tol_sep), gamma=int(gamma),
    )
    return {
        "converged": bool(result.converged), "iterations": int(result.n_iterations),
        "delta": float(result.delta), "final_max_violation": float(result.final_max_violation),
        "is_optimal": None if result.is_optimal is None else bool(result.is_optimal),
        "max_support_over_m_plus_n": float(
            max((rec.support_size for rec in result.iterations), default=result.support_size)
        ) / float(planted.n_eff + planted.m_eff),
        "final_support_over_m_plus_n": float(result.support_size) / float(planted.n_eff + planted.m_eff),
        "peak_memory_mb": float(max((getattr(rec, "peak_memory_mb", 0.0) for rec in result.iterations), default=0.0)),
        "solve_seconds": float(time.perf_counter() - t0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="planting (Monge) parameter sweep")
    parser.add_argument("--families", nargs="+", default=["diagonal", "quadratic", "lowrank"])
    parser.add_argument("--dims", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--g-profiles", nargs="+", default=["zero", "random"])
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--scan", default="numpy", choices=("numpy", "hello", "torch"),
                        help="separation backend (hello/torch are GPU/linear-memory)")
    parser.add_argument("--solver", default="highs", choices=("highs", "cupdlpx"),
                        help="master solver (cupdlpx is the GPU first-order master)")
    parser.add_argument("--prune-strategy", default="none", choices=("none", "tolerance", "hysteresis"))
    parser.add_argument("--tau-prune", type=float, default=1e-12,
                        help="Pruning slack tolerance tau_prune: drop F\\S edges with slack > tau_prune.")
    parser.add_argument("--tol-sep", type=float, default=1e-9,
                        help="Separation tolerance in Separate: candidate iff sigma + delta > tol-sep.")
    parser.add_argument("--gamma", type=int, default=1,
                        help="Select: top-gamma violated edges per row and per column.")
    parser.add_argument("--master-time-limit", type=float, default=60.0,
                        help="Per-master-solve CuPDLPx time budget t (seconds).")
    parser.add_argument("--tie-break", default="none", choices=("none", "regularized", "lex_generic"),
                        help="cupdlpx: 'none' or 'regularized'; 'lex_generic' is HiGHS-only")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="regularizer weight for tie_break='regularized'")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "iot_experiments" / "core" / "out" / "planting")
    args = parser.parse_args(argv)

    records: list[dict[str, object]] = []
    for family in args.families:
        for d in args.dims:
            for n in args.sizes:
                for seed in args.seeds:
                    for g_profile in args.g_profiles:
                        rng = np.random.default_rng(10_000 + int(seed))
                        g = rng.normal(scale=0.5, size=int(n)) if g_profile == "random" else None
                        planted = plant_instance(
                            family=family, n=int(n), point_dimension=int(d), seed=int(seed), g=g)
                        record: dict[str, object] = {
                            "family": family, "d": int(d), "n_requested": int(n), "seed": int(seed),
                            "g_profile": g_profile, "problem_type": planted.problem_type,
                            "mass_splitting": planted.mass_splitting,
                            "n_eff": planted.n_eff, "m_eff": planted.m_eff,
                            "support_size": planted.support_size,
                            "max_dual_violation": planted.max_dual_violation,
                            "max_support_residual": planted.max_support_residual,
                            "gap_quantiles": planted.gap_quantiles,
                            "g_norm": planted.g_norm, "feature_scale": planted.feature_scale,
                            "prune_false": _solve(planted, strategy="none", scan=str(args.scan),
                                                  solver=str(args.solver), tie_break="none",
                                                  epsilon=0.0, tau_prune=float(args.tau_prune),
                                                  tol_sep=float(args.tol_sep), gamma=int(args.gamma),
                                                  master_time_limit=float(args.master_time_limit),
                                                  max_iterations=int(args.max_iterations)),
                            "default": _solve(planted, strategy=str(args.prune_strategy), scan=str(args.scan),
                                              solver=str(args.solver), tie_break=str(args.tie_break),
                                              epsilon=float(args.epsilon), tau_prune=float(args.tau_prune),
                                              tol_sep=float(args.tol_sep), gamma=int(args.gamma),
                                              master_time_limit=float(args.master_time_limit),
                                              max_iterations=int(args.max_iterations)),
                        }
                        records.append(record)
                        df = record["prune_false"]
                        dflt = record["default"]
                        print(f"{family:10s} d={d} n_req={n} n_eff={planted.n_eff} m_eff={planted.m_eff} "
                              f"seed={seed} g={g_profile:6s}: |S|={planted.support_size} "
                              f"cert={planted.max_dual_violation:.1e} gap50={planted.gap_quantiles['0.50']:.2e} | "
                              f"noprune: conv={df['converged']} it={df['iterations']} "
                              f"F/(m+n)={df['max_support_over_m_plus_n']:.2f} t={df['solve_seconds']:.0f}s "
                              f"mem={df['peak_memory_mb']:.0f}MB | "
                              f"strat={args.prune_strategy} tie={args.tie_break} eps={args.epsilon:g} "
                              f"tau={args.tau_prune:g} tolsep={args.tol_sep:g} gamma={args.gamma}: "
                              f"conv={dflt['converged']} it={dflt['iterations']} "
                              f"F/(m+n)={dflt['max_support_over_m_plus_n']:.2f} t={dflt['solve_seconds']:.0f}s "
                              f"mem={dflt['peak_memory_mb']:.0f}MB",
                              flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "planting_sweep.json").write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    print(f"[planting_sweep] wrote {args.out / 'planting_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
