"""
CN: **分数（Kantorovich）方案的参数扫描驱动** —— 构造 `|S| ∈ [n, 2n]` 的 fractional 实例并跑循环。
    与 `planting_sweep.py` 对应，但用 `fractional_planting.plant_fractional`，并额外扫
    `slide-cap` / `split-ratio` / 方向。

EN: **Fractional (Kantorovich) sweep driver** -- build `|S| in [n, 2n]` fractional instances and run the
    loop. Mirrors `planting_sweep.py` but uses `fractional_planting.plant_fractional`, and additionally
    sweeps `slide-cap` / `split-ratio` / direction.

用法 / EN: usage
    python -m iot_experiments.core.fractional_sweep --family quadratic --sizes 3200 --split-fractions 1.0 \
        --scan hello --solver cupdlpx --prune-strategy none --max-iterations 100
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

from iot import solve_inverse_ot  # noqa: E402
from iot.backends.cupdlpx_master import CuPdlpxMasterSolver  # noqa: E402
from iot.exact.master import LinearMasterSolver  # noqa: E402
from iot_experiments.core.fractional_planting import plant_fractional  # noqa: E402


def _solve(planted, *, strategy: str, scan: str, solver: str, tie_break: str,
           epsilon: float, tau_prune: float, tol_sep: float, gamma: int,
           master_time_limit: float, max_iterations: int) -> dict[str, object]:
    """CN: 与 planting_sweep._solve 相同的求解/读数。EN: same solve/read-out as planting_sweep._solve."""
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
        "relative_gap": None if result.relative_gap is None else float(result.relative_gap),
        "max_support_over_m_plus_n": float(
            max((rec.support_size for rec in result.iterations), default=result.support_size)
        ) / float(planted.n_eff + planted.m_eff),
        "peak_memory_mb": float(max((getattr(rec, "peak_memory_mb", 0.0) for rec in result.iterations), default=0.0)),
        "solve_seconds": float(time.perf_counter() - t0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fractional (Kantorovich) parameter sweep")
    parser.add_argument("--family", default="quadratic")
    parser.add_argument("--point-dimension", type=int, default=8)
    parser.add_argument("--sizes", type=int, nargs="+", default=[3200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--split-fractions", type=float, nargs="+", default=[1.0])
    parser.add_argument("--split-ratios", type=float, nargs="+", default=[0.5])
    parser.add_argument("--slide-caps", type=float, nargs="+", default=[4.0])
    parser.add_argument("--direction", default="target_diff", choices=("target_diff",))
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--scan", default="hello", choices=("numpy", "hello", "torch"))
    parser.add_argument("--solver", default="cupdlpx", choices=("highs", "cupdlpx"))
    parser.add_argument("--prune-strategy", default="none", choices=("none", "tolerance", "hysteresis"))
    parser.add_argument("--tau-prune", type=float, default=1e-12)
    parser.add_argument("--tol-sep", type=float, default=1e-9)
    parser.add_argument("--gamma", type=int, default=1)
    parser.add_argument("--tie-break", default="none", choices=("none", "regularized"))
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--master-time-limit", type=float, default=60.0)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "core" / "out" / "fractional")
    args = parser.parse_args(argv)

    records: list[dict[str, object]] = []
    for n in args.sizes:
        for seed in args.seeds:
            for sf in args.split_fractions:
                for sr in args.split_ratios:
                    for cap in args.slide_caps:
                        planted = plant_fractional(
                            family=str(args.family), n=int(n),
                            point_dimension=int(args.point_dimension), seed=int(seed),
                            split_fraction=float(sf), split_ratio=float(sr),
                            direction=str(args.direction), slide_cap=float(cap),
                        )
                        record: dict[str, object] = {
                            "family": str(args.family), "n": int(n), "seed": int(seed),
                            "split_fraction": float(sf), "split_ratio": float(sr),
                            "slide_cap": float(cap), "direction": str(args.direction),
                            "problem_type": planted.problem_type, "mass_splitting": planted.mass_splitting,
                            "n_eff": planted.n_eff, "m_eff": planted.m_eff,
                            "support_size": planted.support_size,
                            "support_over_n": planted.support_over_n,
                            "n_split": planted.n_split, "n_skipped": planted.n_skipped,
                            "max_dual_violation": planted.max_dual_violation,
                            "max_support_residual": planted.max_support_residual,
                            "gap_quantiles": planted.gap_quantiles,
                            "solve": _solve(planted, strategy=str(args.prune_strategy), scan=str(args.scan),
                                            solver=str(args.solver), tie_break=str(args.tie_break),
                                            epsilon=float(args.epsilon), tau_prune=float(args.tau_prune),
                                            tol_sep=float(args.tol_sep), gamma=int(args.gamma),
                                            master_time_limit=float(args.master_time_limit),
                                            max_iterations=int(args.max_iterations)),
                        }
                        records.append(record)
                        s = record["solve"]
                        print(f"{args.family:10s} n={n} seed={seed} sf={sf} sr={sr} cap={cap}: "
                              f"|S|/n={planted.support_over_n:.2f} split={planted.n_split} skip={planted.n_skipped} "
                              f"maxviol={planted.max_dual_violation:.1e} | "
                              f"conv={s['converged']} it={s['iterations']} F/(m+n)={s['max_support_over_m_plus_n']:.2f} "
                              f"isopt={s['is_optimal']} relgap={s['relative_gap']} t={s['solve_seconds']:.0f}s",
                              flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fractional_sweep.json").write_text(json.dumps({"records": records}, indent=2) + "\n",
                                                    encoding="utf-8")
    print(f"[fractional_sweep] wrote {args.out / 'fractional_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
