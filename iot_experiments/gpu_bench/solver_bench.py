"""
CN: 在**冻结快照**上离线基准各求解器：CuPDLPx / HiGHS / (任何可用的) 在 eps ∈ {1e-4..1e-8} 上
    记录时间、δ 误差、向量误差、tight/非 tight 翻转数、最大违反。这是未来任何设计决策的精度-成本表。
EN: Offline solver benchmark on **frozen snapshots**: CuPDLPx / HiGHS / (any available) over
    eps ∈ {1e-4..1e-8}, recording time, delta error, vector error, tight/not-tight flips, max violation.

用法 / EN: usage:
    python -m iot_experiments.gpu_bench.solver_bench            # all snapshots in out/snapshots
    python -m iot_experiments.gpu_bench.solver_bench --snap <run_id>
结果 / results: out/solver_bench/<run_id>.json
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

from iot import (  # noqa: E402
    CuPdlpxMasterSolver,
    EdgeSet,
    InverseOTProblem,
    LinearMasterSolver,
    SimplexTheta,
)
from iot_experiments.core.families import FamilySpec, build_family  # noqa: E402
from iot_experiments.core.marginals import marginal_spec  # noqa: E402
from iot_experiments.core.runner import _hello_plan  # noqa: E402

_SNAP = _REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "snapshots"
_OUT = _REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "solver_bench"
EPS_LIST = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
FLIP_THRESHOLDS = (1e-3, 1e-4)


def _rebuild(meta: dict, arrays) -> InverseOTProblem:
    """CN: 用元数据重建与保存时相同的实例（同 rng 顺序）。EN: Rebuild the same instance (same rng order)."""
    rng = np.random.default_rng(int(meta["seed"]))
    features, theta_set, theta_star = build_family(
        FamilySpec(str(meta["family"]), point_dimension=int(meta["point_dimension"])),
        rng, int(meta["n"]), int(meta["n"]),
    )
    a = marginal_spec(str(meta["marginal"])).weights(int(meta["n"]))
    b = marginal_spec(str(meta["marginal"])).weights(int(meta["n"]))
    plan, _u, _v = _hello_plan(features, theta_star, a, b, device="auto", max_iterations=200)
    _ = arrays  # plan is deterministic from the seed; arrays hold F/solutions
    return InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))


def _metrics(problem: InverseOTProblem, edges: EdgeSet, sol, ref_delta: float, ref_vec: np.ndarray) -> dict[str, object]:
    """CN: 与参考解比较：δ 误差、向量误差、tight 翻转数、最大违反。EN: δ error, vector error, flips, max violation."""
    theta = np.asarray(sol.theta)
    vec = np.concatenate([theta, np.asarray(sol.source_dual), np.asarray(sol.target_dual)])
    m, n = int(problem.n_source), int(problem.n_target)
    all_rows = np.repeat(np.arange(m, dtype=np.int64), n)
    all_cols = np.tile(np.arange(n, dtype=np.int64), m)
    costs = np.asarray(problem.features.costs(theta, all_rows, all_cols))
    scores = np.asarray(sol.source_dual)[all_rows] + np.asarray(sol.target_dual)[all_cols] - costs
    worst = float(np.max(scores + float(sol.delta)))
    # reference (HiGHS) slacks on the same edges
    ref_costs = np.asarray(problem.features.costs(np.asarray(ref_vec[:theta.size]), all_rows, all_cols))
    ref_scores = np.asarray(ref_vec[theta.size:theta.size + m])[all_rows] + np.asarray(ref_vec[theta.size + m:])[all_cols] - ref_costs
    flips = {str(thr): int(np.sum((scores <= thr) != (ref_scores <= thr))) for thr in FLIP_THRESHOLDS}
    return {
        "delta": float(sol.delta), "delta_error_vs_highs": abs(float(sol.delta) - ref_delta),
        "vector_error_vs_highs": float(np.linalg.norm(vec - ref_vec)),
        "max_all_edge_violation": worst, "tight_flips": flips,
    }


def bench_snapshot(run_id: str, *, eps_list=EPS_LIST) -> dict[str, object]:
    """CN: 对单个快照的全部 F 点基准各求解器。EN: Benchmark every F point of one snapshot."""
    meta = json.loads((_SNAP / f"{run_id}.json").read_text(encoding="utf-8"))
    arrays = np.load(_SNAP / f"{run_id}.npz")
    problem = _rebuild(meta, arrays)
    out: dict[str, object] = {"run_id": run_id, "meta": meta, "points": {}}
    for point in ("early", "half", "last"):
        if f"{point}_F_rows" not in arrays:
            continue
        F = EdgeSet.from_arrays(np.asarray(arrays[f"{point}_F_rows"]), np.asarray(arrays[f"{point}_F_cols"]),
                                n_target=int(problem.n_target))
        ref_delta = float(meta["points"][point]["highs_delta"])
        ref_vec = np.concatenate([np.asarray(arrays[f"{point}_theta"]),
                                  np.asarray(arrays[f"{point}_src_dual"]),
                                  np.asarray(arrays[f"{point}_tgt_dual"])])
        entry: dict[str, object] = {"F_size": int(F.size), "solver": {}}
        # HiGHS reference (tie_break none)
        t0 = time.perf_counter()
        hs = LinearMasterSolver(random_seed=int(meta["seed"])).solve(problem, F, tie_break="none")
        entry["solver"]["highs"] = {"seconds": time.perf_counter() - t0,
                                    **_metrics(problem, F, hs, ref_delta, ref_vec)}
        # CuPDLPx over eps
        for eps in eps_list:
            try:
                t0 = time.perf_counter()
                cp = CuPdlpxMasterSolver(params={"eps_optimal_relative": eps, "eps_feasible_relative": eps,
                                                 "time_sec_limit": 60.0}).solve(problem, F, tie_break="none")
                entry["solver"][f"cupdlpx_eps{eps:.0e}"] = {
                    "seconds": time.perf_counter() - t0, "eps": eps,
                    "delta_over_eps": float(cp.delta) / eps,
                    **_metrics(problem, F, cp, ref_delta, ref_vec)}
            except Exception as error:  # noqa: BLE001
                entry["solver"][f"cupdlpx_eps{eps:.0e}"] = {"error": f"{type(error).__name__}: {error}"[:120]}
        out["points"][point] = entry
        print(f"[bench] {run_id} {point} |F|={F.size}: " + " ".join(
            f"{k}={v.get('seconds', float('nan')):.2f}s" for k, v in entry["solver"].items()), flush=True)
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / f"{run_id}.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="offline solver benchmark on frozen snapshots")
    p.add_argument("--snap", nargs="*", default=None, help="run_id(s); default = all in out/snapshots")
    p.add_argument("--eps", type=float, nargs="+", default=list(EPS_LIST))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ids = args.snap if args.snap else sorted(p.stem for p in _SNAP.glob("*.json"))
    if not ids:
        print("no snapshots found; run snapshots.py save first")
        return 1
    for run_id in ids:
        bench_snapshot(str(run_id), eps_list=[float(e) for e in args.eps])
    print(f"[bench] wrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
