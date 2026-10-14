"""
CN: 小规模诊断：一阶 master（CuPDLPx）与精确 master（HiGHS）在「加边/剪边」循环中的差异。
EN: Small-scale diagnostics: first-order master (CuPDLPx) vs exact master (HiGHS) in the
    add/prune loop.

CN: 不改动既有 benchmark 结构；只做测试要求的量测。运行：
      python -m iot_experiments.gpu_bench.loop_diag test1 --n 200 --seed 0
    （test2..test5 随后补充）。所有阈值对一阶求解器不高于 1e-3 的 10× 以内。
EN: Does not change existing benchmarks; only measures what the tests ask. Run:
      python -m iot_experiments.gpu_bench.loop_diag test1 --n 200 --seed 0
    (test2..test5 added next.) Thresholds stay within ~10x the first-order accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import ot
from scipy import sparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import (  # noqa: E402
    CuPdlpxMasterSolver,
    EdgeSet,
    InverseOTProblem,
    LinearMasterSolver,
)
from iot.loop import solve_inverse_ot  # noqa: E402
from iot_experiments.core.costs import blockwise_cost_matrix  # noqa: E402
from iot_experiments.core.families import FamilySpec, build_family  # noqa: E402
from iot_experiments.core.marginals import marginal_spec  # noqa: E402

TOL = 1e-9
TAU_PRUNE = 1e-12
GAMMA = 1
TIGHT_THRESHOLDS = (1e-3, 1e-4)  # 1e-4 is at the edge of the 10x rule for a 1e-4 solver


def edge_hash(keys: np.ndarray) -> str:
    """CN: 边集键的稳定哈希（检测精确重复）。EN: Stable hash of edge keys (detect exact repeats)."""
    return hashlib.sha1(np.sort(np.asarray(keys, dtype="<i8")).tobytes()).hexdigest()[:12]


class RecordingSolver:
    """CN: 包装 master，记录每次 solve 的边集/hash/耗时。EN: Wrap the master; record F/hash/timing per solve."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.edge_sets: list[EdgeSet] = []
        self.hashes: list[str] = []
        self.sizes: list[int] = []
        self.seconds: list[float] = []

    def solve(self, problem, edges, *, tie_break: str = "none"):
        self.edge_sets.append(edges)
        self.hashes.append(edge_hash(edges.keys))
        self.sizes.append(int(edges.size))
        started = time.perf_counter()
        solution = self.inner.solve(problem, edges, tie_break=tie_break)
        self.seconds.append(time.perf_counter() - started)
        return solution


def build_problem(*, family: str, point_dimension: int, marginal: str, n: int, seed: int) -> InverseOTProblem:
    """CN: 小规模实例（POT 真值；n<=800 无妨）。EN: Small instance (POT ground truth; fine at n<=800)."""
    rng = np.random.default_rng(int(seed))
    features, theta_set, theta_star = build_family(
        FamilySpec(family, point_dimension=int(point_dimension)), rng, int(n), int(n)
    )
    a = marginal_spec(marginal).weights(int(n))
    b = marginal_spec(marginal).weights(int(n))
    cost = blockwise_cost_matrix(features, theta_star)
    plan = sparse.csr_matrix(np.asarray(ot.emd(a, b, cost, numItermax=2_000_000)))
    return InverseOTProblem(features=features, theta_set=theta_set, plan=plan)


def run_loop(problem, *, inner, prune: bool, hysteresis: bool, max_iterations: int, scan_backend: str):
    """CN: 跑循环并记录每轮的 (maxviol, delta, |F|, F-hash, master/oracle 时间)。
    EN: Run the loop recording per-iteration (maxviol, delta, |F|, F-hash, master/oracle time)."""
    recorder = RecordingSolver(inner)
    result = solve_inverse_ot(
        problem, tie_break="none", solver=recorder, max_iterations=int(max_iterations),
        tol=TOL, tau_prune=TAU_PRUNE, gamma=GAMMA, validate=False,
        prune=bool(prune), prune_hysteresis=bool(hysteresis), scan_backend=str(scan_backend),
    )
    log: list[dict[str, object]] = []
    for i, rec in enumerate(result.iterations):
        log.append({
            "iteration": i + 1,
            "max_violation": float(rec.max_violation),
            "delta": float(rec.delta),
            "n_edges": int(recorder.sizes[i]) if i < len(recorder.sizes) else int(rec.support_size),
            "edge_hash": recorder.hashes[i] if i < len(recorder.hashes) else "",
            "master_seconds": float(rec.master_time),
            "scan_seconds": float(rec.oracle_time),
        })
    return result, recorder, log


def _all_edges(problem: InverseOTProblem) -> tuple[np.ndarray, np.ndarray]:
    m, n = int(problem.n_source), int(problem.n_target)
    return (
        np.repeat(np.arange(m, dtype=np.int64), n),
        np.tile(np.arange(n, dtype=np.int64), m),
    )


def solution_compare(problem: InverseOTProblem, solutions: dict[str, object]) -> dict[str, object]:
    """CN: 比较冻结 F 上不同 master 解：δ、向量差、最大违反、非支撑边的 tight/非 tight 计数。
    EN: Compare master solutions on a frozen F: delta, vector diffs, max violation, tight counts."""
    m, n = int(problem.n_source), int(problem.n_target)
    rows_all, cols_all = _all_edges(problem)
    support = set(int(k) for k in problem.support_keys())
    off_mask = np.array([int(r) * n + int(c) not in support for r, c in zip(rows_all, cols_all)])
    rows_off, cols_off = rows_all[off_mask], cols_all[off_mask]

    def pack(sol) -> np.ndarray:
        return np.concatenate([np.asarray(sol.theta), np.asarray(sol.source_dual), np.asarray(sol.target_dual)])

    vectors = {name: pack(sol) for name, sol in solutions.items()}
    stats: dict[str, object] = {"delta": {}, "max_violation": {}, "tight_counts": {}, "vectors": {}}
    slacks: dict[str, np.ndarray] = {}
    for name, sol in solutions.items():
        cost_off = np.asarray(problem.features.costs(np.asarray(sol.theta), rows_off, cols_off))
        slack_off = cost_off - np.asarray(sol.source_dual)[rows_off] - np.asarray(sol.target_dual)[cols_off]
        slacks[name] = slack_off
        # max violation over all edges: max(f_i + g_j - c_ij + delta)
        worst = -np.inf
        step = 256
        for start in range(0, m, step):
            stop = min(m, start + step)
            rr = np.repeat(np.arange(start, stop, dtype=np.int64), n)
            cc = np.tile(np.arange(n, dtype=np.int64), stop - start)
            cost = np.asarray(problem.features.costs(np.asarray(sol.theta), rr, cc))
            worst = max(worst, float(np.max(np.asarray(sol.source_dual)[rr] + np.asarray(sol.target_dual)[cc] - cost + float(sol.delta))))
        stats["delta"][name] = float(sol.delta)
        stats["max_violation"][name] = worst
        stats["tight_counts"][name] = {
            str(thr): int(np.sum(slack_off <= thr)) for thr in TIGHT_THRESHOLDS
        }
        stats["vectors"][name] = vectors[name]
    # pairwise vector distances
    dist: dict[str, float] = {}
    names = list(solutions)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist[f"{names[i]}~{names[j]}"] = float(np.linalg.norm(vectors[names[i]] - vectors[names[j]]))
    # tight/not-tight disagreements (off-support edges) between pairs
    disagree: dict[str, object] = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            entry = {}
            for thr in TIGHT_THRESHOLDS:
                ta = slacks[names[a]] <= thr
                tb = slacks[names[b]] <= thr
                entry[str(thr)] = int(np.sum(ta != tb))
            disagree[f"{names[a]}~{names[b]}"] = entry
    return {"stats": {k: v for k, v in stats.items() if k != "vectors"}, "vector_distances": dist,
            "tight_disagreements": disagree, "n_off_support": int(rows_off.size),
            "vector_norms": {k: float(np.linalg.norm(v)) for k, v in vectors.items()}}


def cmd_test1(args: argparse.Namespace) -> int:
    """CN: Test 1 —— HiGHS 跑到大约一半，冻结 F，用两种 master 解同一 F 并比较。
    EN: Test 1 -- run HiGHS to about halfway, freeze F, solve that F with both masters and compare."""
    problem = build_problem(
        family=str(args.family), point_dimension=int(args.point_dimension),
        marginal=str(args.marginal), n=int(args.n), seed=int(args.seed),
    )
    # 1) HiGHS loop to convergence with the shipped pruning, recording every F.
    inner = LinearMasterSolver(random_seed=int(args.seed))
    result, recorder, log = run_loop(
        problem, inner=inner, prune=True, hysteresis=True, max_iterations=int(args.max_iterations),
        scan_backend=str(args.scan),
    )
    iters = int(result.n_iterations)
    half = max(1, (iters + 1) // 2)
    F_half = recorder.edge_sets[half - 1]
    print(f"[test1] HiGHS loop: converged={result.converged} iters={iters}; freeze F at iter {half} (|F|={F_half.size})", flush=True)

    solutions: dict[str, object] = {}
    solutions["highs_lex"] = LinearMasterSolver(random_seed=int(args.seed)).solve(problem, F_half, tie_break="lex")
    solutions["highs_none"] = LinearMasterSolver(random_seed=int(args.seed)).solve(problem, F_half, tie_break="none")
    cupdlpx_note = ""
    try:
        solutions["cupdlpx_default"] = CuPdlpxMasterSolver().solve(problem, F_half, tie_break="none")
        solutions["cupdlpx_eps6"] = CuPdlpxMasterSolver(
            params={"eps_optimal_relative": 1e-6, "eps_feasible_relative": 1e-6}
        ).solve(problem, F_half, tie_break="none")
    except Exception as error:  # noqa: BLE001
        cupdlpx_note = f"CuPDLPx unavailable: {type(error).__name__}: {str(error)[:120]}"
        print(f"[test1] {cupdlpx_note}", flush=True)

    comparison = solution_compare(problem, solutions)
    payload = {
        "test": "test1", "n": int(args.n), "seed": int(args.seed), "family": str(args.family),
        "marginal": str(args.marginal), "point_dimension": int(args.point_dimension),
        "loop_iters": iters, "freeze_iteration": half, "frozen_F_size": int(F_half.size),
        "loop_log": log, "comparison": comparison, "cupdlpx_note": cupdlpx_note,
    }
    out = Path(args.out) if args.out else (_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "diag" / f"test1_n{args.n}_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[test1] wrote {out}", flush=True)
    print("delta:", comparison["stats"]["delta"])
    print("max_violation over all edges:", comparison["stats"]["max_violation"])
    print("off-support tight counts (per threshold):", comparison["stats"]["tight_counts"])
    print("vector distances:", comparison["vector_distances"])
    print("tight/not-tight disagreements:", comparison["tight_disagreements"])
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="small-scale loop diagnostics")
    parser.add_argument("test", choices=("test1",))
    parser.add_argument("--family", default="diagonal")
    parser.add_argument("--point-dimension", type=int, default=8)
    parser.add_argument("--marginal", default="gmm")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--scan", default="numpy")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.test == "test1":
        return cmd_test1(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
