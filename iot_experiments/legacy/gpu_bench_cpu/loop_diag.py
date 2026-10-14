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
    InverseOTError,
    InverseOTProblem,
    LinearMasterSolver,
    MasterSolution,
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
        self.solutions: list[object] = []

    def solve(self, problem, edges, *, tie_break: str = "none"):
        self.edge_sets.append(edges)
        self.hashes.append(edge_hash(edges.keys))
        self.sizes.append(int(edges.size))
        started = time.perf_counter()
        solution = self.inner.solve(problem, edges, tie_break=tie_break)
        self.seconds.append(time.perf_counter() - started)
        self.solutions.append(solution)
        return solution


class PenalizedMasterSolver:
    """
    CN: 目标改为 max δ - (ρ/2)·||(θ,f,g)||²（唯一最优解），用 HiGHS 的 QP 求解。
        CuPDLPx 是 LP 一阶求解器，**不支持**二次目标（Test 2 的 GPU 部分到此为止）。
    EN: Objective max δ - (ρ/2)·||(θ,f,g)||² (unique optimum), solved as a HiGHS QP. CuPDLPx is
        a first-order **LP** solver and does **not** support a quadratic objective (Test 2 GPU
        part stops here).
    """

    def __init__(self, rho: float) -> None:
        self.rho = float(rho)

    def solve(self, problem, edges, *, tie_break: str = "none"):
        import highspy
        from iot.master import build_master_system

        system = build_master_system(problem, edges)
        layout = system.layout
        n = int(layout.size)
        a = sparse.vstack([system.a_eq, system.a_ub], format="csr")
        row_lower = np.concatenate([system.b_eq, np.full(int(system.a_ub.shape[0]), -np.inf)])
        row_upper = np.concatenate([system.b_eq, system.b_ub])
        col_lower = np.array([lo if lo is not None else -np.inf for lo, _ in system.bounds], dtype=float)
        col_upper = np.array([hi if hi is not None else np.inf for _, hi in system.bounds], dtype=float)
        c = np.zeros(n, dtype=float)
        c[int(layout.delta)] = -1.0
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = int(a.shape[0])
        lp.col_cost_ = c
        lp.col_lower_ = col_lower
        lp.col_upper_ = col_upper
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = np.asarray(a.indptr, dtype=np.int32)
        lp.a_matrix_.index_ = np.asarray(a.indices, dtype=np.int32)
        lp.a_matrix_.value_ = np.asarray(a.data, dtype=float)
        highs.passModel(lp)
        cols = np.arange(int(layout.theta_start), n)
        start = np.zeros(n + 1, dtype=np.int32)
        start[cols + 1] = 1
        start = np.cumsum(start).astype(np.int32)
        hessian = highspy.HighsHessian()
        hessian.dim_ = n
        hessian.format_ = highspy.HessianFormat.kTriangular
        hessian.start_ = start
        hessian.index_ = cols.astype(np.int32)
        hessian.value_ = np.full(cols.size, self.rho, dtype=float)
        highs.passHessian(hessian)
        highs.run()
        status = highs.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            raise InverseOTError(f"penalized master not optimal: {status}")
        x = np.asarray(highs.getSolution().col_value, dtype=float)
        return MasterSolution(
            delta=float(x[int(layout.delta)]),
            theta=np.ascontiguousarray(x[int(layout.theta_start):int(layout.source_start)]),
            source_dual=np.ascontiguousarray(x[int(layout.source_start):int(layout.target_start)]),
            target_dual=np.ascontiguousarray(x[int(layout.target_start):]),
            status="optimal",
        )


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
    freeze_index = (iters - 1) if str(getattr(args, "freeze", "half")) == "last" else (half - 1)
    F_half = recorder.edge_sets[freeze_index]
    print(f"[test1] HiGHS loop: converged={result.converged} iters={iters}; freeze F at iter {freeze_index + 1} "
          f"(mode={getattr(args, 'freeze', 'half')}, |F|={F_half.size}, "
          f"maxviol={float(log[freeze_index]['max_violation']):.3e})", flush=True)

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


def cmd_test2(args: argparse.Namespace) -> int:
    """CN: Test 2 —— 二次惩罚 master（唯一解），HiGHS QP，严格剪枝；CuPDLPx 无 QP，GPU 部分停止。
    EN: Test 2 -- quadratic-penalty master (unique), HiGHS QP, strict pruning; CuPDLPx has no QP."""
    problem = build_problem(
        family=str(args.family), point_dimension=int(args.point_dimension),
        marginal=str(args.marginal), n=int(args.n), seed=int(args.seed),
    )
    base_result, _base_rec, base_log = run_loop(
        problem, inner=LinearMasterSolver(random_seed=int(args.seed)),
        prune=True, hysteresis=False, max_iterations=int(args.max_iterations), scan_backend=str(args.scan),
    )
    payload: dict[str, object] = {
        "test": "test2", "n": int(args.n), "seed": int(args.seed), "family": str(args.family),
        "marginal": str(args.marginal),
        "cupdlpx_qp_supported": False,
        "cupdlpx_note": "CuPDLPx is a first-order LP solver (build_cupdlpx_lp is linear); it has "
                        "no quadratic-objective/Hessian path, so the penalty master cannot run on it.",
        "unpenalized": {
            "converged": bool(base_result.converged), "iterations": int(base_result.n_iterations),
            "final_delta": float(base_result.delta), "final_F": int(base_result.support_size),
        },
        "per_rho": {},
    }
    print(f"[test2] unpenalized (strict prune): converged={base_result.converged} "
          f"iters={base_result.n_iterations} final_delta={base_result.delta:.6g} |F|={base_result.support_size}")

    for rho in [float(v) for v in args.rho]:
        result, rec, log = run_loop(
            problem, inner=PenalizedMasterSolver(rho), prune=True, hysteresis=False,
            max_iterations=int(args.max_iterations), scan_backend=str(args.scan),
        )
        objectives: list[float] = []
        for solution in rec.solutions:
            vec = np.concatenate([np.asarray(solution.theta), np.asarray(solution.source_dual),
                                  np.asarray(solution.target_dual)])
            objectives.append(float(solution.delta) - 0.5 * rho * float(vec @ vec))
        deltas = [float(s.delta) for s in rec.solutions]
        monotone = all(objectives[i + 1] <= objectives[i] + 1e-9 for i in range(len(objectives) - 1))
        payload["per_rho"][str(rho)] = {
            "converged": bool(result.converged), "iterations": int(result.n_iterations),
            "final_delta": float(result.delta), "final_F": int(result.support_size),
            "objectives": objectives, "deltas": deltas, "objective_monotone_nonincreasing": bool(monotone),
            "log": log,
        }
        print(f"[test2] rho={rho:g}: converged={result.converged} iters={result.n_iterations} "
              f"final_delta={result.delta:.6g} (unpen {base_result.delta:.6g}) |F|={result.support_size} "
              f"obj_monotone={monotone} obj={[round(o, 5) for o in objectives]}")
    out = Path(args.out) if args.out else (_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "diag" / f"test2_n{args.n}_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[test2] wrote {out}")
    return 0


def cmd_test3(args: argparse.Namespace) -> int:
    """CN: Test 3 —— GPU（CuPDLPx）关剪枝 vs HiGHS（同样关剪枝），n=200/400/800。
    EN: Test 3 -- GPU (CuPDLPx) no-prune vs HiGHS (no-prune), n=200/400/800."""
    rows: list[dict[str, object]] = []
    for n in [int(v) for v in args.sizes]:
        problem = build_problem(
            family=str(args.family), point_dimension=int(args.point_dimension),
            marginal=str(args.marginal), n=int(n), seed=int(args.seed),
        )
        for name, inner in (
            ("highs_noprune", LinearMasterSolver(random_seed=int(args.seed))),
            ("cupdlpx_noprune", CuPdlpxMasterSolver(params={"time_sec_limit": float(args.cupdlpx_cap)})),
        ):
            result, rec, log = run_loop(
                problem, inner=inner, prune=False, hysteresis=False,
                max_iterations=int(args.max_iterations), scan_backend=str(args.scan),
            )
            master = [float(e["master_seconds"]) for e in log]
            scan = [float(e["scan_seconds"]) for e in log]
            solver_eps = 1e-4 if name.startswith("cupdlpx") else 1e-12
            rows.append({
                "n": int(n), "solver": name, "converged": bool(result.converged),
                "iterations": int(result.n_iterations), "final_F": int(result.support_size),
                "mean_master_s": float(np.mean(master)) if master else 0.0,
                "mean_scan_s": float(np.mean(scan)) if scan else 0.0,
                "mean_iter_s": float(np.mean([m + s for m, s in zip(master, scan)])) if master else 0.0,
                "total_s": float(np.sum(master) + np.sum(scan)),
                "edge_hashes": rec.hashes, "final_delta": float(result.delta),
                "solver_eps": solver_eps,
                "delta_over_eps": float(result.delta) / solver_eps if solver_eps > 0 else float("nan"),
                "maxviol_traj": [float(e["max_violation"]) for e in log],
                "F_traj": [int(e["n_edges"]) for e in log],
            })
            r = rows[-1]
            print(f"[test3] n={n:>4} {name:<18} conv={result.converged} it={result.n_iterations:>3} "
                  f"|F|={result.support_size:>5} ms/iter={r['mean_iter_s']*1e3:>8.1f} total={r['total_s']:>7.2f}s "
                  f"delta={result.delta:.4g} delta/eps={r['delta_over_eps']:.3g}")
    out = Path(args.out) if args.out else (_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "diag" / f"test3_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"test": "test3", "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[test3] wrote {out}")
    return 0


class SwitchingSolver:
    """CN: 前 `switch_at-1` 次 solve 用 GPU，之后改用精确 master（F 由循环延续，故是热启动）。
    EN: GPU for the first `switch_at-1` solves, then the exact master (the loop carries F, so the
        exact phase is warm-started)."""

    def __init__(self, gpu, exact, switch_at: int) -> None:
        self.gpu = gpu
        self.exact = exact
        self.switch_at = int(switch_at)
        self.calls = 0

    def solve(self, problem, edges, *, tie_break: str = "none"):
        self.calls += 1
        inner = self.gpu if self.calls < self.switch_at else self.exact
        return inner.solve(problem, edges, tie_break=tie_break)


def _final_repair(problem: InverseOTProblem, solution) -> dict[str, object]:
    """CN: 廉价修复：全边最大违反 v=max(f+g-c+δ)；δ'=δ-v。δ'>=0 即为有效证书。
    EN: Cheap repair: max all-edge violation v=max(f+g-c+δ); δ'=δ-v. δ'>=0 is a valid certificate."""
    m, n = int(problem.n_source), int(problem.n_target)
    f = np.asarray(solution.source_dual)
    g = np.asarray(solution.target_dual)
    worst = -np.inf
    for start in range(0, m, 256):
        stop = min(m, start + 256)
        rr = np.repeat(np.arange(start, stop, dtype=np.int64), n)
        cc = np.tile(np.arange(n, dtype=np.int64), stop - start)
        cost = np.asarray(problem.features.costs(np.asarray(solution.theta), rr, cc))
        worst = max(worst, float(np.max(f[rr] + g[cc] - cost + float(solution.delta))))
    return {"max_all_edge_violation": worst, "delta": float(solution.delta),
            "shifted_delta": float(solution.delta) - worst,
            "shifted_nonnegative": bool(float(solution.delta) - worst >= -1e-12)}


def cmd_test4(args: argparse.Namespace) -> int:
    """CN: Test 4 —— GPU 全程不剪枝，maxviol<1e-3 后切换 HiGHS；并测廉价 δ 修复证书。
    EN: Test 4 -- GPU no-prune throughout, switch to HiGHS once maxviol<1e-3; plus the cheap δ repair."""
    from iot import is_optimal_plan

    problem = build_problem(
        family=str(args.family), point_dimension=int(args.point_dimension),
        marginal=str(args.marginal), n=int(args.n), seed=int(args.seed),
    )
    gpu = CuPdlpxMasterSolver(params={"time_sec_limit": float(args.cupdlpx_cap)})
    result1, rec1, log1 = run_loop(problem, inner=gpu, prune=False, hysteresis=False,
                                   max_iterations=int(args.max_iterations), scan_backend=str(args.scan))
    switch_at = next((i + 1 for i, e in enumerate(log1) if float(e["max_violation"]) < 1e-3), None)
    print(f"[test4] phase1(GPU) iters={result1.n_iterations} converged={result1.converged}; "
          f"first maxviol<1e-3 at call {switch_at}")
    repair = _final_repair(problem, rec1.solutions[-1]) if rec1.solutions else {}
    print(f"[test4] cheap repair on GPU final: {repair}")

    payload: dict[str, object] = {
        "test": "test4", "n": int(args.n), "seed": int(args.seed),
        "phase1_gpu": {"iterations": int(result1.n_iterations), "converged": bool(result1.converged),
                       "max_violation": [float(e["max_violation"]) for e in log1]},
        "switch_at_call": switch_at, "cheap_repair_gpu_final": repair,
    }
    if switch_at is not None:
        switch = SwitchingSolver(
            CuPdlpxMasterSolver(params={"time_sec_limit": float(args.cupdlpx_cap)}),
            LinearMasterSolver(random_seed=int(args.seed)), switch_at=int(switch_at),
        )
        result2, _rec2, log2 = run_loop(problem, inner=switch, prune=False, hysteresis=False,
                                        max_iterations=int(args.max_iterations), scan_backend=str(args.scan))
        nh = max(0, int(result2.n_iterations) - (int(switch_at) - 1))
        check = is_optimal_plan(problem, np.asarray(result2.theta), tol=1e-7, num_itermax=2_000_000)
        payload["phase2_switch"] = {
            "gpu_solves": int(switch_at) - 1, "exact_solves": nh,
            "F_at_switch": int(rec1.sizes[int(switch_at) - 1]) if int(switch_at) - 1 < len(rec1.sizes) else None,
            "iterations": int(result2.n_iterations), "converged": bool(result2.converged),
            "final_F": int(result2.support_size), "is_optimal": bool(check.is_optimal),
            "relative_gap": float(check.relative_gap),
        }
        print(f"[test4] phase2: exact_solves={nh} F_at_switch={payload['phase2_switch']['F_at_switch']} "
              f"iters={result2.n_iterations} is_optimal={check.is_optimal}")
    out = Path(args.out) if args.out else (_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "diag" / f"test4_n{args.n}_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[test4] wrote {out}")
    return 0


class ScheduleSolver:
    """CN: 按预算把 CuPDLPx 的 eps 设为「当前最大违反/10」（夹在 [1e-4, 1e-2]）。
    EN: Drive CuPDLPx's eps to "current max violation / 10" (clamped to [1e-4, 1e-2])."""

    def __init__(self, eps_schedule: list[float], cap: float) -> None:
        self.eps = [float(e) for e in eps_schedule]
        self.cap = float(cap)
        self.calls = 0

    def solve(self, problem, edges, *, tie_break: str = "none"):
        eps = self.eps[min(self.calls, len(self.eps) - 1)]
        self.calls += 1
        return CuPdlpxMasterSolver(
            params={"eps_optimal_relative": eps, "eps_feasible_relative": eps,
                    "time_sec_limit": self.cap}
        ).solve(problem, edges, tie_break=tie_break)


def cmd_test5(args: argparse.Namespace) -> int:
    """CN: Test 5 —— GPU 的精度跟随进展：eps_k = clamp(maxviol_k/10, 1e-4, 1e-2)。
    EN: Test 5 -- GPU accuracy follows progress: eps_k = clamp(maxviol_k/10, 1e-4, 1e-2)."""
    problem = build_problem(
        family=str(args.family), point_dimension=int(args.point_dimension),
        marginal=str(args.marginal), n=int(args.n), seed=int(args.seed),
    )
    result1, rec1, log1 = run_loop(
        problem, inner=CuPdlpxMasterSolver(params={"time_sec_limit": float(args.cupdlpx_cap)}),
        prune=False, hysteresis=False, max_iterations=int(args.max_iterations), scan_backend=str(args.scan),
    )
    schedule = [float(np.clip(float(e["max_violation"]) / 10.0, 1e-4, 1e-2)) for e in log1]
    result2, rec2, log2 = run_loop(
        problem, inner=ScheduleSolver(schedule, float(args.cupdlpx_cap)),
        prune=False, hysteresis=False, max_iterations=int(args.max_iterations), scan_backend=str(args.scan),
    )
    t1 = float(np.sum([e["master_seconds"] + e["scan_seconds"] for e in log1]))
    t2 = float(np.sum([e["master_seconds"] + e["scan_seconds"] for e in log2]))
    payload = {
        "test": "test5", "n": int(args.n), "seed": int(args.seed),
        "default": {"converged": bool(result1.converged), "iterations": int(result1.n_iterations),
                    "total_s": t1, "final_F": int(result1.support_size)},
        "schedule": {"eps_schedule": schedule, "converged": bool(result2.converged),
                     "iterations": int(result2.n_iterations), "total_s": t2, "final_F": int(result2.support_size)},
    }
    print(f"[test5] default eps=1e-4: conv={result1.converged} it={result1.n_iterations} total={t1:.2f}s")
    print(f"[test5] schedule eps: conv={result2.converged} it={result2.n_iterations} total={t2:.2f}s "
          f"eps={[round(e,4) for e in schedule]}")
    out = Path(args.out) if args.out else (_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "diag" / f"test5_n{args.n}_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[test5] wrote {out}")
    return 0


def cmd_test6(args: argparse.Namespace) -> int:
    """
    CN: Test 6 —— (a) 精确收尾证书：GPU 终态 F 上用 HiGHS 解一次并 is_optimal_plan；
        (b) GPU 的 δ 是否正确：与同一 F 上的 HiGHS δ 比较，并跨 seed/family 检查 δ 随 n 的衰减。
    EN: Test 6 -- (a) VALIDATION reference only: one HiGHS solve on the GPU's final F + is_optimal_plan;
        HiGHS is never a production finisher (see certify.py for the HiGHS-free certificate).
        (b) is the GPU's delta right: compare against HiGHS delta on the same F, and check the
        delta(n) decay across seeds/families.
    """
    from iot import is_optimal_plan

    rows: list[dict[str, object]] = []
    for n in [int(v) for v in args.sizes]:
        for seed in [int(v) for v in args.seeds]:
            for family in [str(v) for v in args.families]:
                problem = build_problem(
                    family=family, point_dimension=int(args.point_dimension),
                    marginal=str(args.marginal), n=int(n), seed=int(seed),
                )
                # GPU (no prune) loop, capture the final F
                gpu = CuPdlpxMasterSolver(params={"time_sec_limit": float(args.cupdlpx_cap)})
                r_gpu, rec_gpu, log_gpu = run_loop(
                    problem, inner=gpu, prune=False, hysteresis=False,
                    max_iterations=int(args.max_iterations), scan_backend=str(args.scan),
                )
                final_F = rec_gpu.edge_sets[-1]
                # (a) HiGHS as a VALIDATION reference only (not a production finisher)
                t0 = time.perf_counter()
                exact_sol = LinearMasterSolver(random_seed=int(seed)).solve(problem, final_F, tie_break="none")
                t_exact = time.perf_counter() - t0
                check = is_optimal_plan(problem, np.asarray(exact_sol.theta), tol=1e-7, num_itermax=2_000_000)
                rows.append({
                    "n": int(n), "seed": int(seed), "family": family,
                    "gpu_delta": float(r_gpu.delta), "gpu_iters": int(r_gpu.n_iterations),
                    "gpu_converged": bool(r_gpu.converged), "final_F": int(final_F.size),
                    "highs_delta_on_F": float(exact_sol.delta),
                    "delta_abs_diff": abs(float(r_gpu.delta) - float(exact_sol.delta)),
                    "delta_rel_diff": abs(float(r_gpu.delta) - float(exact_sol.delta)) / max(1e-30, abs(float(exact_sol.delta))),
                    "highs_finisher_seconds": t_exact,
                    "is_optimal": bool(check.is_optimal), "relative_gap": float(check.relative_gap),
                    "delta_over_eps": float(r_gpu.delta) / 1e-4,
                })
                r = rows[-1]
                print(f"[test6] n={n:>5} seed={seed} {family:<9} gpu_delta={r['gpu_delta']:.4g} "
                      f"highs_delta={r['highs_delta_on_F']:.4g} |diff|={r['delta_abs_diff']:.2e} "
                      f"is_optimal={r['is_optimal']} finisher={t_exact:.2f}s", flush=True)
    out = Path(args.out) if args.out else (_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "diag" / "test6_delta_cert.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"test": "test6", "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[test6] wrote {out}")
    return 0


def cmd_test7(args: argparse.Namespace) -> int:
    """CN: Test 7 —— eps 扫描（冻结 F，避免循环差异）：每次 (n, eps) 解同一 F，记录时间、δ/eps、与 HiGHS 的差。
    EN: Test 7 -- eps sweep on a FROZEN F (avoids loop variation): per (n, eps) solve the same F,
        recording time, delta/eps, and the HiGHS delta gap."""
    rows: list[dict[str, object]] = []
    eps_list = [float(v) for v in args.eps]
    for n in [int(v) for v in args.sizes]:
        problem = build_problem(
            family=str(args.family), point_dimension=int(args.point_dimension),
            marginal=str(args.marginal), n=int(n), seed=int(args.seed),
        )
        r_ref, rec_ref, _log = run_loop(
            problem, inner=CuPdlpxMasterSolver(params={"time_sec_limit": float(args.cupdlpx_cap)}),
            prune=False, hysteresis=False, max_iterations=int(args.max_iterations), scan_backend=str(args.scan),
        )
        frozen_F = rec_ref.edge_sets[-1]
        hs = LinearMasterSolver(random_seed=int(args.seed)).solve(problem, frozen_F, tie_break="none")
        delta_highs = float(hs.delta)
        print(f"[test7] n={n} frozen |F|={frozen_F.size} highs_delta={delta_highs:.6g}", flush=True)
        for eps in eps_list:
            solver = CuPdlpxMasterSolver(
                params={"eps_optimal_relative": eps, "eps_feasible_relative": eps,
                        "time_sec_limit": float(args.cupdlpx_cap)}
            )
            t0 = time.perf_counter()
            sol = solver.solve(problem, frozen_F, tie_break="none")
            secs = time.perf_counter() - t0
            rows.append({
                "n": int(n), "eps": eps, "delta": float(sol.delta), "seconds": secs,
                "delta_over_eps": float(sol.delta) / eps,
                "delta_abs_diff_vs_highs": abs(float(sol.delta) - delta_highs),
            })
            r = rows[-1]
            print(f"[test7]   eps={eps:.0e} t={secs:7.2f}s delta={sol.delta:.6g} "
                  f"delta/eps={r['delta_over_eps']:.3g} |diff_vs_highs|={r['delta_abs_diff_vs_highs']:.2e}", flush=True)
    out = Path(args.out) if args.out else (_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "diag" / "test7_eps_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"test": "test7", "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[test7] wrote {out}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="small-scale loop diagnostics")
    parser.add_argument("test", choices=("test1", "test2", "test3", "test4", "test5", "test6", "test7"))
    parser.add_argument("--family", default="diagonal")
    parser.add_argument("--families", nargs="+", default=["diagonal", "quadratic", "lowrank"])
    parser.add_argument("--point-dimension", type=int, default=8)
    parser.add_argument("--marginal", default="gmm")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-4, 1e-5, 1e-6, 1e-7])
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--scan", default="numpy")
    parser.add_argument("--rho", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--cupdlpx-cap", type=float, default=60.0)
    parser.add_argument("--freeze", choices=("half", "last"), default="half")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.test == "test1":
        return cmd_test1(args)
    if args.test == "test2":
        return cmd_test2(args)
    if args.test == "test3":
        return cmd_test3(args)
    if args.test == "test4":
        return cmd_test4(args)
    if args.test == "test5":
        return cmd_test5(args)
    if args.test == "test6":
        return cmd_test6(args)
    if args.test == "test7":
        return cmd_test7(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
