"""
CN: master LP 的隔离微基准：在**同一条边集 F、同一个问题**上比较 HiGHS（CPU）与 CuPDLPx（GPU）。
    这直接检验「大 K 时 GPU LP 追赶/超过 HiGHS」的前提，不夹杂 oracle 扫描。
EN: Isolated master-LP microbenchmark: HiGHS (CPU) vs CuPDLPx (GPU) on the *same problem and
    the same edge set F*. This tests the "GPU LP catches up to HiGHS at large K" premise
    without the oracle scan in the way.

CN: F 取 B = S ∪ R（mandatory）再加 `extra_edges` 条随机非支撑边，使 |F| 可控且可复现。
    `K` 直接由族决定：lowrank/diagonal 的 K 就是参数本身。
EN: F is B = S ∪ R (mandatory) plus `extra_edges` random non-support edges, so |F| is a
    controlled, reproducible knob. `K` is set directly by the family for lowrank/diagonal.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.core.families import FamilySpec, build_family
from iot_experiments.core.marginals import marginal_spec


def _family_for_k(family: str, k: int) -> FamilySpec:
    """CN: 由目标 K 构造族规格。EN: Build a family spec from the target K."""
    if family == "lowrank":
        return FamilySpec("lowrank", lowrank_k=int(k))
    if family == "diagonal":
        return FamilySpec("diagonal", point_dimension=int(k))
    if family == "quadratic":
        # K = d(d+1)/2  =>  d = (-1 + sqrt(1 + 8K)) / 2
        d = int(round((math.isqrt(1 + 8 * int(k)) - 1) / 2))
        if d * (d + 1) // 2 != int(k):
            raise ValueError(f"quadratic cannot realise K={k} (need K=d(d+1)/2)")
        return FamilySpec("quadratic", point_dimension=d)
    raise ValueError(f"unsupported family for K-sweep: {family!r}")


def _cost_matrix(features, theta, *, byte_budget: int = 256 * 1024 * 1024) -> np.ndarray:
    """
    CN: 分块构造 (m,n) 代价矩阵，避免一次性 (m*n, K) 中间张量（K~10^3 时那是数 GB）。
    EN: Blockwise (m,n) cost matrix, avoiding a single (m*n, K) intermediate (GBs at K~10^3).
    """
    m = int(features.n_source)
    n = int(features.n_target)
    kdim = int(features.n_features)
    per_edge_bytes = 8 * max(1, kdim)
    rows_per_block = max(1, int(byte_budget) // max(1, per_edge_bytes) // max(1, n))
    cost = np.empty((m, n), dtype=np.float64)
    for start in range(0, m, rows_per_block):
        stop = min(m, start + rows_per_block)
        rows = np.repeat(np.arange(start, stop, dtype=np.int64), n)
        cols = np.tile(np.arange(n, dtype=np.int64), stop - start)
        cost[start:stop] = np.asarray(features.costs(theta, rows, cols)).reshape(stop - start, n)
    return cost


def _build_problem(*, family: str, k: int, marginal: str, n: int, seed: int, num_itermax: int):
    """CN: 构造一个 (problem, F, |S|)。EN: Build one (problem, F, |S|)."""
    import ot
    from scipy import sparse

    from iot import InverseOTProblem, mandatory_edges

    rng = np.random.default_rng(int(seed))
    spec = _family_for_k(family, int(k))
    features, theta_set, theta_star = build_family(spec, rng, int(n), int(n))
    source_mass = marginal_spec(marginal).weights(int(n))
    target_mass = marginal_spec(marginal).weights(int(n))
    cost = _cost_matrix(features, theta_star)
    plan = np.asarray(ot.emd(source_mass, target_mass, cost, numItermax=int(num_itermax)))
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))
    mandatory, _ = mandatory_edges(problem)
    return problem, mandatory, int(problem.support_size), rng


def _time_backend(solver, problem, edges, repeats: int) -> tuple[float, float, str | None]:
    """CN: 预热一次后重复计时；即便抛错也记录墙钟（例如 HiGHS 命中 time_limit）。
    EN: Warm up once then repeat; record the wall even on failure (e.g. a HiGHS time limit)."""
    from iot import InverseOTError

    error: str | None = None
    try:
        solver.solve(problem, edges, tie_break="none")
    except InverseOTError as exc:
        error = str(exc)[:200]
    walls: list[float] = []
    delta = float("nan")
    for _ in range(int(repeats)):
        started = time.perf_counter()
        try:
            solution = solver.solve(problem, edges, tie_break="none")
            delta = float(solution.delta)
        except InverseOTError as exc:
            error = str(exc)[:200]
        walls.append(time.perf_counter() - started)
    wall = float(np.median(walls)) if walls else float("nan")
    return wall, delta, error


def _install_highs_time_limit(seconds: float | None):
    """
    CN: 给 `iot.master._solve_system` 注入 HiGHS `time_limit`，防止大 |F| 时无限迭代；
        返回原始函数以便还原（None 表示未安装）。
    EN: Inject a HiGHS `time_limit` into `iot.master._solve_system` so a large-|F| LP cannot
        iterate forever; returns the original for restoration (None = not installed).
    """
    if seconds is None:
        return None
    import iot.master as master_module
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, vstack

    original = master_module._solve_system

    def patched(system, objective, *, extra_eq=None, extra_eq_rhs=None, extra_ub=None,
                extra_ub_rhs=None, bounds=None):
        a_eq, b_eq, a_ub, b_ub = system.a_eq, system.b_eq, system.a_ub, system.b_ub
        if extra_eq:
            a_eq = vstack([a_eq, csr_matrix(np.vstack(extra_eq))], format="csr")
            b_eq = np.concatenate([b_eq, np.asarray(extra_eq_rhs, dtype=np.float64)])
        if extra_ub:
            a_ub = vstack([a_ub, csr_matrix(np.vstack(extra_ub))], format="csr")
            b_ub = np.concatenate([b_ub, np.asarray(extra_ub_rhs, dtype=np.float64)])
        result = linprog(
            np.asarray(objective, dtype=np.float64),
            A_ub=(a_ub if a_ub.shape[0] else None), b_ub=(b_ub if a_ub.shape[0] else None),
            A_eq=(a_eq if a_eq.shape[0] else None), b_eq=(b_eq if a_eq.shape[0] else None),
            bounds=system.bounds if bounds is None else bounds,
            method="highs", options={"time_limit": float(seconds)},
        )
        if not result.success or result.x is None:
            return None, None, str(result.message)
        return np.asarray(result.x, dtype=np.float64), float(result.fun), str(result.message)

    master_module._solve_system = patched
    return original


def run_master_microbench(
    *,
    family: str = "lowrank",
    marginal: str = "gmm",
    n: int = 800,
    ks: Sequence[int] = (64, 128, 256, 512, 1024),
    seeds: Sequence[int] = (0,),
    repeats: int = 3,
    extra_edges: int = 0,
    backends: Sequence[str] = ("highs", "cupdlpx"),
    num_itermax: int = 2_000_000,
    cupdlpx_params: dict[str, object] | None = None,
    highs_time_limit: float | None = None,
) -> list[dict[str, object]]:
    """CN: 返回每条 (seed,K,backend) 的记录。EN: Return one record per (seed, K, backend)."""
    from iot import CuPdlpxMasterSolver, EdgeSet, RegularizedMasterSolver

    cupdlpx_params = dict(cupdlpx_params or {})
    records: list[dict[str, object]] = []
    _original_solve = _install_highs_time_limit(highs_time_limit)
    for seed in seeds:
        for k in ks:
            build_started = time.perf_counter()
            problem, mandatory, support_size, rng = _build_problem(
                family=family, k=int(k), marginal=marginal, n=int(n), seed=int(seed),
                num_itermax=int(num_itermax),
            )
            edges = mandatory
            if int(extra_edges) > 0:
                rows = rng.integers(0, int(n), size=int(extra_edges))
                cols = rng.integers(0, int(n), size=int(extra_edges))
                edges = mandatory.union(EdgeSet.from_arrays(rows, cols, n_target=int(n)))
            n_edges = int(edges.size)
            print(
                f"[microbench] K={int(k)} seed={int(seed)} |F|={n_edges} "
                f"build={time.perf_counter() - build_started:.2f}s",
                flush=True,
            )
            delta_ref = float("nan")
            for backend in backends:
                if backend == "highs":
                    solver = RegularizedMasterSolver(epsilon=1e-4, random_seed=int(seed))
                elif backend == "cupdlpx":
                    solver = CuPdlpxMasterSolver(params=dict(cupdlpx_params))
                else:
                    raise ValueError(f"unknown backend {backend!r}")
                wall, delta, error = _time_backend(solver, problem, edges, int(repeats))
                if backend == "highs" and not math.isnan(delta):
                    delta_ref = delta
                print(
                    f"[microbench]   {backend:>8} wall={wall:.3f}s delta={delta:.6f} err={error}",
                    flush=True,
                )
                records.append(
                    {
                        "cupdlpx_params": (dict(cupdlpx_params) if backend == "cupdlpx" else {}),
                        "highs_time_limit": (highs_time_limit if backend == "highs" else None),
                        "family": family,
                        "marginal": marginal,
                        "n": int(n),
                        "k": int(k),
                        "seed": int(seed),
                        "backend": backend,
                        "n_edges": n_edges,
                        "support_size": int(support_size),
                        "repeats": int(repeats),
                        "wall_seconds_median": wall,
                        "delta": delta,
                        "delta_abs_diff_vs_highs": (
                            abs(delta - delta_ref)
                            if (not math.isnan(delta) and not math.isnan(delta_ref))
                            else float("nan")
                        ),
                        "error": error,
                    }
                )
    if _original_solve is not None:
        import iot.master as master_module

        master_module._solve_system = _original_solve
    return records


def main(argv: list[str] | None = None) -> int:
    """CN: 独立 CLI（一般由 run_batch 调用）。EN: Standalone CLI (normally called by run_batch)."""
    import argparse

    parser = argparse.ArgumentParser(description="isolated master-LP microbenchmark")
    parser.add_argument("--family", default="lowrank")
    parser.add_argument("--marginal", default="gmm")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--extra-edges", type=int, default=0)
    parser.add_argument("--backends", nargs="+", default=["highs", "cupdlpx"])
    parser.add_argument("--cupdlpx-time-limit", type=float, default=None, help="CuPDLPx seconds cap")
    parser.add_argument("--cupdlpx-iters", type=int, default=None, help="CuPDLPx iteration cap")
    parser.add_argument("--highs-time-limit", type=float, default=None, help="HiGHS seconds cap")
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from iot_experiments.gpu_bench import gpu

    gpu.configure(args.gpu)
    cupdlpx_params: dict[str, object] = {}
    if args.cupdlpx_time_limit is not None:
        cupdlpx_params["time_sec_limit"] = float(args.cupdlpx_time_limit)
    if args.cupdlpx_iters is not None:
        cupdlpx_params["iteration_limit"] = int(args.cupdlpx_iters)
    records = run_master_microbench(
        family=str(args.family), marginal=str(args.marginal), n=int(args.n),
        ks=[int(v) for v in args.ks], seeds=[int(v) for v in args.seeds],
        repeats=int(args.repeats), extra_edges=int(args.extra_edges),
        backends=[str(b) for b in args.backends], cupdlpx_params=cupdlpx_params,
        highs_time_limit=args.highs_time_limit,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
