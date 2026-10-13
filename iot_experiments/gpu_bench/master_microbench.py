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


def _build_problem(*, family: str, k: int, marginal: str, n: int, seed: int, num_itermax: int):
    """CN: 构造一个 (problem, F, |S|)。EN: Build one (problem, F, |S|)."""
    import ot
    from scipy import sparse

    from iot import (
        InverseOTProblem,
        dense_cost_matrix,
        mandatory_edges,
        EdgeSet,
    )

    rng = np.random.default_rng(int(seed))
    spec = _family_for_k(family, int(k))
    features, theta_set, theta_star = build_family(spec, rng, int(n), int(n))
    source_mass = marginal_spec(marginal).weights(int(n))
    target_mass = marginal_spec(marginal).weights(int(n))
    cost = dense_cost_matrix(features, theta_star)
    plan = np.asarray(ot.emd(source_mass, target_mass, cost, numItermax=int(num_itermax)))
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))
    mandatory, _ = mandatory_edges(problem)
    return problem, mandatory, int(problem.support_size), rng


def _time_backend(solver, problem, edges, repeats: int) -> tuple[float, float, str | None]:
    """CN: 预热一次后重复计时，返回 (中位墙钟, delta, 错误)。EN: Warm up once, repeat, return (median wall, delta, error)."""
    from iot import InverseOTError

    try:
        solver.solve(problem, edges, tie_break="none")
    except InverseOTError as error:
        return float("nan"), float("nan"), str(error)[:200]
    walls: list[float] = []
    delta = float("nan")
    for _ in range(int(repeats)):
        started = time.perf_counter()
        solution = solver.solve(problem, edges, tie_break="none")
        walls.append(time.perf_counter() - started)
        delta = float(solution.delta)
    return float(np.median(walls)), delta, None


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
) -> list[dict[str, object]]:
    """CN: 返回每条 (seed,K,backend) 的记录。EN: Return one record per (seed, K, backend)."""
    from iot import CuPdlpxMasterSolver, EdgeSet, RegularizedMasterSolver

    records: list[dict[str, object]] = []
    for seed in seeds:
        for k in ks:
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
            delta_ref = float("nan")
            for backend in backends:
                if backend == "highs":
                    solver = RegularizedMasterSolver(epsilon=1e-4, random_seed=int(seed))
                elif backend == "cupdlpx":
                    solver = CuPdlpxMasterSolver()
                else:
                    raise ValueError(f"unknown backend {backend!r}")
                wall, delta, error = _time_backend(solver, problem, edges, int(repeats))
                if backend == "highs" and not math.isnan(delta):
                    delta_ref = delta
                records.append(
                    {
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
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from iot_experiments.gpu_bench import gpu

    gpu.configure(args.gpu)
    records = run_master_microbench(
        family=str(args.family), marginal=str(args.marginal), n=int(args.n),
        ks=[int(v) for v in args.ks], seeds=[int(v) for v in args.seeds],
        repeats=int(args.repeats), extra_edges=int(args.extra_edges),
        backends=[str(b) for b in args.backends],
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
