"""
CN: 只构造 master LP（**不求解**），测各阶段耗时与 LP 规模（nnz）。
EN: Build the master LP only (**no solve**); time each phase and report the LP size (nnz).

CN: 用于回答「构造 LP 本身是不是瓶颈」：family 抽样、代价矩阵、POT 真值、mandatory 边、
    `build_master_system`（lil_matrix 组装）各自耗时，以及 nnz = |F|·(K+3)。
EN: Answers "is building the LP itself the bottleneck?": time for family sampling, the cost
    matrix, POT ground truth, mandatory edges, and `build_master_system` (lil assembly), plus
    nnz = |F|·(K+3).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.core.costs import blockwise_cost_matrix  # noqa: E402
from iot_experiments.core.families import build_family  # noqa: E402
from iot_experiments.core.marginals import marginal_spec  # noqa: E402
from iot_experiments.gpu_bench.master_microbench import _family_for_k  # noqa: E402


def probe_one(
    *, family: str, k: int, marginal: str, n: int, extra_edges: int, seed: int, num_itermax: int
) -> dict[str, object]:
    """CN: 构造一次 LP 并返回各阶段耗时与规模。EN: Build one LP and return per-phase timing/size."""
    import ot
    from scipy import sparse

    from iot import EdgeSet, InverseOTProblem, mandatory_edges
    from iot.exact.master import build_master_system

    rng = np.random.default_rng(int(seed))
    started = time.perf_counter()
    features, theta_set, theta_star = build_family(_family_for_k(family, int(k)), rng, int(n), int(n))
    t_family = time.perf_counter() - started

    started = time.perf_counter()
    cost = blockwise_cost_matrix(features, theta_star)
    t_cost = time.perf_counter() - started

    source_mass = marginal_spec(marginal).weights(int(n))
    target_mass = marginal_spec(marginal).weights(int(n))
    started = time.perf_counter()
    plan = np.asarray(ot.emd(source_mass, target_mass, cost, numItermax=int(num_itermax)))
    t_pot = time.perf_counter() - started

    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))
    started = time.perf_counter()
    mandatory, _ = mandatory_edges(problem)
    t_mandatory = time.perf_counter() - started

    edges = mandatory
    if int(extra_edges) > 0:
        rows = rng.integers(0, int(n), size=int(extra_edges))
        cols = rng.integers(0, int(n), size=int(extra_edges))
        edges = mandatory.union(EdgeSet.from_arrays(rows, cols, n_target=int(n)))

    started = time.perf_counter()
    system = build_master_system(problem, edges)
    t_lp = time.perf_counter() - started

    nnz = int(system.a_eq.nnz) + int(system.a_ub.nnz)
    return {
        "family": family, "k": int(k), "n": int(n), "extra_edges": int(extra_edges),
        "k_features": int(problem.n_features), "support_size": int(problem.support_size),
        "n_edges": int(edges.size),
        "t_family_s": t_family, "t_cost_s": t_cost, "t_pot_s": t_pot,
        "t_mandatory_s": t_mandatory, "t_build_lp_s": t_lp,
        "t_total_s": t_family + t_cost + t_pot + t_mandatory + t_lp,
        "n_eq": int(system.a_eq.shape[0]), "n_ub": int(system.a_ub.shape[0]),
        "n_vars": int(system.a_eq.shape[1]), "nnz": nnz,
        "nnz_over_edges_times_k": float(nnz) / max(1.0, float(edges.size) * (int(problem.n_features) + 3)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="master-LP build-only probe")
    parser.add_argument("--family", default="lowrank")
    parser.add_argument("--marginal", default="gmm")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--ks", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--extra-edges", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-itermax", type=int, default=2_000_000)
    args = parser.parse_args(argv)

    header = (
        f"{'K':>5} {'|S|':>6} {'|F|':>7} {'nnz':>11} "
        f"{'fam_s':>6} {'cost_s':>7} {'pot_s':>6} {'mand_s':>6} {'buildLP_s':>9} {'total_s':>8}"
    )
    print(header, flush=True)
    for extra in args.extra_edges:
        for k in args.ks:
            row = probe_one(
                family=str(args.family), k=int(k), marginal=str(args.marginal), n=int(args.n),
                extra_edges=int(extra), seed=int(args.seed), num_itermax=int(args.num_itermax),
            )
            print(
                f"{row['k']:>5} {row['support_size']:>6} {row['n_edges']:>7} {row['nnz']:>11} "
                f"{row['t_family_s']:>6.2f} {row['t_cost_s']:>7.2f} {row['t_pot_s']:>6.2f} "
                f"{row['t_mandatory_s']:>6.2f} {row['t_build_lp_s']:>9.2f} {row['t_total_s']:>8.2f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
