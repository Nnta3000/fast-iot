"""
EN: **T3 restricted edge set (fixed `F`) -- does a sparse master stay cheap, and does it still approximate U?**

    The question this answers is NOT "is T3 finished". It is the two-part question the restricted master has
    to pass before an adaptive loop or a certificate is meaningful:

      1. **Restricted accuracy and cost.** On a fixed edge set `F ⊊ U`, how accurately and how quickly does
         the joint master minimise `Psi_F`?
      2. **Agreement with the full grid.** `Psi_F` is a DIFFERENT objective from `Psi_U`: dropping edges
         removes strictly positive exponential terms, so

             Psi_F(theta, g) <= Psi_U(theta, g)   for every (theta, g),

         and the minimiser of `Psi_F` need not be that of `Psi_U`. A cheap restricted solve proves nothing on
         its own, so every row reports both the restricted optimum AND the full-grid (omitted-edge) quantity
         at the restricted solution.

    Edge sets. For each size and each target fraction, `F` is built from the SUPPORT plus connectors:

        S = {(i, j) : pi_hat[i,j] > 0},   R = connectors making (I, J, S u R) connected (spec section 3),
        B = S u R,  then F = B u (the highest-mass edges of the full-grid plan) up to the requested size.

    `F` always contains `B`, so every source and target node is represented and the bipartite graph is
    connected -- which the joint system requires (one gauge null direction, fixed by `g_0 = 0`).

CN: **T3 受限边集（固定 `F`）—— 稀疏 master 是否既便宜又能近似全网格？**
    `Psi_F` 与 `Psi_U` 是**不同的目标**：去掉边会删掉正指数项，故 `Psi_F <= Psi_U`，两者最优解不必相同。
    因此每行同时记录受限最优值与在全网格上的量。`F` 始终包含 `B = S ∪ R`，保证连通且每个节点都有代表。
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.parameters import cost_std  # noqa: E402
from iot.entropic.joint_reference import (  # noqa: E402
    ReducedProblem,
    evaluate,
    marginals,
    simplex_kkt,
    solve,
)
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.lib.planting import plant_instance  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402

SIZES: tuple[int, ...] = (70, 120, 300)
REL_EPS_GRID: tuple[float, ...] = (0.5, 0.25)
FRACTIONS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20)
POINT_DIMENSION: int = 4
SEEDS: tuple[int, ...] = (0, 1)

PROVENANCE: dict[str, object] = {
    "question": "on a fixed restricted F: how accurate/cheap is the restricted solve, and how well does it approximate the full grid?",
    "edge_rule": "F = B u top-mass edges of the full-grid plan, where B = S u R and R keeps the graph connected",
    "solver": "iot.entropic.joint_reference (SciPy SLSQP, joint reduced objective)",
    "note": "Psi_F <= Psi_U for every point; the restricted optimum is NOT the full-grid optimum",
    "connector_set": "spec section 3: without R the master has an unbounded delta and componentwise shifts",
}


@dataclasses.dataclass(frozen=True)
class RestrictedConfig(RunConfig):
    """EN: knobs. CN: 旋钮。"""

    section = "t3_restricted_f"
    study = "t3_restricted_f"
    point_dimension: int = POINT_DIMENSION
    max_iterations: int = 2000

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """EN: only this study's flags. CN: 只加本脚本特有的参数。"""
        parser.add_argument("--point-dimension", type=int, default=POINT_DIMENSION)
        parser.add_argument("--max-iterations", type=int, default=2000)


def components(rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"],
               n_source: int, n_target: int) -> int:
    """
    EN: number of connected components of the bipartite graph, by union-find. CN: 并查集求连通分量数。
    """
    parent = np.arange(int(n_source) + int(n_target), dtype=np.int64)

    def find(node: int) -> int:
        root = int(node)
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != root:
            parent[node], node = root, int(parent[node])
        return root

    for row, col in zip(np.asarray(rows).tolist(), np.asarray(cols).tolist(), strict=True):
        left, right = find(int(row)), find(int(n_source) + int(col))
        if left != right:
            parent[right] = left
    return int(np.unique([find(node) for node in range(parent.size)]).size)


def build_edge_set(support_rows: Int[np.ndarray, "support"], support_cols: Int[np.ndarray, "support"],
                   grid_rows: Int[np.ndarray, "edges"], grid_cols: Int[np.ndarray, "edges"],
                   plan: Float[np.ndarray, "edges"], n_source: int, n_target: int,
                   fraction: float) -> tuple[Int[np.ndarray, "f"], Int[np.ndarray, "f"]]:
    """
    EN: build `F = B u (top-mass grid edges)` to the requested fraction, guaranteeing connectivity.

        `B = S u R`: `S` is the observed support; `R` connects the components of `S` in a cycle (`i_k` to
        `j_{k+1 mod p}`), exactly as spec section 3 requires. Edges are then added in decreasing `pi_hat`
        order of the FULL-GRID plan until `|F|` reaches `fraction * |U|`. The result is checked for
        connectivity and for full node coverage and the function raises if either fails.

    CN: 构造 `F = B ∪ (质量最高的网格边)`，并保证连通与节点覆盖。
    """
    node_keys = np.asarray(np.asarray(support_rows) * int(n_target) + np.asarray(support_cols),
                           dtype=np.int64)
    base_rows = list(np.asarray(support_rows).tolist())
    base_cols = list(np.asarray(support_cols).tolist())
    base_keys = set(node_keys.tolist())
    # EN: connector set R -- one edge per component boundary, cycling through components.
    # CN: 连接边集 R —— 按分量循环添加。
    labels_parent = np.arange(int(n_source) + int(n_target), dtype=np.int64)

    def find(node: int) -> int:
        root = int(node)
        while labels_parent[root] != root:
            root = int(labels_parent[root])
        return root

    for row, col in zip(base_rows, base_cols, strict=True):
        left, right = find(row), find(int(n_source) + col)
        if left != right:
            labels_parent[right] = left
    roots = [find(node) for node in range(int(n_source) + int(n_target))]
    unique = sorted(set(roots))
    component_sources: dict[int, int] = {}
    component_targets: dict[int, int] = {}
    for node, root in enumerate(roots):
        if node < int(n_source):
            component_sources.setdefault(root, node)
        else:
            component_targets.setdefault(root, node - int(n_source))
    order = [root for root in unique if root in component_sources and root in component_targets]
    for position, root in enumerate(order):
        nxt = order[(position + 1) % len(order)]
        row = component_sources[root]
        col = component_targets[nxt]
        key = row * int(n_target) + col
        if key not in base_keys:
            base_keys.add(key)
            base_rows.append(row)
            base_cols.append(col)

    target = max(len(base_keys), int(round(float(fraction) * int(grid_rows.size))))
    grid_keys = np.asarray(np.asarray(grid_rows) * int(n_target) + np.asarray(grid_cols),
                           dtype=np.int64)
    order_by_mass = np.argsort(-np.asarray(plan, dtype=np.float64))
    taken = list(base_rows), list(base_cols)
    keys = set(base_keys)
    for index in order_by_mass.tolist():
        if len(keys) >= target:
            break
        key = int(grid_keys[index])
        if key in keys:
            continue
        keys.add(key)
        taken[0].append(int(grid_rows[index]))
        taken[1].append(int(grid_cols[index]))
    rows = np.asarray(taken[0], dtype=np.int64)
    cols = np.asarray(taken[1], dtype=np.int64)
    found = components(rows, cols, n_source, n_target)
    if found != 1:
        raise ValueError(f"edge set has {found} components; F must contain the connector set B")
    if np.unique(rows).size != int(n_source) or np.unique(cols).size != n_target:
        raise ValueError("edge set does not cover every source and target node")
    return rows, cols


def _run_case(n: int, seed: int, rel_eps: float, fraction: float, point_dimension: int,
              max_iterations: int) -> dict[str, object]:
    """
    EN: solve the full grid once, then the restricted problem, and report both restricted accuracy and
        full-grid agreement. Failures are recorded, never retried.
    CN: 先解全网格，再解受限问题，同时报告受限精度与全网格一致性。
    """
    started = time.perf_counter()
    planted = plant_instance(family="diagonal", n=int(n), point_dimension=int(point_dimension), seed=int(seed))
    problem = planted.problem
    m, jj, k = int(problem.features.n_source), int(problem.features.n_target), int(planted.theta_star.size)
    all_rows = np.repeat(np.arange(m), jj)
    all_cols = np.tile(np.arange(jj), m)
    full_features = np.asarray(problem.features.features(all_rows, all_cols), dtype=np.float64)
    a = np.asarray(problem.resolved_source_mass, dtype=np.float64)
    b = np.asarray(problem.resolved_target_mass, dtype=np.float64)
    support_rows, support_cols, support_mass = problem.support()
    phi_bar = (np.asarray(support_mass, dtype=np.float64)[:, None]
               * np.asarray(problem.features.features(support_rows, support_cols),
                            dtype=np.float64)).sum(axis=0)
    epsilon = float(rel_eps) * float(cost_std(full_features @ np.full(k, 1.0 / k)))
    full = ReducedProblem(full_features, np.asarray(all_rows, dtype=np.int64),
                          np.asarray(all_cols, dtype=np.int64), a, b, phi_bar, epsilon)
    t0 = time.perf_counter()
    full_run = solve(full, np.full(k, 1.0 / k), np.zeros(jj), label="full", max_iterations=max_iterations)
    full_seconds = time.perf_counter() - t0
    _psi_full, grad_full, _gg_full, full_plan = evaluate(full, full_run.theta, full_run.g)

    rows, cols = build_edge_set(np.asarray(support_rows, dtype=np.int64),
                                np.asarray(support_cols, dtype=np.int64),
                                np.asarray(all_rows, dtype=np.int64), np.asarray(all_cols, dtype=np.int64),
                                full_plan, m, jj, fraction)
    restricted = ReducedProblem(np.asarray(problem.features.features(rows, cols), dtype=np.float64),
                                rows, cols, a, b, phi_bar, epsilon)
    t1 = time.perf_counter()
    restricted_run = solve(restricted, np.full(k, 1.0 / k), np.zeros(jj), label="f",
                           max_iterations=max_iterations)
    restricted_seconds = time.perf_counter() - t1
    psi_f, grad_theta_f, grad_g_f, plan_f = evaluate(restricted, restricted_run.theta, restricted_run.g)
    # EN: the SAME point evaluated on the FULL grid: this is what "omitted edges" cost.
    # CN: 在同一 (theta, g) 上评估全网格目标，即被省略的边带来的差别。
    psi_u_at_theta_f, _gt, _gg, per_edge = evaluate(full, restricted_run.theta, restricted_run.g)
    restricted_plan_sum = float(plan_f.sum())
    full_plan_at_theta_f = float(per_edge.sum())
    row_error, col_error = marginals(restricted, plan_f)
    _multiplier, kkt_violation = simplex_kkt(grad_theta_f, restricted_run.theta)
    # EN: which full-grid edges survived in F, so the omitted plan mass can be measured exactly.
    # CN: F 中保留了哪些全网格边，用于精确测量被省略的质量。
    key_keep = set((rows * jj + cols).tolist())
    kept = np.array([(int(r) * jj + int(c)) in key_keep
                     for r, c in zip(all_rows.tolist(), all_cols.tolist())], dtype=bool)
    omitted_mass_at_theta_f = float(per_edge[~kept].sum())
    total_mass_at_theta_f = full_plan_at_theta_f
    return {
        "n": int(n), "seed": int(seed), "rel_eps": float(rel_eps), "ok": True,
        "fraction_requested": float(fraction),
        "edges_full": int(all_rows.size), "edges_restricted": int(rows.size),
        "fraction_actual": float(rows.size) / float(all_rows.size),
        "dense_variables": int(k + jj - 1), "K": int(k),
        "full_seconds": round(full_seconds, 4),
        "restricted_seconds": round(restricted_seconds, 4),
        "time_ratio": round(restricted_seconds / max(full_seconds, 1e-12), 4),
        "psi_full": float(full_run.psi), "psi_restricted": float(psi_f),
        "psi_full_at_theta_restricted": float(psi_u_at_theta_f),
        "psi_gap_vs_full_optimum": float(psi_u_at_theta_f - full_run.psi),
        "theta_l1_vs_full": float(np.abs(restricted_run.theta - full_run.theta).sum()),
        "row_marginal_error": float(row_error), "col_marginal_error": float(col_error),
        "theta_kkt_violation": float(kkt_violation),
        "restricted_plan_mass": restricted_plan_sum,
        "full_plan_mass_at_theta_restricted": total_mass_at_theta_f,
        "omitted_edge_mass_at_theta_restricted": omitted_mass_at_theta_f,
        "omitted_mass_fraction": float(omitted_mass_at_theta_f / max(total_mass_at_theta_f, 1e-30)),
        "restricted_status": int(restricted_run.status),
        "restricted_message": str(restricted_run.message)[:120],
        "total_seconds": round(time.perf_counter() - started, 4),
    }


def main(argv: list[str] | None = None) -> int:
    """EN: entry point; dispatches per-case workers. CN: 入口；分派单个用例的子进程。"""
    tokens = list(argv) if argv is not None else sys.argv[1:]
    if "--worker" in tokens:
        position = tokens.index("--worker")
        n_text, seed_text, rel_text, frac_text = tokens[position + 1:position + 5]
        record = _run_case(int(n_text), int(seed_text), float(rel_text), float(frac_text),
                           POINT_DIMENSION, 2000)
        print(json.dumps(record))
        return 0

    parser = build_parser(RestrictedConfig)
    args = parser.parse_args(tokens)
    config = RestrictedConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("T3 restricted-F: sparse master accuracy/cost and full-grid agreement")
    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem={preflight.mem_available_gb:.1f} GB")
    log.line(f"sizes={SIZES} rel_eps={REL_EPS_GRID} fractions={FRACTIONS} seeds={SEEDS}")

    records: list[dict[str, object]] = []
    for n in SIZES:
        for seed in SEEDS:
            for rel_eps in REL_EPS_GRID:
                for fraction in FRACTIONS:
                    log.line(f"n={n} seed={seed} rel_eps={rel_eps} frac={fraction}: start")
                    command = [sys.executable, "-m",
                               "iot_experiments.studies.real.t1t2.metric_learning.t3_restricted_f",
                               "--worker", str(n), str(seed), str(rel_eps), str(fraction)]
                    completed = subprocess.run(command, capture_output=True, text=True,
                                               cwd=str(_REPO_ROOT))
                    if completed.returncode != 0:
                        record: dict[str, object] = {
                            "n": n, "seed": seed, "rel_eps": rel_eps, "fraction_requested": fraction,
                            "ok": False, "stop_reason": f"worker exit {completed.returncode}",
                            "stderr": completed.stderr[-400:]}
                    else:
                        try:
                            record = json.loads(completed.stdout.strip().splitlines()[-1])
                        except (ValueError, IndexError):
                            record = {"n": n, "seed": seed, "rel_eps": rel_eps,
                                      "fraction_requested": fraction, "ok": False,
                                      "stop_reason": "unparseable worker output"}
                    records.append(record)
                    if record.get("ok"):
                        log.line(f"    |F|={record['edges_restricted']:>7} "
                                 f"({record['fraction_actual']:.3f}) "
                                 f"t_F={record['restricted_seconds']:>6.3f}s "
                                 f"t_U={record['full_seconds']:>6.3f}s "
                                 f"psi_F={record['psi_restricted']:.6e} "
                                 f"dpsi={record['psi_gap_vs_full_optimum']:+.2e} "
                                 f"thL1={record['theta_l1_vs_full']:.2e} "
                                 f"kept_mass={1-record['omitted_mass_fraction']:.3f}")
                    else:
                        log.line(f"    FAILED: {record.get('stop_reason')}")

    payload = {"records": records, "provenance": PROVENANCE, "preflight": preflight.to_dict(),
               "sizes": list(SIZES), "rel_eps": list(REL_EPS_GRID), "fractions": list(FRACTIONS),
               "seeds": list(SEEDS),
               "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                          for key, v in dataclasses.asdict(config).items()}}
    (run_dir / "restricted_f.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                               encoding="utf-8")
    log.line(f"wrote {run_dir / 'restricted_f.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
