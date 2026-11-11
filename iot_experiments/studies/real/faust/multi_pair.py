"""
EN: **Many plans, one `theta` -- a block-structured LP with a shared weight vector.**

    Fitting on a single pair wastes the other pairs, and concatenating pairs into one plan is wrong (it couples
    a source vertex to another pair's targets). The correct relaxation gives every pair its own potentials and
    its own edge set, but **one shared `theta`**:

        minimise   (1/P) * sum_p R_p(theta, f_p, g_p)
        s.t.       f_p[i] + g_p[j] - theta.phi^p_ij <= 0     for every edge (i,j) in F_p, every pair p
                   sum_k theta_k = 1,  theta_k >= 0

    so the solver sees `P` competitions and must pick weights that work for all of them. The pairs differ by
    subject and pose, so this is the "more diverse competition" argument from `notes/APPLICATIONS.md` §4.1.

    **What this is not.** The edge sets `F_p` are the ones each single-pair loop finished with, not
    regenerated inside the multi-pair problem. Without constraint generation on the joint LP, `theta_multi` is
    a solution of a *restricted* master, so its regret is a lower bound on the joint optimum. The comparison of
    interest -- does a shared `theta` transfer to held-out pairs better than a single-pair `theta`? -- is
    unaffected, because both are evaluated the same way on the same held-out pairs.

    **Memory.** The LP is sparse: `sum_p |F_p|` rows (~4 x 50k at n=1500) with ~16 non-zeros per row. Nothing of
    size `n*n` or `n*n*K` is built.

    CN: **多方案共享 θ**：每个训练对有自己的 `(f_p, g_p)` 与边集，`θ` 共享。`F_p` 取各单对循环最终的边集，
        没有在联合 LP 上做约束生成，所以 `theta_multi` 是**受限** master 的解（regret 是联合最优的下界）；
        但"共享 θ 与单对 θ 在留出对上的比较"不受影响，因为两者评估方式相同。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.multi_pair --n-sub 1500 --pairs 0 13 2 41 5 88 7 52
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from scipy.optimize import linprog

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.edges import EdgeSet  # noqa: E402
from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot.core.parameters import (  # noqa: E402
    PruningParameters,
    RelaxedObjective,
    SolverParameters,
)
from iot.core.problem import InverseOTProblem  # noqa: E402
from iot.core.theta import SimplexTheta  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot.relaxed.master_regret import RegretMasterSolver, regret_objective  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import read_ply  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.studies.real.faust.correspondence import (  # noqa: E402
    _evaluate_pair,
    _sample_index,
    default_pairs,
    descriptors,
    standardise,
)

DEFAULT_ROOT = "/home/agent/Documents/fast-iot/data/faust_extract/MPI-FAUST/training/registrations"

PROVENANCE: dict[str, object] = {
    "question": "does a theta shared across several training pairs transfer to held-out pairs better than a "
                "theta fitted on one pair?",
    "formulation": "block LP: one shared theta, per-pair (f_p, g_p), per-pair edge set F_p, objective is the mean "
                   "of the per-pair regrets",
    "edge_sets": "F_p is the edge set each single-pair loop finished with, NOT regenerated in the joint LP, so "
                 "theta_multi solves a RESTRICTED master and its regret is a lower bound on the joint optimum",
    "solver": "scipy HiGHS on the assembled sparse LP (one solve, no constraint generation)",
    "comparison": "theta_single (fitted on pair 1) vs theta_multi (shared over all P), both evaluated on the "
                  "same held-out test pairs",
}


@dataclasses.dataclass(frozen=True)
class MultiPairConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_multi_pair"
    section: str = "faust_multi_pair"
    root: str = DEFAULT_ROOT
    train_pairs: tuple[int, ...] = (0, 13, 2, 41, 5, 88, 7, 52)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    test_pair_count: int = 8
    n_sub: int = 1500
    max_iterations: int = 30
    per_pair_backend: str = "cupdlpx"   # the per-pair fits; the joint LP itself is solved by HiGHS

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pairs", type=int, nargs="+",
                            default=[0, 13, 2, 41, 5, 88, 7, 52])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--n-sub", type=int, default=1500)
        parser.add_argument("--max-iterations", type=int, default=30)
        parser.add_argument("--per-pair-backend", default="cupdlpx", choices=["cupdlpx", "highs"])


class _RecordingMaster(RegretMasterSolver):
    """CN: 记录最后一次求解用的边集（研究内包装，不改 `src/iot`）。EN: records the last edge set used."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.last_edges: EdgeSet | None = None

    def solve(self, problem: InverseOTProblem, edges: EdgeSet):  # type: ignore[no-untyped-def]
        self.last_edges = edges
        return super().solve(problem, edges)


@dataclasses.dataclass(frozen=True)
class _PairProblem:
    """CN: 一个训练对：特征族、方案、最终边集。EN: one training pair: family, plan, final edge set."""

    features: SquaredDifferenceFeatures
    problem: InverseOTProblem
    edges: EdgeSet
    theta_single: np.ndarray
    regret_single: float


def _solve_multi(pairs: list[_PairProblem], k: int) -> dict[str, object]:
    """
    CN: 装配并求解共享 θ 的块 LP：变量 `[θ | (f_p, g_p) for p]`，每个 `(p, e)` 一条不等式行。
    EN: assemble and solve the shared-θ block LP: variables `[theta | (f_p, g_p) for p]`, one inequality row per
        `(pair, edge)`.
    """
    total_rows = sum(int(p.edges.size) for p in pairs)
    ncols = int(k) + sum(int(p.problem.n_source) + int(p.problem.n_target) for p in pairs)
    rows_idx: list[np.ndarray] = []
    cols_idx: list[np.ndarray] = []
    data: list[np.ndarray] = []
    offset = int(k)
    objective = np.zeros(ncols, dtype=np.float64)
    scale = 1.0 / float(len(pairs))
    # CN: 用 `enumerate` 记录行偏移。`pairs.index(pair)` 会对 dataclass 字段做 `==`，
    #     而字段里含 numpy 数组 ⇒ "truth value ... is ambiguous"（实测崩溃）。
    # EN: track the row offset with `enumerate`. `pairs.index(pair)` compares dataclass fields with `==`, and
    #     those fields hold numpy arrays -> "truth value ... is ambiguous" (measured crash).
    row_offset = 0
    for pair in pairs:
        m = int(pair.problem.n_source)
        n = int(pair.problem.n_target)
        r = np.asarray(pair.edges.rows, dtype=np.int64)
        c = np.asarray(pair.edges.cols, dtype=np.int64)
        n_edges = int(r.size)
        base = np.arange(n_edges, dtype=np.int64) + int(row_offset)
        row_offset += n_edges
        # CN: -phi_e 在 θ 列上。EN: -phi_e on the theta columns.
        phi = np.asarray(pair.features.features(r, c), dtype=np.float64)
        rows_idx.append(np.repeat(base, int(k)))
        cols_idx.append(np.tile(np.arange(int(k), dtype=np.int64), n_edges))
        data.append((-phi).reshape(-1))
        # CN: f_p 与 g_p 的 +1。EN: +1 on f_p and g_p.
        rows_idx.append(base)
        cols_idx.append(offset + r)
        data.append(np.ones(n_edges, dtype=np.float64))
        rows_idx.append(base)
        cols_idx.append(offset + m + c)
        data.append(np.ones(n_edges, dtype=np.float64))
        objective[offset:offset + m] = -np.asarray(pair.problem.resolved_source_mass) * scale
        objective[offset + m:offset + m + n] = -np.asarray(pair.problem.resolved_target_mass) * scale
        objective[: int(k)] += scale * np.asarray(regret_objective(pair.problem)[: int(k)])
        offset += m + n

    a_ub = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows_idx), np.concatenate(cols_idx))),
                             shape=(total_rows, ncols)).tocsr()
    a_eq = sparse.csr_matrix(np.ones((1, int(k)), dtype=np.float64))
    a_eq = sparse.hstack([a_eq, sparse.csr_matrix((1, ncols - int(k)))], format="csr")
    bounds = [(0.0, None)] * int(k) + [(None, None)] * (ncols - int(k))
    result = linprog(objective, A_ub=a_ub, b_ub=np.zeros(total_rows, dtype=np.float64),
                     A_eq=a_eq, b_eq=np.ones(1), bounds=bounds, method="highs")
    if not result.success:
        return {"ok": False, "reason": str(result.message)[:140],
                "shape": (total_rows, ncols), "nnz": int(a_ub.nnz)}
    theta = np.asarray(result.x[: int(k)], dtype=np.float64)
    theta = np.maximum(theta, 0.0)
    total = float(theta.sum())
    theta = theta / total if total > 0 else theta
    return {"ok": True, "theta": theta, "objective": float(result.fun),
            "shape": (total_rows, ncols), "nnz": int(a_ub.nnz)}


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(MultiPairConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = MultiPairConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB")

    root = Path(config.root)
    flat = [int(v) for v in config.train_pairs]
    if len(flat) % 2:
        raise SystemExit("--train-pairs needs an even number of ids")
    train_pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    test_pairs = default_pairs(tuple(int(v) for v in config.test_subjects), int(config.test_pair_count))
    log.line(f"{len(train_pairs)} training pairs {train_pairs}; {len(test_pairs)} held-out pairs")

    built: list[_PairProblem] = []
    for pair in train_pairs:
        a = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        b = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        idx = _sample_index(a, config, 0)
        da, db = standardise(descriptors(a)[idx], descriptors(b)[idx])
        n = int(idx.size)
        plan = sparse.csr_matrix((np.full(n, 1.0 / n), (np.arange(n), np.arange(n))), shape=(n, n))
        family = SquaredDifferenceFeatures(da, db)
        problem = InverseOTProblem(features=family, theta_set=SimplexTheta(int(da.shape[1])), plan=plan)
        master = _RecordingMaster(
            backend=str(config.per_pair_backend), tie_break=False,
            params={"time_sec_limit": 300.0} if str(config.per_pair_backend) == "cupdlpx" else None)
        single = solve_inverse_ot_regret(
            problem,
            objective=RelaxedObjective(eps_r_rel=1e-6, check_pot=False, selection="none"),
            params=SolverParameters(tol=None, gamma=1, max_iterations=int(config.max_iterations),
                                    pruning=PruningParameters(enabled=False)),
            master=master)
        if master.last_edges is None:
            raise SystemExit(f"pair {pair}: no edge set recorded")
        built.append(_PairProblem(features=family, problem=problem, edges=master.last_edges,
                                  theta_single=np.asarray(single.theta, dtype=np.float64),
                                  regret_single=float(single.regret)))
        log.line(f"pair {pair}: |F|={int(master.last_edges.size)} regret_single={float(single.regret):.6g}")

    k = int(built[0].features.n_features)
    multi = _solve_multi(built, k)
    if not multi.get("ok"):
        log.line(f"multi-pair LP failed: {multi.get('reason')} (shape {multi.get('shape')})")
        log.close()
        return 3
    theta_multi = np.asarray(multi["theta"], dtype=np.float64)
    log.line(f"multi LP: shape {multi['shape']} nnz {multi['nnz']:,} objective {multi['objective']:.6g}")
    log.line(f"theta_multi={np.round(theta_multi, 4).tolist()}")
    for pair, piece in zip(train_pairs, built):
        diff = float(np.abs(piece.theta_single - theta_multi).sum())
        log.line(f"  pair {pair}: L1(theta_single, theta_multi) = {diff:.4f}")

    rows: list[RunRecord] = []
    for pair in test_pairs:
        a = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        b = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        thetas = {"multi": theta_multi, "single": built[0].theta_single}
        for r in _evaluate_pair(a, b, config=config, thetas=thetas, seed=0):
            rows.append(r)

    table = summarise(rows, group=("arm",), metrics=("mean", "miss_5pct"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "mean_mean", "miss_5pct_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
