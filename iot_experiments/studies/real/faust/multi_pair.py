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
from iot.core.features import FeatureMap  # noqa: E402
from iot.core.parameters import (  # noqa: E402
    PruningParameters,
    RelaxedObjective,
    SolverParameters,
)
from iot.core.problem import InverseOTProblem  # noqa: E402
from iot.core.theta import SimplexTheta  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot.relaxed.master_regret import regret_objective  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.master_probe import RecordingMaster  # noqa: E402
from iot_experiments.harness.faust import (  # noqa: E402
    default_registrations_root,
    geodesic_matrix,
    read_ply,
    surface_area,
)
from iot_experiments.harness.faust_match import geodesic_errors, match  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.studies.real.faust.correspondence import (  # noqa: E402
    _sample_index,
    default_pairs,
    descriptors,
    standardise,
)
# EN: the dictionary DEFINITIONS live in one place only. Importing the gate's builder (a private helper, but the
#     single source of truth) guarantees the two studies cannot drift apart.
# CN: 字典定义只有一处。复用 gate 的构造器（私有函数，但唯一真源），保证两个研究不会漂移。
from iot_experiments.studies.real.faust.oracle_ceiling import _dictionaries  # noqa: E402

DEFAULT_ROOT = str(default_registrations_root())

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
    "cost_family": "selectable with --family; the definitions are imported from the oracle-ceiling gate so the two "
                   "studies cannot drift. Default diag13 = the incumbent 13-feature diagonal family",
    "joint_feasibility": "reported on the FULL edge set with exact POT per pair, NOT as the restricted-master LP "
                         "objective, which is only a lower bound on the true joint regret",
    "scales": "the shared object is theta. Each pair's Gaussian bandwidths / hat-basis extent are derived from "
              "that pair's own nearest-neighbour distances, because they are normalisations, not parameters",
    "flatness": "every score is reported with the cost's identity gap; a rich family can certify a plan by tying "
                "all candidates (measured median gap 1e-5), after which the decoded map is an arbitrary optimum",
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
    # EN: which cost family the shared theta is fitted over. `diag13` is the incumbent; `profile8` is the hat
    #     basis that was bimodal in the oracle gate (7/24 wins, then catastrophic losses through flat costs).
    # CN: 共享 θ 所用的代价族。`diag13` 是现有基线；`profile8` 是 oracle 门中呈双峰的帽基（24 行赢 7 行，
    #     但通过"平代价"惨败）。
    family: str = "diag13"
    bands: int = 5
    basis_size: int = 8

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
        parser.add_argument("--family", default="diag13",
                            choices=["diag13", "hinge8", "profile8", "group+diag", "group_gauss",
                                     "gauss", "gauss+cos"])
        parser.add_argument("--bands", type=int, default=5)
        parser.add_argument("--basis-size", type=int, default=8)


@dataclasses.dataclass(frozen=True)
class _PairProblem:
    """CN: 一个训练对：特征族、方案、最终边集。EN: one training pair: family, plan, final edge set."""

    features: FeatureMap
    problem: InverseOTProblem
    edges: EdgeSet
    theta_single: np.ndarray
    regret_single: float


def _uniform_regret(pair: _PairProblem) -> float:
    """
    CN: 该对的 `R_uniform`：均匀 θ 与一组可行对偶下的 regret，用作**这一对的尺度**。跨对比较必须用它归一，
        因为原始 regret 的量级随实例差 3 倍（实测 0.00079 / 0.00074 / 0.0024 / 0.0024），直接平均会让最难的两对
        独占目标。
    EN: the pair's `R_uniform`: the regret of uniform theta at a feasible dual, used as **this pair's scale**.
        Cross-pair averaging requires it because raw regrets differ by 3x across pairs (measured 0.00079,
        0.00074, 0.0024, 0.0024), and a plain mean lets the two hardest pairs dominate the objective.
    """
    from iot.relaxed.master_regret import regret_objective, support_duals  # noqa: PLC0415

    uniform = np.full(int(pair.features.n_features), 1.0 / float(pair.features.n_features))
    f_u, g_u = support_duals(pair.problem, uniform, pair.edges)
    objective = np.asarray(regret_objective(pair.problem), dtype=np.float64)
    value = float(objective @ np.concatenate([uniform, f_u, g_u]))
    return value if np.isfinite(value) and value > 0.0 else float("nan")


def _solve_multi(pairs: list[_PairProblem], k: int, *, scales: list[float] | None = None
                 ) -> dict[str, object]:
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
    # CN: 每对先除以自己的 `R_uniform`，于是目标 = mean_p R_p / R_uniform_p（无量纲）。
    # EN: divide each pair by its own `R_uniform`, so the objective is mean_p R_p / R_uniform_p (dimensionless).
    weights = [1.0 / float(len(pairs))] * len(pairs) if scales is None else list(scales)
    # CN: 用 `enumerate` 记录行偏移。`pairs.index(pair)` 会对 dataclass 字段做 `==`，
    #     而字段里含 numpy 数组 ⇒ "truth value ... is ambiguous"（实测崩溃）。
    # EN: track the row offset with `enumerate`. `pairs.index(pair)` compares dataclass fields with `==`, and
    #     those fields hold numpy arrays -> "truth value ... is ambiguous" (measured crash).
    row_offset = 0
    for pair, pair_weight in zip(pairs, weights):
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
        objective[offset:offset + m] = -np.asarray(pair.problem.resolved_source_mass) * pair_weight
        objective[offset + m:offset + m + n] = -np.asarray(pair.problem.resolved_target_mass) * pair_weight
        objective[: int(k)] += pair_weight * np.asarray(regret_objective(pair.problem)[: int(k)])
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


def _family(name: str, a: np.ndarray, b: np.ndarray, *, bands: int, basis_size: int) -> FeatureMap:
    """
    EN: build one named dictionary for a pair, reusing the oracle gate's definitions so the two studies stay
        comparable. The per-pair *scales* (Gaussian bandwidths, hat-basis extent) are derived from the pair's own
        nearest-neighbour distances, so a theta transfers but the scale does not; that is a property of the
        dictionary, and it is stated in the provenance rather than hidden.
    CN: 为一对形状构造指定字典，复用 oracle gate 的定义以保证可比。逐对**尺度**（高斯带宽、帽基范围）来自该对
        自身的最近邻距离，所以迁移的是 θ、不是尺度；这是字典的性质，写进 provenance 而不是藏起来。
    """
    available = dict(_dictionaries(a, b, bands=int(bands), basis_size=int(basis_size)))
    if name not in available:
        raise SystemExit(f"unknown family {name!r}; known: {sorted(available)}")
    return available[name]


def _cost_rows(feature_map: FeatureMap, theta: np.ndarray, n: int, *, block: int = 64
               ) -> np.ndarray:
    """
    EN: the full **square** `n x n` cost, built in row blocks. Materialising `(n*n, K)` at once is 600 MB for
        `K = 117` at `n = 800`; this path never does, so the rich families are as cheap in memory as the diagonal
        one. Every use in this study is square (source and target are the same vertex subsample).
    CN: 完整 `n x n` 代价，**按行分块**构造。一次性物化 `(n*n, K)` 在 `n=800, K=117` 时是 600 MB；本路径不做，
        所以丰富族与对角族的内存开销相同。
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    out = np.empty((int(n), int(n)), dtype=np.float64)
    cols_index = np.arange(int(n), dtype=np.int64)
    for start in range(0, int(n), int(block)):
        stop = min(int(n), start + int(block))
        rows_index = np.repeat(np.arange(start, stop, dtype=np.int64), int(n))
        cols = np.tile(cols_index, stop - start)
        values = feature_map.costs(theta, rows_index, cols)
        out[start:stop, :] = np.asarray(values, dtype=np.float64).reshape(stop - start, int(n))
    return out


def _flatness(cost: np.ndarray) -> dict[str, float]:
    """
    EN: **how tied is the cost?** This is the diagnostic the oracle gate was missing, and it explains that
        result: a rich dictionary can certify the observed plan by making every candidate equally good, after
        which the decoded optimal plan is an arbitrary optimum and the map is bad *while the certificate is
        excellent*.

        Sign convention: `gap = c_ii - min_{j != i} c_ij`. **Negative** means the correct partner is strictly
        cheaper than every alternative (good); **zero** means it is tied with some other candidate, so the
        optimal plan is not unique. Only `|gap|` may be compared to a tolerance — an earlier version tested
        `gap < 1e-6` and so counted every *strictly better* diagonal as "tied" (`tied = 1.000` for any
        well-separated cost). That is a measurement bug, caught by `tests/iot_multi_pair_test.py`.
    CN: **代价有多"平"？** 符号约定：`gap = c_ii - min_{j≠i} c_ij`，**负**表示正确伙伴严格更便宜（好），**零**
        表示与某个候选打平（最优方案不唯一）。只有 `|gap|` 才能与容差比较——早期版本用 `gap < 1e-6`，于是把每个
        "严格更好"的对角都算成打平（任何分离良好的代价都 `tied=1.000`）。这是测量 bug，由测试捕获。
    """
    n = int(cost.shape[0])
    off_diagonal = np.array(cost, dtype=np.float64, copy=True)
    np.fill_diagonal(off_diagonal, np.inf)
    gap = cost[np.arange(n), np.arange(n)] - off_diagonal.min(axis=1)
    return {"row_gap_median": float(np.median(gap)),
            "row_gap_p90": float(np.percentile(gap, 90.0)),
            "tied_fraction_1e6": float((np.abs(gap) < 1e-6).mean())}


def _score_pair(a, b, *, family: str, config, thetas: dict[str, np.ndarray], seed: int,
                bands: int, basis_size: int) -> list[RunRecord]:
    """
    EN: score supplied thetas on one pair with **its own** dictionary, using exact OT and the geodesic metric.
        Mirrors `correspondence._evaluate_pair` but goes through `FeatureMap.costs`, because that helper builds
        the cost from a diagonal weight vector and would silently score a rich family as if it were diagonal.
        Also returns `nn_*` (argmin, no transport) as the pairing-gate control.
    CN: 用该对**自身**的字典评估给定 θ，精确 OT + 测地线度量。与 `_evaluate_pair` 平行，但走 `FeatureMap.costs`，
        因为那个助手按对角权重构造代价，会把丰富族当作对角族评估。同时给出 `nn_*`（argmin，无传输）作对照。
    """
    index = _sample_index(a, config, seed)
    da, db = standardise(descriptors(a)[index], descriptors(b)[index])
    n = int(index.size)
    geo = geodesic_matrix(b.vertices, b.faces, index)
    norm = float(np.sqrt(surface_area(b.vertices, b.faces)))
    feature_map = _family(family, da, db, bands=bands, basis_size=basis_size)
    everything = {"uniform": np.full(int(feature_map.n_features), 1.0 / float(feature_map.n_features)),
                  **{key: value for key, value in thetas.items() if value is not None}}
    geometry = f"pair{a.path.stem[7:]}-{b.path.stem[7:]}"
    rows: list[RunRecord] = []
    for name, theta in everything.items():
        cost = _cost_rows(feature_map, theta, n)
        flat = _flatness(cost)
        for mode, prediction in (("ot", match(cost)), ("nn", np.argmin(cost, axis=1).astype(np.int64))):
            label = f"nn_{name}" if mode == "nn" else name
            rows.append(RunRecord(
                study=str(config.study), geometry=geometry, arm=label, k=int(feature_map.n_features),
                m=int(n), draw=int(seed), metrics=geodesic_errors(prediction, geo, norm),
                config=config.to_dict(),
                extra={"family": family, "n_features": int(feature_map.n_features), "mode": mode, **flat}))
    return rows


def _cross_regret(pairs: list[_PairProblem], thetas: list[np.ndarray], scales: list[float]
                  ) -> np.ndarray:
    """
    CN: **交叉 regret 矩阵** `M[p, q] = R_p(theta_q) / R_uniform_p`，其中 `R_p(θ) = <θ, φ̄_p> − OT_p(c_p(θ))`
        （`pi_hat` 是最优方案时该值为 0）。用 POT 精确求解，故 θ 来自另一对时也能给出诚实的量。
    EN: the **cross-regret matrix** `M[p, q] = R_p(theta_q) / R_uniform_p`, with
        `R_p(theta) = <theta, phi_bar_p> - OT_p(c_p(theta))` (zero iff `pi_hat` is optimal). Exact POT, so a
        theta borrowed from another pair is still measured honestly.
    """
    import ot  # noqa: PLC0415
    from iot.relaxed.master_regret import regret_objective  # noqa: PLC0415

    # CN: 行 = 被评估的对（P），列 = θ 的来源（P 个单对 θ + 共享 θ ⇒ P+1）。两者大小不同，
    #     之前按 (P, P) 分配导致 `index 4 is out of bounds`（实测崩溃）。
    # EN: rows = the pair being evaluated (P), columns = where theta came from (P single thetas + the shared
    #     one => P+1). They differ; allocating `(P, P)` raised `index 4 is out of bounds` (measured crash).
    out = np.full((len(pairs), len(thetas)), np.nan, dtype=np.float64)
    for p, pair in enumerate(pairs):
        phi_bar = np.asarray(regret_objective(pair.problem)[: int(pair.features.n_features)],
                             dtype=np.float64)
        a = np.asarray(pair.problem.resolved_source_mass, dtype=np.float64)
        b = np.asarray(pair.problem.resolved_target_mass, dtype=np.float64)
        a = a / a.sum()
        b = b / b.sum()
        n_src = int(pair.features.n_source)
        for q, theta in enumerate(thetas):
            # EN: blocked, so a rich family does not allocate `(n*n, K)` just to measure one theta.
            # CN: 分块，避免为了量一个 θ 就分配 `(n*n, K)`。
            cost = _cost_rows(pair.features, theta, n_src)
            transport = float(ot.emd2(a, b, np.ascontiguousarray(cost), numItermax=10_000_000))
            out[p, q] = (float(phi_bar @ np.asarray(theta, dtype=np.float64)) - transport) / scales[p]
    return out


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
    log.line(f"cost family={config.family!r}  bands={config.bands}  basis={config.basis_size}  "
             f"n_sub={config.n_sub}  per-pair iterations={config.max_iterations}")

    built: list[_PairProblem] = []
    for pair in train_pairs:
        a = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        b = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        idx = _sample_index(a, config, 0)
        da, db = standardise(descriptors(a)[idx], descriptors(b)[idx])
        n = int(idx.size)
        plan = sparse.csr_matrix((np.full(n, 1.0 / n), (np.arange(n), np.arange(n))), shape=(n, n))
        family = _family(str(config.family), da, db, bands=int(config.bands),
                         basis_size=int(config.basis_size))
        problem = InverseOTProblem(features=family, theta_set=SimplexTheta(int(family.n_features)),
                                   plan=plan)
        master = RecordingMaster(
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
        log.line(f"pair {pair}: K={int(family.n_features)} |F|={int(master.last_edges.size)} "
                 f"regret_single={float(single.regret):.6g}")

    k = int(built[0].features.n_features)
    scales = [_uniform_regret(piece) for piece in built]
    log.line("per-pair R_uniform (the normalising scale): "
             + ", ".join(f"{s:.4g}" for s in scales))
    if not all(np.isfinite(scales)):
        log.line("a pair has a non-positive R_uniform; falling back to the unnormalised objective")
        scales = None  # type: ignore[assignment]
    multi = _solve_multi(built, k, scales=scales)
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
    # CN: 交叉 regret 矩阵：行 = 评估用的对，列 = θ 的来源（单对 θ 们 + 共享 θ）。
    # EN: cross-regret matrix: rows = the pair being evaluated, columns = where the theta came from.
    thetas_cols = [piece.theta_single for piece in built] + [theta_multi]
    scale_column = [float(v) if np.isfinite(v) else 1.0 for v in scales] if scales else [1.0] * len(built)
    matrix = _cross_regret(built, thetas_cols, scale_column)
    log.line("\ncross-regret R_p(theta_q) / R_uniform_p   (rows = evaluated pair, cols = theta source)")
    header = "        " + "".join(f"{'th'+str(q):>10}" for q in range(len(thetas_cols)))
    log.line(header + "   (last column = shared theta)")
    for p, pair in enumerate(train_pairs):
        log.line(f"pair{str(pair):<10}" + "".join(f"{matrix[p, q]:>10.4f}" for q in range(len(thetas_cols))))
    rows.append(RunRecord(study=str(config.study), geometry="cross_regret", arm="matrix", k=k,
                          m=int(config.n_sub), draw=0, metrics={},
                          config=config.to_dict(),
                          extra={"matrix": matrix.tolist(), "train_pairs": [list(p) for p in train_pairs],
                                 "scales": [float(v) for v in (scales or [])]}))

    # EN: the honest joint-feasibility question, per training pair: the **full-edge** normalised regret of the
    #     shared theta (NOT the restricted LP value) plus the flatness of the cost it induces. The restricted
    #     master can only be optimistic, so the LP objective alone must never be reported as "satisfies all".
    # CN: 逐训练对的诚实联合可行性：共享 θ 在**全边集**上的归一化 regret（不是受限 LP 值），以及它诱导代价的
    #     平坦度。受限 master 只会偏乐观，故绝不能只凭 LP 目标宣称"满足所有对"。
    diagnostics: dict[str, dict[str, float]] = {}
    for p, (pair_id, piece) in enumerate(zip(train_pairs, built)):
        n_pair = int(piece.problem.n_source)
        flat_multi = _flatness(_cost_rows(piece.features, theta_multi, n_pair))
        flat_single = _flatness(_cost_rows(piece.features, piece.theta_single, n_pair))
        diagnostics[f"pair{list(pair_id)}"] = {
            "full_edge_regret_multi": float(matrix[p, len(built)]),
            "full_edge_regret_single": float(matrix[p, p]),
            **{f"multi_{key}": value for key, value in flat_multi.items()},
            **{f"single_{key}": value for key, value in flat_single.items()},
        }
        log.line(f"  pair {pair_id}: full-edge regret multi={matrix[p, len(built)]:.4f} "
                 f"single={matrix[p, p]:.4f}  flatness median gap={flat_multi['row_gap_median']:.2e} "
                 f"tied={flat_multi['tied_fraction_1e6']:.3f}")
    rows.append(RunRecord(study=str(config.study), geometry="joint_diagnostics", arm="diagnostics",
                          k=k, m=int(config.n_sub), draw=0, metrics={}, config=config.to_dict(),
                          extra={"family": str(config.family), "pairs": diagnostics,
                                 "theta_multi": theta_multi.tolist(),
                                 "l1_to_uniform": float(np.abs(theta_multi - 1.0 / k).sum())}))

    for pair in test_pairs:
        a = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        b = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        thetas = {"multi": theta_multi, "single": built[0].theta_single}
        rows.extend(_score_pair(a, b, family=str(config.family), config=config, thetas=thetas, seed=0,
                                bands=int(config.bands), basis_size=int(config.basis_size)))

    table = summarise(rows, group=("arm",), metrics=("mean", "miss_5pct"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "mean_mean", "miss_5pct_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
