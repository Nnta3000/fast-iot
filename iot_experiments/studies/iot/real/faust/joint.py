"""
EN: **Shared-`theta` machinery, protocol-agnostic.** Block LP over a set of *problems* (each with its own plan,
    potentials and edge set), plus the max-margin/soft-margin variant, the blocked cost planner, the flatness
    diagnostic and the uniform-regret scale. Nothing here knows whether the problems are FAUST pairs or
    template targets; the protocol lives in the driver.

    Originally `multi_pair.py`, which also carried a P1 (pairwise-proxy) driver. That driver has been retired:
    the protocol is now `challenge.py`, which follows FAUST's own two-group structure (intra/inter, ordered).

CN: **共享 θ 机制，与协议无关**：一组 problem 上的块 LP、最大间隔/软间隔变体、分块代价、平坦度诊断与均匀 regret
    尺度。这里不关心 problem 是 FAUST 的配对还是模板目标；协议在 driver 里。原为 `multi_pair.py`，其中的 P1
    （成对代理）driver 已退役，协议改由 `challenge.py` 承担（遵循 FAUST 自身的 intra/inter 两组、有向配对）。

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
        .venv/bin/python -u -m iot_experiments.studies.iot.real.faust.multi_pair --n-sub 1500 --pairs 0 13 2 41 5 88 7 52
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
from iot.core.problem import InverseOTProblem  # noqa: E402

from iot.relaxed.master_regret import regret_objective  # noqa: E402
from iot_experiments.harness.faust import geodesic_matrix, surface_area  # noqa: E402
from iot_experiments.harness.faust_match import geodesic_errors, match  # noqa: E402
from iot_experiments.harness.records import RunRecord  # noqa: E402
from iot_experiments.studies.iot.real.faust.correspondence import (  # noqa: E402
    _sample_index,
    descriptors,
    standardise,
)
# EN: the dictionary DEFINITIONS live in one place only. Importing the gate's builder (a private helper, but the
#     single source of truth) guarantees the two studies cannot drift apart.
# CN: 字典定义只有一处。复用 gate 的构造器（私有函数，但唯一真源），保证两个研究不会漂移。
from iot_experiments.studies.iot.real.faust.oracle_ceiling import _dictionaries  # noqa: E402

@dataclasses.dataclass(frozen=True)
class PairProblem:
    """CN: 一个训练对：特征族、方案、最终边集。EN: one training pair: family, plan, final edge set."""

    features: FeatureMap
    problem: InverseOTProblem
    edges: EdgeSet
    theta_single: np.ndarray
    regret_single: float
    loss: np.ndarray | None = None


def _uniform_regret(pair: PairProblem) -> float:
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


def _solve_multi(pairs: list[PairProblem], k: int, *, scales: list[float] | None = None
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


def _cross_regret(pairs: list[PairProblem], thetas: list[np.ndarray], scales: list[float]
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


def _solve_shared_margin(pairs: list[PairProblem], k: int, losses: list[np.ndarray]
                         ) -> dict[str, object]:
    """
    EN: **joint soft-margin master** — the non-vacuous version of "one theta for many pairs".

            minimise   sum_p xi_p
            s.t.       c_ii(theta) - g_i + g_j + loss_ij - c_ij(theta) <= xi_p     for all (i,j) in F_p, i != j
                       sum_k theta_k = 1,  theta >= 0,  g_p[0] = 0,  xi_p >= 0

        One `xi` **per pair**, so `xi_p` is how much the loss-scaled margin is violated for pair p, and `sum xi`
        is the joint infeasibility. `delta = 1` is fixed rather than maximised: the hard-margin version
        (`xi = 0`, maximise delta) is *infeasible* as soon as any plan is not exactly optimal, which is the usual
        case — measured on `diag13`, where the fitted theta has regret 1.9e-2 and the hard LP returned
        "infeasible". A fixed unit margin plus slack is always feasible and gives a **continuous** quantity,
        which is what a "how many pairs can share one theta" question needs.

        The potentials: the observed plan is the identity, a perfect matching, so `f_i + g_i = c_ii` determines
        `f` once `g` is chosen; substituting removes `f`. `g_p[0] = 0` fixes the remaining global shift.

        A constant or zero cost gives `theta`-terms of 0 and then `loss_ij <= xi_p` for every edge, so
        `xi_p >= max_ij loss_ij`: the degenerate direction is *punished* instead of certified. That is the whole
        point — the regret objective could be satisfied by it, this one cannot.

    CN: **联合软间隔 master**。每对**一个** `xi_p`，`xi_p` 是该对损失缩放间隔的违反量，`sum xi` 是联合不可行度。
        固定 `delta = 1` 而不是最大化：硬间隔版本（`xi = 0`）只要有一个方案不是精确最优就**不可行**（diag13 实测
        regret 1.9e-2，硬 LP 返回 infeasible）。固定单位间隔 + 松弛永远可行，并给出**连续**量，这正是"多少对能共享
        一个 θ"需要的东西。常数或零代价会给出 `loss_ij <= xi_p`，故 `xi_p` 至少是最大损失——退化方向被惩罚而不是
        被认证。
    """
    k = int(k)
    total_rows = 0
    for pair in pairs:
        rows = np.asarray(pair.edges.rows, dtype=np.int64)
        cols = np.asarray(pair.edges.cols, dtype=np.int64)
        total_rows += int(np.count_nonzero(rows != cols))
    n_pairs = len(pairs)
    ncols = k + n_pairs + int(sum(int(piece.problem.n_target) for piece in pairs))
    objective = np.zeros(ncols, dtype=np.float64)
    objective[k:k + n_pairs] = 1.0  # EN: sum of the per-pair slacks. CN: 逐对松弛之和。

    rows_index: list[np.ndarray] = []
    cols_index: list[np.ndarray] = []
    data: list[np.ndarray] = []
    b_ub_parts: list[np.ndarray] = []
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_values: list[float] = []
    eq_rhs: list[float] = []
    for column in range(k):
        eq_rows.append(0)
        eq_cols.append(column)
        eq_values.append(1.0)
    eq_rhs.append(1.0)
    next_eq = 1
    row_base = 0
    g_offset = k + n_pairs
    for pair_index, (piece, loss) in enumerate(zip(pairs, losses)):
        rows = np.asarray(piece.edges.rows, dtype=np.int64)
        cols = np.asarray(piece.edges.cols, dtype=np.int64)
        keep = rows != cols
        rows, cols = rows[keep], cols[keep]
        n_edges = int(rows.size)
        base = np.arange(n_edges, dtype=np.int64) + row_base
        row_base += n_edges
        phi = np.asarray(piece.features.features(rows, cols), dtype=np.float64)
        phi_support = np.asarray(piece.features.features(rows, rows), dtype=np.float64)
        rows_index.append(np.repeat(base, k))
        cols_index.append(np.tile(np.arange(k, dtype=np.int64), n_edges))
        data.append((phi_support - phi).reshape(-1))
        rows_index.append(base)
        cols_index.append(g_offset + rows)
        data.append(-np.ones(n_edges, dtype=np.float64))
        rows_index.append(base)
        cols_index.append(g_offset + cols)
        data.append(np.ones(n_edges, dtype=np.float64))
        # EN: -xi_p on this pair's slack column, and the loss on the right-hand side (b_ub = -loss).
        # CN: 该对的松弛列取 -1，损失移到右端（b_ub = -loss）。
        rows_index.append(base)
        cols_index.append(np.full(n_edges, k + pair_index, dtype=np.int64))
        data.append(-np.ones(n_edges, dtype=np.float64))
        b_ub_parts.append(-np.asarray(loss[rows, cols], dtype=np.float64))
        eq_rows.append(next_eq)
        eq_cols.append(g_offset)
        eq_values.append(1.0)
        eq_rhs.append(0.0)
        next_eq += 1
        g_offset += int(piece.problem.n_target)

    a_ub = sparse.coo_matrix((np.concatenate(data),
                              (np.concatenate(rows_index), np.concatenate(cols_index))),
                             shape=(total_rows, ncols)).tocsr()
    a_eq = sparse.coo_matrix((np.asarray(eq_values, dtype=np.float64),
                              (np.asarray(eq_rows, dtype=np.int64),
                               np.asarray(eq_cols, dtype=np.int64))),
                             shape=(next_eq, ncols)).tocsr()
    bounds = [(0.0, 1.0)] * k + [(0.0, None)] * n_pairs + [(None, None)] * (ncols - k - n_pairs)
    result = linprog(objective, A_ub=a_ub, b_ub=np.concatenate(b_ub_parts),
                     A_eq=a_eq, b_eq=np.asarray(eq_rhs, dtype=np.float64),
                     bounds=bounds, method="highs")
    if not result.success:
        return {"ok": False, "reason": str(result.message)[:160],
                "shape": (total_rows, ncols), "nnz": int(a_ub.nnz)}
    theta = np.maximum(np.asarray(result.x[:k], dtype=np.float64), 0.0)
    total = float(theta.sum())
    theta = theta / total if total > 0 else theta
    return {"ok": True, "theta": theta,
            "slack": np.asarray(result.x[k:k + n_pairs], dtype=np.float64),
            "potentials": np.asarray(result.x[k + n_pairs:], dtype=np.float64),
            "shape": (total_rows, ncols), "nnz": int(a_ub.nnz)}


def _verified_slack(feature_map: FeatureMap, theta: np.ndarray, g: np.ndarray, loss: np.ndarray,
                    n: int) -> float:
    """
    EN: the worst margin violation **on all `n*n` edges**, using the potentials from the restricted master.

        The master only saw `F_p`, so its `xi` is an underestimate of the true joint infeasibility. This
        evaluates the same inequality everywhere and returns the largest violation (negative or zero means the
        certificate really holds on every edge):
        `max_{i != j} [ loss_ij - (c_ij - c_ii + g_i - g_j) ]`.

    CN: 用受限 master 的势函数，在**全部 `n*n` 边**上计算最坏间隔违反量。master 只看到 `F_p`，故其 `xi` 低估了
        真实的联合不可行度。这里在全边上求最大的违反量（≤0 表示证书在全边上确实成立）。
    """
    cost = _cost_rows(feature_map, theta, n)
    diagonal = np.diag(cost)
    margin = cost - diagonal[:, None] + g[:, None] - g[None, :]
    denominator = np.array(loss, dtype=np.float64, copy=True)
    np.fill_diagonal(denominator, np.inf)
    violation = denominator - margin
    np.fill_diagonal(violation, -np.inf)
    return float(violation.max())


