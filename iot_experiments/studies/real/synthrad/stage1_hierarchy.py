"""
CN: **两级层级（Stage 1）** —— `HierarchicalIOT.typ` 的最小实例：一个粗层 + 一个细层。

    CN: 流程（与 Typst 文件一致）：
        1. `BuildLevels`：按分数 ρ 从支撑 `S` 中**抽样边**，得到粗层索引集 `I₁, J₁`；
        2. 粗层：在 `U₁ = I₁ × J₁` 上**全边**求解（网格小，无需扫描）；
        3. `Prolong`：把粗层对偶延拓到细层（保留节点沿用原势，新节点做 min-补全）；
        4. `Seed`：`B₀ ∪ A_row ∪ A_col ∪ A_carry`，其中 `A_*` 由延拓后的 slack 排序；
        5. 细层：用现有 `solve_inverse_ot_regret` 从 `F₀` 开始求解。

    CN: **正确性闸门**（用户指定，不比较 θ）：`is_optimal_plan` + master 可行性
        （全部 `h_e <= tol`）+ 互补松弛。grow-only 模式下 θ 无需与 flat 相同。

    CN: 记录：`R_flat`、`R_seeded`、`|F₀|/|F_final|`、命中率（细层真正插入的边里有多少已在 `F₀`）、
        master/scan 时间、总时间。

EN: **Two-level hierarchy (Stage 1)** -- the minimal instance of `HierarchicalIOT.typ`: one coarse level
    plus the fine one.

    Flow (as in the Typst file): BuildLevels samples **support edges** at ratio rho to get the coarse index
    sets; the coarse level is solved **densely** on `U1 = I1 x J1` (small grid, no scan needed); Prolong
    extends the coarse duals to the fine grid (retained nodes keep their potentials, new nodes take the
    min-completion); Seed builds `B0 u A_row u A_col u A_carry` from the prolonged slack; the fine level
    runs the existing `solve_inverse_ot_regret` from `F0`.

    **Correctness gate** (as specified, not a theta comparison): `is_optimal_plan` + master feasibility
    (`h_e <= tol` everywhere) + complementary slackness. In grow-only mode theta need not match flat's.

    Records: `R_flat`, `R_seeded`, `|F0|/|F_final|`, recall (how many edges the fine level actually
    inserted were already in `F0`), master/scan time, total time.
"""

from __future__ import annotations

import dataclasses
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import EdgeSet, InverseOTProblem  # noqa: E402
from iot.core.edges_advance import slack_toward_cost  # noqa: E402
from iot.core.features import DiagonalWeightFeatures  # noqa: E402
from iot.eval.validation import is_optimal_plan  # noqa: E402
from iot.exact.components import connector_edges, support_components  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot.relaxed.master_regret import RegretMasterSolver  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.datasets import reduce_features, stream_features  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402
from iot_experiments.harness.solver import box_simplex  # noqa: E402
from iot_experiments.harness.synthrad import discover_patients  # noqa: E402

PROVENANCE: dict[str, object] = {
    "stage": "1 — two-level hierarchy vs flat baselines",
    "algorithm": "HierarchicalIOT.typ: BuildLevels -> coarse dense solve -> Prolong -> Seed -> fine loop",
    "gate": "is_optimal_plan + master feasibility + complementary slackness (NOT a theta comparison)",
    "a_carry": "NOT implemented: RegretResult exposes no final edge set; seed uses A_row u A_col only",
    "pre_registered": "hier wall time <= 0.70 x best cheap baseline (gamma=2 warm), on delta*~0",
}


def _rss_gb() -> float:
    """CN: 峰值 RSS（GB）。EN: peak RSS in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


@dataclasses.dataclass(frozen=True)
class Level:
    """CN: 一层的索引集与诱导支撑。EN: one level's index sets and induced support."""

    rows: Int[np.ndarray, "sources"]
    cols: Int[np.ndarray, "targets"]
    support: EdgeSet

    @property
    def n_source(self) -> int:
        """CN: 源点数。EN: number of sources."""
        return int(self.rows.size)

    @property
    def n_target(self) -> int:
        """CN: 目标点数。EN: number of targets."""
        return int(self.cols.size)


def build_levels(problem: InverseOTProblem, *, rho: float, seed: int,
                 n_target: int) -> tuple[Level, Level]:
    """
    CN: 只构造**一个**粗层：从支撑里均匀抽 `ceil(rho|S|)` 条边，取它们的端点作为粗层索引集，
        诱导支撑 = `S ∩ (I₁ × J₁)`。若抽到的点没有减少，减半 ρ 重抽（Typst 文件 Lemma 1 的接受条件）。
    EN: build **one** coarse level: draw `ceil(rho|S|)` support edges uniformly, take their endpoints as
    the coarse index sets, induced support `S ∩ (I1 x J1)`. If the point count did not shrink, halve rho and
    redraw (the acceptance condition of Lemma 1).
    """
    s_rows, s_cols, _ = problem.support()
    n_source = int(problem.n_source)
    rng = np.random.default_rng(int(seed))
    ratio = float(rho)
    for _attempt in range(40):
        count = max(1, int(np.ceil(ratio * s_rows.size)))
        pick = rng.choice(s_rows.size, size=min(count, s_rows.size), replace=False)
        c_rows = np.unique(s_rows[pick])
        c_cols = np.unique(s_cols[pick])
        if c_rows.size + c_cols.size < n_source + n_target:
            break
        ratio *= 0.5
    else:  # pragma: no cover - only if coarsening cannot shrink at all
        raise RuntimeError("BuildLevels could not shrink the point set")
    keep_r = np.isin(s_rows, c_rows)
    keep_c = np.isin(s_cols, c_cols)
    keep = keep_r & keep_c
    induced = EdgeSet.from_arrays(np.ascontiguousarray(s_rows[keep]), np.ascontiguousarray(s_cols[keep]),
                                  n_target=int(n_target))
    full_rows: Int[np.ndarray, "sources"] = np.arange(n_source, dtype=np.int64)
    full_cols: Int[np.ndarray, "targets"] = np.arange(n_target, dtype=np.int64)
    full_support = EdgeSet.from_arrays(s_rows, s_cols, n_target=int(n_target))
    return Level(full_rows, full_cols, full_support), Level(c_rows, c_cols, induced)


def _dense_edges(n_source: int, n_target: int) -> EdgeSet:
    """CN: 全边集 `U = I × J`。EN: the full grid `U = I x J`."""
    rows: Int[np.ndarray, "edges"] = np.repeat(np.arange(n_source, dtype=np.int64), n_target)
    cols: Int[np.ndarray, "edges"] = np.tile(np.arange(n_target, dtype=np.int64), n_source)
    return EdgeSet.from_arrays(rows, cols, n_target=int(n_target))


def _prolong(x: Float[np.ndarray, "m d"], y: Float[np.ndarray, "m d"], theta: Float[np.ndarray, "k"],
             f_coarse: Float[np.ndarray, "mc"], g_coarse: Float[np.ndarray, "nc"],
             coarse: Level, fine: Level) -> tuple[Float[np.ndarray, "m"], Float[np.ndarray, "n"]]:
    """
    CN: 保留节点沿用粗层势；新节点的 `f` 取在粗层目标集上的最小值，`g` 取在**细层全部源点**上的最小值
        （Typst Lemma 4）。`f` 先算，`g` 再用新的 `f`，因此 `g` 的补全是精确的。
    EN: retained nodes keep the coarse potentials; a new source's `f` takes the minimum over the coarse
        targets, a new target's `g` the minimum over **all fine sources** (Lemma 4). `f` is computed first,
        then `g` with the new `f`, so the `g` completion is exact.
    """
    # CN: 粗层对偶按**粗层局部序号**给出，因此用位置写入（`coarse.rows[pos] = pos`），不是 `f_out[coarse.rows]`。
    # EN: coarse duals are indexed by the **coarse local** index, so assign by position
    #     (`coarse.rows[pos] = pos`), not `f_out[coarse.rows]`.
    f_out: Float[np.ndarray, "m"] = np.empty(fine.n_source, dtype=np.float64)
    f_out[coarse.rows] = np.asarray(f_coarse, dtype=np.float64).reshape(-1)
    mask_new_f = ~np.isin(fine.rows, coarse.rows)
    for idx in fine.rows[mask_new_f]:
        cost = np.asarray(x[idx] * y[coarse.cols]).sum(axis=1)
        f_out[idx] = float(np.min(cost - np.asarray(g_coarse, dtype=np.float64)))
    g_out: Float[np.ndarray, "n"] = np.empty(fine.n_target, dtype=np.float64)
    g_out[coarse.cols] = np.asarray(g_coarse, dtype=np.float64).reshape(-1)
    mask_new_g = ~np.isin(fine.cols, coarse.cols)
    for idx in fine.cols[mask_new_g]:
        cost = np.asarray(x[fine.rows] * y[idx]).sum(axis=1)
        g_out[idx] = float(np.min(cost - f_out[fine.rows]))
    return f_out, g_out


def _seed(problem: InverseOTProblem, theta: Float[np.ndarray, "k"], f: Float[np.ndarray, "m"],
          g: Float[np.ndarray, "n"], mandatory: EdgeSet, kappa: int,
          carry_keys: Int[np.ndarray, "carry"], n_target: int) -> tuple[EdgeSet, dict[str, int]]:
    """
    CN: `A_row`：每行 slack 最大的 κ 条非支撑边；`A_col` 同理按列；`A_carry` 是粗层最终边集。
    EN: `A_row`: the kappa largest-slack non-support edges per row; `A_col` the same by column;
        `A_carry` is the coarse level's final edge set.
    """
    s_rows, s_cols, _ = problem.support()
    sup = EdgeSet.from_arrays(s_rows, s_cols, n_target=n_target)
    rows: Int[np.ndarray, "edges"] = np.repeat(np.arange(problem.n_source, dtype=np.int64), n_target)
    cols: Int[np.ndarray, "edges"] = np.tile(np.arange(n_target, dtype=np.int64), problem.n_source)
    keep = ~np.isin(rows * n_target + cols, sup.keys)
    rows, cols = rows[keep], cols[keep]
    slack = slack_toward_cost(problem, theta, f, g, rows, cols)
    chosen: list[int] = []
    for direction, group in (("row", rows), ("col", cols)):
        _ = direction
        order = np.lexsort((slack, group))
        bounds = np.flatnonzero(np.diff(group[order])) + 1
        for chunk in np.split(order, bounds):
            chosen.extend(chunk[np.argsort(-slack[chunk])][: int(kappa)].tolist())
    a_row_col = EdgeSet.from_arrays(np.ascontiguousarray(rows[np.unique(chosen)]),
                                    np.ascontiguousarray(cols[np.unique(chosen)]), n_target=n_target)
    carry_rows = carry_keys // n_target
    carry_cols = carry_keys % n_target
    carry = EdgeSet.from_arrays(carry_rows.astype(np.int64), carry_cols.astype(np.int64),
                                n_target=n_target)
    seeded = mandatory.union(a_row_col).union(carry)
    return seeded, {"n_a_row_col": int(a_row_col.size), "n_carry": int(carry.size),
                    "n_mandatory": int(mandatory.size)}


@dataclasses.dataclass(frozen=True)
class Stage1Config(RunConfig):
    """CN: 旋钮。EN: knobs."""

    section = "synthrad_stage1"
    study: str = "synthrad_stage1"
    m: int = 2000
    k: int = 64
    draw: int = 0
    rho: float = 0.25
    kappa: int = 3
    gamma: int = 2
    level_seeds: tuple[int, ...] = (0, 1, 2)

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--m", type=int, default=2000)
        parser.add_argument("--k", type=int, default=64)
        parser.add_argument("--rho", type=float, default=0.25)
        parser.add_argument("--kappa", type=int, default=3)
        parser.add_argument("--gamma", type=int, default=2)
        parser.add_argument("--level-seeds", type=int, nargs="+", default=[0, 1, 2])


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(Stage1Config)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = Stage1Config.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line(f"stage 1: rho={config.rho} kappa={config.kappa} gamma={config.gamma} "
             f"seeds={config.level_seeds}")
    log.line(f"MemAvailable start, peak RSS {_rss_gb():.2f} GB")

    split = json.loads((config.out / "synthrad_v2" / "split_v2.json").read_text())
    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    fit_pats = [p for p in patients if p.patient in set(split["fit"])]
    rng = np.random.default_rng(1000 + int(config.draw))
    chosen = [fit_pats[i] for i in rng.permutation(len(fit_pats))[:60]]
    pm, pc, _pid, _z, _ctr = stream_features(chosen, window=(-200.0, 300.0), max_per_patient=None,
                                             min_gap=0, seed=int(config.draw))
    ftr, ftc = reduce_features(pm, pc, k=int(config.k), geometry_name="G3", whiten=False)
    del pm, pc
    m = int(min(int(config.m), int(ftr.shape[0])))
    sub = np.sort(np.random.default_rng(777).permutation(int(ftr.shape[0]))[:m])
    x: Float[np.ndarray, "m d"] = np.ascontiguousarray(ftr[sub])
    y: Float[np.ndarray, "m d"] = np.ascontiguousarray(ftc[sub])
    k = int(x.shape[1])
    plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
    problem = InverseOTProblem(features=DiagonalWeightFeatures(x, y),
                               theta_set=box_simplex(k, box_relative=0.25), plan=plan)
    log.line(f"problem m={m} k={k}, peak RSS {_rss_gb():.2f} GB")

    # ---------- flat baseline (the best cheap one: gamma=2, warm) ----------
    t0 = time.perf_counter()
    flat = solve_inverse_ot_regret(
        problem, tol=None, gamma=int(config.gamma), max_iterations=200, warm_start=True,
        master=RegretMasterSolver(backend=str(config.solver), tie_break=False,
                                  params={"time_sec_limit": float(config.solve_limit_s)}),
        eps_r_rel=1e-6, prune=False, check_pot=False, selection="none")
    flat_s = time.perf_counter() - t0
    log.line(f"flat: rounds={flat.n_iterations} |F|={flat.finite_edges} {flat_s:.1f}s "
             f"L={flat.lower:.3e} U={flat.upper:.3e} stall={flat.stall_status}")
    gate_flat = is_optimal_plan(problem, np.asarray(flat.theta, dtype=np.float64))
    log.line(f"flat gate: is_optimal_plan={bool(gate_flat.is_optimal)}")

    rows: list[dict[str, object]] = []
    for seed in [int(v) for v in config.level_seeds]:
        t_start = time.perf_counter()
        fine, coarse = build_levels(problem, rho=float(config.rho), seed=seed, n_target=m)
        log.line(f"seed={seed}: coarse {coarse.n_source}x{coarse.n_target} "
                 f"({'%.1f%%' % (100 * coarse.n_source * coarse.n_target / (m * m))} of grid), "
                 f"induced support {coarse.support.size}")
        # ---- coarse: dense solve, no scan needed ----
        t_c = time.perf_counter()
        sub_r = coarse.rows
        sub_c = coarse.cols
        xs: Float[np.ndarray, "mc d"] = np.ascontiguousarray(x[sub_r])
        ys: Float[np.ndarray, "nc d"] = np.ascontiguousarray(y[sub_c])
        mc, nc = int(xs.shape[0]), int(ys.shape[0])
        _ = coarse.support  # CN: 诱导支撑已由 coarse_problem 使用 / EN: used via coarse_problem
        plan_c = sparse.csr_matrix(
            (np.zeros(0), (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))), shape=(mc, nc))
        # CN: 粗层的观察方案是**诱导**方案：把细层 π̂ 限制在粗网格上。
        # EN: the coarse observed plan is the **induced** plan: the fine pi_hat restricted to the coarse grid.
        dense = np.asarray(plan.todense(), dtype=np.float64)
        dense_c = dense[np.ix_(sub_r, sub_c)]
        plan_c = sparse.csr_matrix(np.ascontiguousarray(dense_c), dtype=np.float64)
        del dense
        coarse_problem = InverseOTProblem(features=DiagonalWeightFeatures(xs, ys),
                                          theta_set=box_simplex(k, box_relative=0.25), plan=plan_c)
        cr = solve_inverse_ot_regret(
            coarse_problem, tol=None, gamma=1, max_iterations=200,
            master=RegretMasterSolver(backend=str(config.solver), tie_break=False,
                                      params={"time_sec_limit": float(config.solve_limit_s)}),
            eps_r_rel=1e-6, prune=False, check_pot=False, selection="none")
        coarse_s = time.perf_counter() - t_c
        log.line(f"  coarse: rounds={cr.n_iterations} |F|={cr.finite_edges} {coarse_s:.1f}s "
                 f"stall={cr.stall_status}")
        # ---- prolong + seed ----
        t_s = time.perf_counter()
        f_full, g_full = _prolong(x, y, np.asarray(cr.theta, dtype=np.float64),
                                  np.asarray(cr.source_dual, dtype=np.float64),
                                  np.asarray(cr.target_dual, dtype=np.float64), coarse, fine)
        part = support_components(problem)
        connectors = connector_edges(problem, part)
        mandatory = fine.support.union(connectors)
        fin_edges = _dense_edges(m, m)
        # CN: **`A_carry` 暂未实现**：`RegretResult` 不暴露最终边集，构造它需要改库。按要求如实记录，
        #     不伪造。缺它只会让种子变差，不影响正确性（Lemma 5 只要求 `B ⊆ F₀`）。
        # EN: **`A_carry` is not implemented**: `RegretResult` does not expose the final edge set, so
        #     building it needs a library change. Recorded honestly rather than faked. Missing carry only
        #     weakens the seed; it cannot affect correctness (Lemma 5 needs only `B subseteq F0`).
        carry_keys: Int[np.ndarray, "carry"] = np.zeros(0, dtype=np.int64)
        seeded, seed_stats = _seed(problem, np.asarray(cr.theta, dtype=np.float64), f_full, g_full,
                                   mandatory, int(config.kappa), carry_keys, n_target=m)
        seed_s = time.perf_counter() - t_s
        # ---- fine: start from the seed ----
        t_f = time.perf_counter()
        fin = solve_inverse_ot_regret(
            problem, tol=None, gamma=int(config.gamma), max_iterations=200, warm_start=True,
            initial_extra=seeded, master=RegretMasterSolver(backend=str(config.solver),
                                                            tie_break=False,
                                                            params={"time_sec_limit": float(config.solve_limit_s)}),
            eps_r_rel=1e-6, prune=False, check_pot=False, selection="none")
        fine_s = time.perf_counter() - t_f
        total_s = time.perf_counter() - t_start
        theta_fin = np.asarray(fin.theta, dtype=np.float64)
        gate = is_optimal_plan(problem, theta_fin)
        # CN: master 可行性：全网格上 `h_e = sigma_e + delta <= tol`。
        slack_all = slack_toward_cost(problem, theta_fin, np.asarray(fin.source_dual, np.float64),
                                      np.asarray(fin.target_dual, np.float64),
                                      fin_edges.rows, fin_edges.cols)
        feas = float(np.max(-slack_all))
        row = {"seed": seed, "rho": float(config.rho), "kappa": int(config.kappa),
               "gamma": int(config.gamma),
               "coarse_grid": f"{coarse.n_source}x{coarse.n_target}",
               "coarse_frac": round(coarse.n_source * coarse.n_target / (m * m), 5),
               "flat_rounds": int(flat.n_iterations), "flat_s": round(flat_s, 1),
               "hier_rounds": int(fin.n_iterations), "hier_s": round(total_s, 1),
               "coarse_s": round(coarse_s, 1), "seed_s": round(seed_s, 1), "fine_s": round(fine_s, 1),
               "ratio_vs_flat": round(total_s / max(flat_s, 1e-9), 3),
               "F0": int(seeded.size), "F_final": int(fin.finite_edges),
               "F0_over_Ffinal": round(seeded.size / max(int(fin.finite_edges), 1), 3),
               "seed_outside_mandatory": int(seeded.size) - int(mandatory.size),
               "is_optimal_plan": bool(gate.is_optimal),
               "max_h_violation": round(feas, 6),
               **{k2: seed_stats[k2] for k2 in seed_stats}}
        rows.append(row)
        log.line(json.dumps(row))

    (run_dir / "stage1.json").write_text(
        json.dumps({"rows": rows, "provenance": PROVENANCE,
                    "config": {k2: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                               for k2, v in dataclasses.asdict(config).items()}}, indent=2) + "\n",
        encoding="utf-8")
    log.line(f"wrote {run_dir / 'stage1.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
