"""
CN: **用"它能不能完成任务"来评判二次 iOT**，而不是用收敛性。

    CN: 做法与 F2（`metric_learning`）完全一致：在同一个 400 对的留出池上，比较二次 iOT 的代价
        与 cosine 的代价在 **matching hit** 上的差别。iOT 只需给出 θ；是否收敛只记录、不判定。

    CN: 两条路径都跑，**同一问题**：
        (a) 松弛循环 `solve_inverse_ot_regret`（prune=False）—— 生产路径；
        (b) 全边 LP `fit_full_lp` —— 仅作 m 足够小、内存允许时的参照（m=300、K=2080 会 OOM，故只在 m≤100 跑）。

    EN: **Judge quadratic iOT by whether it does the task**, not by convergence.

    Same protocol as F2 (`metric_learning`): on the same held-out pool of 400 pairs, compare the hit rate of the
    quadratic-iOT cost against the cosine cost. iOT only has to produce a theta; convergence is recorded, not
    used as a verdict.

    Both engines run on the **same problem** when memory allows:
        (a) the relaxed loop `solve_inverse_ot_regret` (prune=False) -- the production path;
        (b) the all-edges LP `fit_full_lp` -- reference only, and only at m where it fits (m=300, K=2080 OOMs).
"""

from __future__ import annotations

import dataclasses
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import ot
import scipy.sparse as sparse
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import InverseOTProblem  # noqa: E402
from iot.core.features import QuadraticFeatures  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot.relaxed.master_regret import RegretMasterSolver  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402
from iot_experiments.harness.solver import (  # noqa: E402
    box_simplex, diagonal_cost, fit_full_lp, matching_hit, rank_metrics)
from iot_experiments.harness.synthrad import discover_patients  # noqa: E402
from iot_experiments.harness.datasets import reduce_features, stream_features  # noqa: E402

PROVENANCE: dict[str, object] = {
    "question": "does quadratic iOT do the task (hit rate) on the same held-out pool as F2?",
    "family": "QuadraticFeatures: c_ij = sum_{a<=b} theta_ab (x_ia-y_ja)(x_ib-y_jb), K = d(d+1)/2",
    "production": "relaxed loop solve_inverse_ot_regret, prune=False",
    "reference": "all-edges LP fit_full_lp, only where memory allows",
    "metric": "matching_hit on the held-out pool of 400, cosine as the baseline",
}


def _rss_gb() -> float:
    """CN: 峰值 RSS（GB）。EN: peak RSS in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _metrics(cost: Float[np.ndarray, "m n"], draw: int) -> dict[str, float]:
    """
    CN: 与 F2 `metric_learning.evaluate` 同一口径：hit、R@1、margin-P@50，覆盖子集随 draw 变化。
    EN: same recipe as F2's `metric_learning.evaluate`: hit, R@1, margin-P@50, with the draw-dependent
        coverage subset.
    """
    values = np.asarray(cost, dtype=np.float64)
    out = {"hit": matching_hit(values, seed=int(draw))}
    out.update(rank_metrics(values))
    row_s = np.sort(values, axis=1)
    col_s = np.sort(values, axis=0)
    margin = np.minimum(row_s[:, 1] - row_s[:, 0], col_s[1, :] - col_s[0, :])
    n = int(values.shape[0])
    weights = np.ones(n, dtype=np.float64) / float(n)
    perm = np.random.default_rng(int(draw)).permutation(n)
    plan = ot.emd(weights, weights, values[:, perm], numItermax=10_000_000)
    hits = np.argmax(plan, axis=1) == np.argsort(perm)
    order = np.argsort(-margin, kind="stable")
    take = max(1, int(round(0.5 * n)))
    out["P@50"] = float(np.mean(np.asarray(hits, dtype=np.float64)[order[:take]]))
    return out


def _quadratic_cost(x: Float[np.ndarray, "n d"], y: Float[np.ndarray, "n d"],
                    theta: Float[np.ndarray, "K"]) -> Float[np.ndarray, "n n"]:
    """
    CN: `c_ij = Σ_{a≤b} θ_ab (x_ia-y_ja)(x_ib-y_jb)`，把 θ 摊回对称矩阵（上三角、行主序）。
    EN: `c_ij = sum_{a<=b} theta_ab (x_ia-y_ja)(x_ib-y_jb)`, θ scattered back into a symmetric matrix.
    """
    d = int(x.shape[1])
    # CN: θ 的顺序是 `QuadraticFeatures.pairs()` 的**行主序上三角** (a ≤ b)，不是 `np.tril_indices` 的
    #     列主序；用错顺序会静默打乱 θ 与特征的对应。
    # EN: theta is ordered like `QuadraticFeatures.pairs()` -- **row-major upper triangle** (a <= b) --
    #     not `np.tril_indices`' column-major order; the wrong order silently permutes theta vs features.
    upper = np.triu_indices(d)
    sym = np.zeros((d, d), dtype=np.float64)
    # CN: 与 `QuadraticFeatures.costs` 同一定义：`c = diff^T M diff`，`M_aa = theta_aa`，
    #     `M_ab = theta_ab / 2`（a != b），因此对称求和后每个交叉项恰好计一次。
    # EN: same definition as `QuadraticFeatures.costs`: `c = diff^T M diff` with `M_aa = theta_aa` and
    #     `M_ab = theta_ab / 2` for a != b, so the symmetric sum counts each cross term exactly once.
    flat = np.asarray(theta, dtype=np.float64)
    if flat.size != upper[0].size:
        raise ValueError(f"theta has {flat.size} entries, expected {upper[0].size}")
    diag = upper[0] == upper[1]
    values = np.where(diag, flat, 0.5 * flat)
    sym[upper[0], upper[1]] = values
    sym[upper[1], upper[0]] = values
    # CN: 已经**对称写入**且交叉项已减半，这里不能再做 `sym + sym.T`（那会把交叉项再翻一倍）。
    # EN: the matrix is already filled symmetrically with halved off-diagonals; adding `sym + sym.T`
    #     here would double the cross terms again.
    n = int(x.shape[0])
    out = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, 512):
        stop = min(start + 512, n)
        diff = x[start:stop, None, :] - y[None, :, :]
        # CN: diff 形状 (block, n, d)；两个 d 轴用不同字母，才能同时收缩成 (block, n)。
        # EN: diff is (block, n, d); give the two d axes different letters so both contract to (block, n).
        out[start:stop] = np.einsum("bcd,de,bce->bc", diff, sym, diff)
    return out


@dataclasses.dataclass(frozen=True)
class QuadTaskConfig(RunConfig):
    """CN: 本脚本的旋钮。EN: this script's knobs."""

    section = "synthrad_quad_task"
    study: str = "synthrad_quad_task"
    quad_k: int = 64
    rel: float = 0.25
    ms: tuple[int, ...] = (50, 100, 300)
    draw: int = 0
    pool_pairs: int = 400
    max_iterations: int = 200
    run_dense: bool = True
    dense_max_m: int = 100

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--quad-k", type=int, default=64)
        parser.add_argument("--rel", type=float, default=0.25)
        parser.add_argument("--ms", type=int, nargs="+", default=[50, 100, 300])
        parser.add_argument("--draw", type=int, default=0)
        parser.add_argument("--max-iterations", type=int, default=200)
        parser.add_argument("--no-dense", action="store_true", help="skip the all-edges reference")
        parser.add_argument("--dense-max-m", type=int, default=100,
                            help="largest m for the all-edges reference; its lil assembly is O(E*K) "
                                 "and OOMs above roughly m=300, K=2000")


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(QuadTaskConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = QuadTaskConfig.from_namespace(args, parser)
    # CN: 输出目录按 `d` 分键，否则下一个 `d` 会覆盖上一个的 `quad_task.json`
    #     （d=64 的结果就是这样被 d=32 覆盖掉的，只剩 job log）。
    # EN: key the output directory by `d`; otherwise the next `d` overwrites the previous `quad_task.json`
    #     (which is exactly how the d=64 record was lost to the d=32 run, surviving only in a job log).
    run_dir = config.run_dir / f"d{int(config.quad_k)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("production engine: relaxed loop (prune=False); dense LP is a reference only")
    log.line(f"config: d={config.quad_k} rel={config.rel} ms={config.ms} draw={config.draw} "
             f"pool={config.pool_pairs}")
    log.line(f"starting RSS {_rss_gb():.2f} GB, peak so far {_rss_gb():.2f} GB")

    split = json.loads((config.out / "synthrad_v2" / "split_v2.json").read_text())
    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    fit_pats = [p for p in patients if p.patient in set(split["fit"])]
    rng = np.random.default_rng(1000 + int(config.draw))
    chosen = [fit_pats[i] for i in rng.permutation(len(fit_pats))[:60]]
    log.line(f"streaming features for {len(chosen)} FIT patients (d={config.quad_k})")
    pm, pc, _pid, _z, _ctr = stream_features(chosen, window=(-200.0, 300.0), max_per_patient=None,
                                             min_gap=0, seed=int(config.draw))
    ftr, ftc = reduce_features(pm, pc, k=int(config.quad_k), geometry_name="G3", whiten=False)
    del pm, pc
    gc.collect()
    log.line(f"features ready {ftr.shape}, peak {_rss_gb():.2f} GB")

    # CN: L2 归一化，与 F2 的池空间一致；cosine 基线就是 uniform θ 的对角代价。
    def _normalise(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12))

    x_all, y_all = _normalise(ftr), _normalise(ftc)
    order = np.random.default_rng(int(config.seed) + 7).permutation(x_all.shape[0])
    pool_idx = np.sort(order[: int(config.pool_pairs)])
    rest = np.sort(order[int(config.pool_pairs):])
    x_pool, y_pool = x_all[pool_idx], y_all[pool_idx]
    uni = np.ones(int(config.quad_k), dtype=np.float64) / float(config.quad_k)
    log.line(f"pool: {x_pool.shape[0]} pairs (cosine baseline is computed per draw inside the loop)")

    rows: list[dict[str, object]] = []
    for m in [int(v) for v in config.ms]:
        rng_m = np.random.default_rng(int(config.draw))
        pick = np.sort(rng_m.choice(rest, size=m, replace=False))
        x: Float[np.ndarray, "m d"] = np.ascontiguousarray(x_all[pick])
        y: Float[np.ndarray, "m d"] = np.ascontiguousarray(y_all[pick])
        family = QuadraticFeatures(x, y)
        K = int(family.n_features)
        plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
        problem = InverseOTProblem(features=family,
                                   theta_set=box_simplex(K, box_relative=float(config.rel)),
                                   plan=plan)
        log.line(f"--- m={m} d={config.quad_k} K={K} ---")

        # CN: 每个 draw 有独立的候选打乱与 50% 覆盖子集，所以 baseline 也按 draw 重算。
        # EN: each draw gets its own column shuffle and 50% coverage subset, so the baseline is per draw too.
        def _baseline(draw: int) -> dict[str, float]:
            cost_u = diagonal_cost(x_pool, y_pool, uni)
            out = {"hit": matching_hit(cost_u, seed=draw)}
            out.update(rank_metrics(cost_u))
            row_s = np.sort(cost_u, axis=1)
            col_s = np.sort(cost_u, axis=0)
            margin = np.minimum(row_s[:, 1] - row_s[:, 0], col_s[1, :] - col_s[0, :])
            n_p = int(cost_u.shape[0])
            w = np.ones(n_p) / n_p
            perm = np.random.default_rng(draw).permutation(n_p)
            plan_u = ot.emd(w, w, cost_u[:, perm], numItermax=10_000_000)
            hits_u = np.argmax(plan_u, axis=1) == np.argsort(perm)
            order_m = np.argsort(-margin, kind="stable")
            kk = max(1, int(round(0.5 * n_p)))
            out["P@50"] = float(np.mean(np.asarray(hits_u, dtype=np.float64)[order_m[:kk]]))
            return {f"cos_{name}": float(v) for name, v in out.items()}

        # (a) relaxed loop — the production path
        t0 = time.perf_counter()
        cos_by_draw = {d: _baseline(d) for d in range(3)}
        row: dict[str, object] = {"m": m, "K": K,
                                  "cos_hit_by_draw": [round(cos_by_draw[d]["cos_hit"], 4)
                                                      for d in range(3)]}
        try:
            res = solve_inverse_ot_regret(
                problem, tol=None, gamma=1, max_iterations=int(config.max_iterations),
                master=RegretMasterSolver(backend=str(config.solver), tie_break=False,
                                          params={"time_sec_limit": float(config.solve_limit_s)}),
                eps_r_rel=1e-6, prune=False, check_pot=False, selection="none")
            theta_loop = np.asarray(res.theta, dtype=np.float64)
            cost_loop = _quadratic_cost(x_pool, y_pool, theta_loop)
            loop_metrics = {d: _metrics(cost_loop, d) for d in range(3)}
            row.update({"loop_delta_hit_by_draw": [round(loop_metrics[d]["hit"]
                                                         - cos_by_draw[d]["cos_hit"], 4)
                                                   for d in range(3)],
                        "loop_status": "ok", "loop_rounds": int(res.n_iterations),
                        "loop_F": int(res.finite_edges), "loop_converged": bool(res.converged),
                        "loop_stall": str(res.stall_status),
                        "loop_width_rel": float((res.upper - res.lower) / max(float(res.upper), 1e-30)),
                        "loop_secs": round(time.perf_counter() - t0, 1),
                        "loop_theta_l1": round(float(np.abs(theta_loop).sum()), 6)})
            row.update({f"loop_{k}": round(float(np.mean([loop_metrics[d][k] for d in range(3)])), 4)
                        for k in ("R@1", "P@50")})
        except Exception as error:  # noqa: BLE001 - reported, never substituted
            row.update({"loop_status": f"FAILED: {type(error).__name__}: {str(error)[:100]}",
                        "loop_secs": round(time.perf_counter() - t0, 1)})

        # (b) all-edges LP reference — only when it fits in memory
        if config.run_dense and m <= int(config.dense_max_m):
            t0 = time.perf_counter()
            try:
                fit = fit_full_lp(x, y, solver=str(config.solver),
                                  solve_limit_s=float(config.solve_limit_s),
                                  box_relative=float(config.rel), features=family)
                if fit.status == "ok":
                    cost_d = _quadratic_cost(x_pool, y_pool, np.asarray(fit.theta, dtype=np.float64))
                    dense_metrics = {d: _metrics(cost_d, d) for d in range(3)}
                    row.update({"dense_status": "ok",
                                "dense_delta_hit_by_draw": [round(dense_metrics[d]["hit"]
                                                                  - cos_by_draw[d]["cos_hit"], 4)
                                                            for d in range(3)],
                                "dense_R@1": round(float(np.mean([dense_metrics[d]["R@1"]
                                                                 for d in range(3)])), 4),
                                "dense_P@50": round(float(np.mean([dense_metrics[d]["P@50"]
                                                                  for d in range(3)])), 4),
                                "dense_secs": round(time.perf_counter() - t0, 1)})
                else:
                    row.update({"dense_status": fit.status[:90]})
            except Exception as error:  # noqa: BLE001
                row.update({"dense_status": f"FAILED: {type(error).__name__}: {str(error)[:90]}"})

        row["peak_gb"] = round(_rss_gb(), 2)
        rows.append(row)
        log.line(json.dumps(row))
        del family, problem
        gc.collect()

    payload = {"rows": rows, "provenance": PROVENANCE,
               "config": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in dataclasses.asdict(config).items()}}
    (run_dir / "quad_task.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                                   encoding="utf-8")
    log.line(f"wrote {run_dir / 'quad_task.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
