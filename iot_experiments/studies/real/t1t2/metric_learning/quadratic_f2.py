"""
CN: **二次 iOT 在 F2 的设置里**——数据、池、几何、CV、指标全部照搬
    `studies/real/t1t2/metric_learning/run.py`，**唯一**改动是代价族与求解引擎：
        - 代价族：`QuadraticFeatures`（`K = d(d+1)/2`，对 θ 线性）；
        - 引擎：松弛循环 `solve_inverse_ot_regret`（`prune=False`），**不用** `fit_full_lp`。

    CN: 因此这里的 Δhit 可以**直接**与 F2 的 iOT/InfoNCE/residual 行比较：余弦基线是同一个
        （池、G2 几何、L2 归一化都一样），`rel` 的选择规则、K 折、pool EMD 指标也一致。

    CN: θ 的维度不同（对角 `k` vs 二次 `d(d+1)/2`），因此二次臂的 `d` 不等于 F2 的 `k`；
        用 `--quad-k` 指定特征宽度，并在记录里同时写明 `k`（PCA 宽度）与 `K`（θ 长度）。

EN: **Quadratic iOT inside F2's setup** -- data, pool, geometry, CV and metrics are taken verbatim from
    `studies/real/t1t2/metric_learning/run.py`. The only changes are the cost family and the engine:
        - family: `QuadraticFeatures` (`K = d(d+1)/2`, linear in theta);
        - engine: the relaxed loop `solve_inverse_ot_regret` (`prune=False`), **not** `fit_full_lp`.

    So its Δhit is **directly** comparable to F2's iOT / InfoNCE / residual rows: the cosine baseline is the
    same object (same pool, same G2 geometry, same L2 normalisation), and `rel` selection, the K-fold and the
    pool-EMD metrics match.

    The theta dimension differs (diagonal `k` vs quadratic `d(d+1)/2`), so a quadratic arm's `d` is not F2's
    `k`; `--quad-k` sets the feature width and every record states both `k` (PCA width) and `K` (theta length).
"""

from __future__ import annotations

import dataclasses
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
from iot_experiments.harness.registry import apply_geometry, features, fit_geometry  # noqa: E402
from iot_experiments.harness.solver import (  # noqa: E402
    box_simplex, diagonal_cost, matching_hit, rank_metrics)
from iot_experiments.studies.real.t1t2.metric_learning.run import pair_hit  # noqa: E402

PROVENANCE: dict[str, object] = {
    "setup": "verbatim from studies/real/t1t2/metric_learning: dinov2-pool4 PCA, pool of 400 held out, "
             "geometry G2 from the pool, labelled subsets per draw, cosine baseline, pool EMD metrics",
    "family": "QuadraticFeatures: c_ij = sum_{a<=b} theta_ab (x_ia-y_ja)(x_ib-y_jb), K = d(d+1)/2",
    "engine": "solve_inverse_ot_regret (relaxed loop, prune=False); fit_full_lp NOT used",
    "rel_selection": "same K-fold as select_iot (REL_GRID + 0.05), but each fold solved by the loop",
    "comparable_to": "F2's iOT / infonce / residual64 rows at the same m",
}

REL_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9)
FOLDS: int = 5
HOLDOUT_PAIRS: int = 400


def _rss_gb() -> float:
    """CN: 峰值 RSS（GB）。EN: peak RSS in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _quadratic_cost(x: Float[np.ndarray, "n d"], y: Float[np.ndarray, "n d"],
                    theta: Float[np.ndarray, "K"]) -> Float[np.ndarray, "n n"]:
    """
    CN: 与 `QuadraticFeatures.costs` 同一定义：`c = diff^T M diff`，`M_aa = theta_aa`，
        `M_ab = theta_ab/2`；θ 的顺序是 `pairs()` 的行主序上三角。
    EN: identical to `QuadraticFeatures.costs`: `c = diff^T M diff` with `M_aa = theta_aa` and
        `M_ab = theta_ab/2`; theta is ordered like `pairs()` (row-major upper triangle).
    """
    d = int(x.shape[1])
    upper = np.triu_indices(d)
    flat = np.asarray(theta, dtype=np.float64)
    if flat.size != upper[0].size:
        raise ValueError(f"theta has {flat.size} entries, expected {upper[0].size}")
    values = np.where(upper[0] == upper[1], flat, 0.5 * flat)
    matrix = np.zeros((d, d), dtype=np.float64)
    matrix[upper[0], upper[1]] = values
    matrix[upper[1], upper[0]] = values
    n = int(x.shape[0])
    out = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        diff = x[start:stop, None, :] - y[None, :, :]
        out[start:stop] = np.einsum("bcd,de,bce->bc", diff, matrix, diff)
    return out


def _fit_loop(x: Float[np.ndarray, "m d"], y: Float[np.ndarray, "m d"], *, rel: float, solver: str,
              solve_limit_s: float, max_iterations: int) -> tuple[Float[np.ndarray, "K"] | None, dict[str, object]]:
    """
    CN: 用松弛循环解一次；返回 `(theta, 诊断)`。失败返回 `None`，**绝不**回退成 uniform θ。
    EN: one relaxed-loop solve; returns `(theta, diagnostics)`. On failure returns `None` and **never**
        substitutes a uniform theta.
    """
    m = int(x.shape[0])
    family = QuadraticFeatures(x, y)
    K = int(family.n_features)
    plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
    problem = InverseOTProblem(features=family,
                               theta_set=box_simplex(K, box_relative=float(rel)), plan=plan)
    t0 = time.perf_counter()
    try:
        res = solve_inverse_ot_regret(
            problem, tol=None, gamma=1, max_iterations=int(max_iterations),
            master=RegretMasterSolver(backend=str(solver), tie_break=False,
                                      params={"time_sec_limit": float(solve_limit_s)}),
            eps_r_rel=1e-6, prune=False, check_pot=False, selection="none")
        return np.asarray(res.theta, dtype=np.float64), {
            "status": "ok", "rounds": int(res.n_iterations), "F": int(res.finite_edges),
            "converged": bool(res.converged), "stall": str(res.stall_status),
            "width_rel": float((res.upper - res.lower) / max(float(res.upper), 1e-30)),
            "secs": round(time.perf_counter() - t0, 2)}
    except Exception as error:  # noqa: BLE001 - reported, never substituted
        return None, {"status": f"FAILED: {type(error).__name__}: {str(error)[:100]}",
                      "secs": round(time.perf_counter() - t0, 2)}


def _evaluate(cost: Float[np.ndarray, "n n"], *, draw: int, fraction: float = 0.5) -> dict[str, float]:
    """
    CN: F2 `metric_learning.evaluate` 的同一实现：hit、R@1、margin-P@fraction。
    EN: byte-for-byte the recipe of F2's `metric_learning.evaluate`: hit, R@1, margin-P@fraction.
    """
    values = np.asarray(cost, dtype=np.float64)
    hit = matching_hit(values, seed=int(draw))
    ranks = rank_metrics(values)
    row_sorted = np.sort(values, axis=1)
    col_sorted = np.sort(values, axis=0)
    margin = np.minimum(row_sorted[:, 1] - row_sorted[:, 0], col_sorted[1, :] - col_sorted[0, :])
    n = int(values.shape[0])
    weights = np.ones(n, dtype=np.float64) / float(n)
    perm = np.random.default_rng(int(draw)).permutation(n)
    plan = ot.emd(weights, weights, values[:, perm], numItermax=10_000_000)
    hits = np.argmax(plan, axis=1) == np.argsort(perm)
    order = np.argsort(-margin, kind="stable")
    take = max(1, int(round(float(fraction) * n)))
    return {"hit": hit, "R@1": float(ranks["R@1"]),
            f"P@{int(round(fraction * 100))}": float(np.mean(np.asarray(hits, np.float64)[order[:take]]))}


@dataclasses.dataclass(frozen=True)
class QuadF2Config(RunConfig):
    """CN: 本脚本的旋钮（其余全部来自 F2 的默认值）。EN: this script's knobs; the rest are F2's defaults."""

    section = "t1t2_quad_f2"
    study: str = "t1t2_quad_f2"
    quad_k: int = 64
    ms: tuple[int, ...] = (50, 100, 300)
    draws: tuple[int, ...] = (0, 1, 2)
    holdout_pairs: int = HOLDOUT_PAIRS
    holdout_draw: int = 0
    max_iterations: int = 200

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--quad-k", type=int, default=64,
                            help="PCA width d of the quadratic family (theta length K = d(d+1)/2)")
        parser.add_argument("--ms", type=int, nargs="+", default=[50, 100, 300])
        parser.add_argument("--holdout-pairs", type=int, default=HOLDOUT_PAIRS)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(QuadF2Config)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = QuadF2Config.from_namespace(args, parser)
    run_dir = config.run_dir / f"d{int(config.quad_k)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("setup: F2 (t1t2 metric learning), quadratic family, relaxed loop engine, prune=False")
    log.line(f"config: d={config.quad_k} ms={config.ms} draws={config.draws} "
             f"holdout_pairs={config.holdout_pairs} holdout_draw={config.holdout_draw}")

    k = int(config.quad_k)
    x, y = features("dinov2-pool4", dim=k)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_subjects = int(x.shape[0])
    log.line(f"features: {x.shape} (dinov2-pool4, d={k}); peak {_rss_gb():.2f} GB")

    pool_rng = np.random.default_rng(10_000 + int(config.holdout_draw))
    pool_order = pool_rng.permutation(n_subjects)
    pool_subjects = np.sort(pool_order[: int(config.holdout_pairs)])
    remaining = np.sort(pool_order[int(config.holdout_pairs):])
    stats = fit_geometry(x[pool_subjects], y[pool_subjects], "G2")
    x_pool, y_pool = apply_geometry(x[pool_subjects], y[pool_subjects], stats)
    for arr in (x_pool, y_pool):
        norms = np.linalg.norm(np.asarray(arr, dtype=np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-8):
            raise SystemExit(f"pool space not L2-normalised: [{norms.min():.6f}, {norms.max():.6f}]")
    uniform = np.ones(k, dtype=np.float64) / float(k)
    cosine_pool = diagonal_cost(x_pool, y_pool, uniform)
    log.line(f"pool: {len(pool_subjects)} pairs; labelled pool {len(remaining)}")

    rows: list[dict[str, object]] = []
    for m in [int(v) for v in config.ms]:
        for draw in [int(d) for d in config.draws]:
            label_rng = np.random.default_rng(draw)
            labelled = np.sort(label_rng.choice(remaining, size=m, replace=False))
            x_fit, y_fit = apply_geometry(x[labelled], y[labelled], stats)
            if np.intersect1d(labelled, pool_subjects).size:
                raise SystemExit("labelled and pool subjects overlap")
            base = _evaluate(cosine_pool, draw=draw)

            # CN: `rel` 用与 `select_iot` 相同的 K 折规则选择，但每折由**循环**求解。
            # EN: pick `rel` with the same K-fold rule as `select_iot`, but each fold is solved by the loop.
            folds = []
            for fold in range(FOLDS):
                take = (np.arange(x_fit.shape[0]) % FOLDS) == fold
                folds.append((~take, take))
            best: tuple[float, float] | None = None
            for rel in list(REL_GRID) + [0.05]:
                scores: list[float] = []
                for train, val in folds:
                    theta, diag = _fit_loop(x_fit[train], y_fit[train], rel=float(rel),
                                            solver=str(config.solver),
                                            solve_limit_s=float(config.solve_limit_s),
                                            max_iterations=int(config.max_iterations))
                    if theta is None:
                        scores.append(float("nan"))
                        continue
                    scores.append(pair_hit(_quadratic_cost(x_fit[val], y_fit[val], theta)))
                score = float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")
                if not np.isfinite(score):
                    continue
                if best is None or score > best[1]:
                    best = (float(rel), score)
            rel_chosen = best[0] if best is not None else 0.25

            theta, diag = _fit_loop(x_fit, y_fit, rel=rel_chosen, solver=str(config.solver),
                                    solve_limit_s=float(config.solve_limit_s),
                                    max_iterations=int(config.max_iterations))
            if theta is None:
                rows.append({"m": m, "k": k, "K": k * (k + 1) // 2, "draw": draw,
                             "rel": rel_chosen, "status": diag["status"], "hit": None,
                             "d_hit": None, "cos_hit": base["hit"]})
                log.line(f"[d={k} m={m} draw={draw}] FAILED {diag['status']}")
                continue

            met = _evaluate(_quadratic_cost(x_pool, y_pool, theta), draw=draw)
            row = {"m": m, "k": k, "K": k * (k + 1) // 2, "draw": draw, "rel": rel_chosen,
                   "status": "ok", "hit": met["hit"], "R@1": met["R@1"], "P@50": met["P@50"],
                   "cos_hit": base["hit"], "cos_R@1": base["R@1"],
                   "d_hit": met["hit"] - base["hit"],
                   "d_hit_pts": round(100.0 * (met["hit"] - base["hit"]), 1),
                   "cv_score": best[1] if best is not None else None,
                   "rounds": diag["rounds"], "F": diag["F"], "converged": diag["converged"],
                   "stall": diag["stall"], "width_rel": round(float(diag["width_rel"]), 4),
                   "secs": diag["secs"], "theta_l1": round(float(np.abs(theta).sum()), 6),
                   "peak_gb": round(_rss_gb(), 2)}
            rows.append(row)
            log.line(json.dumps(row))

    payload = {"rows": rows, "provenance": PROVENANCE,
               "config": {kk: (list(vv) if isinstance(vv, tuple) else str(vv) if isinstance(vv, Path) else vv)
                          for kk, vv in dataclasses.asdict(config).items()}}
    (run_dir / "quad_f2.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                          encoding="utf-8")
    log.line(f"wrote {run_dir / 'quad_f2.json'}; peak {_rss_gb():.2f} GB")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
