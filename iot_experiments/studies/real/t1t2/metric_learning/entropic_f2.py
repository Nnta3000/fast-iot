"""
CN: **熵 OT 在 F2 的设置里** —— 数据、池、几何、指标全部照搬
    `studies/real/t1t2/metric_learning/run.py`，唯一的改动是**引擎**：

        - 精确/松弛 iOT：解一个 master，给 θ 与证书（`delta` 或区间 `[L,U]`）；
        - 这里：`solve_entropic` —— Sinkhorn 给出 `P_ε(θ)`，投影梯度最小化
          `L_ε(θ) = <C_θ, π̂> - <C_θ, P_ε(θ)>`。**没有 master，没有证书。**

    CN: 因此这里的 Δhit 可以**直接**与 F2 的 iOT/InfoNCE/residual 行比较：余弦基线是同一个
        （同池、同 G2 几何、同 L2 归一化、同 `matching_hit`）。

    CN: `relative_epsilon` 用与 iOT 的 `rel` 同样的 **K 折**规则在标注对上选择，池不参与，
        所以选择过程与 F2 一样是"标注对内"的。池只用于最终评分。

    CN: **诚实标注**：本臂不产生证书。它测的是**泛化**（留出池上的命中率），不是最优性。

EN: **Entropic OT inside F2's setup** -- data, pool, geometry and metrics are taken verbatim from
    `studies/real/t1t2/metric_learning/run.py`; only the **engine** changes:

        - exact/relaxed iOT: solve a master, returning theta and a certificate (`delta` or `[L,U]`);
        - here: `solve_entropic` -- Sinkhorn gives `P_eps(theta)`, projected gradient minimises
          `L_eps(theta) = <C_theta, pi_hat> - <C_theta, P_eps(theta)>`. **No master, no certificate.**

    So the Δhit is **directly** comparable to F2's iOT / InfoNCE / residual rows: same pool, same G2
    geometry, same L2 normalisation, same `matching_hit`.

    `relative_epsilon` is chosen by the same **K-fold** rule used for iOT's `rel`, on labelled pairs only, so
    the pool never enters selection. The pool scores only.

    **Stated plainly: this arm produces no certificate.** It measures **generalisation** (held-out pool hit
    rate), not optimality.
"""

from __future__ import annotations

import dataclasses
import json
import resource
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import InverseOTProblem  # noqa: E402
from iot.core.features import DiagonalWeightFeatures  # noqa: E402
from iot.core.theta import SimplexTheta  # noqa: E402
from iot.core.parameters import EntropicObjective  # noqa: E402
from iot.entropic import solve_entropic  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402
from iot_experiments.harness.registry import apply_geometry, features, fit_geometry  # noqa: E402
from iot_experiments.harness.solver import diagonal_cost, matching_hit, rank_metrics  # noqa: E402
from iot_experiments.studies.real.t1t2.metric_learning.run import pair_hit  # noqa: E402

FOLDS: int = 5
HOLDOUT_PAIRS: int = 400
# CN: 相对 ε 的候选网格；小 ε 更接近无正则化但 Sinkhorn 迭代数急剧上升。
# EN: candidate grid for the relative epsilon; smaller is closer to unregularised but costs iterations.
# CN: 实测 ε=0.1 需要 >20000 步 Sinkhorn 才收敛（marginal err 9e-7），故网格止于 0.25。
#     EN: measured: epsilon=0.1 needs >20000 Sinkhorn iterations (marginal err 9e-7), so the
#     grid stops at 0.25.
EPS_GRID: tuple[float, ...] = (0.5, 0.25)

PROVENANCE: dict[str, object] = {
    "setup": "verbatim from studies/real/t1t2/metric_learning: dinov2-pool4 PCA, pool of 400 held out, "
             "geometry G2 from the pool, labelled subsets per draw, cosine baseline, pool EMD metrics",
    "engine": "solve_entropic (Sinkhorn + projected gradient); no master, NO CERTIFICATE",
    "objective": "min L_eps(theta) = <C_theta, pi_hat> - <C_theta, P_eps(theta)>",
    "eps_selection": "same K-fold as select_iot, on labelled pairs only; relative_epsilon multiplied by "
                     "std(cost) internally",
    "measures": "generalisation (held-out pool hit rate), not optimality",
}


def _rss_gb() -> float:
    """CN: 峰值 RSS（GB）。EN: peak RSS in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _evaluate(cost: Float[np.ndarray, "n n"], *, draw: int, fraction: float = 0.5) -> dict[str, float]:
    """
    CN: 与 F2 `metric_learning.evaluate` 同一实现：hit、R@1、margin-P@50。
    EN: byte-for-byte the recipe of F2's `metric_learning.evaluate`: hit, R@1, margin-P@50.
    """
    import ot  # noqa: PLC0415

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


def _fit(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, relative_epsilon: float,
         max_steps: int, learning_rate: float, device: str,
         sinkhorn_iterations: int) -> tuple[Float[np.ndarray, "k"] | None, dict[str, object]]:
    """
    CN: 在标注对上做一次熵拟合。`DiagonalWeightFeatures`（与 F2 的 iOT 同一族，θ 为对角权重）。
    EN: one entropic fit on the labelled pairs. `DiagonalWeightFeatures` (same family as F2's iOT, theta is
        the diagonal weights).
    """
    import scipy.sparse as sparse  # noqa: PLC0415

    m = int(x.shape[0])
    k = int(x.shape[1])
    uniform_plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
    problem = InverseOTProblem(features=DiagonalWeightFeatures(x, y),
                               # CN: 熵拟合用**核心**单纯形（`fit_entropic` 只接受它）；harness 的
                               #     `box_simplex` 是另一个类，不能混用。
                               # EN: the entropic fit uses the **core** simplex (`fit_entropic` accepts only
                               #     that); the harness `box_simplex` is a different class and cannot be used.
                               theta_set=SimplexTheta(dimension=k), plan=uniform_plan)
    try:
        result = solve_entropic(problem, objective=EntropicObjective(
            relative_epsilon=float(relative_epsilon), learning_rate=float(learning_rate),
            max_steps=int(max_steps), sinkhorn_iterations=int(sinkhorn_iterations), device=str(device)))
    except Exception as error:  # noqa: BLE001 - reported, never substituted
        return None, {"status": f"FAILED: {type(error).__name__}: {str(error)[:100]}"}
    if not np.all(np.isfinite(result.theta)):
        return None, {"status": "FAILED: non-finite theta"}
    return np.asarray(result.theta, dtype=np.float64), {
        "status": "ok", "steps": int(result.steps), "converged": bool(result.converged),
        "stall": str(result.stall_reason), "cost_std": float(result.cost_std),
        "final_gradient": float(result.steps_record[-1].gradient_norm) if result.steps_record else float("nan")}


def _diagonal_cost(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"],
                   theta: Float[np.ndarray, "k"]) -> Float[np.ndarray, "m m"]:
    """CN: 对角族代价 `c_ij = Σ_a θ_a (x_ia - y_ja)^2`（池上的方阵）。EN: diagonal cost on the pool."""
    return np.asarray(diagonal_cost(x, y, np.asarray(theta, dtype=np.float64)), dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class EntropicF2Config(RunConfig):
    """CN: 旋钮。EN: knobs."""

    section = "t1t2_entropic_f2"
    study: str = "t1t2_entropic_f2"
    k: int = 64
    ms: tuple[int, ...] = (50, 100, 300)
    draw: int = 0
    holdout_pairs: int = HOLDOUT_PAIRS
    holdout_draw: int = 0
    max_steps: int = 200
    learning_rate: float = 0.05
    sinkhorn_iterations: int = 20000

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--k", type=int, default=64, help="PCA width (F2's k)")
        parser.add_argument("--ms", type=int, nargs="+", default=[50, 100, 300])
        parser.add_argument("--draw", type=int, default=0)
        parser.add_argument("--max-steps", type=int, default=200)
        parser.add_argument("--learning-rate", type=float, default=0.05)
        parser.add_argument("--sinkhorn-iterations", type=int, default=20000)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(EntropicF2Config)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = EntropicF2Config.from_namespace(args, parser)
    run_dir = config.run_dir / f"k{int(config.k)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("setup: F2 (t1t2 metric learning), ENTROPIC engine (no master, no certificate)")
    log.line(f"config: k={config.k} ms={config.ms} draws={config.draw} eps_grid={EPS_GRID} "
             f"lr={config.learning_rate} max_steps={config.max_steps}")

    k = int(config.k)
    x, y = features("dinov2-pool4", dim=k)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    log.line(f"features: {x.shape}; peak {_rss_gb():.2f} GB")

    pool_rng = np.random.default_rng(10_000 + int(config.holdout_draw))
    pool_order = pool_rng.permutation(int(x.shape[0]))
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
        for draw in [int(d) for d in [config.draw]]:
            label_rng = np.random.default_rng(draw)
            labelled = np.sort(label_rng.choice(remaining, size=m, replace=False))
            x_fit, y_fit = apply_geometry(x[labelled], y[labelled], stats)
            if np.intersect1d(labelled, pool_subjects).size:
                raise SystemExit("labelled and pool subjects overlap")
            base = _evaluate(cosine_pool, draw=draw)

            # CN: `relative_epsilon` 用 K 折在**标注对**上选，池不参与。
            # EN: pick `relative_epsilon` by K-fold on the **labelled pairs** only; the pool never enters.
            take = np.arange(x_fit.shape[0]) % FOLDS
            best: tuple[float, float] | None = None
            for eps in EPS_GRID:
                scores: list[float] = []
                for fold in range(FOLDS):
                    train, val = take != fold, take == fold
                    theta, diag = _fit(x_fit[train], y_fit[train], relative_epsilon=eps,
                                       max_steps=int(config.max_steps),
                                       learning_rate=float(config.learning_rate),
                                       device=str(config.device),
                                       sinkhorn_iterations=int(config.sinkhorn_iterations))
                    if theta is None:
                        continue
                    scores.append(pair_hit(_diagonal_cost(x_fit[val], y_fit[val], theta)))
                if not scores:
                    continue
                score = float(np.mean(scores))
                if best is None or score > best[1]:
                    best = (float(eps), score)
            eps_chosen = best[0] if best is not None else 0.25

            theta, diag = _fit(x_fit, y_fit, relative_epsilon=eps_chosen,
                               max_steps=int(config.max_steps),
                               learning_rate=float(config.learning_rate), device=str(config.device),
                               sinkhorn_iterations=int(config.sinkhorn_iterations))
            if theta is None:
                rows.append({"m": m, "k": k, "draw": draw, "eps": eps_chosen,
                             "status": diag["status"]})
                log.line(f"[k={k} m={m} draw={draw}] FAILED {diag['status']}")
                continue
            met = _evaluate(_diagonal_cost(x_pool, y_pool, theta), draw=draw)
            row = {"m": m, "k": k, "draw": draw, "eps_chosen": eps_chosen,
                   "train_pair_hit": round(float(best[1]), 4) if best else None,
                   "status": "ok", "hit": round(float(met["hit"]), 4),
                   "R@1": round(float(met["R@1"]), 4), "P@50": round(float(met["P@50"]), 4),
                   "cos_hit": round(float(base["hit"]), 4),
                   "d_hit_pts": round(100.0 * (float(met["hit"]) - float(base["hit"])), 1),
                   **{key: diag[key] for key in ("steps", "converged", "stall", "cost_std",
                                                 "final_gradient")},
                   "peak_gb": round(_rss_gb(), 2)}
            rows.append(row)
            log.line(json.dumps(row))

    (run_dir / "entropic_f2.json").write_text(
        json.dumps({"rows": rows, "provenance": PROVENANCE,
                    "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                               for key, v in dataclasses.asdict(config).items()}}, indent=2) + "\n",
        encoding="utf-8")
    log.line(f"wrote {run_dir / 'entropic_f2.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
