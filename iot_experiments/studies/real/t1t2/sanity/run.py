"""
EN: **Sanity checks for the relaxed / entropic inverse-OT arm on real T1/T2 pairs.**

    These are the checks that were asked for and never run. Each is a recorded experiment, not a probe:
    it writes `RunRecord`s, asserts its own chance floor, and its verdict rule is fixed in `PROVENANCE`
    before the run.

      A. `null`      -- capacity. Fit on the TRUE pairs and on a PERMUTED pairing at the same `(m, K)`,
                        score both on the same held-out pool. If the fit carries no signal they match.
      B. `curve`     -- sample-limited vs feature-limited. Extend F2's learning curve past `m = 300` to
                        `m = 600` (the whole labelled pool), and look for saturation against cosine.
      C. `baselines` -- wrong baseline. The SAME objective ("estimate the plan") under three losses on the
                        same features: row-softmax cross-entropy (logistic / InfoNCE), doubly-stochastic
                        Sinkhorn cross-entropy, and exact inverse OT (the relaxed regret LP).

    Hypotheses D (pool-size transfer) and E (identifiability) live in `studies/real/t1t2/sanity/ident.py`
    and `studies/real/t1t2/sanity/transfer.py`, under the same study name `sanity`.

    Everything here reuses the F2 protocol verbatim (DINOv2-pool4 PCA, geometry G2 from the pool, the one
    fixed 400-pair pool, `rel` chosen by 5-fold on the labelled pairs only), so a row is directly
    comparable to F2 with no extra configuration.

    **Cost.** The relaxed loop evaluates `sigma` over all `m x n` edges: at `m = 600, n = 400` that is
    240k edges per round and a few GB of GPU memory. Budgeted in the study `README.md`.

    CN: **T1/T2 真实配对上的合理性检查**：置换零假设（容量）、学习曲线外推（样本还是特征受限）、
    同一目标下三种损失（行 softmax / Sinkhorn / 精确 iOT）的对照。协议与 F2 完全一致，可直接比较。
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.harness.registry import features  # noqa: E402
from iot_experiments.harness.solver import (  # noqa: E402
    diagonal_cost,
    fit_full_lp,
    matching_hit,
    rank_metrics,
)
from iot_experiments.studies.real.t1t2.entropic_metric.run import kfold_indices  # noqa: E402
from iot_experiments.studies.real.t1t2.metric_learning.run import select_iot  # noqa: E402
from iot_experiments.studies.real.t1t2.metric_learning.run import (  # noqa: E402
    REL_GRID,
    pair_hit,
)

FOLDS: int = 5
HOLDOUT_PAIRS: int = 400
_SCORING_DRAWS: tuple[int, ...] = (0, 1, 2)   # every cost is scored on the same three column shuffles
MIN_GAIN: float = 0.03

PROVENANCE: dict[str, object] = {
    "question": "does the relaxed/EOT fit carry signal, is the curve saturating, and does the OT structure "
                "beat the same rows-only reweighting?",
    "protocol": "verbatim from studies/real/t1t2/metric_learning: dinov2-pool4 PCA, geometry G2 from the "
                "held-out pool, ONE fixed 400-pair pool, labelled subjects disjoint from it",
    "null_rule": "a fit carries signal iff pool hit(true pairs) >= pool hit(permuted pairs) + 0.03; "
                 "COSINE (label-free) and UNIFORM (theta = theta_ref, never touches pi_hat) are references",
    "curve_rule": "a cell is SATURATED if hit(m_end) - hit(m) < 0.03 (one MIN_GAIN) and RISING otherwise",
    "baseline_rule": "row-softmax vs iot is the pre-registered comparison (iot wins a cell if it is >= "
                     "row-softmax + 0.03 on hit and R@1 agrees in direction); row-softmax is the "
                     "zero-temperature limit of its own Sinkhorn arm",
    "correction": "the zero-cost and near-degenerate cases show the assignment rule's own bias: a constant "
                  "cost gives hit 0.04 / 0.00 on a 50x50 pool, and a near-degenerate squared cost gives 1.0 "
                  "under min-argmin. Every cell records its own floor before a hit is read",
    "not_comparable_to": "the SynthRAD study (registered pairs, patient-level yardstick)",
}


# ------------------------------------------------------------------------------------------------ helpers


def _cell_scores(cost: Float[np.ndarray, "candidates candidates"]) -> dict[str, float]:
    """
    CN: 池上的读数（三种列打乱取均值）：`hit`（精确 EMD）与 `R@1`（自由逐行 argmin）。与 F2 `evaluate` 同配方。
    EN: pool readings averaged over the three column shuffles: `hit` (exact EMD) and `R@1` (free per-row
        argmin). The same recipe as F2's `evaluate`, minus the confidence score this study does not use.
    """
    values = np.asarray(cost, dtype=np.float64)
    hits = [matching_hit(values, seed=int(d)) for d in _SCORING_DRAWS]
    ranks = [rank_metrics(values)["R@1"] for _ in _SCORING_DRAWS]
    return {"hit": float(np.mean(hits)), "R@1": float(np.mean(ranks))}


def _squared_distance(a: Float[np.ndarray, "candidates k"], b: Float[np.ndarray, "candidates k"]
                      ) -> Float[np.ndarray, "candidates candidates"]:
    """CN: 未加权的对角族代价 `Sum_a (x_ia-y_ja)^2`。EN: the unweighted diagonal-family cost."""
    xa = np.asarray(a, dtype=np.float64)
    ya = np.asarray(b, dtype=np.float64)
    return ((xa ** 2).sum(axis=1)[:, None] - 2.0 * (xa @ ya.T) + (ya ** 2).sum(axis=1)[None, :])


def select_rel_and_fit(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, solver: str,
                       solve_limit_s: float, seed: int, max_iterations: int) -> dict[str, object]:
    """
    CN: `rel` 用与 `select_iot` 相同的 5 折规则选（只在标注对上，按验证折配对命中率），再在全标注对上重拟合。
        求解器用 `fit_full_lp`（**对角**族，`|theta| = k`），与 F2 的 `iot` 臂同一个引擎，因此 TRUE / PERM 两组
        只差"标注对不对"。失败时 `theta=None`，**绝不**回退成 uniform。

    EN: `rel` chosen by the same 5-fold rule as `select_iot` (labelled pairs only, validation-fold pairing
        hit), then refit on all labelled pairs. The engine is `fit_full_lp` (the **diagonal** family,
        `|theta| = k`), the same engine as F2's `iot` arm, so TRUE and PERM differ only in the labels.
        On failure `theta=None`; uniform is **never** substituted.
    """
    folds = kfold_indices(int(x.shape[0]), folds=FOLDS, seed=int(seed))
    best: tuple[float, float] | None = None
    for rel in list(REL_GRID) + [0.05]:
        scores: list[float] = []
        for train, drop in folds:
            mask = np.ones(int(x.shape[0]), dtype=bool)
            mask[train] = False
            mask[drop] = False
            if not mask.any():
                continue
            res = fit_full_lp(x[train], y[train], solver=str(solver), solve_limit_s=float(solve_limit_s),
                              box_relative=float(rel))
            if not str(res.status).startswith("ok"):
                scores.append(float("nan"))
                continue
            scores.append(pair_hit(diagonal_cost(x[mask], y[mask], res.theta)))
        score = float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")
        if np.isfinite(score) and (best is None or score > best[1]):
            best = (float(rel), score)
    rel_chosen = best[0] if best is not None else 0.25
    res = fit_full_lp(x, y, solver=str(solver), solve_limit_s=float(solve_limit_s),
                      box_relative=rel_chosen)
    theta = np.asarray(res.theta, dtype=np.float64) if str(res.status).startswith("ok") else None
    return {"rel": rel_chosen, "cv_score": None if best is None else best[1], "theta": theta,
            "status": str(res.status)}


def _device(name: str) -> str:
    """CN: `auto` → 有 CUDA 就用。EN: `auto` → CUDA when present."""
    return name if name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")


def _cost_t(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CN: 对角族代价矩阵（torch，可微）。EN: the diagonal-family cost matrix in torch, differentiable."""
    return (x * x).sum(1)[:, None] - 2.0 * (x @ y.T) + (y * y).sum(1)[None, :]


def fit_row_softmax_ce(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, tau: float,
                       epochs: int, lr: float, device: str
                       ) -> tuple[Float[np.ndarray, "k"], dict[str, float]]:
    """
    EN: **row-softmax cross-entropy** (the InfoNCE-style arm): logits `-C_ij / tau`, loss = mean_i
        CE(row_i, target i). Source marginal only, no transport. `epochs = 0` returns uniform `theta`
        (== cosine), so early stopping can never be worse than the reference.

    CN: **行 softmax 交叉熵**（InfoNCE 式）：logits `-C_ij/τ`，逐行对目标 i 做 CE。只有行边缘，没有输运。
        `epochs=0` 返回 uniform θ（= cosine），因此早停不可能比参照更差。
    """
    dev = _device(device)
    xm = torch.tensor(np.asarray(x, dtype=np.float32), device=dev)
    ym = torch.tensor(np.asarray(y, dtype=np.float32), device=dev)
    m, k = int(xm.shape[0]), int(xm.shape[1])
    theta = torch.full((k,), 1.0 / k, device=dev)
    if int(epochs) > 0:
        param = torch.nn.Parameter(theta.clone())
        opt = torch.optim.Adam([param], lr=float(lr))
        target = torch.arange(m, device=dev)
        for _ in range(int(epochs)):
            loss = torch.nn.functional.cross_entropy(-_cost_t(xm, ym) / float(tau), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                param.clamp_(min=1e-8)
                param.div_(param.sum())
        theta = param.detach()
    return theta.cpu().numpy().astype(np.float64), {"tau": float(tau), "epochs": float(epochs)}


def fit_sinkhorn_ce(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, tau: float, epochs: int,
                    lr: float, iterations: int, device: str
                    ) -> tuple[Float[np.ndarray, "k"], dict[str, float]]:
    """
    EN: **doubly-stochastic Sinkhorn cross-entropy**: `P = Sinkhorn(exp(-C/tau))` over a fixed number of
        differentiable iterations, loss = mean_i CE(P row i, i). Source **and** target marginals, so the
        competition between candidates lives in the plan, as in OT.

    CN: **双随机 Sinkhorn 交叉熵**：固定可微迭代数，损失为 `P` 各行对目标的 CE。**有**行与列两个边缘，
        因此候选之间的竞争在计划里，与 OT 一致。
    """
    dev = _device(device)
    xm = torch.tensor(np.asarray(x, dtype=np.float32), device=dev)
    ym = torch.tensor(np.asarray(y, dtype=np.float32), device=dev)
    m, k = int(xm.shape[0]), int(xm.shape[1])
    param = torch.nn.Parameter(torch.full((k,), 1.0 / k, device=dev))
    opt = torch.optim.Adam([param], lr=float(lr))
    target = torch.arange(m, device=dev)
    mass = torch.full((m,), 1.0 / m, device=dev)
    for _ in range(int(epochs)):
        cost = _cost_t(xm, ym)
        kernel = torch.exp(-(cost - cost.min()) / max(float(tau), 1e-12))
        u = torch.ones_like(mass)
        v = torch.ones_like(mass)
        for _it in range(int(iterations)):
            u = mass / (kernel @ v + 1e-30)
            v = mass / (kernel.T @ u + 1e-30)
        plan = u[:, None] * kernel * v[None, :]
        plan = plan / (plan.sum() + 1e-30)
        loss = torch.nn.functional.cross_entropy(torch.log(plan + 1e-30), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            param.clamp_(min=1e-8)
            param.div_(param.sum())
    return param.detach().cpu().numpy().astype(np.float64), {"tau": float(tau), "epochs": float(epochs)}


# ------------------------------------------------------------------------------------------------- config


@dataclasses.dataclass(frozen=True)
class SanityConfig(RunConfig):
    """CN: `--check` 选子研究；其余旋钮各子研究各取所需。EN: `--check` picks the sub-study."""

    study: str = "sanity"
    section: str = "sanity"
    check: Literal["null", "curve", "baselines"] = "null"
    feature: str = "dinov2-pool4"
    ks: tuple[int, ...] = (64, 128, 256)
    ms: tuple[int, ...] = (50, 100, 300)
    holdout_pairs: int = HOLDOUT_PAIRS
    holdout_draw: int = 0
    max_iterations: int = 40
    epochs_grid: tuple[int, ...] = (0, 10, 40)
    lr: float = 0.05
    tau_grid: tuple[float, ...] = (1.0, 0.1)
    sinkhorn_iterations: int = 8

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--check", default="null", choices=["null", "curve", "baselines"])
        parser.add_argument("--feature", default="dinov2-pool4", choices=["dinov2-pool4"])
        parser.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256])
        parser.add_argument("--ms", type=int, nargs="+", default=[50, 100, 300])
        parser.add_argument("--holdout-pairs", type=int, default=HOLDOUT_PAIRS)
        parser.add_argument("--holdout-draw", type=int, default=0)
        parser.add_argument("--max-iterations", type=int, default=40)
        parser.add_argument("--epochs-grid", type=int, nargs="+", default=[0, 10, 40])
        parser.add_argument("--lr", type=float, default=0.05)
        parser.add_argument("--tau-grid", type=float, nargs="+", default=[1.0, 0.1])
        parser.add_argument("--sinkhorn-iterations", type=int, default=8)


class SanityError(RuntimeError):
    """CN: 读数不可信（例如标注对与池重叠）。EN: the readings cannot be trusted (e.g. pool overlap)."""


# ------------------------------------------------------------------------------------------------ driver


def _load(config: SanityConfig, k: int, log: RunLog) -> dict[str, object]:
    """
    CN: 按 F2 的规则载入特征并切池（池与标注候选不相交）。
    EN: load features and cut the pool exactly as F2 does; the pool and the labelling candidates are
        disjoint by construction, and `check_null` re-asserts it per draw.
    """
    x, y = features(str(config.feature), dim=int(k))
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pool_rng = np.random.default_rng(10_000 + int(config.holdout_draw))
    pool_order = pool_rng.permutation(int(x.shape[0]))
    pool_subjects = np.sort(pool_order[: int(config.holdout_pairs)])
    remaining = np.sort(pool_order[int(config.holdout_pairs):])
    log.line(f"[K={k}] features {x.shape}; pool {len(pool_subjects)}; labelled candidates "
             f"{len(remaining)}")
    return {"x": x, "y": y, "pool_subjects": pool_subjects, "remaining": remaining, "k": int(k)}


def _geometry(data: dict[str, object], labelled: np.ndarray) -> tuple[Float[np.ndarray, "subjects k"],
                                                                    Float[np.ndarray, "subjects k"]]:
    """
    CN: 用 F2 的 G2（在池上拟合、对任意集合只变换不重拟合）把这个集合放到评分空间。
    EN: put a set in the scoring space with F2's G2 (fitted on the pool, transform-only for other sets).
    """
    from iot_experiments.harness.registry import apply_geometry, fit_geometry  # noqa: PLC0415

    x, y, pool = data["x"], data["y"], data["pool_subjects"]
    stats = fit_geometry(x[pool], y[pool], "G2")
    xu, yu = apply_geometry(x[labelled], y[labelled], stats)
    for arr in (xu, yu):
        norms = np.linalg.norm(np.asarray(arr, dtype=np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-8):
            raise SanityError(f"scoring space not L2-normalised: [{norms.min():.6f}, {norms.max():.6f}]")
    return np.asarray(xu, dtype=np.float64), np.asarray(yu, dtype=np.float64)


def _record(arm: str, k: int, m: int, draw: int, metrics: dict[str, float], config: SanityConfig,
            **extra: object) -> RunRecord:
    """CN: 一条读数。EN: one reading."""
    return RunRecord(study=str(config.study), geometry=f"G2|pool{int(config.holdout_pairs)}",
                     arm=str(arm), k=int(k), m=int(m), draw=int(draw), metrics=metrics,
                     config=config.to_dict(), extra=dict(extra))


def check_null(config: SanityConfig, log: RunLog, data_by_k: dict[int, dict[str, object]]
               ) -> list[RunRecord]:
    """
    EN: **A -- the permutation null.** For each `(K, m, draw)`: pool hit of `COSINE` and `UNIFORM`
        (references that never see the labelling) and of the fit on the TRUE pairs and on a PERMUTED pairing.
        The verdict is per cell: `TRUE - PERM >= 0.03` on every draw means the fit carries signal.

    CN: **A —— 置换零假设**：同一 (K, m) 上，真标注与打乱标注的池命中率比较。
    """
    rows: list[RunRecord] = []
    for k, data in data_by_k.items():
        pool = data["pool_subjects"]
        x_pool, y_pool = _geometry(data, pool)
        base = _cell_scores(diagonal_cost(x_pool, y_pool, np.ones(int(k)) / float(k)))
        log.line(f"[K={k}] cosine hit={base['hit']:.4f} R@1={base['R@1']:.4f}")
        for m in [int(v) for v in config.ms]:
            for draw in [int(d) for d in config.draws]:
                select = np.random.default_rng(draw).choice(int(data["remaining"].size), size=m,
                                                            replace=False)
                labelled = np.asarray(data["remaining"])[select]
                if np.intersect1d(labelled, pool).size:
                    raise SanityError("labelled and pool subjects overlap")
                x_fit, y_fit = _geometry(data, labelled)
                ref = _squared_distance(x_fit, y_fit)[np.arange(m), np.arange(m)]
                rows.append(_record("COSINE", k, m, draw, base, config, label_free=True))
                uniform_row = _reliable_cell(diagonal_cost(x_pool, y_pool, np.ones(int(k)) / float(k)))
                rows.append(_record("UNIFORM", k, m, draw, uniform_row, config, label_free=True))
                log.line(f"[K={k} m={m} d={draw}] UNIFORM hit={uniform_row['hit']:.4f} "
                         f"(rel-MSE on labelled pairs {float(np.mean(ref)):.4f})")

                perm = np.random.default_rng(10_000 + 7 * int(draw) + int(m)).permutation(m)
                for arm, y_used in (("TRUE", y_fit), ("PERM", y_fit[perm])):
                    fit = select_rel_and_fit(x_fit, y_used, solver=str(config.solver),
                                             solve_limit_s=float(config.solve_limit_s),
                                             seed=int(draw), max_iterations=int(config.max_iterations))
                    if fit["theta"] is None:
                        log.line(f"[K={k} m={m} d={draw}] {arm} FAILED {fit['status']}")
                        rows.append(_record(arm, k, m, draw, {}, config, excluded=True,
                                            status=str(fit["status"])))
                        continue
                    met = _reliable_cell(diagonal_cost(x_pool, y_pool, fit["theta"]))
                    rows.append(_record(arm, k, m, draw, met, config, rel=fit["rel"],
                                        cv_score=fit["cv_score"],
                                        theta_l1=float(np.abs(np.asarray(fit["theta"])).sum())))
                    log.line(f"[K={k} m={m} d={draw}] {arm:<4} hit={met['hit']:.4f} "
                             f"R@1={met['R@1']:.4f} rel={fit['rel']}")
    return rows


def _reliable_cell(cost: Float[np.ndarray, "candidates candidates"]) -> dict[str, float]:
    """
    CN: 一个单元的三个读数，外加该单元自己的**机会下限**（常数代价的 hit/R@1）。没有这一列，命中率无从解释。
    EN: a cell's readings plus that cell's own **chance floor** (the hit/R@1 of a constant cost). Without
        that column a hit rate cannot be interpreted: the assignment rule's own bias is the floor.
    """
    pool = int(np.asarray(cost).shape[0])
    floor = _cell_scores(np.zeros((pool, pool), dtype=np.float64))
    met = _cell_scores(cost)
    return {**met, "floor_hit": floor["hit"], "floor_R@1": floor["R@1"]}


def check_curve(config: SanityConfig, log: RunLog, data_by_k: dict[int, dict[str, object]]
                ) -> list[RunRecord]:
    """
    EN: **B -- the learning curve.** `m` over `--ms`; the labelled pool is `n_subjects - 400` subjects, so
        the run that uses this passes `--ms 50 100 300 600` with one draw. Records `COSINE` once per `K`
        and F2's `iot` arm (`select_iot`, dense `fit_full_lp`, the arm F2 actually published) per `m`;
        the analysis labels a cell SATURATED or RISING from the `m_end - m` increment.

    CN: **B —— 学习曲线**：把 F2 的 m 网格向上延伸（标注池只有 600 个受试者，故一个 draw）。
    """
    rows: list[RunRecord] = []
    for k, data in data_by_k.items():
        x_pool, y_pool = _geometry(data, data["pool_subjects"])
        base = _cell_scores(diagonal_cost(x_pool, y_pool, np.ones(int(k)) / float(k)))
        rows.append(_record("COSINE", k, min(config.ms), 0, base, config, label_free=True))
        log.line(f"[K={k}] cosine hit={base['hit']:.4f} R@1={base['R@1']:.4f}")
        for m in [int(v) for v in config.ms]:
            draw = int(config.draws[0])
            m_eff = int(min(m, data["remaining"].size))
            select = np.random.default_rng(draw).choice(int(data["remaining"].size), size=m_eff,
                                                        replace=False)
            x_fit, y_fit = _geometry(data, np.asarray(data["remaining"])[select])
            t0 = time.perf_counter()
            fit = select_iot(x_fit, y_fit, solver=str(config.solver),
                             solve_limit_s=float(config.solve_limit_s), seed=draw)
            if fit.status != "ok":
                log.line(f"[K={k} m={m}] iot FAILED {fit.status}")
                rows.append(_record("iot", k, m, draw, {}, config, excluded=True, status=str(fit.status)))
                continue
            met = _reliable_cell(fit.cost(x_pool, y_pool))
            rows.append(_record("iot", k, m, draw, met, config,
                                seconds=round(time.perf_counter() - t0, 2),
                                selected=fit.selected or {}, n_params=int(fit.n_params)))
            log.line(f"[K={k} m={m}] iot hit={met['hit']:.4f} R@1={met['R@1']:.4f} "
                     f"({100.0 * (met['hit'] - base['hit']):+.1f} pts vs cosine) "
                     f"{time.perf_counter() - t0:.0f}s sel={fit.selected}")
    return rows


def check_baselines(config: SanityConfig, log: RunLog, data_by_k: dict[int, dict[str, object]]
                    ) -> list[RunRecord]:
    """
    EN: **C -- the same objective under three losses.** `COSINE`; `rowsoftmax` and `sinkhorn` choose
        `(tau, epochs)` by 5-fold on the labelled pairs with `epochs = 0` (== cosine) a candidate; `iot`
        uses `select_iot` with its own 5-fold. All are scored on the same pool and the same three column
        shuffles, so the only difference between the arms is the **loss**.

    CN: **C —— 同一目标、三种损失**：行 softmax CE、双随机 Sinkhorn CE、精确 iOT（松弛后悔 LP）。
    """
    rows: list[RunRecord] = []
    for k, data in data_by_k.items():
        x_pool, y_pool = _geometry(data, data["pool_subjects"])
        base = _cell_scores(diagonal_cost(x_pool, y_pool, np.ones(int(k)) / float(k)))
        rows.append(_record("COSINE", k, min(config.ms), 0, base, config, label_free=True))
        log.line(f"[K={k}] cosine hit={base['hit']:.4f} R@1={base['R@1']:.4f}")
        for m in [int(v) for v in config.ms]:
            for draw in [int(d) for d in config.draws]:
                select = np.random.default_rng(draw).choice(int(data["remaining"].size), size=m,
                                                            replace=False)
                x_fit, y_fit = _geometry(data, np.asarray(data["remaining"])[select])
                folds = kfold_indices(m, folds=FOLDS, seed=draw)

                def cv(engine) -> tuple[float, float, int]:
                    """CN: 每个 (tau, epochs) 的验证折命中率；返回 (tau, epochs, score)。EN: grid CV."""
                    best = (float("nan"), 0.0, -1.0)
                    for tau in config.tau_grid:
                        for epochs in config.epochs_grid:
                            scores = [pair_hit(diagonal_cost(
                                x_fit[drop], y_fit[drop],
                                engine(x_fit[train], y_fit[train], tau=tau, epochs=epochs,
                                       lr=float(config.lr), device=str(config.device))[0]))
                                for train, drop in folds]
                            good = float(np.mean(scores))
                            if not np.isfinite(best[2]) or good > best[2]:
                                best = (float(tau), float(epochs), good)
                    return best

                def sinkh(*_a, tau: float, epochs: int, lr: float, device: str):
                    return fit_sinkhorn_ce(*_a, tau=tau, epochs=epochs, lr=lr,
                                           iterations=int(config.sinkhorn_iterations), device=device)

                for arm, engine in (("rowsoftmax", fit_row_softmax_ce), ("sinkhorn", sinkh)):
                    tau_c, epochs_c, cv_score = cv(engine)
                    theta, _diag = engine(x_fit, y_fit, tau=tau_c, epochs=epochs_c,
                                          lr=float(config.lr), device=str(config.device))
                    met = _reliable_cell(diagonal_cost(x_pool, y_pool, theta))
                    rows.append(_record(arm, k, m, draw, met, config, tau=tau_c,
                                        epochs=int(epochs_c), cv_score=cv_score))
                    log.line(f"[K={k} m={m} d={draw}] {arm:<10} hit={met['hit']:.4f} "
                             f"R@1={met['R@1']:.4f} (tau={tau_c:g} epochs={int(epochs_c)} "
                             f"cv={cv_score:.3f})")

                fit = select_iot(x_fit, y_fit, solver=str(config.solver),
                                 solve_limit_s=float(config.solve_limit_s), seed=draw)
                if fit.status != "ok":
                    rows.append(_record("iot", k, m, draw, {}, config, excluded=True,
                                        status=str(fit.status)))
                    continue
                met = _reliable_cell(fit.cost(x_pool, y_pool))
                rows.append(_record("iot", k, m, draw, met, config, selected=fit.selected or {}))
                log.line(f"[K={k} m={m} d={draw}] {'iot':<10} hit={met['hit']:.4f} "
                         f"R@1={met['R@1']:.4f} sel={fit.selected}")
    return rows


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(SanityConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = SanityConfig.from_namespace(args, parser)
    run_dir = config.run_dir / str(config.check)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB")
    if not pre.ok:
        log.line(f"preflight failures: {list(pre.failures)}")
        log.close()
        return 2
    log.line(f"check={config.check} ks={config.ks} ms={config.ms} draws={config.draws}")

    data_by_k = {int(k): _load(config, int(k), log) for k in config.ks}
    engines = {"null": check_null, "curve": check_curve, "baselines": check_baselines}
    rows = engines[str(config.check)](config, log, data_by_k)

    table = summarise(rows, group=("arm", "k", "m"), metrics=("hit", "R@1", "floor_hit"))
    log.line("\n" + format_table(table, ("arm", "k", "m", "n_draws", "hit_mean", "R@1_mean",
                                          "floor_hit_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
