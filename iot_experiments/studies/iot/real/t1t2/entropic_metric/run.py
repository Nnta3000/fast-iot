"""
EN: **Does the SMOOTH ENTORPIC joint master recover useful metric-learning weights on T1/T2?**

    This is NOT the `iot` arm of `metric_learning`. That arm is the **relaxed regret LP** over `m*m` edges
    with a box `intersect` simplex; it has no temperature at all. This study uses the **entropic** objective

        Psi(theta, f, g) = <phi_bar, theta> - a.f - b.g + eps * sum_e exp(sigma_e / eps),

    restricted to the **pure simplex** (`sum(theta) = 1`, `theta >= 0`, NO box and NO upper bound), and asks
    the same downstream question: does the cost it produces recover T1/T2 pairs?

    **Protocol.** Everything about data, geometry, splits, scoring and records is inherited unchanged from
    `metric_learning`:

      * features `dinov2-pool4` + PCA at dimension `K` (`harness/registry.features`);
      * geometry `G2`, statistics fitted on the **held-out pool** (`harness/registry.fit_geometry`);
      * ONE fixed held-out pool shared by every `m` (`--holdout-pairs`, `--holdout-draw`);
      * labelled subjects drawn from the remaining subjects, disjoint from the pool;
      * scoring by `matching_hit` / `rank_metrics` on the pool, exactly as the other arms.

    **Epsilon protocol (new, and specified before any run).** There is no temperature in the relaxed LP, so
    `eps` needs its own pre-registered rule. It is:

      * **scale**: `eps = rel_eps * cost_std(c(theta_uniform))`, with `cost_std` the **population** standard
        deviation of the diagonal cost at the uniform `theta`. `theta_ref` is fixed at uniform for the whole
        solve, so `eps` never drifts with `theta` (a defect we hit before);
      * **selection**: `rel_eps` is chosen by 5-fold CV on the **labelled pairs only**, over a fixed grid,
        exactly like `rel` for the relaxed arm;
      * **refit**: after selection, the master is refit on all labelled pairs at the chosen `rel_eps`;
      * **no pool contact**: the held-out pool is never used for selection.

    **Arms this study records.** `entropic` (this master, with the CV-selected `rel_eps`) and `cosine`
    (label-free reference, identical to the other study's cosine by construction). The relaxed `iot` numbers
    are NOT recomputed here; the existing `metric_learning` records are the reference and are quoted with
    their own config.

CN: **平滑熵正则联合 master 在 T1/T2 上能学到有用的度量权重吗？**
    注意：这**不是** `metric_learning` 的 `iot` 臂。那一个臂是**放松遗憾 LP**（盒∩单纯形，无温度）。
    本研究用**熵正则**目标，约束是**纯单纯形**（无盒、无上界）。数据、几何、切分、评分、记录全部沿用
    `metric_learning`。`eps` 的协议在本文件里**先于运行**写定：`eps = rel_eps * cost_std(c(uniform))`，
    `rel_eps` 用**只在标注对**上的 5 折 CV 选择，留出池不参与选择。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.parameters import cost_std  # noqa: E402
from iot.entropic.joint_reference import ReducedProblem, evaluate, simplex_kkt, solve  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.harness.registry import apply_geometry, features, fit_geometry  # noqa: E402
from iot_experiments.harness.solver import diagonal_cost, matching_hit, rank_metrics  # noqa: E402

PROVENANCE: dict[str, object] = {
    "objective": "entropic joint master, PURE SIMPLEX (no box, no upper bound)",
    "not_the_same_as": "the 'iot' arm of metric_learning, which is the relaxed regret LP with a box",
    "features": "DINOv2 vit_small_patch14_pool4 + PCA (basis on all T1+T2 images, label-free)",
    "geometry": "G2, statistics from the held-out pool",
    "pool": "ONE fixed held-out pool shared by every m (labelled subjects disjoint)",
    "epsilon_scale": "eps = rel_eps * cost_std(c(theta_uniform)); theta_ref fixed at uniform, so eps cannot drift",
    "epsilon_selection": "rel_eps chosen by 5-fold CV on the LABELLED pairs only; the pool never touches it",
    "solver": "iot.entropic.joint_reference (SciPy SLSQP on the reduced joint objective)",
    "chance_hit": "1 / holdout_pairs",
}

REL_EPS_GRID: tuple[float, ...] = (0.1, 0.25, 0.5)
FOLDS: int = 5
HOLDOUT_PAIRS: int = 400


class EntropicMetricError(RuntimeError):
    """EN: a precondition of the study failed. CN: 研究的前置条件失败。"""


def fit_entropic(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, rel_eps: float,
                 max_iterations: int) -> tuple[Float[np.ndarray, "k"], dict[str, float]]:
    """
    EN: fit theta on `m` labelled pairs by minimising the entropic `Psi` on the m x m diagonal plan.

        The source and target marginals are uniform over the same `m` subjects, matching the diagonal plan
        the relaxed arm uses. Returns `(theta, diagnostics)`.

    CN: 在 `m` 个标注对上最小化熵正则 `Psi`（m×m 对角方案，均匀边缘）。返回 `(theta, 诊断量)`。
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    m, k = int(xa.shape[0]), int(xa.shape[1])
    rows: Int[np.ndarray, "edges"] = np.repeat(np.arange(m), m)
    cols: Int[np.ndarray, "edges"] = np.tile(np.arange(m), m)
    # EN: diagonal features: phi_e = (x_i - y_j)^2 computed with the same factorised identity as
    #     `diagonal_cost`, so the cost the master sees is exactly the cost used for scoring.
    # CN: 对角特征，与 `diagonal_cost` 用同一分解式，保证 master 看到的代价就是评分用的代价。
    diff = xa[rows] - ya[cols]
    cost_features = np.asarray(diff * diff, dtype=np.float64).reshape(-1, k)
    mass = np.full(m, 1.0 / m, dtype=np.float64)
    # EN: the observed feature average under the diagonal plan: (1/m) sum_i phi_ii.
    diagonal = np.arange(m)
    phi_bar = cost_features[diagonal * m + diagonal].mean(axis=0)
    scale = cost_std(cost_features @ np.full(k, 1.0 / k))
    problem = ReducedProblem(cost_features, rows.astype(np.int64), cols.astype(np.int64),
                             mass, mass, phi_bar, float(rel_eps) * scale)
    run = solve(problem, np.full(k, 1.0 / k), np.zeros(m), label="fit",
                max_iterations=int(max_iterations))
    _psi, grad_theta, _grad_g, _plan = evaluate(problem, run.theta, run.g)
    _multiplier, kkt = simplex_kkt(grad_theta, run.theta)
    return np.asarray(run.theta, dtype=np.float64), {
        "rel_eps": float(rel_eps), "eps_absolute": float(rel_eps) * scale,
        "cost_std": float(scale), "psi": float(run.psi),
        "theta_kkt_violation": float(kkt), "theta_min": float(np.min(run.theta)),
        "theta_max": float(np.max(run.theta)),
        "status": float(run.status), "iterations": float(run.iterations),
    }


def kfold_indices(m: int, *, folds: int, seed: int) -> list[tuple[Int[np.ndarray, "train"],
                                                                 Int[np.ndarray, "val"]]]:
    """
    EN: contiguous index folds, as in `metric_learning`: validation and training are disjoint and cover all.
    CN: 与 `metric_learning` 相同的连续折：训练与验证不相交且覆盖全部。
    """
    order = np.random.default_rng(int(seed)).permutation(int(m))
    chunks = np.array_split(order, int(folds))
    out = []
    for index in range(len(chunks)):
        val = np.sort(chunks[index])
        train = np.sort(np.concatenate([c for j, c in enumerate(chunks) if j != index]))
        out.append((train.astype(np.int64), val.astype(np.int64)))
    return out


def select_rel_eps(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, seed: int,
                   max_iterations: int, grid: tuple[float, ...] = REL_EPS_GRID
                   ) -> tuple[float, dict[str, float]]:
    """
    EN: choose `rel_eps` by 5-fold CV **on the labelled pairs only**.

        For each candidate, solve on the training folds and score the held-out fold by `matching_hit` on
        the diagonal cost. Ties go to the SMALLER `rel_eps` (the sharper, more deterministic problem), which
        is fixed here before any run so the rule cannot be tuned to the result afterwards.

    CN: `rel_eps` 只用**标注对**做 5 折 CV 选择；同分取更小的 `rel_eps`（更尖锐）。此规则先行写定。
    """
    folds = kfold_indices(int(x.shape[0]), folds=FOLDS, seed=int(seed))
    scores: dict[float, float] = {}
    # EN: record the fold iteration counts per candidate -- a step explosion at small eps shows here first.
    # CN: 记录每个候选的折内迭代数；小 eps 的步数爆炸会最先在这里看出。
    fold_iterations: dict[float, float] = {}
    fold_failures: dict[float, float] = {}
    for rel_eps in grid:
        fold_scores: list[float] = []
        counts: list[float] = []
        failures = 0
        for train, val in folds:
            theta, info = fit_entropic(x[train], y[train], rel_eps=float(rel_eps),
                                       max_iterations=int(max_iterations))
            counts.append(float(info["iterations"]))
            if int(info["status"]) != 0:
                failures += 1
                fold_scores.append(float("nan"))
                continue
            fold_scores.append(matching_hit(diagonal_cost(x[val], y[val], theta), seed=int(seed)))
        finite = [v for v in fold_scores if np.isfinite(v)]
        fold_iterations[float(rel_eps)] = float(np.mean(counts)) if counts else float("nan")
        fold_failures[float(rel_eps)] = float(failures)
        if finite:
            scores[float(rel_eps)] = float(np.mean(finite))
    if not scores:
        raise EntropicMetricError("every rel_eps candidate failed on every fold")
    best = max(scores.values())
    chosen = min([r for r, v in scores.items() if v >= best - 1e-12])
    diagnostics: dict[str, float] = {"cv_hit": scores[float(chosen)]}
    for rel_eps_value in grid:
        tag = f"{rel_eps_value:g}"
        diagnostics[f"cv_hit_rel{tag}"] = scores.get(float(rel_eps_value), float("nan"))
        diagnostics[f"cv_fold_iters_rel{tag}"] = fold_iterations.get(float(rel_eps_value),
                                                                     float("nan"))
        diagnostics[f"cv_fold_failures_rel{tag}"] = fold_failures.get(float(rel_eps_value),
                                                                      float("nan"))
    return float(chosen), diagnostics


@dataclasses.dataclass(frozen=True)
class EntropicMetricConfig(RunConfig):
    """
    EN: knobs for the entropic metric-learning arm.

        **These MUST be real dataclass fields.** `RunConfig.from_namespace` builds the config from
        `dataclasses.fields(cls)`, so a flag that only exists in `add_study_arguments` is parsed and then
        **silently discarded**. That happened here: `--ks 64 128 256` produced a run at K = 16, 32 with no
        error. `main()` now asserts the parsed values reached the config.
    CN: **必须是真正的 dataclass 字段。** `from_namespace` 按 `dataclasses.fields` 构造，只在
        `add_study_arguments` 里声明的参数会被解析后**静默丢弃**（确实发生过：`--ks 64 128 256` 跑成了
        K=16/32）。`main()` 现在会断言解析值已进入配置。
    """

    section = "entropic_metric"
    study = "entropic_metric"
    feature: str = "dinov2-pool4"
    ks: tuple[int, ...] = (16, 32)
    ms: tuple[int, ...] = (50, 100)
    holdout_pairs: int = HOLDOUT_PAIRS
    holdout_draw: int = 0
    max_iterations: int = 500
    rel_eps_grid: tuple[float, ...] = REL_EPS_GRID

    def __post_init__(self) -> None:
        """EN: normalise tuples and validate. CN: 归一化元组并校验。"""
        object.__setattr__(self, "ks", tuple(int(v) for v in self.ks))
        object.__setattr__(self, "ms", tuple(int(v) for v in self.ms))
        object.__setattr__(self, "rel_eps_grid", tuple(float(v) for v in self.rel_eps_grid))
        if not self.ks or not self.ms:
            raise ValueError("ks and ms must be non-empty")
        if not self.rel_eps_grid:
            raise ValueError("rel_eps_grid must be non-empty")
        if int(self.holdout_pairs) < 1 or int(self.max_iterations) < 1:
            raise ValueError("holdout_pairs and max_iterations must be positive")

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """EN: only this study's flags. CN: 只加本研究特有的参数。"""
        parser.add_argument("--feature", default="dinov2-pool4", choices=["dinov2-pool4"])
        parser.add_argument("--ks", type=int, nargs="+", default=[16, 32])
        parser.add_argument("--ms", type=int, nargs="+", default=[50, 100])
        parser.add_argument("--holdout-pairs", type=int, default=HOLDOUT_PAIRS)
        parser.add_argument("--holdout-draw", type=int, default=0)
        parser.add_argument("--max-iterations", type=int, default=500)
        parser.add_argument("--rel-eps-grid", type=float, nargs="+", default=list(REL_EPS_GRID))


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = build_parser(EntropicMetricConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = EntropicMetricConfig.from_namespace(args, parser)
    # EN: **guard against silently dropped flags.** `from_namespace` only sees dataclass fields; if a
    #     study flag is added to `add_study_arguments` but not to the dataclass, the CLI value is parsed
    #     and discarded. Compare the two and refuse to run rather than produce a mislabelled result.
    # CN: **防止参数被静默丢弃**：若 CLI 值与配置不一致，直接拒绝运行，避免产出标签错误的记录。
    for flag in ("ks", "ms", "holdout_pairs", "holdout_draw", "max_iterations"):
        parsed = getattr(args, flag, None)
        carried = getattr(config, flag, None)
        if parsed is not None and carried is not None:
            left = tuple(int(v) for v in parsed) if isinstance(parsed, (list, tuple)) else int(parsed)
            right = tuple(int(v) for v in carried) if isinstance(carried, (list, tuple)) else int(carried)
            if left != right:
                raise SystemExit(
                    f"flag --{flag}={parsed!r} did not reach the config ({carried!r}); "
                    f"declare it as a dataclass field on {type(config).__name__}")
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line(f"entropic metric learning on T1/T2: ks={config.ks} ms={config.ms} draws={config.draws} "
             f"rel_eps_grid={config.rel_eps_grid}")
    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem={preflight.mem_available_gb:.1f} GB")

    records: list[RunRecord] = []
    for k in config.ks:
        x, y = features(str(config.feature), dim=int(k))
        n_subjects = int(x.shape[0])
        pool_rng = np.random.default_rng(10_000 + int(config.holdout_draw))
        pool_order = pool_rng.permutation(n_subjects)
        pool_subjects = np.sort(pool_order[: int(config.holdout_pairs)])
        remaining = np.sort(pool_order[int(config.holdout_pairs):])
        stats = fit_geometry(x[pool_subjects], y[pool_subjects], "G2")
        x_pool, y_pool = apply_geometry(x[pool_subjects], y[pool_subjects], stats)
        uniform = np.ones(int(k)) / float(k)
        cosine_pool = diagonal_cost(x_pool, y_pool, uniform)

        for m in config.ms:
            for draw in config.draws:
                label_rng = np.random.default_rng(int(draw))
                labelled = np.sort(label_rng.choice(remaining, size=int(m), replace=False))
                if np.intersect1d(labelled, pool_subjects).size:
                    raise EntropicMetricError("labelled and pool subjects overlap")
                x_fit, y_fit = apply_geometry(x[labelled], y[labelled], stats)
                log.line(f"[K={k} m={m} d={draw}] start "
                         f"(eps grid {config.rel_eps_grid})")
                started = time.perf_counter()
                try:
                    rel_eps, sel = select_rel_eps(x_fit, y_fit, seed=int(draw),
                                                  max_iterations=int(config.max_iterations),
                                                  grid=tuple(float(v) for v in config.rel_eps_grid))
                    theta, info = fit_entropic(x_fit, y_fit, rel_eps=rel_eps,
                                               max_iterations=int(config.max_iterations))
                except EntropicMetricError as error:
                    records.append(RunRecord(study=str(config.study), geometry="G2", arm="entropic",
                                             k=int(k), m=int(m), draw=int(draw), metrics={},
                                             config=config.to_dict(), excluded=True,
                                             status=f"FAILED {error}"))
                    log.line(f"[K={k} m={m} d={draw}] entropic FAILED {error}")
                    continue
                seconds = time.perf_counter() - started
                cost_pool = diagonal_cost(x_pool, y_pool, theta)
                metrics = {"hit": float(matching_hit(cost_pool, seed=int(draw))),
                           "R@1": float(rank_metrics(cost_pool)["R@1"])}
                in_sample = float(matching_hit(diagonal_cost(x_fit, y_fit, theta), seed=int(draw)))
                excluded = int(info["status"]) != 0
                records.append(RunRecord(
                    study=str(config.study), geometry="G2", arm="entropic", k=int(k), m=int(m),
                    draw=int(draw), metrics=metrics, config=config.to_dict(), excluded=excluded,
                    status="ok" if not excluded else "FAILED solver status",
                    extra={"n_params": int(k), "seconds": round(seconds, 2),
                           "in_sample_hit": in_sample, "rel_eps": rel_eps, **sel, **info}))
                log.line(f"[K={k} m={m} d={draw}] entropic rel_eps={rel_eps:g} "
                         f"hit={metrics['hit']:.4f} R@1={metrics['R@1']:.4f} "
                         f"in={in_sample:.4f} kkt={info['theta_kkt_violation']:.1e} "
                         f"{seconds:.1f}s")

                base = {"hit": float(matching_hit(cosine_pool, seed=int(draw))),
                        "R@1": float(rank_metrics(cosine_pool)["R@1"])}
                records.append(RunRecord(study=str(config.study), geometry="G2", arm="cosine",
                                         k=int(k), m=int(m), draw=int(draw), metrics=base,
                                         config=config.to_dict(), extra={"n_params": 0}))
                log.line(f"[K={k} m={m} d={draw}] cosine   hit={base['hit']:.4f} R@1={base['R@1']:.4f}")

    # EN: cosine is label-free, so for a given K it must be identical across m and draw. Any spread means
    #     the pool or the geometry was contaminated -- diagnose before reading the table.
    # CN: cosine 无标签，同一 K 下必须与 m/draw 无关；有波动说明池或几何被污染。
    for k in config.ks:
        hits = [r.metrics["hit"] for r in records if r.arm == "cosine" and r.k == k]
        if hits and (max(hits) - min(hits)) > 1e-12:
            raise EntropicMetricError(f"cosine varied at K={k}: {min(hits):.6f}..{max(hits):.6f}")

    write_records(run_dir / "records.json", records, provenance=PROVENANCE)
    rows = [r.to_dict() for r in records if not r.excluded]
    log.line(format_table(rows, ["arm", "k", "m", "draw"]))
    # EN: `summarise` returns one dict per group with `hit_mean` / `hit_std` / `n_draws`.
    # CN: `summarise` 每组返回一个 dict，含 `hit_mean`/`hit_std`/`n_draws`。
    for row in summarise(records, group=("arm", "k", "m")):
        log.line(f"  arm={row['arm']:>9} K={row['k']:>3} m={row['m']:>4} "
                 f"hit={row.get('hit_mean', float('nan')):.4f} "
                 f"R@1={row.get('R@1_mean', float('nan')):.4f} "
                 f"n={row['n_draws']}")
    log.line(f"wrote {run_dir / 'records.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
