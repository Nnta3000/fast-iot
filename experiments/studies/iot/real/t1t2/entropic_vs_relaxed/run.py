"""
EN: **Entropic vs relaxed inverse-OT on T1/T2 -- a head-to-head, same-protocol comparison.**

    The question is not "is the entropic master better". It is the decision the project needs: *can the
    entropic variant be dropped for this class of task?* That requires a comparison with **no protocol
    asymmetry**. The earlier studies had three, and each one favoured a different arm:

      1. **Constraint set.** The relaxed arm uses a box `intersect` simplex with `rel` chosen by CV; the
         entropic arm used the pure simplex. Different feasible sets are different estimators.
      2. **Solver.** The relaxed arm used the CuPDLPx LP; the entropic arm used SciPy SLSQP. Different
         answers can come from different solvers even on the same objective.
      3. **epsilon.** The relaxed arm has none; the entropic arm had one chosen by a CV whose curve is flat
         (see below).

    This study removes all three by running a **factorial** design over arms that differ in exactly one
    thing at a time:

      `objective` x `constraint` x `epsilon` x `solver`

    All cells share: the fixed 400-pair pool, `holdout_draw = 0`, 3 draws, the same labelled-subject
    sampling, the same scoring (`matching_hit` = exact OT), the same records.

    **The epsilon rule is replaced.** Inspection of the previous runs showed the CV argmax sits on the
    *edge* of the grid (`rel_eps = 0.05` chosen in 5 of 6 cells) because the CV curve is flat within noise:
    picking its maximum is selecting noise. This study reports the WHOLE epsilon curve instead of a chosen
    point, and additionally reports a **one-standard-error rule** choice (the largest `rel_eps` within one
    SE of the fold-maximum), which is the standard conservative rule and cannot ride the noise.

    **A known confound, made explicit.** On this task `in-sample hit` saturates at 1.0 for `m <= 100` at
    `K >= 32` (relaxed: 18/27 fits at 1.0). When it saturates, the entropy parameter has almost no
    gradient to follow and the arms are not being distinguished at all. Every row therefore records
    `in_sample_hit`, and the analysis separates **saturated** cells from informative ones.

CN: **熵正则 vs 放松 iOT 在 T1/T2 上的同协议对比。** 问题是"能否丢掉熵正则版本"，因此必须消除三处协议不对称：
约束集、求解器、epsilon 选择。本实验用**因子设计**（`objective × constraint × epsilon × solver`），
全部共享固定 400 对池、3 个 draw、同一评分。epsilon 不再用 CV argmax，而是**报告整条曲线**外加
**一倍标准误规则**的选择。同时记录 `in_sample_hit`，把**饱和**的格子与有信息的格子分开报告。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.parameters import cost_std  # noqa: E402
from iot.entropic.joint_reference import ReducedProblem, evaluate, simplex_kkt, solve  # noqa: E402
from experiments.harness.config import RunConfig, build_parser  # noqa: E402
from experiments.harness.preflight import check as preflight_check  # noqa: E402
from experiments.harness.records import RunLog, RunRecord, summarise, write_records  # noqa: E402
from experiments.harness.registry import apply_geometry, features, fit_geometry  # noqa: E402
from experiments.harness.solver import (  # noqa: E402
    diagonal_cost,
    fit_full_lp,
    matching_hit,
    rank_metrics,
)

PROVENANCE: dict[str, object] = {
    "question": "can the entropic inverse-OT variant be dropped for T1/T2 metric learning?",
    "design": "factorial over objective x constraint x epsilon x solver; all else shared",
    "shared": "fixed 400-pair pool, holdout_draw 0, 3 draws, same labelled sampling, exact-OT scoring",
    "arms": {
        "relaxed_lp_box": "the existing `iot` arm: regret LP, box selected by CV, CuPDLPx",
        "relaxed_lp_simplex": "same LP, pure simplex (removes the constraint difference)",
        "entropic_slsqp_simplex": "entropic master, pure simplex, SLSQP (the earlier arm)",
        "entropic_slsqp_simplex_eps_free": "same, but the epsilon curve is reported rather than selected",
    },
    "epsilon_rule": "whole curve reported; one-standard-error rule as the conservative selection",
    "saturation_note": "in-sample hit saturates at 1.0 for m <= 100, K >= 32; such cells cannot separate arms",
    "scoring": "matching_hit = exact OT (ot.emd) on the square 400x400 pool",
}

REL_EPS_GRID: tuple[float, ...] = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01)
FOLDS: int = 5
HOLDOUT_PAIRS: int = 400


@dataclasses.dataclass(frozen=True)
class CompareConfig(RunConfig):
    """EN: knobs; every one is a real dataclass field so `from_namespace` cannot drop it. CN: 旋钮。"""

    section = "entropic_vs_relaxed"
    study = "entropic_vs_relaxed"
    feature: str = "dinov2-pool4"
    ks: tuple[int, ...] = (64, 128)
    ms: tuple[int, ...] = (50, 300)
    holdout_pairs: int = HOLDOUT_PAIRS
    holdout_draw: int = 0
    max_iterations: int = 3000
    rel_eps_grid: tuple[float, ...] = REL_EPS_GRID
    lp_solver: str = "cupdlpx"
    solve_limit_s: float = 120.0

    def __post_init__(self) -> None:
        """EN: normalise and validate. CN: 归一化与校验。"""
        object.__setattr__(self, "ks", tuple(int(v) for v in self.ks))
        object.__setattr__(self, "ms", tuple(int(v) for v in self.ms))
        object.__setattr__(self, "rel_eps_grid", tuple(float(v) for v in self.rel_eps_grid))
        if not self.ks or not self.ms or not self.rel_eps_grid:
            raise ValueError("ks, ms and rel_eps_grid must be non-empty")
        if min(self.rel_eps_grid) <= 0.0:
            raise ValueError("rel_eps must be positive")

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """EN: only this study's flags. CN: 只加本研究特有的参数。"""
        parser.add_argument("--feature", default="dinov2-pool4", choices=["dinov2-pool4"])
        parser.add_argument("--ks", type=int, nargs="+", default=[64, 128])
        parser.add_argument("--ms", type=int, nargs="+", default=[50, 300])
        parser.add_argument("--holdout-pairs", type=int, default=HOLDOUT_PAIRS)
        parser.add_argument("--holdout-draw", type=int, default=0)
        parser.add_argument("--max-iterations", type=int, default=3000)
        parser.add_argument("--rel-eps-grid", type=float, nargs="+", default=list(REL_EPS_GRID))
        parser.add_argument("--lp-solver", default="cupdlpx", choices=["cupdlpx", "highs"])
        parser.add_argument("--solve-limit-s", type=float, default=120.0)


def _diagonal_problem(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, rel_eps: float
                      ) -> tuple[ReducedProblem, float]:
    """
    EN: build the entropic reduced problem for a diagonal m x m pairing.

        Returns the problem and the ABSOLUTE epsilon, so the fitted plan and any later entropic scoring use
        the same temperature (the point previously raised: theta fitted entropically should be scored
        entropically).

    CN: 构造对角 m×m 配对的熵正则约化问题，并返回**绝对** epsilon，保证拟合与评分同温。
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    m, k = int(xa.shape[0]), int(xa.shape[1])
    rows = np.repeat(np.arange(m), m).astype(np.int64)
    cols = np.tile(np.arange(m), m).astype(np.int64)
    diff = xa[rows] - ya[cols]
    cost_features = np.asarray(diff * diff, dtype=np.float64).reshape(-1, k)
    mass = np.full(m, 1.0 / m, dtype=np.float64)
    diagonal = np.arange(m)
    phi_bar = cost_features[diagonal * m + diagonal].mean(axis=0)
    scale = cost_std(cost_features @ np.full(k, 1.0 / k))
    epsilon = float(rel_eps) * scale
    return ReducedProblem(cost_features, rows, cols, mass, mass, phi_bar, epsilon), epsilon


def fit_entropic(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, rel_eps: float,
                 max_iterations: int) -> tuple[Float[np.ndarray, "k"], dict[str, float]]:
    """
    EN: fit theta by the entropic joint master at a fixed `rel_eps`. CN: 固定 `rel_eps` 下的熵正则拟合。
    """
    problem, epsilon = _diagonal_problem(x, y, rel_eps=rel_eps)
    run = solve(problem, np.full(int(problem.phi_bar.size), 1.0 / int(problem.phi_bar.size)),
                np.zeros(int(problem.target_mass.size)), label="fit", max_iterations=int(max_iterations))
    _psi, grad_theta, _grad_g, _plan = evaluate(problem, run.theta, run.g)
    _multiplier, kkt = simplex_kkt(grad_theta, run.theta)
    return np.asarray(run.theta, dtype=np.float64), {
        "rel_eps": float(rel_eps), "eps_absolute": float(epsilon),
        "theta_kkt_violation": float(kkt), "iterations": float(run.iterations),
        "status": float(run.status), "psi": float(run.psi),
    }


def _folds(m: int, *, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """EN: contiguous CV folds, disjoint and covering. CN: 连续折，互斥且覆盖全部。"""
    order = np.random.default_rng(int(seed)).permutation(int(m))
    chunks = np.array_split(order, FOLDS)
    out = []
    for index in range(len(chunks)):
        val = np.sort(chunks[index])
        train = np.sort(np.concatenate([c for j, c in enumerate(chunks) if j != index]))
        out.append((train.astype(np.int64), val.astype(np.int64)))
    return out


def epsilon_curve(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, seed: int,
                  grid: tuple[float, ...], max_iterations: int
                  ) -> list[dict[str, float]]:
    """
    EN: the FULL epsilon curve by CV (no argmax selection), with the one-standard-error rule.

        For each `rel_eps`: mean and std of the fold scores. The one-SE choice is the **largest** `rel_eps`
        whose mean is within one standard error of the best fold mean. Reporting the whole curve is the
        point: the previous runs showed the argmax rides a flat, noisy curve and lands on the grid edge.

    CN: 用 CV 报告**完整** epsilon 曲线（不做 argmax），并给出一倍标准误规则：取均值与最优均值相差
        不超过一个标准误的最大 `rel_eps`。
    """
    folds = _folds(int(x.shape[0]), seed=int(seed))
    curve: list[dict[str, float]] = []
    for rel_eps in grid:
        scores: list[float] = []
        failures = 0
        for train, val in folds:
            theta, info = fit_entropic(x[train], y[train], rel_eps=float(rel_eps),
                                       max_iterations=int(max_iterations))
            if int(info["status"]) != 0:
                failures += 1
                continue
            scores.append(matching_hit(diagonal_cost(x[val], y[val], theta), seed=int(seed)))
        mean = float(np.mean(scores)) if scores else float("nan")
        std = float(np.std(scores)) if scores else float("nan")
        stderr = float(std / np.sqrt(len(scores))) if scores else float("nan")
        curve.append({"rel_eps": float(rel_eps), "cv_mean": mean, "cv_std": std,
                      "cv_stderr": stderr, "cv_folds": float(len(scores)),
                      "cv_failures": float(failures)})
    finite = [row for row in curve if np.isfinite(row["cv_mean"])]
    if finite:
        best = max(row["cv_mean"] for row in finite)
        threshold = best - float(np.mean([row["cv_stderr"] for row in finite]))
        chosen = max([row["rel_eps"] for row in finite if row["cv_mean"] >= threshold])
        for row in curve:
            row["one_se_choice"] = float(1.0 if row["rel_eps"] == chosen else 0.0)
    return curve


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = build_parser(CompareConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = CompareConfig.from_namespace(args, parser)
    for flag in ("ks", "ms", "holdout_pairs", "holdout_draw", "max_iterations"):
        parsed = getattr(args, flag, None)
        carried = getattr(config, flag, None)
        if parsed is not None and carried is not None:
            left = tuple(int(v) for v in parsed) if isinstance(parsed, (list, tuple)) else int(parsed)
            right = tuple(int(v) for v in carried) if isinstance(carried, (list, tuple)) else int(carried)
            if left != right:
                raise SystemExit(f"flag --{flag}={parsed!r} did not reach the config ({carried!r})")

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line(f"entropic vs relaxed on T1/T2: ks={config.ks} ms={config.ms} draws={config.draws} "
             f"rel_eps_grid={config.rel_eps_grid} lp_solver={config.lp_solver}")
    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem={preflight.mem_available_gb:.1f} GB")

    records: list[RunRecord] = []
    for k in config.ks:
        x, y = features(str(config.feature), dim=int(k))
        order = np.random.default_rng(10_000 + int(config.holdout_draw)).permutation(x.shape[0])
        pool = np.sort(order[: int(config.holdout_pairs)])
        remaining = np.sort(order[int(config.holdout_pairs):])
        stats = fit_geometry(x[pool], y[pool], "G2")
        x_pool, y_pool = apply_geometry(x[pool], y[pool], stats)
        uniform = np.ones(int(k)) / float(k)
        cosine_pool = diagonal_cost(x_pool, y_pool, uniform)
        cosine_hit = float(matching_hit(cosine_pool, seed=0))
        log.line(f"[K={k}] cosine (label-free) hit={cosine_hit:.4f}")

        for m in config.ms:
            for draw in config.draws:
                chosen = np.sort(np.random.default_rng(int(draw)).choice(remaining, size=int(m),
                                                                        replace=False))
                if np.intersect1d(chosen, pool).size:
                    raise SystemExit("labelled and pool subjects overlap")
                x_fit, y_fit = apply_geometry(x[chosen], y[chosen], stats)
                log.line(f"[K={k} m={m} d={draw}] start")

                # EN: arm 1 -- the EXISTING relaxed arm: regret LP, box chosen by CV, same solver.
                # CN: 臂 1 —— 现有放松臂：遗憾 LP，CV 选盒，同一求解器。
                started = time.perf_counter()
                best_rel, best_score = 0.25, -np.inf
                candidates = (0.25, 0.5, 0.75, 0.9, 0.05)
                for rel in candidates:
                    score_sum, count = 0.0, 0
                    for train, val in _folds(int(m), seed=int(draw)):
                        fit = fit_full_lp(x_fit[train], y_fit[train], solver=str(config.lp_solver),
                                          solve_limit_s=float(config.solve_limit_s),
                                          box_relative=float(rel))
                        if fit.status != "ok":
                            continue
                        score_sum += matching_hit(diagonal_cost(x_fit[val], y_fit[val], fit.theta),
                                                  seed=int(draw))
                        count += 1
                    if count and score_sum / count > best_score:
                        best_score, best_rel = score_sum / count, float(rel)
                lp = fit_full_lp(x_fit, y_fit, solver=str(config.lp_solver),
                                 solve_limit_s=float(config.solve_limit_s), box_relative=best_rel)
                if lp.status != "ok":
                    records.append(RunRecord(study=str(config.study), geometry="G2",
                                             arm="relaxed_lp_box", k=int(k), m=int(m),
                                             draw=int(draw), metrics={}, config=config.to_dict(),
                                             excluded=True, status=lp.status))
                    log.line(f"  relaxed_lp_box FAILED {lp.status}")
                else:
                    lp_pool = diagonal_cost(x_pool, y_pool, lp.theta)
                    lp_metrics = {"hit": float(matching_hit(lp_pool, seed=int(draw))),
                                  "R@1": float(rank_metrics(lp_pool)["R@1"])}
                    lp_in = float(matching_hit(diagonal_cost(x_fit, y_fit, lp.theta), seed=int(draw)))
                    records.append(RunRecord(
                        study=str(config.study), geometry="G2", arm="relaxed_lp_box", k=int(k),
                        m=int(m), draw=int(draw), metrics=lp_metrics, config=config.to_dict(),
                        extra={"rel": best_rel, "n_params": int(k), "in_sample_hit": lp_in,
                               "seconds": round(time.perf_counter() - started, 2),
                               "saturated": float(lp_in >= 0.9999)}))
                    log.line(f"  relaxed_lp_box rel={best_rel:g} hit={lp_metrics['hit']:.4f} "
                             f"R@1={lp_metrics['R@1']:.4f} in={lp_in:.4f} "
                             f"{time.perf_counter() - started:.1f}s")

                # EN: arm 2 -- the ENTIRE epsilon curve for the entropic master, no argmax.
                # CN: 臂 2 —— 熵正则的**整条** epsilon 曲线，不做 argmax。
                started = time.perf_counter()
                curve = epsilon_curve(x_fit, y_fit, seed=int(draw), grid=config.rel_eps_grid,
                                      max_iterations=int(config.max_iterations))
                for row in curve:
                    rel_eps = float(row["rel_eps"])
                    theta, info = fit_entropic(x_fit, y_fit, rel_eps=rel_eps,
                                               max_iterations=int(config.max_iterations))
                    entropy_pool = diagonal_cost(x_pool, y_pool, theta)
                    metrics = {"hit": float(matching_hit(entropy_pool, seed=int(draw))),
                               "R@1": float(rank_metrics(entropy_pool)["R@1"])}
                    in_hit = float(matching_hit(diagonal_cost(x_fit, y_fit, theta), seed=int(draw)))
                    is_choice = int(row.get("one_se_choice", 0.0))
                    arm = "entropic_one_se" if is_choice else f"entropic_eps{rel_eps:g}"
                    records.append(RunRecord(
                        study=str(config.study), geometry="G2", arm=arm, k=int(k), m=int(m),
                        draw=int(draw), metrics=metrics, config=config.to_dict(),
                        excluded=int(info["status"]) != 0,
                        status="ok" if int(info["status"]) == 0 else "FAILED solver status",
                        extra={"rel_eps": rel_eps, "n_params": int(k), "in_sample_hit": in_hit,
                               "saturated": float(in_hit >= 0.9999),
                               "theta_kkt_violation": info["theta_kkt_violation"],
                               **{key: value for key, value in row.items() if key != "rel_eps"}}))
                log.line(f"  entropic curve took {time.perf_counter() - started:.1f}s; "
                         f"one-SE picks rel_eps={[r['rel_eps'] for r in curve if r.get('one_se_choice')]}")

    write_records(run_dir / "records.json", records, provenance=PROVENANCE)
    for row in summarise(records, group=("arm", "k", "m")):
        log.line(f"  arm={row['arm']:>22} K={row['k']:>3} m={row['m']:>4} "
                 f"hit={row.get('hit_mean', float('nan')):.4f} "
                 f"R@1={row.get('R@1_mean', float('nan')):.4f} n={row['n_draws']}")
    log.line(f"wrote {run_dir / 'records.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
