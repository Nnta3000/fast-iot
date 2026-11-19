"""
EN: **Planted shared-`theta` instance: does the relaxed arm pass its own sanity check?**

    A trip-count matrix is not an optimal-transport plan, so the NYC taxi instance cannot say whether the relaxed
    arm recovers a cost when its assumptions hold. Here they hold by construction: the observed plans are **exactly
    optimal** at one planted `theta*`, so `R(theta*) = 0` on every window.

    Protocol:
      1. draw `n` points, build the distance matrix and the feature family (unit-norm columns, the same gauge as the
         taxi study);
      2. plant `theta*` on the simplex and build `c* = sum_k theta*_k Phi_k`;
      3. draw `windows` pairs of marginals and take `P_l = EMD(a_l, b_l; c*)` — the observed plans;
      4. leave one window out: fit `theta` on the rest with the relaxed arm, evaluate on the held-out window;
      5. repeat for every fold.

    Readings on the held-out window, from `theta` and the held-out marginals only:

      `R_test`, `share = R/R_independent`, `plan_l1` against the planted held-out plan,
      `plan_cost_rel` = excess cost of the predicted plan under `c*`, `cost_rel_scaled` = the recovered cost up to
      the positive scale that leaves every OT plan unchanged, `theta_l1`, and the **diagonal mass** of both plans
      (a degenerate cost that only penalises staying in place makes any diagonal-free plan optimal, so the diagonal
      mass says whether the instance is vulnerable to that).

    Three knobs exist because the first version of this test had two weaknesses:

      * `--same-zone-weight` (default 0.0): the original planted cost put 0.30 on the same-zone column, so the
        planted plans avoided the diagonal — exactly the property a same-zone-only degenerate cost also explains.
        With 0.0 that escape is closed;
      * `--windows`: with more plans the admissible set of costs narrows, so `theta_l1` and `cost_rel_scaled` should
        fall if the method becomes reliable with data;
      * `--fresh-points`: every window gets its own point cloud (same `theta*`), so the held-out plan is for a
        **new** geometry — generalisation to new pairs, not just to new marginals of the same points.

    `--recovery-only` skips the folds and fits on all windows, which is the right shape for the "does theta converge
    with more windows" question.

    **Control (`--arm control`).** The same pipeline with plans generated at `c* + 0.5*s^3`, a direction the family
    does not contain; a reading that cannot separate the control from the planted case is not diagnostic.

CN: **植入共享 `theta` 实例：松弛臂能否通过自身的健全性检查？** 行程计数矩阵不是最优传输计划，故纽约出租车实例无法回答
    "假设成立时该臂能否恢复代价"。这里假设按构造成立：观测计划恰好最优于一个植入的 `theta*`。流程与读数见上。三个旋钮来自
    第一版测试的两个弱点：`--same-zone-weight`（默认 0.0；原版把 0.30 放在同 zone 列上，使植入计划避开对角线 —— 而这正是
    仅惩罚对角线的退化代价也能解释的性质）；`--windows`（计划越多，可行代价集越窄）；`--fresh-points`（每窗新点云，同一
    `theta*`，故留出计划属于**新几何** —— 迁移到新配对，而不只是同一点集的新 marginal）。`--recovery-only` 跳过折、在全窗上
    拟合，用于"theta 是否随窗口增多而收敛"。对照组用族外代价生成计划；若某读数无法区分两者，它就没有诊断力。

    Run / 用法:
        python -m iot_experiments.studies.synthetic.planted_shared_theta.run --same-zone-weight 0.0
        python -m iot_experiments.studies.synthetic.planted_shared_theta.run --fresh-points
        python -m iot_experiments.studies.synthetic.planted_shared_theta.run --recovery-only --windows 6 12 25 50
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import ot

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.relaxed.multi_plan import (  # noqa: E402
    distance_features,
    fit_shared_theta,
    regret_of,
    regret_share,
)
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)

PROVENANCE: dict[str, object] = {
    "question": "when observed plans are generated exactly by the relaxed arm's own model, does a theta fitted on "
                "some windows transfer to a held-out one?",
    "generator": "n points on [0,1)^2 (shared, or fresh per window with --fresh-points); s = d/max(d); features "
                 "{s, s^2, log(1+s), same-zone} scaled to unit Frobenius norm; theta* on the simplex; "
                 "P_l = exact EMD(a_l, b_l; c(theta*)); marginals ~ Dirichlet(1)+0.1",
    "estimator": "relaxed shared-theta arm: min sum_l [<c(theta), P_l> - OT_l(c)], Frank-Wolfe on the simplex, "
                 "exact OT forward model",
    "readings": "R_test; share = R/R_independent; plan_l1 and plan_cost_rel against the planted held-out plan; "
                "cost_rel_scaled = min_alpha ||alpha c_hat - c*||/||c*||; theta_l1; diagonal mass of both plans",
    "control": "the same pipeline with plans from c* plus a seeded random symmetric cost (no small family can "
               "represent it); readings that cannot separate the two arms are not diagnostic",
    "status": "this is a sanity check: it says the pipeline is not broken and can learn a cost when its assumptions "
              "hold. It does not say the method predicts real trip flows (the taxi study addresses that, and it "
              "does not).",
}


@dataclasses.dataclass(frozen=True)
class PlantedConfig(RunConfig):
    """EN: knobs. CN: 旋钮。"""

    study: str = "planted_shared_theta"
    section: str = "planted_shared_theta"
    arms: tuple[str, ...] = ("planted", "control")
    ns: tuple[int, ...] = (40,)
    widths: tuple[int, ...] = (4,)
    windows: int = 6
    iterations: int = 300
    seeds: tuple[int, ...] = (0, 1, 2)
    same_zone_weight: float = 0.0
    diagonal_offset: float = 0.0
    fresh_points: bool = False
    recovery_only: bool = False

    def __post_init__(self) -> None:
        """EN: normalise and validate. CN: 归一化与校验。"""
        object.__setattr__(self, "arms", tuple(str(v) for v in self.arms))
        object.__setattr__(self, "seeds", tuple(int(v) for v in self.seeds))
        object.__setattr__(self, "ns", tuple(int(v) for v in self.ns))
        object.__setattr__(self, "widths", tuple(int(v) for v in self.widths))
        object.__setattr__(self, "same_zone_weight", float(self.same_zone_weight))
        object.__setattr__(self, "diagonal_offset", float(self.diagonal_offset))
        if float(self.diagonal_offset) < 0.0:
            raise ValueError("diagonal_offset must be non-negative")
        if set(self.arms) - {"planted", "control"}:
            raise ValueError(f"unknown arms: {sorted(self.arms)}")
        if any(int(v) < 4 for v in self.ns) or int(self.windows) < 2:
            raise ValueError("every n >= 4 and windows >= 2 are required")
        if any(int(v) not in {4, 8} for v in self.widths):
            raise ValueError(f"widths must be 4 or 8; got {self.widths}")
        if not 0.0 <= float(self.same_zone_weight) <= 1.0:
            raise ValueError("same_zone_weight must be in [0, 1]")

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """EN: only this study's flags. CN: 只加本研究特有的参数。"""
        parser.add_argument("--arms", nargs="+", default=["planted", "control"])
        parser.add_argument("--ns", type=int, nargs="+", default=[40])
        parser.add_argument("--widths", type=int, nargs="+", default=[4])
        parser.add_argument("--windows", type=int, default=6)
        parser.add_argument("--iterations", type=int, default=300)
        parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        parser.add_argument("--same-zone-weight", type=float, default=0.0)
        parser.add_argument("--diagonal-offset", type=float, default=0.0)
        parser.add_argument("--fresh-points", action="store_true")
        parser.add_argument("--recovery-only", action="store_true")


def theta_star(same_zone_weight: float, width: int = 4) -> np.ndarray:
    """
    EN: the planted parameter, normalised to the simplex: `(0.10, 0.55, 0.35, w_same)` scaled to sum to 1.

        The default `w_same = 0` removes the same-zone component, because with it the planted plans avoid the
        diagonal and a degenerate cost that only penalises the diagonal explains them as well.

    CN: 植入参数，归一化到单纯形：`(0.10, 0.55, 0.35, w_same)` 除以和。默认 `w_same=0` 去掉同 zone 分量，因为带该分量时
        植入计划会避开对角线，而仅惩罚对角线的退化代价同样能解释它们。
    """
    # EN: padded to `width`: the extra profiles of the 8-column family carry zero weight, so the *cost function* is
    #     the same in both cells and the sweep measures only what extra parameters do.
    # CN: 补零到 `width`：8 列族中多出的剖面权重为 0，故两个格子的**代价函数**相同，扫描只测量多出的参数带来的影响。
    weights = np.array([0.10, 0.55, 0.35, float(same_zone_weight), 0.0, 0.0, 0.0, 0.0])[: int(width)]
    return weights / weights.sum()


def geometry(n: int, seed: int, window: int, *, fresh: bool, diagonal_offset: float = 0.0) -> np.ndarray:
    """
    EN: the distance matrix of one window: the same points for every window, or a fresh cloud per window.
    CN: 一个窗的距离矩阵：所有窗共用同一批点，或每窗新取一批。
    """
    rng = np.random.default_rng((seed, window) if fresh else (seed, 0))
    points = rng.random((n, 2))
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distance, 0.0)
    return distance + float(diagonal_offset)


def marginals(n: int, windows: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """EN: `windows` pickup/drop-off pairs with mass on every zone. CN: 每个 zone 都有质量的若干对 marginal。"""
    rng = np.random.default_rng((seed, 10_000))
    pairs = []
    for _ in range(int(windows)):
        source = rng.random(n) + 0.1
        target = rng.random(n) + 0.1
        pairs.append((source / source.sum(), target / target.sum()))
    return pairs


def observed_plans(distances: list[np.ndarray], theta: np.ndarray,
                   pairs: list[tuple[np.ndarray, np.ndarray]], *, outside: bool, width: int = 4,
                   seed: int = 0
                   ) -> list[np.ndarray]:
    """
    EN: `P_l = EMD(a_l, b_l; c)` on window `l`'s geometry. `outside=True` uses a different cost for the plans: a
        seeded **random symmetric** matrix added to `c`, which no small feature family can represent.

        The first version added `0.5*s^3`; that was valid for the 4-column family (where `s^3` is not a column) but
        **not** for the 8-column one, where `s^3` *is* a column — the "control" then became representable and its
        plans were reproduced exactly, which is why the control is now a random cost.

    CN: 在窗 `l` 的几何上取 `P_l = EMD(a_l,b_l;c)`；`outside=True` 时用不同的代价生成计划：在 `c` 上加一个带种子的
        **随机对称**矩阵，任何小特征族都无法表示它。第一版加的是 `0.5*s³`：对 4 列族有效（`s³` 不是列），但对 8 列族
        **无效**（`s³` 正是其中一列），"对照"因此变得可表示、计划被精确复现 —— 故改用随机代价。
    """
    plans = []
    for position, (distance, (source, target)) in enumerate(zip(distances, pairs, strict=True)):
        features = distance_features(distance, width=int(width))
        cost = np.tensordot(features, theta, axes=([2], [0]))
        if outside:
            rng = np.random.default_rng((int(seed), int(position), 4242))
            noise = rng.random(cost.shape)
            cost = cost + 0.5 * (noise + noise.T)
        plans.append(np.asarray(ot.emd(source, target, cost, numItermax=2_000_000), dtype=np.float64))
    return plans


def diagonal_mass(plan: np.ndarray) -> float:
    """EN: share of the plan's mass that stays in the same zone. CN: 停留在同一 zone 的质量占比。"""
    total = float(np.sum(plan))
    return float(np.trace(np.asarray(plan, dtype=np.float64)) / total) if total > 0 else float("nan")


def _evaluate_cell(
    n: int, width: int, seed: int, config: "PlantedConfig", planted: np.ndarray,
    *, log: RunLog,
) -> list[RunRecord]:
    """
    EN: one `(n, width, seed)` cell: build the geometry and plans for each arm, then score every fold (or fit on all
        windows when `--recovery-only`).

    CN: 一个 `(n, width, seed)` 格子：为每个臂构造几何与计划，然后对每折打分（`--recovery-only` 时在全窗上拟合）。
    """
    distances = [geometry(int(n), int(seed), w, fresh=bool(config.fresh_points),
                          diagonal_offset=float(config.diagonal_offset))
                 for w in range(int(config.windows))]
    pairs = marginals(int(n), int(config.windows), int(seed))
    records: list[RunRecord] = []
    label = f"n{n}_k{width}_seed{seed}"
    for arm in config.arms:
        plans = observed_plans(distances, planted, pairs, outside=(arm == "control"), width=int(width),
                               seed=int(seed))
        masses = [diagonal_mass(plan) for plan in plans]
        log.line(f"[{label} {arm:>7}] diagonal mass min/median/max "
                 f"{min(masses):.3f}/{float(np.median(masses)):.3f}/{max(masses):.3f}")
        if bool(config.recovery_only):
            started = time.perf_counter()
            fit = fit_shared_theta(np.stack(plans), tuple(distances), iterations=int(config.iterations),
                                   width=int(width))
            features = distance_features(distances[0], width=int(width))
            truth = np.tensordot(features, planted, axes=([2], [0]))
            cost_hat = np.tensordot(features, fit.theta, axes=([2], [0]))
            alpha = float(np.sum(cost_hat * truth) / np.sum(cost_hat ** 2))
            records.append(RunRecord(
                study=str(config.study), geometry=label, arm=str(arm), k=int(fit.n_features), m=int(n),
                draw=0,
                metrics={"train_objective": float(fit.objective),
                         "theta_l1": float(np.abs(fit.theta - planted).sum()),
                         "cost_rel_scaled": float(np.linalg.norm(alpha * cost_hat - truth)
                                                  / np.linalg.norm(truth))},
                config=config.to_dict(),
                extra={"theta": [float(v) for v in fit.theta], "theta_star": [float(v) for v in planted],
                       "seconds": round(time.perf_counter() - started, 3)}))
            continue
        for held_out in range(len(plans)):
            train = [k for k in range(len(plans)) if k != held_out]
            started = time.perf_counter()
            fit = fit_shared_theta(np.stack([plans[k] for k in train]),
                                   tuple(distances[k] for k in train),
                                   iterations=int(config.iterations), width=int(width))
            source, target = pairs[held_out]
            observed = plans[held_out]
            features = distance_features(distances[held_out], width=int(width))
            predicted = fit.predict(source, target, features=features)
            truth = np.tensordot(features, planted, axes=([2], [0]))
            cost_hat = np.tensordot(features, fit.theta, axes=([2], [0]))
            alpha = float(np.sum(cost_hat * truth) / np.sum(cost_hat ** 2))
            optimum = float(ot.emd2(source, target, truth, numItermax=2_000_000))
            share, _fraction = regret_share(fit.theta, features, observed)
            records.append(RunRecord(
                study=str(config.study), geometry=label, arm=str(arm), k=int(fit.n_features), m=int(n),
                draw=int(held_out),
                metrics={
                    "train_objective": float(fit.objective),
                    "regret_test": float(regret_of(fit.theta, features, observed)),
                    "regret_share": float(share),
                    "plan_l1": float(np.abs(predicted - observed).sum() / observed.sum()),
                    "plan_cost_rel": (float(np.sum(truth * predicted)) / optimum - 1.0
                                      if optimum > 0 else float("nan")),
                    "cost_rel_scaled": float(np.linalg.norm(alpha * cost_hat - truth)
                                             / np.linalg.norm(truth)),
                    "theta_l1": float(np.abs(fit.theta - planted).sum()),
                    "diagonal_mass_observed": diagonal_mass(observed),
                    "diagonal_mass_predicted": diagonal_mass(predicted),
                },
                config=config.to_dict(),
                extra={"theta": [float(v) for v in fit.theta], "theta_star": [float(v) for v in planted],
                       "seconds": round(time.perf_counter() - started, 3)}))
    return records


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = build_parser(PlantedConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = PlantedConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line(f"ns={config.ns} widths={config.widths} windows={config.windows} seeds={config.seeds} "
             f"arms={config.arms} iterations={config.iterations} "
             f"same_zone_weight={config.same_zone_weight:g} diagonal_offset={config.diagonal_offset:g} "
             f"fresh_points={config.fresh_points} recovery_only={config.recovery_only}")

    records: list[RunRecord] = []
    for n in config.ns:
        for width in config.widths:
            planted = theta_star(float(config.same_zone_weight), int(width))
            log.line(f"--- n={n} K={width}: theta*={np.round(planted, 4).tolist()}")
            for seed in config.seeds:
                records.extend(_evaluate_cell(int(n), int(width), int(seed), config, planted, log=log))

    write_records(run_dir / "records.json", records, provenance=PROVENANCE)
    group = ("arm", "geometry") if (bool(config.recovery_only) or len(config.ns) * len(config.widths) > 1) \
        else ("arm",)
    table = summarise(records, group=group,
                      metrics=("train_objective", "regret_share", "plan_l1", "plan_cost_rel",
                               "cost_rel_scaled", "theta_l1", "diagonal_mass_observed"))
    log.line("\n" + format_table(table, tuple(group) + ("n_draws", "train_objective_mean", "regret_share_mean",
                                                        "plan_l1_mean", "plan_cost_rel_mean",
                                                        "cost_rel_scaled_mean", "theta_l1_mean",
                                                        "diagonal_mass_observed_mean")))
    log.line(f"wrote {run_dir / 'records.json'} ({len(records)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
