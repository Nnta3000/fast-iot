"""
EN: **NYC taxi: leave-one-window-out**, with the *distance-imputation* and *distance-scope* knobs made explicit.

    Two decisions were implicit in the first version of this study and both change the numbers:

      * **Scope.** The pairwise distance is a mean over trips. `other_windows` (default) builds it from the training
        trips of the **five other** windows only, so a held-out window contributes nothing to the features it is
        scored on; `all_train` uses all six windows' training trips, which leaks the held-out window's *support*
        (its 20 % test trips never enter). Measured: 709-1767 pairs per fold differ in support between the two, and
        they carry 0.06-0.18 % of each held-out window's trip mass.
      * **Unsupported pairs.** 46 428 of 69 169 pairs (67 %) have no training trip. `median` (default) fills their
        distance with the median observed distance — one stated assumption. `mask` keeps the distance columns at 0
        there and adds one indicator column equal to 1 exactly on those pairs; since `theta >= 0` the mask can only
        *add* cost, so it can never make an unexplored pair cheaper than an observed one, and it is reported as a
        sensitivity rather than as the model.

    A third variant, `--restrict-edges`, forbids every pair with no training trip (the edge-set formulation, and the
    analogue of the graph restriction in Stuart & Wolfram). Held-out windows do contain trips on pairs outside the
    training support, so those cannot be carried by the predicted plan; the fraction of held-out mass on forbidden
    pairs is recorded per fold.

    Scoring is unchanged: `rmse`, `correlation`, `cpc` (the columns M-IOT's Table 1 reports) and the formulation's own
    `regret_ratio = R_test / R_uniform`, with `R` computed on the same feature matrix and edge set the fit used.

CN: **纽约出租车：留一时间窗**，把**距离插补**与**距离口径**两个此前隐含的决定显式化。`other_windows`（默认）只用其余
    五个窗的训练行程构造成对距离，故留出窗对它被评分的特征没有任何贡献；`all_train` 用全部六窗的训练行程，会泄漏留出窗的
    **支撑**（其 20% 测试行程从不进入）。实测：每折 709–1767 对在两口径下支撑不同，承载该窗 0.06–0.18% 的行程量。
    无训练行程的对 46 428/69 169（67%）：`median`（默认）用观测距离中位数补齐（一个明确假设）；`mask` 让距离列在这些对上为
    0 并追加一列指示（这些对上为 1），因 `theta>=0` 掩码只能**增加**代价、绝不能让未探索的对更便宜，故作为敏感性报告。
    第三种 `--restrict-edges` 禁止所有无训练行程的对（边集形式，Stuart & Wolfram 图限制的类比）；留出窗确实有落在训练支撑
    之外的行程，那些无法被预测计划承载，故每折记录被禁对上的留出质量比例。

    Run / 用法:
        python -m iot_experiments.studies.related.nyc_taxi.run --study related_nyc_taxi_imputation
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.external.metrics import align_to_marginals, flow_scores  # noqa: E402
from iot.external.methods.gravity import fit_gravity  # noqa: E402
from iot.relaxed.multi_plan import (  # noqa: E402
    fit_shared_theta,
    mixed_features,
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
from iot_experiments.studies.related.nyc_taxi.instance import Instance, WINDOWS  # noqa: E402

DEFAULT_INSTANCE = "data/nyc_taxi/instance_n263_k6.npz"

PROVENANCE: dict[str, object] = {
    "question": "does the relaxed (regret) formulation generalise across time windows on NYC taxi flows, and how "
                "much of the answer is the distance imputation rather than the method?",
    "protocol": "leave-one-window-out over six windows; fit on five; predict the sixth at its own marginals; average "
                "over the six folds",
    "estimator": "relaxed shared-theta arm (src/iot/relaxed/multi_plan): min sum_l [<c,pi_hat_l> - OT_l(c)] over "
                 "theta on the simplex, Frank-Wolfe, exact OT forward model",
    "distance_scope": "other_windows (default, leak-free: the held-out window contributes no trips to the distance) "
                      "vs all_train (sensitivity: all six windows' train trips, leaks the held-out support)",
    "unsupported_pairs": "median (default) fills the 46 428 pairs with no training trip at the median observed "
                         "distance; mask (sensitivity) keeps them at zero distance and adds an indicator column, "
                         "which with theta >= 0 can only raise their cost",
    "edge_restriction": "--restrict-edges forbids pairs with no training trip (large finite cost); the mass the "
                        "forward plan places there and the held-out mass on forbidden pairs are both recorded",
    "native_metric": "regret_ratio = R_test / R_uniform on the held-out window, R = <c,pi_hat> - OT(c) on the same "
                     "feature matrix and edge set as the fit",
    "flow_metrics": "RMSE, Pearson correlation, CPC (iot.external.metrics), the columns M-IOT's Table 1 reports",
    "published_reference": "NYC taxi, n=100, K=6 (Bao, Li & Sun 2026, Table 1): M-IOT 4.45 / 0.9989 / 0.9566; "
                           "naive average IOT 4.76 / 0.9988 / 0.9489; gravity 37.12 / 0.9076 / 0.6568",
    "instance": "n_zones=263 (official TLC ids; the paper's n=100 merge rule is not stated), 2016-01, 80/20 split "
                "within each window",
    "gauge": "theta on the simplex with non-negative features: it excludes the zero cost and, with the constant "
             "column off, the constant cost. The cost is identified only up to that choice, which is stated",
    "caveat": "absolute RMSE is not comparable to the published table (different n, month, preprocessing); "
              "correlation and CPC are the scale-free columns",
}


@dataclasses.dataclass(frozen=True)
class NycTaxiConfig(RunConfig):
    """EN: knobs. CN: 旋钮。"""

    study: str = "related_nyc_taxi"
    section: str = "related_nyc_taxi"
    instance: str = DEFAULT_INSTANCE
    arms: tuple[str, ...] = ("independence", "gravity", "shared_theta")
    family: str = "distance_power"
    imputations: tuple[str, ...] = ("median", "mask")
    scopes: tuple[str, ...] = ("other_windows", "all_train")
    geometries: tuple[str, ...] = ("real", "shuffled")
    restrict_edges: bool = False
    iterations: int = 200

    def __post_init__(self) -> None:
        """EN: normalise and validate. CN: 归一化与校验。"""
        for name in ("arms", "imputations", "scopes", "geometries"):
            object.__setattr__(self, name, tuple(str(v) for v in getattr(self, name)))
        if set(self.arms) - {"independence", "gravity", "shared_theta"}:
            raise ValueError(f"unknown arms: {sorted(self.arms)}")
        if set(self.imputations) - {"median", "mask"}:
            raise ValueError(f"imputations must be median/mask; got {self.imputations}")
        if set(self.scopes) - {"other_windows", "all_train"}:
            raise ValueError(f"scopes must be other_windows/all_train; got {self.scopes}")
        if set(self.geometries) - {"real", "shuffled"}:
            raise ValueError(f"geometries must be real/shuffled; got {self.geometries}")
        if not Path(self.instance).is_file():
            raise ValueError(f"instance not found: {self.instance} (build it with "
                             f"python -m iot_experiments.studies.related.nyc_taxi.instance)")

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """EN: only this study's flags. CN: 只加本研究特有的参数。"""
        parser.add_argument("--instance", default=DEFAULT_INSTANCE)
        parser.add_argument("--arms", nargs="+", default=["independence", "gravity", "shared_theta"])
        parser.add_argument("--family", default="distance_power")
        parser.add_argument("--imputations", nargs="+", default=["median", "mask"])
        parser.add_argument("--scopes", nargs="+", default=["other_windows", "all_train"])
        parser.add_argument("--geometries", nargs="+", default=["real", "shuffled"])
        parser.add_argument("--restrict-edges", action="store_true")
        parser.add_argument("--iterations", type=int, default=200)


def _distance(instance: Instance, scope: str, train: list[int]) -> np.ndarray:
    """
    EN: the pairwise mean distance for one fold under one scope; `NaN` where no training trip supports the pair.

        `other_windows` sums the stored per-window moments of the five training windows, `all_train` sums all six.
        Both leave the held-out window's **test** trips out by construction; they differ only in whether its *train*
        trips are used, which is exactly the leak being measured.

    CN: 某折在某口径下的成对平均距离；无训练行程支撑的对为 `NaN`。`other_windows` 汇总五个训练窗的逐窗矩，`all_train`
        汇总全部六窗。两者按构造都不含留出窗的**测试**行程；差别只在于是否使用它的**训练**行程，这正是被测量的泄漏。
    """
    if scope == "other_windows":
        sums = instance.window_distance_sum[train].sum(axis=0)
        counts = instance.window_distance_count[train].sum(axis=0)
    elif scope == "all_train":
        sums = instance.window_distance_sum.sum(axis=0)
        counts = instance.window_distance_count.sum(axis=0)
    else:
        raise ValueError(f"unknown scope {scope!r}")
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def _independence(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """EN: `IPF(ones)` — the no-information model. CN: `IPF(ones)`，无信息模型。"""
    return align_to_marginals(np.ones((source.size, target.size), dtype=np.float64), source, target)


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = build_parser(NycTaxiConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = NycTaxiConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    instance = Instance.load(Path(config.instance))
    log.line(f"instance {config.instance}: n_zones={instance.n_zones} windows={instance.n_windows}")
    log.line(f"arms={config.arms} scopes={config.scopes} imputations={config.imputations} "
             f"restrict_edges={config.restrict_edges}; forward model: exact OT")

    names = [name for name, *_ in WINDOWS]
    records: list[RunRecord] = []
    for held_out in range(instance.n_windows):
        train = [k for k in range(instance.n_windows) if k != held_out]
        train_flows = np.stack([instance.train_flows[k] for k in train])
        source = instance.test_source[held_out]
        target = instance.test_target[held_out]
        observed = instance.test_flows[held_out]
        predictions: dict[str, np.ndarray] = {}
        extras: dict[str, dict[str, object]] = {}
        if "independence" in config.arms:
            predictions["independence"] = _independence(source, target)
        for scope in config.scopes:
            distance_real = _distance(instance, scope, train)
            for geometry in config.geometries:
                if geometry == "real":
                    distance = distance_real
                else:
                    # EN: **the shuffle control.** Permute the distance VALUES among the pairs that have a training
                    #     trip, keeping the support (which pairs are observed) exactly as it is: the flow keeps its real
                    #     labels and the mask/imputation stay aligned with the data, so the only thing destroyed is the
                    #     assignment of distances to geography. If the held-out regret share does not move, the number
                    #     carries no geographic information. (Permuting the NaN pattern too would confound geography with
                    #     support, which is a property of the data rather than of the map.)
                    # CN: **置换对照。** 只在**有训练行程**的对之间置换距离**取值**，支撑（哪些对可观测）保持不变：流量保留
                    #     真实标号，掩码/插补仍与数据对齐，唯一被破坏的是"距离与地理的对应"。若留出 regret share 不动，
                    #     该数字不含地理信息。（连 NaN 模式一起置换会把地理与支撑混淆，而支撑是数据的性质、不是地图的。）
                    rng = np.random.default_rng(1000 + held_out)
                    observed_pairs = ~np.isnan(distance_real)
                    shuffled = np.full_like(distance_real, np.nan)
                    values = distance_real[observed_pairs]
                    shuffled[observed_pairs] = rng.permutation(values)
                    distance = shuffled
                unsupported = int(np.count_nonzero(np.isnan(distance)))
                if "gravity" in config.arms and geometry == "real":
                    # EN: gravity is a flow predictor, not a regret arm; it is fitted on the real geometry only.
                    # CN: 引力模型是流量预测器，不是 regret 臂；只在真实几何上拟合。
                    started = time.perf_counter()
                    fit = fit_gravity(instance.train_flows[train], distance)
                    arm = f"gravity_{scope}"
                    predictions[arm] = fit.predict(source, target, distance)
                    extras[arm] = {"alpha": fit.alpha, "beta": fit.beta, "gamma": fit.gamma,
                                   "distance_fill": fit.distance_fill,
                                   "unsupported_pairs": unsupported,
                                   "seconds": round(time.perf_counter() - started, 3)}
                if "shared_theta" in config.arms:
                    for imputation in config.imputations:
                        started = time.perf_counter()
                        fit_shared = fit_shared_theta(train_flows, distance, family=str(config.family),
                                                      iterations=int(config.iterations),
                                                      unsupported=str(imputation))
                        arm = f"shared_theta_{geometry}_{scope}_{imputation}"
                        predictions[arm] = fit_shared.predict(source, target)
                        features = mixed_features(distance, train_flows, family=str(config.family),
                                                  unsupported=str(imputation))
                        uniform_theta = np.full(features.shape[2], 1.0 / features.shape[2])
                        regret_test = regret_of(fit_shared.theta, features, observed)
                        regret_uniform = regret_of(uniform_theta, features, observed)
                        share, fraction = regret_share(fit_shared.theta, features, observed)
                        columns = ["s", "s^2", "log1p(s)", "same_zone"]
                        if str(imputation) == "mask":
                            columns.append("mask")
                        extras[arm] = {
                            "scope": scope, "geometry": geometry, "imputation": imputation,
                            "theta": [float(v) for v in fit_shared.theta],
                            "theta_columns": columns,
                            "selected_column": columns[int(np.argmax(fit_shared.theta))],
                            "train_objective": fit_shared.objective,
                            "regret_test": regret_test, "regret_uniform": regret_uniform,
                            "regret_ratio": regret_test / regret_uniform if regret_uniform > 0 else float("nan"),
                            "regret_share": share, "regret_fraction": fraction,
                            "unsupported_pairs": unsupported,
                            "seconds": round(time.perf_counter() - started, 3),
                        }
                        log.line(f"[{names[held_out]:>16}] {arm}: theta={np.round(fit_shared.theta, 3).tolist()} "
                                 f"({extras[arm]['selected_column']}) F_train={fit_shared.objective:.4e} "
                                 f"share={share:.4f} fraction={fraction:.4f} unsupported={unsupported}")
                    if bool(config.restrict_edges) and geometry == "real":
                        started = time.perf_counter()
                        fit_shared = fit_shared_theta(train_flows, distance, family=str(config.family),
                                                      iterations=int(config.iterations), unsupported="median",
                                                      restrict_edges=True)
                        arm = f"shared_theta_{geometry}_{scope}_edges"
                        predictions[arm] = fit_shared.predict(source, target)
                        allowed = ~np.isnan(distance)
                        features = mixed_features(distance, train_flows, family=str(config.family),
                                                  unsupported="median")
                        uniform_theta = np.full(features.shape[2], 1.0 / features.shape[2])
                        regret_test = regret_of(fit_shared.theta, features, observed, allowed=allowed)
                        regret_uniform = regret_of(uniform_theta, features, observed, allowed=allowed)
                        share, fraction = regret_share(fit_shared.theta, features, observed, allowed=allowed)
                        forbidden = ~allowed
                        extras[arm] = {
                            "scope": scope, "geometry": geometry, "imputation": "median",
                            "edges": "train_support",
                            "theta": [float(v) for v in fit_shared.theta],
                            "train_objective": fit_shared.objective,
                            "forbidden_forward_mass": fit_shared.forbidden_mass,
                            "test_mass_on_forbidden": float(observed[forbidden].sum()),
                            "test_mass_fraction_on_forbidden": (float(observed[forbidden].sum())
                                                                / float(observed.sum()) if observed.sum() else 0.0),
                            "regret_test": regret_test, "regret_uniform": regret_uniform,
                            "regret_ratio": regret_test / regret_uniform if regret_uniform > 0 else float("nan"),
                            "regret_share": share, "regret_fraction": fraction,
                            "seconds": round(time.perf_counter() - started, 3),
                        }
                        log.line(f"[{names[held_out]:>16}] {arm}: F_train={fit_shared.objective:.4e} "
                                 f"share={share:.4f} forbidden_mass={fit_shared.forbidden_mass:.3e}")
        for arm, prediction in predictions.items():
            scores = flow_scores(prediction, observed)
            metrics: dict[str, float] = scores.to_dict()
            if arm in extras and "regret_ratio" in extras[arm]:
                metrics["regret_ratio"] = float(extras[arm]["regret_ratio"])  # type: ignore[arg-type]
                metrics["regret_share"] = float(extras[arm]["regret_share"])  # type: ignore[arg-type]
                metrics["regret_fraction"] = float(extras[arm]["regret_fraction"])  # type: ignore[arg-type]
            records.append(RunRecord(
                study=str(config.study), geometry=f"holdout_{names[held_out]}", arm=str(arm),
                k=int(len(extras[arm]["theta"])) if arm in extras and "theta" in extras[arm] else 0,
                m=instance.n_zones, draw=held_out, metrics=metrics,
                config=config.to_dict(), extra=extras.get(arm, {})))

    write_records(run_dir / "records.json", records, provenance=PROVENANCE)
    table = summarise(records, group=("arm",), metrics=("rmse", "correlation", "cpc", "regret_ratio",
                                                         "regret_share", "regret_fraction"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "rmse_mean", "correlation_mean", "cpc_mean",
                                         "regret_ratio_mean", "regret_share_mean")))
    log.line("published reference (their n=100, their month): M-IOT 4.45/0.9989/0.9566, "
             "naive IOT 4.76/0.9988/0.9489, gravity 37.12/0.9076/0.6568")
    log.line(f"wrote {run_dir / 'records.json'} ({len(records)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
