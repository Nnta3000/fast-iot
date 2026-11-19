"""
EN: **NYC taxi: leave-one-window-out** — the comparison table for the construct in `instance.py`, with the gravity
    baseline from `src/iot/external/methods` and our **relaxed** shared-`theta` arm from `src/iot/relaxed/multi_plan`.

    Protocol, one fold per held-out time window:

      1. fit on the flows of the other five windows;
      2. predict the held-out window at its own marginals;
      3. score two ways:
           `rmse`, `correlation`, `cpc` — the flow metrics M-IOT's Table 1 reports, so the arms are comparable to
             each other and to the published regime;
           `regret` and `regret_ratio = R_test / R_uniform` — the relaxed formulation's own quantity: how far the
             held-out flow is from being optimal under the cost fitted on the other windows, relative to the cost
             at uniform `theta`. This is the arm's native generalisation measure and does not depend on a decoder.
      4. average over the six folds.

    The forward model is exact OT throughout: a predicted plan has at most `m + n - 1` positive entries, which is a
    real restriction against a flow that occupies thousands of cells. It is reported as it is; the formulation's own
    question is the regret column, not the flow metrics.

CN: **纽约出租车：留一时间窗** —— `instance.py` 构造实例上的比较表，基线为 `src/iot/external/methods` 中的引力模型，我们
    的**松弛**共享 `theta` 臂来自 `src/iot/relaxed/multi_plan`。协议（每个留出窗一个折）：在其余五窗上拟合；在留出窗自身
    marginal 下预测；两种打分 —— `rmse`/`correlation`/`cpc`（M-IOT 表 1 的流量指标，使各臂之间及其与已发表区间可比），
    以及 `regret` 与 `regret_ratio = R_test/R_uniform`（松弛形式自身的量：留出流量在其余窗拟合出的代价下离最优有多远，
    相对均匀 `theta` 归一；这是该臂的原生泛化度量，不依赖解码器）；六折平均。前向模型始终是精确 OT：预测计划最多
    `m+n-1` 个正元素，对占据数千格子的流量是真实限制，如实报告；该形式自身的问题是 regret 那一列。

    Run / 用法:
        python -m iot_experiments.studies.related.nyc_taxi.run --instance data/nyc_taxi/instance_n263_k6.npz
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
from iot.relaxed.multi_plan import fit_shared_theta, mixed_features, regret_of  # noqa: E402
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
                "does the cost it fits score on a flow metric against a gravity baseline?",
    "protocol": "leave-one-window-out over six windows; fit on five; predict the sixth at its own marginals; "
                "average over the six folds",
    "estimator": "relaxed shared-theta arm (src/iot/relaxed/multi_plan): min sum_l [<c,pi_hat_l> - OT_l(c)] over "
                 "theta on the simplex, Frank-Wolfe, exact OT forward model",
    "native_metric": "regret_ratio = R_test / R_uniform on the held-out window, R = <c,pi_hat> - OT(c); 0 would "
                     "mean the held-out flow is optimal under the fitted cost, 1 means no better than uniform theta",
    "flow_metrics": "RMSE, Pearson correlation and CPC (iot.external.metrics), the columns M-IOT's Table 1 reports",
    "published_reference": "NYC taxi, n=100, K=6 (Bao, Li & Sun 2026, Table 1): M-IOT 4.45 / 0.9989 / 0.9566; "
                           "naive average IOT 4.76 / 0.9988 / 0.9489; gravity 37.12 / 0.9076 / 0.6568",
    "instance": "n_zones=263 (official TLC ids; the paper's n=100 merge rule is not stated), 2016-01, 80/20 split "
                "within each window, distance = mean reported trip_distance per pair over TRAIN trips",
    "gauge": "theta on the simplex, features non-negative: it excludes the zero cost and fixes one linear functional. "
             "The cost is therefore identified only up to that choice, which is stated rather than assumed",
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
    families: tuple[str, ...] = ("distance_power", "distance_lowrank")
    rank: int = 4
    iterations: int = 200

    def __post_init__(self) -> None:
        """EN: normalise and validate. CN: 归一化与校验。"""
        object.__setattr__(self, "arms", tuple(str(v) for v in self.arms))
        object.__setattr__(self, "families", tuple(str(v) for v in self.families))
        allowed = {"independence", "gravity", "shared_theta"}
        unknown = set(self.arms) - allowed
        if unknown:
            raise ValueError(f"unknown arms: {sorted(unknown)}; known: {sorted(allowed)}")
        if not Path(self.instance).is_file():
            raise ValueError(f"instance not found: {self.instance} (build it with "
                             f"python -m iot_experiments.studies.related.nyc_taxi.instance)")

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """EN: only this study's flags. CN: 只加本研究特有的参数。"""
        parser.add_argument("--instance", default=DEFAULT_INSTANCE)
        parser.add_argument("--arms", nargs="+", default=["independence", "gravity", "shared_theta"])
        parser.add_argument("--families", nargs="+", default=["distance_power", "distance_lowrank"])
        parser.add_argument("--rank", type=int, default=4)
        parser.add_argument("--iterations", type=int, default=200)


def _independence(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """EN: `IPF(ones)` — the no-information model. CN: `IPF(ones)`，无信息模型。"""
    ones = np.ones((source.size, target.size), dtype=np.float64)
    return align_to_marginals(ones, source, target)


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
    log.line(f"arms={config.arms} folds={instance.n_windows} (leave-one-window-out); forward model: exact OT")

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
        if "gravity" in config.arms:
            started = time.perf_counter()
            fit = fit_gravity(instance.train_flows[train], instance.distance)
            predictions["gravity"] = fit.predict(source, target, instance.distance)
            extras["gravity"] = {"alpha": fit.alpha, "beta": fit.beta, "gamma": fit.gamma,
                                 "seconds": round(time.perf_counter() - started, 3)}
            log.line(f"[{names[held_out]:>16}] gravity fit: alpha={fit.alpha:.3f} beta={fit.beta:.3f} "
                     f"gamma={fit.gamma:.3f}")
        if "shared_theta" in config.arms:
            for family in config.families:
                started = time.perf_counter()
                fit_shared = fit_shared_theta(train_flows, instance.distance, family=str(family),
                                              rank=int(config.rank), iterations=int(config.iterations))
                arm = f"shared_theta_{family}"
                predictions[arm] = fit_shared.predict(source, target)
                # EN: the formulation's own held-out reading, with the SAME features the fit used (for the
                #     low-rank family those depend on this fold's training flows, so they are rebuilt here).
                # CN: 该形式自身的留出读数，使用与拟合相同的特征（低秩族的特征依赖本折训练流量，故在此重建）。
                features = mixed_features(instance.distance, train_flows, family=str(family),
                                          rank=int(config.rank))
                uniform_theta = np.full(features.shape[2], 1.0 / features.shape[2])
                regret_test = regret_of(fit_shared.theta, features, observed)
                regret_uniform = regret_of(uniform_theta, features, observed)
                extras[arm] = {
                    "family": str(family),
                    "theta": [float(v) for v in fit_shared.theta],
                    "train_objective": fit_shared.objective,
                    "regret_test": regret_test,
                    "regret_uniform": regret_uniform,
                    "regret_ratio": regret_test / regret_uniform if regret_uniform > 0 else float("nan"),
                    "seconds": round(time.perf_counter() - started, 3),
                }
                log.line(f"[{names[held_out]:>16}] {arm}: theta={np.round(fit_shared.theta, 3).tolist()} "
                         f"F_train={fit_shared.objective:.4e} R_test={regret_test:.4e} "
                         f"R_uniform={regret_uniform:.4e} ratio={extras[arm]['regret_ratio']:.5f}")
        for arm, prediction in predictions.items():
            scores = flow_scores(prediction, observed)
            metrics: dict[str, float] = scores.to_dict()
            if arm in extras and "regret_ratio" in extras[arm]:
                metrics["regret_ratio"] = float(extras[arm]["regret_ratio"])  # type: ignore[arg-type]
            records.append(RunRecord(
                study=str(config.study), geometry=f"holdout_{names[held_out]}", arm=str(arm),
                k=int(len(extras[arm]["theta"])) if arm in extras and "theta" in extras[arm] else 0,
                m=instance.n_zones, draw=held_out, metrics=metrics,
                config=config.to_dict(), extra=extras.get(arm, {})))
            log.line(f"[{names[held_out]:>16}] {arm:>26} rmse={scores.rmse:9.2f} corr={scores.correlation:.4f} "
                     f"cpc={scores.cpc:.4f}")

    write_records(run_dir / "records.json", records, provenance=PROVENANCE)
    table = summarise(records, group=("arm",), metrics=("rmse", "correlation", "cpc", "regret_ratio"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "rmse_mean", "correlation_mean", "cpc_mean",
                                         "regret_ratio_mean")))
    log.line("published reference (their n=100, their month): M-IOT 4.45/0.9989/0.9566, "
             "naive IOT 4.76/0.9988/0.9489, gravity 37.12/0.9076/0.6568")
    log.line(f"wrote {run_dir / 'records.json'} ({len(records)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
