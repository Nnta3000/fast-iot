"""
CN: **`|F|` 的占比随规模的变化** —— 只测量。

    CN: 问题：若 T3 需要 `|F|` 约占网格 20%，这个比例在 `m` 变大时是**保持**还是**缩小**？
        这决定 T3 在规模上是否可行。

    CN: 两件事分别量，因为它们回答的是同一个问题的两面：
        1. **内层代价**：达到给定边缘误差所需扫描次数随 `m` 的变化（在固定"每节点边数"下）。
           若扫描次数随 `m` 增长，那局部 `|F|` 也救不了。
        2. **`|F|` 需求**：达到给定边缘误差所需的**最小每节点边数**随 `m` 的变化。
           若这个数量随 `m` 下降，则 `|F|` 占比 ∝ (e/n)·m/(m·m) = (e/n)/m 会缩小。

    CN: 实例为合成对角族（`x, y ~ N(0, 0.1)`），因为这里要的是**规模趋势**，不是某个真实数据集的绝对值。

EN: **How `|F|`'s fraction changes with scale** -- measurement only.

    Question: if T3 needs `|F|` at roughly 20 % of the grid, does that fraction **stay** or **shrink** as `m`
    grows? That decides whether T3 is viable at scale.

    Two things are measured separately, because they are two sides of one question:
      1. **inner cost**: sweeps needed to reach a given marginal error, as a function of `m`, at a fixed
         "edges per node". If the sweep count grows with `m`, a local `|F|` does not save it.
      2. **`|F|` requirement**: the **minimum edges per node** needed to reach a given marginal error, as a
         function of `m`. Note the fraction is `(e/n) * m / (m * m) = (e/n) / m`, so a **constant** `e/n`
         already shrinks as `1/m`.

    The instance is synthetic (diagonal family, `x, y ~ N(0, 0.1)`) because the question is a **scaling trend**,
    not the absolute value on any one dataset.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.entropic.inner_f import _evaluate_on_f, _segment_sum  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402
from iot_experiments.studies.real.t1t2.metric_learning.t3_observations import (  # noqa: E402
    _edges_for, _instance)

SIZES: tuple[int, ...] = (40, 80, 160, 320)
EDGES_PER_ROW_GRID: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
REL_EPS_GRID: tuple[float, ...] = (0.25, 0.1)
TARGET: float = 1e-3
"""CN: 目标边缘误差（相对）。根据 "内层误差 -> 外层梯度" 的传播测量，1e-3 已足够。"""

PROVENANCE: dict[str, object] = {
    "purpose": "does |F| as a FRACTION of the grid shrink with m?",
    "instance": "synthetic diagonal family, x, y ~ N(0, 0.1^k), uniform marginals",
    "eps": "absolute, frozen: rel_eps * std(cost at uniform theta)",
    "measure": "minimum edges-per-node reaching a relative marginal error of TARGET, and the sweeps needed",
    "note": "fraction = edges_per_node / m, so a CONSTANT edges-per-node already shrinks as 1/m",
}


def _sweeps_to_target(m: int, k: int, cost: Float[np.ndarray, "m m"], edges, eps: float,
                      target: float, cap: int) -> tuple[int | None, float, float]:
    """
    CN: 交替 Sinkhorn 达到 `target` 所需扫描次数；返回 `(sweeps 或 None, 最终误差, 秒数)`。
    EN: sweeps to reach `target`; returns `(sweeps or None, final error, seconds)`.
    """
    a: Float[np.ndarray, "m"] = np.full(m, 1.0 / m)
    cost_f = np.asarray(cost[edges.rows, edges.cols], dtype=np.float64)
    g = np.zeros(m, dtype=np.float64)
    started = time.perf_counter()
    error = float("inf")
    for sweep in range(1, int(cap) + 1):
        _f, log_pi, _cs, _v = _evaluate_on_f(cost_f, edges.rows, edges.cols, a, a, eps, g)
        pi = np.exp(log_pi)
        g = g - eps * np.log(np.maximum(_segment_sum(pi, edges.cols, m) / a, 1e-300))
        _f, _lp, colsum, _v = _evaluate_on_f(cost_f, edges.rows, edges.cols, a, a, eps, g)
        error = float(np.max(np.abs((colsum - a) / a)))
        if error <= target:
            return sweep, error, time.perf_counter() - started
    return None, error, time.perf_counter() - started


@dataclasses.dataclass(frozen=True)
class ScalingConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    section = "t3_scaling"
    study: str = "t3_scaling"
    k: int = 5

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--k", type=int, default=5)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(ScalingConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = ScalingConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line(f"scaling probe: target relative marginal error = {TARGET:.0e}")
    log.line(f"sizes={SIZES} edges/node grid={EDGES_PER_ROW_GRID} rel_eps={REL_EPS_GRID}")

    records: list[dict[str, object]] = []
    for m in SIZES:
        k = int(config.k)
        x, y, feats = _instance(int(m), k, int(config.seed))
        theta = np.full(k, 1.0 / k)
        cost = np.asarray(feats @ theta, dtype=np.float64).reshape(int(m), int(m))
        std = float((feats @ theta).std())
        for rel_eps in REL_EPS_GRID:
            eps = float(rel_eps) * std
            cap = 4000 if m <= 160 else 2000
            minimal: int | None = None
            for extra in EDGES_PER_ROW_GRID:
                edges = _edges_for(int(m), int(extra), cost)
                sweeps, error, seconds = _sweeps_to_target(int(m), k, cost, edges, eps, TARGET, cap)
                entry = {
                    "m": int(m), "rel_eps": float(rel_eps), "eps_absolute": eps,
                    "extra_edges_per_row": int(extra),
                    "edges_per_node": round(edges.size / (2.0 * m), 3),
                    "F_fraction": round(edges.size / (m * m), 6),
                    "sweeps": sweeps, "final_marginal_error": error,
                    "reached_target": sweeps is not None, "seconds": round(seconds, 2),
                }
                records.append(entry)
                if sweeps is not None and minimal is None:
                    minimal = int(extra)
                log.line(f"m={m:>4} rel={rel_eps:<5} extra={extra:>3} e/n={entry['edges_per_node']:>6.2f} "
                         f"|F|%={100 * entry['F_fraction']:>7.3f} sweeps={str(sweeps):>6} "
                         f"err={error:.1e} {seconds:.1f}s")
            log.line(f"  => m={m} rel={rel_eps}: minimal extra edges/row reaching {TARGET:.0e} = {minimal}")

    payload = {"records": records, "provenance": PROVENANCE, "target": TARGET,
               "sizes": list(SIZES), "edges_per_row_grid": list(EDGES_PER_ROW_GRID),
               "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                          for key, v in dataclasses.asdict(config).items()}}
    (run_dir / "scaling.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                         encoding="utf-8")
    log.line(f"wrote {run_dir / 'scaling.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
