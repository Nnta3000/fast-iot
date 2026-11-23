"""
CN: **EOT 内层求解的观测记录** —— 只测量，不下结论。

    CN: 记录四件事，全部写成 JSON：
        1. 消去 `f` 后 `g` 块 Hessian 的**特征值谱**（条件数）随 `|F|` 的变化；
        2. `pi` 的稀疏程度（低于 1e-6·max 的比例）；
        3. 交替 Sinkhorn 达到给定边缘误差所需的**扫描次数**，以及该扫描次数随 `range/eps` 的变化；
        4. 内层边缘误差如何传播到外层 `theta` 梯度。

    CN: **`|F|` 按"每节点边数"构造**：`B = S ∪ R`（支撑 + 连接环），再加每行按代价最小的 `k` 条边。

    CN: 不做切面判定、不写可行性结论；只报数字与它们的配置。

EN: **Observations on EOT's inner solve** -- measurement only, no conclusions.

    Four things are recorded, all to JSON:
      1. the **eigenvalue spectrum** (condition number) of the `g`-block Hessian after eliminating `f`, as a
         function of `|F|`;
      2. the sparsity of `pi` (fraction below 1e-6 * max);
      3. the number of alternating-Sinkhorn **sweeps** needed to reach a given marginal error, and how that
         count scales with `range/eps`;
      4. how the inner marginal error propagates to the outer `theta` gradient.

    **`|F|` is built by "edges per node"**: `B = S ∪ R` (support plus the connector cycle), then the `k`
    cheapest edges per row are added.

    No feasibility verdict is drawn here; only numbers and their configuration.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.edges import EdgeSet  # noqa: E402
from iot.entropic.inner_f import _evaluate_on_f, _segment_sum  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402

REL_EPS_GRID: tuple[float, ...] = (0.25, 0.1, 0.05)
EXTRA_EDGES_PER_ROW: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 14)
SWEEP_CAP: int = 20_000
TARGET_ERRORS: tuple[float, ...] = (1e-2, 1e-3, 1e-4)

PROVENANCE: dict[str, object] = {
    "purpose": "observations on EOT's inner solve: Hessian spectrum, pi sparsity, Sinkhorn sweep counts, "
               "gradient propagation",
    "instance": "synthetic: x, y ~ N(0, 0.1^2) in R^K, cost = weighted squared differences (diagonal family)",
    "edges": "B = support (identity) + connector cycle; plus the k cheapest edges per row",
    "eps": "ABSOLUTE and frozen: eps = rel_eps * std(cost at uniform theta)",
    "not_included": "no feasibility statement; no comparison across solvers beyond what is recorded",
    "sweeps": "alternating Sinkhorn scaling g <- g - eps*log(colsum/b); stopping on relative marginal error",
}


def _instance(m: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CN: 合成实例 `(x, y, 特征行)`。EN: synthetic instance `(x, y, feature rows)`."""
    rng = np.random.default_rng(int(seed))
    x: Float[np.ndarray, "m k"] = rng.normal(size=(m, k)) * 0.1
    y: Float[np.ndarray, "m k"] = rng.normal(size=(m, k)) * 0.1
    rows: Int[np.ndarray, "edges"] = np.repeat(np.arange(m), m)
    cols: Int[np.ndarray, "edges"] = np.tile(np.arange(m), m)
    feats: Float[np.ndarray, "edges k"] = np.stack(
        [(x[rows, a] - y[cols, a]) ** 2 for a in range(k)], axis=1)
    return x, y, feats


def _edges_for(m: int, extra: int, cost: Float[np.ndarray, "m m"]) -> EdgeSet:
    """
    CN: `B = 支撑 ∪ 连接环`，再加每行 `extra` 条最便宜的边（跳过已在 B 中的）。
    EN: `B = support ∪ connector cycle`, plus the `extra` cheapest edges per row not already in `B`.
    """
    base_r = np.arange(m, dtype=np.int64)
    base_c = np.arange(m, dtype=np.int64)
    cycle_r = np.arange(m, dtype=np.int64)
    cycle_c = (np.arange(m, dtype=np.int64) + 1) % m
    rows = [base_r, cycle_r]
    cols = [base_c, cycle_c]
    if extra > 0:
        add_r: list[int] = []
        add_c: list[int] = []
        for i in range(m):
            taken = {i, (i + 1) % m}
            for j in np.argsort(cost[i]):
                if int(j) not in taken:
                    add_r.append(i)
                    add_c.append(int(j))
                    taken.add(int(j))
                    if len(taken) >= 2 + extra:
                        break
        rows.append(np.asarray(add_r, dtype=np.int64))
        cols.append(np.asarray(add_c, dtype=np.int64))
    return EdgeSet.from_arrays(np.concatenate(rows), np.concatenate(cols), n_target=m)


def _g_block_hessian(pi: Float[np.ndarray, "edges"], edges: EdgeSet, n: int, m: int,
                     eps: float) -> Float[np.ndarray, "n n"]:
    """
    CN: 消去 `f` 后 `g` 块的稠密 Hessian（仅用于测谱，`n` 小时可行）。
    EN: the dense `g`-block Hessian after eliminating `f` (for the spectrum only; feasible at small `n`).
    """
    row_pi = _segment_sum(pi, edges.rows, m)
    safe = np.maximum(row_pi, 1e-300)
    hessian = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        v = np.zeros(n)
        v[j] = 1.0
        s_e = v[edges.cols]
        row_mean = _segment_sum(pi * s_e, edges.rows, m) / safe
        hessian[:, j] = _segment_sum(pi * (s_e - row_mean[edges.rows]), edges.cols, n) / eps
    return 0.5 * (hessian + hessian.T)


def _sinkhorn_sweeps(edges: EdgeSet, cost_f: Float[np.ndarray, "edges"],
                     a: Float[np.ndarray, "m"], b: Float[np.ndarray, "n"], eps: float,
                     targets: tuple[float, ...]) -> tuple[dict[str, int | None], float]:
    """
    CN: 交替 Sinkhorn，返回每个目标误差达到时的扫描次数（未达到为 `None`）与最终误差。
    EN: alternating Sinkhorn; per target, the sweep count reached (`None` if not) and the final error.
    """
    n = int(b.size)
    g = np.zeros(n, dtype=np.float64)
    hit: dict[str, int | None] = {f"{t:.0e}": None for t in targets}
    error = float("inf")
    for sweep in range(1, int(SWEEP_CAP) + 1):
        _f, log_pi, _cs, _v = _evaluate_on_f(cost_f, edges.rows, edges.cols, a, b, eps, g)
        pi = np.exp(log_pi)
        g = g - eps * np.log(np.maximum(_segment_sum(pi, edges.cols, n) / b, 1e-300))
        _f, _log_pi, colsum, _v = _evaluate_on_f(cost_f, edges.rows, edges.cols, a, b, eps, g)
        error = float(np.max(np.abs((colsum - b) / b)))
        for target in targets:
            if hit[f"{target:.0e}"] is None and error <= target:
                hit[f"{target:.0e}"] = sweep
        if all(v is not None for v in hit.values()):
            break
    return hit, error


@dataclasses.dataclass(frozen=True)
class EOTObsConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    section = "eot_inner_observations"
    study: str = "eot_inner_observations"
    m: int = 40
    k: int = 5

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--m", type=int, default=40)
        parser.add_argument("--k", type=int, default=5)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(EOTObsConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = EOTObsConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("EOT inner-solve observations: Hessian spectrum, pi sparsity, sweep counts, gradient propagation")

    m = int(config.m)
    k = int(config.k)
    x, y, feats = _instance(m, k, int(config.seed))
    theta = np.full(k, 1.0 / k)
    cost_flat = feats @ theta
    cost = np.asarray(cost_flat, dtype=np.float64).reshape(m, m)
    a: Float[np.ndarray, "m"] = np.full(m, 1.0 / m)
    b: Float[np.ndarray, "n"] = np.full(m, 1.0 / m)
    std = float(cost_flat.std())
    log.line(f"m=n={m} K={k}; cost std at uniform theta = {std:.4e}")

    rows: list[dict[str, object]] = []
    for rel_eps in REL_EPS_GRID:
        eps = float(rel_eps) * std
        for extra in EXTRA_EDGES_PER_ROW:
            edges = _edges_for(m, int(extra), cost)
            cost_f = np.asarray(cost[edges.rows, edges.cols], dtype=np.float64)
            feats_f = np.asarray([0.0])
            _ = feats_f
            # CN: 在 g=0 处取 pi，测谱与稀疏度（与求解器无关）。
            # EN: take pi at g=0 for the spectrum and sparsity (independent of any solver).
            _f, log_pi, _cs, _v = _evaluate_on_f(cost_f, edges.rows, edges.cols, a, b, eps,
                                                np.zeros(m, dtype=np.float64))
            pi = np.exp(log_pi)
            hessian = _g_block_hessian(pi, edges, m, m, eps)
            eigenvalues = np.linalg.eigvalsh(hessian)
            top = float(eigenvalues.max())
            positive = eigenvalues[eigenvalues > top * 1e-15] if top > 0 else np.zeros(0)
            bottom = float(positive.min()) if positive.size else float("nan")
            condition = float(top / bottom) if bottom and np.isfinite(bottom) and bottom > 0 else None
            hit, final_error = _sinkhorn_sweeps(edges, cost_f, a, b, eps, TARGET_ERRORS)
            row = {
                "rel_eps": float(rel_eps), "eps_absolute": eps, "extra_edges_per_row": int(extra),
                "edges_per_node": round(edges.size / (2.0 * m), 3),
                "F_size": int(edges.size), "F_fraction": round(edges.size / (m * m), 5),
                "cost_range_on_F": float(cost_f.max() - cost_f.min()),
                "range_over_eps": float((cost_f.max() - cost_f.min()) / eps),
                "hessian_lambda_max": top, "hessian_lambda_min_positive": bottom,
                "hessian_condition": condition,
                "hessian_is_zero": bool(top == 0.0),
                "pi_fraction_below_1e6_of_max": float(np.mean(pi < 1e-6 * max(float(pi.max()), 1e-300))),
                "pi_max": float(pi.max()),
                "sweeps_to_target": hit,
                "final_marginal_error": final_error,
            }
            rows.append(row)
            log.line(
                f"rel={rel_eps:<5} extra={extra:>2} |F|={100 * row['F_fraction']:>5.1f}% "
                f"e/n={row['edges_per_node']:>5.2f} r/eps={row['range_over_eps']:>6.1f} "
                f"cond={('%.2e' % condition) if condition else '   zero'} "
                f"sweeps(1e-2/1e-3/1e-4)={hit['1e-02']}/{hit['1e-03']}/{hit['1e-04']} "
                f"err={final_error:.1e}")

    # CN: 梯度传播：在 |F| 固定时，把内层误差与外层梯度误差对上。
    # EN: gradient propagation: at fixed |F|, map inner marginal error to outer gradient error.
    propagation: list[dict[str, object]] = []
    for rel_eps in REL_EPS_GRID:
        eps = float(rel_eps) * std
        edges = _edges_for(m, 6, cost)
        cost_f = np.asarray(cost[edges.rows, edges.cols], dtype=np.float64)
        feats_f = np.asarray([(x[i, A] - y[j, A]) ** 2
                              for i, j in zip(edges.rows, edges.cols) for A in range(k)],
                             dtype=np.float64).reshape(-1, k)
        g = np.zeros(m, dtype=np.float64)
        trace: list[tuple[int, float, np.ndarray]] = []
        for sweep in range(1, 6001):
            _f, log_pi, _cs, _v = _evaluate_on_f(cost_f, edges.rows, edges.cols, a, b, eps, g)
            pi = np.exp(log_pi)
            g = g - eps * np.log(np.maximum(_segment_sum(pi, edges.cols, m) / b, 1e-300))
            _f, log_pi, colsum, _v = _evaluate_on_f(cost_f, edges.rows, edges.cols, a, b, eps, g)
            pi = np.exp(log_pi)
            trace.append((sweep, float(np.max(np.abs((colsum - b) / b))), pi @ feats_f))
        reference = trace[-1][2]
        for sweep, error, grad in trace:
            if sweep in (1, 10, 100, 500, 2000, 6000):
                propagation.append({
                    "rel_eps": float(rel_eps), "sweeps": int(sweep),
                    "marginal_error": error,
                    "gradient_error_inf": float(np.abs(grad - reference).max()),
                    "gradient_scale_inf": float(np.abs(reference).max()),
                    "extra_edges_per_row": 6,
                })
        log.line(f"propagation rel={rel_eps}: recorded {len([p for p in propagation if p['rel_eps'] == rel_eps])} points")

    payload = {"rows": rows, "propagation": propagation, "provenance": PROVENANCE,
               "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                          for key, v in dataclasses.asdict(config).items()},
               "sweep_cap": SWEEP_CAP, "target_errors": list(TARGET_ERRORS),
               "cost_std_uniform": std}
    (run_dir / "observations.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                               encoding="utf-8")
    log.line(f"wrote {run_dir / 'observations.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
