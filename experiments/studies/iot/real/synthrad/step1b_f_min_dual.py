"""
CN: **Step 1b —— 用牛顿/L-BFGS 在双变量上解，以到达小 eps**。

    CN: 为什么换求解器：原版交替 Sinkhorn 在最坏情况下需要约 (代价幅度)/eps 次扫描，所以
        eps=0.1 达不到。这里把 `(f, g)` 当作**无约束**未知量，直接最小化对偶函数

            D(f,g) = -a·f - b·g + eps * sum_ij exp((f_i + g_j - C_ij)/eps)        (计数测度)

        它的梯度是 `(-a_i + sum_j pi_ij, -b_j + sum_i pi_ij)`，`pi = exp(sigma/eps)`，因此平稳点
        正是 Sinkhorn 的固定点。Hessian 是 `pi` 的一个加权图 Laplacian，是半正定的，只差一个
        平移零空间 —— 用 L-BFGS 就够，且不需要显式 Hessian。

    CN: 无约束的含义：`(f, g)` 只在 `sum f + sum g` 的方向上无界，但那个方向不改变 `pi`，
        所以对偶值有界；这里不额外固定规范（与 Step 1 一致，避免破坏 pi 的缩放）。

EN: **Step 1b -- solve on the dual with Newton/L-BFGS to reach small epsilon.**

    Why change the solver: plain alternating Sinkhorn needs about (cost range)/epsilon sweeps in the worst
    case, so epsilon=0.1 is not reached. Here `(f, g)` are treated as **unconstrained** unknowns and the dual

        D(f,g) = -a.f - b.g + eps * sum_ij exp((f_i + g_j - C_ij)/eps)        (counting measure)

    is minimised directly. Its gradient is `(-a_i + sum_j pi_ij, -b_j + sum_i pi_ij)` with
    `pi = exp(sigma/eps)`, so a stationary point is exactly the Sinkhorn fixed point. Its Hessian is a
    `pi`-weighted graph Laplacian (PSD up to the shift null space), so L-BFGS suffices and no Hessian is
    formed.

    "Unconstrained": `(f, g)` is unbounded only along `sum f + sum g`, and that direction does not change
    `pi`, so the dual value is bounded below. No gauge is imposed here, consistent with Step 1, so that the
    scale of `pi` is never disturbed.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import DiagonalWeightFeatures  # noqa: E402
from experiments.harness.config import RunConfig, build_parser  # noqa: E402
from experiments.harness.datasets import reduce_features, stream_features  # noqa: E402
from experiments.harness.records import RunLog  # noqa: E402
from experiments.harness.synthrad import discover_patients  # noqa: E402

PROVENANCE: dict[str, object] = {
    "stage": "1b — |F_min| at smaller epsilon, dual solved by L-BFGS instead of Sinkhorn sweeps",
    "dual": "D(f,g) = -a.f - b.g + eps*sum exp((f+g-C)/eps)   (counting measure)",
    "diagnostics": "median per-row gap between best and 2nd-best REDUCED cost; effective neighbours e^H",
    "gate": "drop the restricted loop if |F_min|/(m+n) > ~12 at an epsilon where the fit is good",
}

REL_EPS_GRID: tuple[float, ...] = (0.25, 0.1, 0.05, 0.025)
ETA_GRID: tuple[float, ...] = (1e-2, 1e-3, 1e-4)


def solve_dual(cost: Float[np.ndarray, "m n"], a: Float[np.ndarray, "m"], b: Float[np.ndarray, "n"],
               *, eps: float, x0: np.ndarray | None = None, max_iterations: int = 20000,
               tolerance: float = 1e-10) -> tuple[Float[np.ndarray, "m"], Float[np.ndarray, "n"],
                                                 Float[np.ndarray, "m n"], float, bool]:
    """
    CN: 用 L-BFGS 解对偶，返回 `(f, g, pi, 相对边缘误差, converged)`。
        `pi = exp((f+g-C)/eps)` 行和应为 `a`、列和应为 `b`。
    EN: L-BFGS on the dual, returning `(f, g, pi, relative marginal error, converged)`.
        `pi = exp((f+g-C)/eps)` must have row sums `a` and column sums `b`.
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    m = int(a.size)
    eps = float(eps)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        # CN: 逐行稳定（**版本 1**，已验证与 Sinkhorn 一致到 3.5e-10）：`pi = exp((sigma-shift_i)/eps)`
        #     再乘 `exp(shift_i/eps)`；因为每行只减自己的最大值，`exp(shift_i/eps)` 与行内项同尺度，
        #     不会像全局 shift 那样溢出。整行直接融合成 `expo_row = exp(shift_i/eps) * exp(...)`。
        # EN: row-wise stabilisation (the **validated** version, agreeing with Sinkhorn to 3.5e-10):
        #     `pi = exp((sigma - shift_i)/eps) * exp(shift_i/eps)`; subtracting only each row's own max keeps
        #     `exp(shift_i/eps)` on the same scale as the row entries, so it does not overflow.
        f = flat[:m]
        g = flat[m:]
        sigma = f[:, None] + g[None, :] - cost
        shift = sigma.max(axis=1, keepdims=True)
        expo = np.exp((sigma - shift) / eps)
        row_scale = np.exp(shift.reshape(-1) / eps)
        row_sums = row_scale * expo.sum(axis=1)
        col_sums = (row_scale[:, None] * expo).sum(axis=0)
        value = float(-a @ f - b @ g + eps * float(row_sums.sum()))
        grad = np.concatenate([-a + row_sums, -b + col_sums])
        return value, grad

    start = np.zeros(2 * m) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(-1)
    result = minimize(objective, start, jac=True, method="L-BFGS-B",
                      options={"maxiter": int(max_iterations), "maxcor": 20,
                               "ftol": 1e-18, "gtol": float(tolerance)})
    flat = np.asarray(result.x, dtype=np.float64)
    f, g = flat[:m], flat[m:]
    sigma = f[:, None] + g[None, :] - cost
    shift = sigma.max(axis=1, keepdims=True)
    pi = np.exp((sigma - shift) / eps) * np.exp(shift / eps)
    err = float(max(np.abs((pi.sum(1) - a) / a).max(), np.abs((pi.sum(0) - b) / b).max()))
    return f, g, pi, err, err <= 1e-4


def _row_gap(cost: Float[np.ndarray, "m n"], f: Float[np.ndarray, "m"],
             g: Float[np.ndarray, "n"]) -> Float[np.ndarray, "m"]:
    """
    CN: 每行**约化代价** `c_ij - f_i - g_j` 中最好与次好的差。这才是决定稀疏性的尺度。
    EN: per-row gap between the best and second-best **reduced** cost `c_ij - f_i - g_j`. This is the scale
        that governs sparsity.
    """
    reduced = np.asarray(cost, dtype=np.float64) - f[:, None] - g[None, :]
    part = np.partition(reduced, 1, axis=1)
    return part[:, 1] - part[:, 0]


def _row_entropy_neighbours(pi: Float[np.ndarray, "m n"]) -> Float[np.ndarray, "m"]:
    """CN: 每行的有效邻居数 `e^H`。EN: per-row effective neighbour count `e^H`."""
    rows = np.asarray(pi, dtype=np.float64) / np.maximum(pi.sum(axis=1, keepdims=True), 1e-300)
    entropy = -np.sum(np.where(rows > 0, rows * np.log(np.maximum(rows, 1e-300)), 0.0), axis=1)
    return np.exp(entropy)


@dataclasses.dataclass(frozen=True)
class Step1bConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    section = "synthrad_step1b"
    study: str = "synthrad_step1b"
    m: int = 600
    k: int = 64
    draw: int = 0

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--m", type=int, default=600)
        parser.add_argument("--k", type=int, default=64)
        parser.add_argument("--draw", type=int, default=0)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(Step1bConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = Step1bConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("step 1b: |F_min| at small epsilon, dual by L-BFGS (reaches where Sinkhorn sweeps cannot)")

    split = json.loads((config.out / "synthrad_v2" / "split_v2.json").read_text())
    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    fit_pats = [p for p in patients if p.patient in set(split["fit"])]
    rng = np.random.default_rng(1000 + int(config.draw))
    chosen = [fit_pats[i] for i in rng.permutation(len(fit_pats))[:60]]
    pm, pc, _pid, _z, _ctr = stream_features(chosen, window=(-200.0, 300.0), max_per_patient=None,
                                             min_gap=0, seed=int(config.draw))
    ftr, ftc = reduce_features(pm, pc, k=int(config.k), geometry_name="G3", whiten=False)
    del pm, pc
    m = int(min(int(config.m), int(ftr.shape[0])))
    sub = np.sort(np.random.default_rng(777).permutation(int(ftr.shape[0]))[:m])
    x: Float[np.ndarray, "m d"] = np.ascontiguousarray(ftr[sub])
    y: Float[np.ndarray, "m d"] = np.ascontiguousarray(ftc[sub])
    k = int(x.shape[1])
    a: Float[np.ndarray, "m"] = np.full(m, 1.0 / m)
    b: Float[np.ndarray, "n"] = np.full(m, 1.0 / m)
    theta_uniform: Float[np.ndarray, "k"] = np.full(k, 1.0 / k)
    rows: Int[np.ndarray, "edges"] = np.repeat(np.arange(m), m)
    cols: Int[np.ndarray, "edges"] = np.tile(np.arange(m), m)
    cost = np.asarray(DiagonalWeightFeatures(x, y).costs(theta_uniform, rows, cols),
                      dtype=np.float64).reshape(m, m)
    std = max(float(cost.std()), 1e-12)
    log.line(f"m={m} k={k} cost std (uniform theta) = {std:.6e}")

    table: list[dict[str, object]] = []
    x0 = None
    for rel in REL_EPS_GRID:
        eps = float(rel) * std
        t0 = time.perf_counter()
        f, g, pi, err, converged = solve_dual(cost, a, b, eps=eps, x0=x0, max_iterations=40000)
        x0 = np.concatenate([f, g])
        dt = time.perf_counter() - t0
        gap = _row_gap(cost, f, g)
        neighbours = _row_entropy_neighbours(pi)
        base = {"rel_eps": rel, "eps": eps, "marginal_error": err, "converged": converged,
                "seconds": round(dt, 1),
                "median_row_gap": float(np.median(gap)),
                "eps_over_gap": float(eps / max(float(np.median(gap)), 1e-300)),
                "median_neighbours": float(np.median(neighbours)),
                "max_sigma": float((f[:, None] + g[None, :] - cost).max())}
        if not converged:
            log.line(f"rel_eps={rel:<6} eps={eps:.3e}: UNCONVERGED (marg err {err:.2e}, {dt:.1f}s)")
            table.append({**base, "status": "unconverged"})
            continue
        order = np.argsort(-pi.reshape(-1))
        values = pi.reshape(-1)[order]
        cumulative = np.cumsum(values)
        total = float(cumulative[-1])
        for eta in ETA_GRID:
            kk = int(np.searchsorted(cumulative, total * (1.0 - float(eta)), side="left")) + 1
            kk = min(kk, int(values.size))
            table.append({**base, "status": "ok", "eta": eta, "F_min_k": kk,
                          "F_min_k_over_nodes": kk / float(2 * m),
                          "excluded_fraction": float(total - cumulative[kk - 1]) / max(total, 1e-300)})
        best = [t for t in table if t.get("rel_eps") == rel and t["status"] == "ok"]
        log.line(f"rel_eps={rel:<6} eps={eps:.3e} marg={err:.1e} gap={base['median_row_gap']:.3e} "
                 f"eps/gap={base['eps_over_gap']:.3f} nbours={base['median_neighbours']:.1f} "
                 f"maxsig={base['max_sigma']:+.2e} {dt:.0f}s | "
                 + "  ".join(f"eta={t['eta']:g}->{t['F_min_k_over_nodes']:.2f}/node" for t in best))

    (run_dir / "step1b.json").write_text(
        json.dumps({"rows": table, "provenance": PROVENANCE, "cost_std_uniform": std,
                    "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                               for key, v in dataclasses.asdict(config).items()},
                    "m": m, "k": k}, indent=2, default=str) + "\n", encoding="utf-8")
    log.line(f"wrote {run_dir / 'step1b.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
