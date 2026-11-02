"""
CN: **松弛循环的数值诊断** —— 代价尺度的分块估计、regret 恒等式断言、logit 指纹。

    CN: 这些都不是算法本身，只是**测量**：代价尺度决定扫描容差，恒等式断言捕捉 master 与循环之间的
        不一致。抽出来是为了让 `loop_regret.py` 只剩控制流。

    CN: 全部按**行分块**，内存与 `m·n` 无关——此前一次性求值全部边在 m=5000 时需要约 13 GB（实测 OOM）。

EN: **Numerical diagnostics for the relaxed loop** -- blockwise cost-scale estimate, the regret identity
    assertion, and an edge-set fingerprint.

    None of these is the algorithm; they are **measurements**: the cost scale sets the scan tolerance and the
    identity assertion catches a disagreement between master and loop. They are extracted so that
    `loop_regret.py` keeps only control flow.

    Everything works **in row blocks**, so memory is independent of `m*n`; evaluating every edge at once
    needed ~13 GB at m=5000 (measured OOM).
"""

from __future__ import annotations

import hashlib

import numpy as np
from jaxtyping import Float, Int

from ..core.edges import EdgeSet
from ..core.problem import InverseOTProblem

IDENTITY_RTOL: float = 1e-10


def uniform_theta(problem: InverseOTProblem) -> Float[np.ndarray, "k"]:
    """CN: 单纯形上的均匀 θ。EN: uniform theta on the simplex."""
    k = int(problem.n_features)
    value = np.full(k, 1.0 / float(k), dtype=np.float64)
    if not problem.theta_set.contains(value, tol=1e-9):
        raise RuntimeError("the reference theta (uniform) is not in Theta; eps_R has no scale")
    return value


def median_cost_blockwise(problem: InverseOTProblem, theta: Float[np.ndarray, "k"], *,
                          m: int, n: int, rows_per_block: int = 256,
                          max_samples: int = 2_000_000) -> float:
    """
    CN: uniform θ 下 `|c|` 中位数的**分块流式**估计，内存与 `m·n` 无关。每个块取等距子样本，
        累积到 `max_samples`；对中位数而言这个样本量远超所需。
    EN: a **blockwise streaming** estimate of the median `|c|` at uniform theta, with memory independent of
        `m*n`. Each block contributes an evenly spaced subsample, accumulated up to `max_samples`; for a
        median that is far more than needed.
    """
    samples: list[Float[np.ndarray, "block"]] = []
    kept = 0
    cols = np.arange(n)
    for start in range(0, m, int(rows_per_block)):
        stop = min(start + int(rows_per_block), m)
        rows: Int[np.ndarray, "block n"] = np.repeat(np.arange(start, stop), n)
        tiled: Int[np.ndarray, "block n"] = np.tile(cols, stop - start)
        cost = np.asarray(problem.features.costs(theta, rows, tiled), dtype=np.float64)
        if cost.size:
            stride = max(1, cost.size // 4096)
            samples.append(np.abs(cost[::stride]))
            kept += samples[-1].size
        if kept >= int(max_samples):
            break
    if not samples:
        raise RuntimeError("could not sample any cost value; the problem is empty")
    pool: Float[np.ndarray, "samples"] = np.concatenate(samples)
    return float(np.median(pool))


def assert_regret_identity(problem: InverseOTProblem, theta: Float[np.ndarray, "k"],
                           source_dual: Float[np.ndarray, "m"], target_dual: Float[np.ndarray, "n"],
                           support_rows: Int[np.ndarray, "support"],
                           support_cols: Int[np.ndarray, "support"],
                           support_values: Float[np.ndarray, "support"], regret: float, *,
                           rtol: float = IDENTITY_RTOL) -> None:
    """
    CN: 断言 `R = -sum_S pi_e sigma_e`（相对容差）。

        CN: **容差必须由调用方给出**：精确 LP 上与 `rtol` 同量级（~1e-10），但一阶求解器（CuPDLPx）
            只把 `sigma <= 0` 满足到求解精度（~1e-7），恒等式只能到同一量级。用同一个 `rtol`
            会在 CuPDLPx 上**误报**。
    EN: assert `R = -sum_S pi_e sigma_e` within a relative tolerance.

        **The tolerance must come from the caller**: on an exact LP it is ~1e-10, but a first-order solver
        (CuPDLPx) satisfies `sigma <= 0` only to its own accuracy (~1e-7), so the identity is only good to
        the same order. Using the exact tolerance fires spuriously on CuPDLPx.
    """
    sigma: Float[np.ndarray, "support"] = (
        np.asarray(source_dual, dtype=np.float64)[support_rows]
        + np.asarray(target_dual, dtype=np.float64)[support_cols]
        - np.asarray(problem.features.costs(theta, support_rows, support_cols), dtype=np.float64))
    values = np.asarray(support_values, dtype=np.float64)
    lhs = float(regret)
    rhs = -float(np.dot(values, sigma))
    cost_support = np.asarray(problem.features.costs(theta, support_rows, support_cols),
                              dtype=np.float64)
    # CN: 与原来一致的尺度：1、|sum pi c|、|sum pi f|、|sum pi g| 的最大值。
    # EN: the same scale as before: max of 1, |sum pi c|, |sum pi f|, |sum pi g|.
    scale = max(1.0,
                abs(float(np.dot(values, cost_support))),
                abs(float(np.dot(values, np.asarray(source_dual, dtype=np.float64)[support_rows]))),
                abs(float(np.dot(values, np.asarray(target_dual, dtype=np.float64)[support_cols]))),
                abs(lhs), abs(rhs))
    if abs(lhs - rhs) > float(rtol) * scale:
        raise RuntimeError(
            f"regret identity violated: R={lhs!r} vs {rhs!r} "
            f"(rel={abs(lhs - rhs) / scale:.3e}, rtol={rtol:.1e}, scale={scale:.3e})")


def fingerprint(edges: EdgeSet) -> str:
    """CN: `F_t` 的稳定哈希。EN: a stable hash of `F_t`."""
    return hashlib.sha256(np.ascontiguousarray(edges.keys).tobytes()).hexdigest()


__all__ = ["IDENTITY_RTOL", "assert_regret_identity", "fingerprint", "median_cost_blockwise",
           "uniform_theta"]
