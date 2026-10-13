"""
CN: 线性内存工具：分块代价矩阵与**分块**对偶证书（永不物化 m×n 或 (m·n,K)）。
EN: Linear-memory helpers: a blockwise cost matrix and a **blockwise** dual certificate
    (never materialise `m×n` or `(m·n, K)`).

CN: 目标路径（HELLO 真值 + GPU 扫描 + GPU master）全程 O(m+n) 内存；这些函数让校验/证书也保持
    线性内存，而不是退回 POT + dense cost。
EN: The target path (HELLO ground truth + GPU scan + GPU master) is O(m+n) memory throughout;
    these keep validation/certification linear too, instead of falling back to POT + dense cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iot import FeatureMap, dense_cost_matrix

DEFAULT_BYTE_BUDGET: int = 256 * 1024 * 1024


def _rows_per_block(features: FeatureMap, byte_budget: int) -> int:
    """CN: 每块的源行数，使 (batch·n, K) 约在 byte_budget 内。EN: Rows per block for a ~byte_budget tile."""
    n_target = max(1, int(features.n_target))
    per_edge = 8 * max(1, int(features.n_features))
    per_row = per_edge * n_target
    return max(1, int(byte_budget) // max(1, per_row))


def blockwise_cost_matrix(
    features: FeatureMap, theta: np.ndarray, *, byte_budget: int = DEFAULT_BYTE_BUDGET
) -> np.ndarray:
    """CN: 分块构造 (m,n) 代价矩阵（仅在确实需要稠密时用，例如小规模 POT 参照）。
    EN: Blockwise (m,n) cost matrix (only when dense is truly needed, e.g. small-n POT reference)."""
    m = int(features.n_source)
    n = int(features.n_target)
    rows_per_block = min(m, _rows_per_block(features, int(byte_budget)))
    cost = np.empty((m, n), dtype=np.float64)
    for start in range(0, m, rows_per_block):
        stop = min(m, start + rows_per_block)
        rows = np.repeat(np.arange(start, stop, dtype=np.int64), n)
        cols = np.tile(np.arange(n, dtype=np.int64), stop - start)
        cost[start:stop] = np.asarray(features.costs(theta, rows, cols)).reshape(stop - start, n)
    return cost


@dataclass(frozen=True)
class DualFeasibility:
    """CN: 独立对偶证书（不依赖 POT，线性内存）。EN: Independent dual certificate (POT-free, linear memory)."""

    certified: bool
    max_dual_violation: float
    support_residual: float


def dual_feasibility(
    features: FeatureMap,
    theta: np.ndarray,
    source_dual: np.ndarray,
    target_dual: np.ndarray,
    support_rows: np.ndarray,
    support_cols: np.ndarray,
    *,
    feasibility_tol: float = 1e-6,
    tightness_tol: float = 1e-6,
    byte_budget: int = DEFAULT_BYTE_BUDGET,
) -> DualFeasibility:
    """
    CN: 用给定对偶势 (u,v) 分块验证 pi_hat 最优：max(u_i+v_j-c_ij) <= tol 且支撑上取等。
        完全不物化 m×n，也不调用 POT。
    EN: Verify pi_hat optimality from given duals (u,v), blockwise: max(u_i+v_j-c_ij) <= tol and
        equality on the support. Never materialises m×n and never calls POT.
    """
    m = int(features.n_source)
    n = int(features.n_target)
    u = np.asarray(source_dual, dtype=np.float64).reshape(-1)
    v = np.asarray(target_dual, dtype=np.float64).reshape(-1)
    rows_per_block = min(m, _rows_per_block(features, int(byte_budget)))
    worst = -np.inf
    for start in range(0, m, rows_per_block):
        stop = min(m, start + rows_per_block)
        rows = np.repeat(np.arange(start, stop, dtype=np.int64), n)
        cols = np.tile(np.arange(n, dtype=np.int64), stop - start)
        cost = np.asarray(features.costs(theta, rows, cols))
        worst = max(worst, float(np.max(u[rows] + v[cols] - cost)))
    if support_rows.size:
        cost_support = np.asarray(features.costs(theta, support_rows, support_cols))
        support_residual = float(np.max(np.abs(u[support_rows] + v[support_cols] - cost_support)))
    else:
        support_residual = 0.0
    certified = bool(worst <= float(feasibility_tol) and support_residual <= float(tightness_tol))
    return DualFeasibility(
        certified=certified,
        max_dual_violation=float(worst),
        support_residual=support_residual,
    )


__all__ = [
    "DEFAULT_BYTE_BUDGET",
    "DualFeasibility",
    "blockwise_cost_matrix",
    "dense_cost_matrix",
    "dual_feasibility",
]
