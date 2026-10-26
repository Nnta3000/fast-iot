"""
CN: 在全部边上分块计算对偶分数 sigma，返回违例集合 Q 与逐行/逐列 top-gamma 候选 V。
EN: Compute the dual score sigma on all edges blockwise and return the violated set Q
    with its per-row/per-column top-gamma candidates V.

CN: 本模块只调用 `FeatureMap.costs`，不物化 m x n 矩阵，也不假设代价的具体结构。
EN: This module only calls `FeatureMap.costs`; it never materializes an m x n matrix and
    makes no assumption about the cost structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.edges import EdgeSet
from .master import MasterSolution
from ..core.problem import InverseOTProblem


@dataclass(frozen=True)
class ViolationScan:
    """
    CN: 一次全边扫描的结果：违例候选 V、违例数与最大违反量。
    EN: Result of one full-edge scan: violation candidates V, violation count, and the
        maximum violation.
    """

    rows: np.ndarray
    cols: np.ndarray
    violation_count: int
    max_violation: float
    scanned_edges: int

    @property
    def has_violations(self) -> bool:
        return int(self.violation_count) > 0 or self.rows.size > 0


def _merge_topk(
    current_values: np.ndarray,
    current_indices: np.ndarray,
    new_values: np.ndarray,
    new_indices: np.ndarray,
    *,
    gamma: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([current_values, new_values], axis=1)
    indices = np.concatenate([current_indices, new_indices], axis=1)
    order = np.argsort(-values, axis=1, kind="stable")[:, : int(gamma)]
    return np.take_along_axis(values, order, axis=1), np.take_along_axis(indices, order, axis=1)


def _tile_shape(n_source: int, n_target: int, *, max_tile_elements: int) -> tuple[int, int]:
    block_rows = min(int(n_source), 256)
    block_cols = min(int(n_target), max(1, int(max_tile_elements) // max(1, block_rows)))
    return max(1, block_rows), max(1, block_cols)


def scan_violations(
    problem: InverseOTProblem,
    master: MasterSolution,
    *,
    excluded: EdgeSet,
    tol: float,
    gamma: int = 1,
    max_tile_elements: int = 1 << 20,
) -> ViolationScan:
    """
    CN: 扫描所有不在 `excluded` 中的边，返回满足 sigma + delta > tol 的候选。
    EN: Scan every edge not in `excluded` and return the candidates with
        sigma + delta > tol.
    """
    if int(gamma) < 1:
        raise ValueError("gamma must be >= 1")
    n_source = int(problem.n_source)
    n_target = int(problem.n_target)
    theta = np.asarray(master.theta, dtype=np.float64)
    source_dual = np.asarray(master.source_dual, dtype=np.float64)
    target_dual = np.asarray(master.target_dual, dtype=np.float64)
    delta = float(master.delta)

    row_values = np.full((n_source, int(gamma)), -np.inf, dtype=np.float64)
    row_indices = np.full((n_source, int(gamma)), -1, dtype=np.int64)
    col_values = np.full((n_target, int(gamma)), -np.inf, dtype=np.float64)
    col_indices = np.full((n_target, int(gamma)), -1, dtype=np.int64)

    violation_count = 0
    max_violation = -np.inf
    scanned = 0
    block_rows, block_cols = _tile_shape(n_source, n_target, max_tile_elements=int(max_tile_elements))

    for row_start in range(0, n_source, block_rows):
        row_stop = min(n_source, row_start + block_rows)
        rows = np.arange(row_start, row_stop, dtype=np.int64)
        local_row_values = row_values[row_start:row_stop]
        local_row_indices = row_indices[row_start:row_stop]
        for col_start in range(0, n_target, block_cols):
            col_stop = min(n_target, col_start + block_cols)
            cols = np.arange(col_start, col_stop, dtype=np.int64)
            grid_rows = np.repeat(rows, cols.size)
            grid_cols = np.tile(cols, rows.size)
            costs = problem.features.costs(theta, grid_rows, grid_cols)
            sigma = source_dual[grid_rows] + target_dual[grid_cols] - costs
            keys = grid_rows * n_target + grid_cols
            inside = excluded.contains(keys)
            sigma = np.where(inside, -np.inf, sigma)
            scanned += int(keys.size - int(np.count_nonzero(inside)))

            filtered = sigma + delta
            finite = np.isfinite(filtered)
            if bool(finite.any()):
                max_violation = max(max_violation, float(np.max(filtered[finite])))
            violated = filtered > float(tol)
            violation_count += int(np.count_nonzero(violated))
            candidate = np.where(violated, sigma, -np.inf).reshape(rows.size, cols.size)

            tile_row_values, tile_row_indices = _tile_topk(candidate, cols, gamma=int(gamma))
            local_row_values, local_row_indices = _merge_topk(
                local_row_values, local_row_indices, tile_row_values, tile_row_indices, gamma=int(gamma)
            )

            tile_col_values, tile_col_indices = _tile_topk(candidate.T, rows, gamma=int(gamma))
            updated_col_values, updated_col_indices = _merge_topk(
                col_values[col_start:col_stop],
                col_indices[col_start:col_stop],
                tile_col_values,
                tile_col_indices,
                gamma=int(gamma),
            )
            col_values[col_start:col_stop] = updated_col_values
            col_indices[col_start:col_stop] = updated_col_indices
        row_values[row_start:row_stop] = local_row_values
        row_indices[row_start:row_stop] = local_row_indices

    candidate_rows: list[np.ndarray] = []
    candidate_cols: list[np.ndarray] = []
    valid_rows = row_values > -np.inf
    if bool(valid_rows.any()):
        owners = np.broadcast_to(np.arange(n_source, dtype=np.int64)[:, None], row_values.shape)
        candidate_rows.append(owners[valid_rows])
        candidate_cols.append(row_indices[valid_rows])
    valid_cols = col_values > -np.inf
    if bool(valid_cols.any()):
        owners = np.broadcast_to(np.arange(n_target, dtype=np.int64)[:, None], col_values.shape)
        candidate_rows.append(col_indices[valid_cols])
        candidate_cols.append(owners[valid_cols])
    if candidate_rows:
        rows_out = np.concatenate(candidate_rows)
        cols_out = np.concatenate(candidate_cols)
    else:
        rows_out = np.zeros(0, dtype=np.int64)
        cols_out = np.zeros(0, dtype=np.int64)
    return ViolationScan(
        rows=np.ascontiguousarray(rows_out, dtype=np.int64),
        cols=np.ascontiguousarray(cols_out, dtype=np.int64),
        violation_count=int(violation_count),
        max_violation=(-np.inf if max_violation == -np.inf else float(max_violation)),
        scanned_edges=int(scanned),
    )


def _tile_topk(
    values: np.ndarray,
    global_indices: np.ndarray,
    *,
    gamma: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 在 tile 上取每行（值为行）的 top-gamma，返回值和全局索引。
    EN: Take each row's top-gamma on the tile and return values with global indices.
    """
    count = values.shape[0]
    if count < 1:
        empty_values = np.zeros((0, int(gamma)), dtype=np.float64)
        empty_indices = np.zeros((0, int(gamma)), dtype=np.int64)
        return empty_values, empty_indices
    order = np.argsort(-values, axis=1, kind="stable")[:, : int(gamma)]
    best_values = np.take_along_axis(values, order, axis=1)
    indices = np.broadcast_to(global_indices[None, :], values.shape)
    best_indices = np.take_along_axis(indices, order, axis=1)
    return best_values, np.ascontiguousarray(best_indices, dtype=np.int64)


__all__ = [
    "ViolationScan",
    "scan_violations",
]
