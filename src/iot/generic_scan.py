"""
CN: **通用 GPU/torch 分离扫描** —— 后端无关（CUDA 或 CPU 上的 torch），对**任意** `FeatureMap`
    （即任意线性于 theta 的代价）都能工作；当该族有双线性嵌入时自动走 `O(m n d)` 的 matmul 路径。

设计目标（与项目框架一致）：
  1. 基础路径对**任何** `c_ij(theta) = <Phi_ij, theta>` 都成立，最坏 `O(m n K)`，分块、线性内存；
  2. 有嵌入时（diagonal / quadratic / lowrank / PSD）自动用 matmul，降到 `O(m n d)`；
  3. 纯 torch 即可跑（CUDA 有则用 CUDA，否则 CPU），不要求 native 扩展；
  4. 只依赖 `torch`，不 import `hello_ot`（保持依赖边界）。

CN: 语义与 `oracle.scan_violations`（numpy 参考实现）**逐边一致**：F 内的边在 top-k 之前排除，
    且只有违例边（`sigma > tol`）参与候选；每个源行/目标列保留分数 top-`gamma` 的候选（并集）。
    `max_violation = max_全网格 sigma + delta`。

EN: **Generic GPU/torch separation scan** -- backend-agnostic (torch on CUDA or CPU), works for **any**
    `FeatureMap` (any cost linear in theta); when the family exposes a bilinear embedding it takes the
    `O(m n d)` matmul path automatically.

    Goals: (1) the base path holds for any `c_ij(theta) = <Phi_ij, theta>`, worst case `O(m n K)`,
    blockwise and linear-memory; (2) families with an embedding (diagonal / quadratic / lowrank / PSD)
    use the matmul path at `O(m n d)`; (3) pure torch works (CUDA when present, else CPU), no native
    extension required; (4) depends only on `torch`, never imports `hello_ot`.

    Semantics match `oracle.scan_violations` (the numpy reference) edge-for-edge: F edges are excluded
    before the top-k, only violated edges (`sigma > tol`) are candidates, each source row / target
    column keeps its top-`gamma` by score (union), and `max_violation = max_grid sigma + delta`.

用法 / EN: usage
    from iot.generic_scan import scan_violations_torch
    scan = scan_violations_torch(problem, master, excluded=edges, tol=1e-9, gamma=1, device="auto")
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .bilinear import bilinear_embedding
from .edges import EdgeSet
from .features import FeatureMap
from .master import MasterSolution
from .oracle import ViolationScan
from .problem import InverseOTProblem
from .torch_runtime import require_torch, resolve_torch_device


def _tile_shape(n_source: int, n_target: int, *, max_tile_bytes: int) -> tuple[int, int]:
    """CN: FP64 tile 形状，控制在字节预算内。EN: FP64 tile shape bounded by the byte budget."""
    tile_elements = max(1, int(max_tile_bytes) // 8)
    source_block = min(int(n_source), 256)
    target_block = min(int(n_target), max(1, tile_elements // max(1, source_block)))
    return max(1, source_block), max(1, target_block)


def _merge_topk(torch: Any, current_values: Any, current_indices: Any, new_values: Any,
                new_indices: Any, *, k: int) -> tuple[Any, Any]:
    """CN: 逐行合并 top-k。EN: Merge top-k per row."""
    values = torch.cat((current_values, new_values), dim=1)
    indices = torch.cat((current_indices, new_indices), dim=1)
    kept_values, order = torch.topk(values, k=int(k), dim=1, largest=True, sorted=True)
    return kept_values, torch.gather(indices, 1, order)


def _tile_keys(torch: Any, score: Any, s0: int, t0: int, n_target: int, device: Any) -> Any:
    """CN: tile 的全局 key（用于 F 判定）。EN: Global keys of the tile (for the F test)."""
    rows_idx = torch.arange(s0, s0 + score.shape[0], dtype=torch.int64, device=device)[:, None]
    cols_idx = torch.arange(t0, t0 + score.shape[1], dtype=torch.int64, device=device)[None, :]
    return rows_idx * int(n_target) + cols_idx


def _mask_tile(torch: Any, score: Any, excluded: EdgeSet, s0: int, t0: int, n_target: int,
               *, tol: float, device: Any) -> Any:
    """
    CN: 与 numpy 扫描一致：F 内的边置 -inf，且非违例边（sigma <= tol）也置 -inf。
    EN: Match the numpy scan: set F edges to -inf, and non-violated edges (sigma <= tol) to -inf too.
    """
    keys = _tile_keys(torch, score, s0, t0, n_target, device).reshape(-1).detach().cpu().numpy()
    inside = torch.as_tensor(excluded.contains(keys).reshape(score.shape), dtype=torch.bool, device=device)
    keep = (~inside) & (score > float(tol))
    return torch.where(keep, score, torch.full_like(score, -torch.inf))


def _max_over_non_f(torch: Any, score: Any, excluded: EdgeSet, s0: int, t0: int, n_target: int,
                    device: Any) -> Any:
    """CN: 非 F 边上的最大 sigma（用于 max_violation，与 numpy 一致）。EN: Max sigma over non-F edges (for max_violation)."""
    keys = _tile_keys(torch, score, s0, t0, n_target, device).reshape(-1).detach().cpu().numpy()
    inside = torch.as_tensor(excluded.contains(keys).reshape(score.shape), dtype=torch.bool, device=device)
    masked = torch.where(inside, torch.full_like(score, -torch.inf), score)
    return torch.amax(masked)


def _select_candidates(torch: Any, row_values: Any, row_indices: Any, col_values: Any, col_indices: Any,
                       n_source: int, n_target: int) -> tuple[np.ndarray, np.ndarray]:
    """CN: 从 top-γ 表收集候选（行 + 列并集）。EN: Collect candidates from the top-γ tables (rows + columns)."""
    row_mask = row_values > -torch.inf
    col_mask = col_values > -torch.inf
    device = row_values.device
    rows = torch.cat((
        torch.arange(n_source, dtype=torch.int64, device=device)[:, None].expand(-1, row_values.shape[1])[row_mask],
        col_indices[col_mask]))
    cols = torch.cat((
        row_indices[row_mask],
        torch.arange(n_target, dtype=torch.int64, device=device)[:, None].expand(-1, col_values.shape[1])[col_mask]))
    r = rows.detach().cpu().numpy().astype(np.int64)
    c = cols.detach().cpu().numpy().astype(np.int64)
    return r, c


def _dedupe(rows: np.ndarray, cols: np.ndarray, n_target: int) -> tuple[np.ndarray, np.ndarray]:
    """CN: 去重（按 key 排序）。EN: Deduplicate (sorted by key)."""
    if rows.size == 0:
        return rows, cols
    keys = rows * n_target + cols
    _, first = np.unique(keys, return_index=True)
    order = np.argsort(keys[first], kind="stable")
    idx = first[order]
    return np.ascontiguousarray(rows[idx], dtype=np.int64), np.ascontiguousarray(cols[idx], dtype=np.int64)


def _scan(
    problem: InverseOTProblem, master: MasterSolution, excluded: EdgeSet, *, tol: float, gamma: int,
    device: Any, max_tile_bytes: int,
) -> ViolationScan:
    """
    CN: 统一扫描核心：有双线性嵌入走 matmul，否则逐 tile 调 `features.costs`；两者分块遍历一致。
    EN: Unified scan core: matmul when a bilinear embedding exists, else per-tile `features.costs`;
        both share the same blockwise traversal.
    """
    torch = require_torch()
    features: FeatureMap = problem.features
    n_source, n_target = int(problem.n_source), int(problem.n_target)
    f = torch.as_tensor(np.asarray(master.source_dual, dtype=np.float64), dtype=torch.float64, device=device)
    g = torch.as_tensor(np.asarray(master.target_dual, dtype=np.float64), dtype=torch.float64, device=device)

    embed = bilinear_embedding(features, master.theta)
    if embed is not None:
        src = torch.as_tensor(embed.source, dtype=torch.float64, device=device)
        tgt = torch.as_tensor(embed.target, dtype=torch.float64, device=device)
        a_off = torch.as_tensor(embed.source_offset, dtype=torch.float64, device=device)
        b_off = torch.as_tensor(embed.target_offset, dtype=torch.float64, device=device)
        scale = float(embed.dot_scale)
    else:
        src = tgt = a_off = b_off = None
        scale = 0.0

    row_values = torch.full((n_source, int(gamma)), -torch.inf, dtype=torch.float64, device=device)
    row_indices = torch.full((n_source, int(gamma)), -1, dtype=torch.int64, device=device)
    col_values = torch.full((n_target, int(gamma)), -torch.inf, dtype=torch.float64, device=device)
    col_indices = torch.full((n_target, int(gamma)), -1, dtype=torch.int64, device=device)
    max_violation = torch.full((), -torch.inf, dtype=torch.float64, device=device)
    violation_count = torch.zeros((), dtype=torch.int64, device=device)
    src_block, tgt_block = _tile_shape(n_source, n_target, max_tile_bytes=int(max_tile_bytes))
    for s0 in range(0, n_source, src_block):
        s1 = min(n_source, s0 + src_block)
        local_rv, local_ri = row_values[s0:s1], row_indices[s0:s1]
        for t0 in range(0, n_target, tgt_block):
            t1 = min(n_target, t0 + tgt_block)
            if embed is not None:
                cost = a_off[s0:s1, None] + b_off[None, t0:t1] - scale * (src[s0:s1] @ tgt[t0:t1].T)
            else:
                rr = np.repeat(np.arange(s0, s1, dtype=np.int64), t1 - t0)
                cc = np.tile(np.arange(t0, t1, dtype=np.int64), s1 - s0)
                cost_np = np.asarray(features.costs(master.theta, rr, cc), dtype=np.float64).reshape(s1 - s0, t1 - t0)
                cost = torch.as_tensor(cost_np, dtype=torch.float64, device=device)
            score = f[s0:s1, None] + g[None, t0:t1] - cost
            max_violation = torch.maximum(
                max_violation, _max_over_non_f(torch, score, excluded, s0, t0, n_target, device))
            score = _mask_tile(torch, score, excluded, s0, t0, n_target, tol=float(tol), device=device)
            violation_count += torch.count_nonzero(score > -torch.inf)
            local_rv, local_ri = _merge_topk(
                torch, local_rv, local_ri, score,
                torch.arange(t0, t1, dtype=torch.int64, device=device)[None, :].expand(s1 - s0, -1), k=int(gamma))
            updated_cv, updated_ci = _merge_topk(
                torch, col_values[t0:t1], col_indices[t0:t1], score.T,
                torch.arange(s0, s1, dtype=torch.int64, device=device)[None, :].expand(t1 - t0, -1), k=int(gamma))
            col_values[t0:t1], col_indices[t0:t1] = updated_cv, updated_ci
        row_values[s0:s1], row_indices[s0:s1] = local_rv, local_ri

    rows, cols = _select_candidates(torch, row_values, row_indices, col_values, col_indices,
                                    n_source, n_target, )
    rows, cols = _dedupe(rows, cols, n_target)
    maxviol = float(max_violation.detach().cpu().item()) + float(master.delta)
    return ViolationScan(
        rows=rows,
        cols=cols,
        violation_count=int(violation_count.detach().cpu().item()),
        max_violation=float(maxviol),
        scanned_edges=int(n_source) * int(n_target) - int(excluded.size),
    )


def scan_violations_torch(
    problem: InverseOTProblem,
    master: MasterSolution,
    *,
    excluded: EdgeSet,
    tol: float,
    gamma: int = 1,
    device: str = "auto",
    max_tile_bytes: int = 64 * 1024 * 1024,
) -> ViolationScan:
    """
    CN: 通用 torch 分离扫描：有嵌入走 `O(m n d)` matmul，否则通用 `O(m n K)`；CUDA/CPU 均可。
    EN: Generic torch separation scan: `O(m n d)` matmul when an embedding exists, else generic
        `O(m n K)`; works on CUDA or CPU.
    """
    if int(gamma) < 1:
        raise ValueError("gamma must be >= 1")
    dev = resolve_torch_device(device)
    return _scan(problem, master, excluded, tol=float(tol), gamma=int(gamma),
                 device=dev, max_tile_bytes=int(max_tile_bytes))


__all__ = ["scan_violations_torch"]
