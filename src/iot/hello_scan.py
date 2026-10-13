"""
CN: 逆 OT 分离 oracle 的快速（torch/GPU）后端：复用 HELLO-OT 的分块双向违例扫描
    `hello_ot.kernels.torch_scan.bidirectional_violation_scan`。这是 `src/iot` 的第三个
    （也是最后一个）允许 import `hello_ot` 的适配器。
EN: Fast (torch/GPU) backend for the inverse-OT separation oracle, reusing HELLO-OT's blockwise
    bidirectional violation scan. This is the third and last `src/iot` adapter allowed to
    import `hello_ot`.

CN: 适用范围 / Scope
--------------------
只有能写成内积形式的特征族才能用：对 `DiagonalWeightFeatures`，
    c_ij(theta) = sum_k theta_k (x_ik - y_jk)^2
                = sum_k theta_k x_ik^2 + sum_k theta_k y_jk^2
                  - 2 <sqrt(theta) x_i, sqrt(theta) y_j>,
这正是 `score_family="inner_product"`、`dot_scale=2` 的代价。其他族退回 CPU 扫描。
EN: Only inner-product feature families are supported: for `DiagonalWeightFeatures`,
    c_ij(theta) = sum_k theta_k (x_ik - y_jk)^2 is `source_offset + target_offset -
    dot_scale * <source, target>` with `source = sqrt(theta) x`, `target = sqrt(theta) y`,
    `dot_scale = 2`. Other families fall back to the CPU scan.

CN: 正确性 / Correctness
------------------------
HELLO 扫描返回所有边上 `sigma > tol - delta` 的逐行/逐列 top-gamma 候选，外加全边最大
    `sigma`。由于可行 master 在 F 上满足 `sigma + delta <= 0`（S 上 `sigma = 0`），被违反的边
    必然不在 F（sigma > 0），因此按 F 过滤候选即可，且过滤后的候选与 CPU 扫描一致。若过滤后
    为空，则回退到 CPU 扫描一次，以取得权威的停机判定与真实 maxviol。
EN: The scan returns per-row/col top-gamma candidates with `sigma > tol - delta` plus the global
    max `sigma`. A feasible master has `sigma + delta <= 0` on `F` (and `sigma = 0` on `S`), so
    violated edges have `sigma > 0` and lie off `F`; filtering candidates by `F` therefore
    matches the CPU scan. If nothing survives, we fall back to the CPU scan once to get the
    authoritative stop decision and the true max violation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .edges import EdgeSet
from .features import DiagonalWeightFeatures, FeatureMap
from .master import MasterSolution
from .oracle import ViolationScan, scan_violations
from .problem import InverseOTError, InverseOTProblem


def fast_scan_supported(features: FeatureMap) -> bool:
    """CN: 该特征族是否有快速（内积）扫描。EN: Whether the family has a fast inner-product scan."""
    return isinstance(features, DiagonalWeightFeatures)


def _load_torch_scan() -> Any:
    """
    CN: 惰性加载 HELLO-OT 的 torch 扫描；缺失时抛出带安装提示的错误。
    EN: Lazily load HELLO-OT's torch scan; raise with an install hint when missing.
    """
    try:
        from hello_ot.kernels.torch_scan import bidirectional_violation_scan
    except ImportError as error:  # pragma: no cover - environment dependent
        raise InverseOTError(
            "the fast oracle scan requires `hello_ot` (and torch); install HELLO-OT or use "
            "scan_backend='numpy'."
        ) from error
    return bidirectional_violation_scan


def scan_violations_fast(
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
    CN: 用 HELLO 的 torch/GPU 分块扫描计算违例集合与逐行/逐列 top-gamma 候选；不支持时回退 CPU。
    EN: Compute the violated set and per-row/col top-gamma candidates with HELLO's torch/GPU
        blockwise scan; fall back to CPU when unsupported.
    """
    if not fast_scan_supported(problem.features):
        return scan_violations(problem, master, excluded=excluded, tol=float(tol), gamma=int(gamma))
    if int(gamma) < 1:
        raise ValueError("gamma must be >= 1")

    bidirectional_violation_scan = _load_torch_scan()
    features = problem.features
    assert isinstance(features, DiagonalWeightFeatures)
    source = np.ascontiguousarray(features.source_points, dtype=np.float64)
    target = np.ascontiguousarray(features.target_points, dtype=np.float64)
    theta = np.ascontiguousarray(np.asarray(master.theta, dtype=np.float64).reshape(-1))
    root = np.sqrt(np.clip(theta, 0.0, None))
    scaled_source = np.ascontiguousarray(source * root[None, :], dtype=np.float64)
    scaled_target = np.ascontiguousarray(target * root[None, :], dtype=np.float64)
    source_offset = np.ascontiguousarray((source * source) @ theta, dtype=np.float64)
    target_offset = np.ascontiguousarray((target * target) @ theta, dtype=np.float64)

    # CN: F 里的边（尤其 S 上 sigma=0 的边）可能占据 top-k 名额，把阈值附近（sigma<0）的真实
    #     违例边挤掉。由可行性，F\S 上的 sigma <= -delta < tol-delta，永不入选；只有 S 上的
    #     sigma=0 会污染。因此按「支撑行/列的最大度数」多取即可，过滤 F 后截到 top-gamma。
    # EN: Edges in F can occupy top-k slots; by feasibility only support edges (sigma=0) can
    #     contaminate, since F\S edges have sigma <= -delta < tol-delta. So over-fetch by the max
    #     support degree per row/col, then filter F and trim to top-gamma.
    n_source = int(problem.n_source)
    n_target = int(problem.n_target)
    support_rows, support_cols, _support_values = problem.support()
    support_rows = np.asarray(support_rows, dtype=np.int64).reshape(-1)
    support_cols = np.asarray(support_cols, dtype=np.int64).reshape(-1)
    sup_row = int(np.max(np.bincount(support_rows, minlength=n_source))) if support_rows.size else 0
    sup_col = int(np.max(np.bincount(support_cols, minlength=n_target))) if support_cols.size else 0
    topk = int(gamma) + max(sup_row, sup_col)

    scan = bidirectional_violation_scan(
        source_points=scaled_source,
        target_points=scaled_target,
        source_offset=source_offset,
        target_offset=target_offset,
        source_dual=master.source_dual,
        target_dual=master.target_dual,
        score_family="inner_product",
        cost_type="l2",
        dot_scale=2.0,
        topk=int(topk),
        theta=float(tol) - float(master.delta),
        device=str(device),
        max_tile_bytes=int(max_tile_bytes),
    )
    rows = np.asarray(scan.rows.detach().cpu().numpy(), dtype=np.int64)
    cols = np.asarray(scan.cols.detach().cpu().numpy(), dtype=np.int64)
    if rows.size:
        keys = rows * n_target + cols
        keep = ~excluded.contains(keys)
        rows = rows[keep]
        cols = cols[keep]
    if rows.size == 0:
        # CN: 没有候选 -> 停机；用 CPU 扫描给出权威的 maxviol（仅此一次）。
        # EN: No candidates -> stop; use one CPU scan for the authoritative max violation.
        return scan_violations(problem, master, excluded=excluded, tol=float(tol), gamma=int(gamma))

    costs = features.costs(theta, rows, cols)
    sigma = master.source_dual[rows] + master.target_dual[cols] - costs

    def _top_indices(owner: np.ndarray, gamma_count: int) -> np.ndarray:
        """CN: 每个 owner 的 sigma top-gamma 的下标。EN: Indices of the top-gamma sigma per owner."""
        order = np.lexsort((-sigma, owner))
        sorted_owner = owner[order]
        starts = np.r_[0, np.nonzero(sorted_owner[1:] != sorted_owner[:-1])[0] + 1]
        counts = np.diff(np.r_[starts, sorted_owner.size])
        rank = np.arange(sorted_owner.size) - np.repeat(starts, counts)
        return order[rank < int(gamma_count)]

    row_sel = _top_indices(rows, int(gamma))
    col_sel = _top_indices(cols, int(gamma))
    selected = np.unique(np.concatenate([row_sel, col_sel]))
    rows = np.ascontiguousarray(rows[selected], dtype=np.int64)
    cols = np.ascontiguousarray(cols[selected], dtype=np.int64)
    sigma = sigma[selected]
    max_violation = float(np.max(sigma) + float(master.delta))
    return ViolationScan(
        rows=rows,
        cols=cols,
        violation_count=int(rows.size),
        max_violation=max_violation,
        scanned_edges=int(problem.n_source) * n_target,
    )


__all__ = [
    "fast_scan_supported",
    "scan_violations_fast",
]
