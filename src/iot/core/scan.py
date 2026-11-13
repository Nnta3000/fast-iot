"""
CN: **两个循环共用的分离扫描** —— 把"按后端挑选扫描实现"这段逻辑从 `exact/loop.py` 抽出来，
    因为它是精确版与松弛版**唯一**需要完全一致的部分。此前它只活在精确循环里，松弛循环只调用
    numpy 版，于是松弛路径**静默地没有 GPU 扫描**——这是两份代码分叉造成的真实缺陷，不是风格问题。

    CN: 语义与 `exact/loop.py` 原有的实现**逐字一致**：同样的三个后端、同样的参数传递、
        `excluded=edges`、`tol`、`gamma`。抽出时不改任何行为，因此精确循环的数值必须逐位不变。

EN: **The separation scan shared by both loops** -- the "pick a backend" dispatch, lifted out of
    `exact/loop.py` because it is the one part the exact and relaxed loops must agree on exactly.
    It previously lived only in the exact loop, while the relaxed loop called the NumPy version, so the
    relaxed path **silently had no GPU scan** -- a real defect caused by the two copies diverging, not a
    matter of style.

    The semantics are **verbatim** those of `exact/loop.py`: the same three backends, the same argument
    passing, `excluded=edges`, `tol`, `gamma`. Nothing changes behaviour, so the exact loop's numbers
    must be bit-identical afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from jaxtyping import Float, Int

from .edges import EdgeSet
from .problem import InverseOTProblem
from ..backends.bilinear import bilinear_embedding
from ..exact.oracle import scan_violations

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..exact.oracle import ViolationScan

ScanBackend = Literal["numpy", "hello", "torch"]
"""CN: 三个合法后端。EN: the three legal backends."""


def scan_for_violations(
    problem: InverseOTProblem,
    dual: Any,
    *,
    excluded: EdgeSet,
    tol: float,
    gamma: int,
    scan_backend: ScanBackend = "numpy",
    scan_device: str = "auto",
    scan_max_tile_bytes: int = 1 << 26,
    max_tile_elements: int = 1 << 20,
    mass_epsilon: float | None = None,
) -> "ViolationScan":
    """
    CN: 用指定后端计算 `sigma = f + g - c(theta)` 并在 `excluded` 之外找违反边。
        三个后端的返回类型相同，因此调用方不必关心用的是哪一个。
    EN: compute `sigma = f + g - c(theta)` on the edges outside `excluded` with the requested backend.
        All three return the same type, so callers do not care which one ran.

    CN: `mass_epsilon` 只由 **numpy** 后端实现（逐点超出质量）；其他后端会忽略它并返回 `None`，
        调用方必须检查。加这个参数是为了 EOT 的停止判据。
    CN: `scan_backend` 只接受 `numpy` / `hello` / `torch`；非法值抛 `ValueError`，**绝不**静默回退——
        静默回退会让"我在 GPU 上"变成一句谎话。
    EN: `scan_backend` accepts only `numpy` / `hello` / `torch`; anything else raises `ValueError` and
        **never** falls back silently, because a silent fallback turns "running on GPU" into a lie.
    """
    if str(scan_backend) not in ("numpy", "hello", "torch"):
        raise ValueError("scan_backend must be 'numpy', 'hello' or 'torch'")
    if str(scan_backend) == "torch":
        from ..backends.generic_scan import scan_violations_torch  # noqa: PLC0415

        return scan_violations_torch(
            problem,
            dual,
            excluded=excluded,
            tol=float(tol),
            gamma=int(gamma),
            device=str(scan_device),
            max_tile_bytes=int(scan_max_tile_bytes),
        )
    if str(scan_backend) == "hello":
        from ..backends.hello_scan import scan_violations_fast  # noqa: PLC0415

        return scan_violations_fast(
            problem,
            dual,
            excluded=excluded,
            tol=float(tol),
            gamma=int(gamma),
            device=str(scan_device),
            max_tile_bytes=int(scan_max_tile_bytes),
        )
    return scan_violations(
        problem,
        dual,
        excluded=excluded,
        tol=float(tol),
        gamma=int(gamma),
        max_tile_elements=int(max_tile_elements),
        mass_epsilon=mass_epsilon,
    )


def maximum_slack(
    problem: InverseOTProblem,
    theta: Float[np.ndarray, "k"],
    source_dual: Float[np.ndarray, "m"],
    target_dual: Float[np.ndarray, "n"],
    rows: Int[np.ndarray, "edges"],
    cols: Int[np.ndarray, "edges"],
    *,
    block: int = 2_000_000,
) -> float:
    """
    CN: `max sigma`，**分块**计算，内存与边数无关。两个循环都需要它（松弛循环用于 `U` 的上界），
        此前松弛循环自带一份 `_sigma_max`，精确循环在没有违反边时直接用自己的扫描结果。
    EN: `max sigma`, computed **in blocks** so memory is independent of the edge count. Both loops need
        it (the relaxed loop for the upper bound `U`); the relaxed loop previously carried its own
        `_sigma_max` while the exact loop reused its scan result.
    """
    worst = -np.inf
    total = int(rows.size)
    for start in range(0, total, int(block)):
        stop = min(start + int(block), total)
        cost = np.asarray(problem.features.costs(theta, rows[start:stop], cols[start:stop]),
                          dtype=np.float64)
        value = np.asarray(source_dual, dtype=np.float64)[rows[start:stop]] + \
            np.asarray(target_dual, dtype=np.float64)[cols[start:stop]] - cost
        worst = max(worst, float(value.max()) if value.size else -np.inf)
    return float(worst)


def maximum_slack_grid(
    problem: InverseOTProblem,
    theta: Float[np.ndarray, "k"],
    source_dual: Float[np.ndarray, "m"],
    target_dual: Float[np.ndarray, "n"],
    *,
    block: int = 2_000_000,
) -> float:
    """
    CN: **整张网格上的 `max sigma`，不物化索引进数组**。按行块生成 `(i, j)`，所以内存上界是
        `block * K`（加上 `costs` 内部的临时量），与网格大小无关。

        CN: 与 `maximum_slack(problem, ..., rows, cols)` **取值完全相同**：命中同一组边，`max` 与顺序无关。
            存在的唯一理由是内存——`rows/cols` 两个 int64 数组是 `2*8` 字节/边，n=6890 时是 760 MB，
            而且会被保存整个求解过程。

    EN: `max sigma` over the **whole grid**, without materialising the index arrays. Row blocks of `(i, j)`
        are generated, so the bound is `block * K` (plus the temporaries inside `costs`), independent of the
        grid size.

        The value is **identical** to `maximum_slack(problem, ..., rows, cols)`: the same set of edges is
        visited and `max` does not depend on order. The only reason this exists is memory -- the two int64
        index arrays cost `2*8` bytes per edge (760 MB at n=6890) and used to be kept for the whole solve.
    """
    n_source = int(problem.n_source)
    n_target = int(problem.n_target)
    if n_source < 1 or n_target < 1:
        return -np.inf
    # CN: **嵌入快路**：族有双线性嵌入时按块做 `src @ tgt.T`，不物化 `(block, K)` 特征。n=2000 时每块
    #     从 208 MB 降到 16 MB，实测该函数 6.2 s -> 2.5 s（总拟合 28 s -> 22 s）。
    # EN: **embedding fast path**: with a bilinear embedding the block is one matmul and no `(block, K)`
    #     feature temporary -- 208 MB -> 16 MB per block at n=2000; measured 6.2 s -> 2.5 s for this function.
    embed = bilinear_embedding(problem.features, np.asarray(theta, dtype=np.float64))
    f = np.asarray(source_dual, dtype=np.float64)
    g = np.asarray(target_dual, dtype=np.float64)
    rows_per_block = max(1, min(n_source, int(block) // max(1, n_target)))
    worst = -np.inf
    for row_start in range(0, n_source, rows_per_block):
        row_stop = min(n_source, row_start + rows_per_block)
        if embed is not None:
            cost = (embed.source_offset[row_start:row_stop, None] + embed.target_offset[None, :]
                    - embed.dot_scale * (embed.source[row_start:row_stop] @ embed.target.T))
            value = f[row_start:row_stop, None] + g[None, :] - cost
            if value.size:
                worst = max(worst, float(value.max()))
            continue
        rows = np.arange(row_start, row_stop, dtype=np.int64)
        grid_rows = np.repeat(rows, n_target)
        grid_cols = np.tile(np.arange(n_target, dtype=np.int64), rows.size)
        cost = np.asarray(problem.features.costs(theta, grid_rows, grid_cols), dtype=np.float64)
        value = (f[grid_rows] + g[grid_cols] - cost)
        if value.size:
            worst = max(worst, float(value.max()))
    return worst


def maximum_slack_outside_grid(
    problem: InverseOTProblem,
    theta: Float[np.ndarray, "k"],
    source_dual: Float[np.ndarray, "m"],
    target_dual: Float[np.ndarray, "n"],
    excluded: EdgeSet,
    *,
    block: int = 2_000_000,
) -> float:
    """
    CN: 网格上**不属于 `excluded`** 的边的 `max sigma`，按行块生成索引。与"先物化全图键、再 `setdiff1d`"
        取值完全相同（命中同一组边，`max` 与顺序无关），但不分配 `m*n` 的键数组（n=6890 时 380 MB，
        且 `setdiff1d` 还要再排序一份）。

    EN: `max sigma` over the grid edges **not in `excluded`**, with indices generated per row block. Value is
        identical to "materialise all grid keys, then `setdiff1d`" (same edge set, `max` is order-invariant),
        but it does not allocate an `m*n` key array (380 MB at n=6890) nor a sorted copy for `setdiff1d`.
    """
    n_source = int(problem.n_source)
    n_target = int(problem.n_target)
    if n_source < 1 or n_target < 1:
        return -np.inf
    rows_per_block = max(1, min(n_source, int(block) // max(1, n_target)))
    # CN: **嵌入快路**：块内一次 matmul 得到整个块的代价，再用每块布尔掩码把 F 内的边置为 -inf。
    #     `excluded.rows` 按键排序，因此 `keys = row*n + col` 的排序等价于按行排序，可以直接 searchsorted
    #     切出本块涉及的 F 边——不再为整块生成 2M 个 int64 键并做 searchsorted。
    # EN: **embedding fast path**: one matmul per block, then a per-block boolean mask marks F edges as -inf.
    #     `excluded.rows` is sorted by key and `keys = row*n + col`, so it is also sorted by row; the block's F
    #     edges are a searchsorted slice -- no 2M int64 key array and no searchsorted over the whole block.
    embed = bilinear_embedding(problem.features, np.asarray(theta, dtype=np.float64))
    f = np.asarray(source_dual, dtype=np.float64)
    g = np.asarray(target_dual, dtype=np.float64)
    ex_rows = np.asarray(excluded.rows, dtype=np.int64)
    ex_cols = np.asarray(excluded.cols, dtype=np.int64)
    worst = -np.inf
    for row_start in range(0, n_source, rows_per_block):
        row_stop = min(n_source, row_start + rows_per_block)
        if embed is not None:
            lo = int(np.searchsorted(ex_rows, row_start, side="left"))
            hi = int(np.searchsorted(ex_rows, row_stop, side="left"))
            keep = np.ones((row_stop - row_start, n_target), dtype=bool)
            keep[ex_rows[lo:hi] - row_start, ex_cols[lo:hi]] = False
            if not bool(keep.any()):
                continue
            cost = (embed.source_offset[row_start:row_stop, None] + embed.target_offset[None, :]
                    - embed.dot_scale * (embed.source[row_start:row_stop] @ embed.target.T))
            value = f[row_start:row_stop, None] + g[None, :] - cost
            value[~keep] = -np.inf
            worst = max(worst, float(value.max()))
            continue
        rows = np.arange(row_start, row_stop, dtype=np.int64)
        grid_rows = np.repeat(rows, n_target)
        grid_cols = np.tile(np.arange(n_target, dtype=np.int64), rows.size)
        keys = grid_rows * n_target + grid_cols
        outside = ~np.asarray(excluded.contains(keys), dtype=bool)
        if not bool(outside.any()):
            continue
        r_o = grid_rows[outside]
        c_o = grid_cols[outside]
        cost = np.asarray(problem.features.costs(theta, r_o, c_o), dtype=np.float64)
        value = (f[r_o] + g[c_o] - cost)
        if value.size:
            worst = max(worst, float(value.max()))
    return worst


__all__ = [
    "ScanBackend",
    "maximum_slack",
    "maximum_slack_grid",
    "maximum_slack_outside_grid",
    "scan_for_violations",
]
