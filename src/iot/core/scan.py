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
) -> "ViolationScan":
    """
    CN: 用指定后端计算 `sigma = f + g - c(theta)` 并在 `excluded` 之外找违反边。
        三个后端的返回类型相同，因此调用方不必关心用的是哪一个。
    EN: compute `sigma = f + g - c(theta)` on the edges outside `excluded` with the requested backend.
        All three return the same type, so callers do not care which one ran.

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


__all__ = ["ScanBackend", "maximum_slack", "scan_for_violations"]
