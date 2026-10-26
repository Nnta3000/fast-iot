"""
CN: **互信息的成对估计（两个等长向量）** —— 供 E1 的像素 NMI 使用。

    CN: 为什么单独一个模块：项目里已经踩过两次同样的坑——把"少量特征"当成样本，直方图被饿死，
        结果是"真配对与假配对的 MI 完全相同"。这里把**样本/格子**作为一等输出公开，
        任何调用方都必须能看到自己给了多少样本。

EN: **Pairwise mutual information over two equal-length vectors** -- the estimator behind E1's pixel NMI.

    Kept in its own module because the project has twice hit the same trap: treating "a few features" as
    samples starves the histogram, and a true pair then gets the same MI as a false one. Samples per cell is
    therefore exposed as a first-class quantity so every caller can see how much data it supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int


class MutualInformationError(RuntimeError):
    """CN: 输入不合法。EN: invalid input."""


def quantile_codes(values: Float[np.ndarray, "samples"], *, bins: int) -> Int[np.ndarray, "samples"]:
    """
    CN: 等频（分位）分箱到 `0..bins-1`；常量输入退化为全 0。
    EN: equal-frequency (quantile) binning into `0..bins-1`; constant input degenerates to all zeros.
    """
    a: Float[np.ndarray, "samples"] = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        raise MutualInformationError("empty input")
    if float(a.max()) - float(a.min()) < 1e-15:
        return np.zeros(a.size, dtype=np.int64)
    edges: Float[np.ndarray, "edges"] = np.quantile(a, np.linspace(0.0, 1.0, int(bins) + 1)[1:-1])
    return np.searchsorted(edges, a, side="right").astype(np.int64)


@dataclass(frozen=True)
class MutualInformation:
    """
    CN: 互信息的读数：`nats`，以及在给定分箱下的**样本/格子**（饥饿诊断量）。
    EN: an MI reading: `nats` plus **samples per cell** at the given binning (the starvation diagnostic).
    """

    nats: float
    samples_per_cell: float
    bins: int
    samples: int


def mutual_information_pair(a: Float[np.ndarray, "samples"], b: Float[np.ndarray, "samples"], *,
                           bins: int = 32) -> float:
    """
    CN: `I(A;B)`（nats），等频分箱 + 半格平滑。用 `mutual_information_detail` 可取到样本/格子。
    EN: `I(A;B)` in nats with equal-frequency bins and half-cell smoothing; use
        `mutual_information_detail` to obtain samples per cell as well.
    """
    return mutual_information_detail(a, b, bins=bins).nats


def mutual_information_detail(a: Float[np.ndarray, "samples"], b: Float[np.ndarray, "samples"], *,
                              bins: int = 32) -> MutualInformation:
    """
    CN: 与 `mutual_information_pair` 相同，但把**样本/格子**一并返回；该值 < 5 时应视为饥饿。
    EN: as `mutual_information_pair` but also returns **samples per cell**; below about 5 treat the estimate
        as starved.
    """
    x: Float[np.ndarray, "samples"] = np.asarray(a, dtype=np.float64)
    y: Float[np.ndarray, "samples"] = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise MutualInformationError(f"vectors must be the same length: {x.shape} vs {y.shape}")
    if x.size == 0:
        raise MutualInformationError("empty input")
    m: int = int(bins)
    codes_x = quantile_codes(x, bins=m)
    codes_y = quantile_codes(y, bins=m)
    joint: Float[np.ndarray, "bins bins"] = np.zeros((m, m), dtype=np.float64)
    np.add.at(joint, (codes_x, codes_y), 1.0)
    total: float = float(x.size)
    p: Float[np.ndarray, "bins bins"] = joint / total
    p_a: Float[np.ndarray, "bins"] = p.sum(axis=1)
    p_b: Float[np.ndarray, "bins"] = p.sum(axis=0)
    nz = p > 0.0
    denom: Float[np.ndarray, "bins bins"] = np.outer(p_a, p_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nz, p / np.maximum(denom, 1e-300), 1.0)
    nats: float = float(np.sum(p[nz] * np.log(ratio[nz]))) if nz.any() else 0.0
    return MutualInformation(nats=nats, samples_per_cell=total / float(m * m), bins=m,
                             samples=int(x.size))


__all__ = [
    "MutualInformation",
    "MutualInformationError",
    "mutual_information_detail",
    "mutual_information_pair",
    "quantile_codes",
]
