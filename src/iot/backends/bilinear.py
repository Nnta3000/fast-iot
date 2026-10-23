"""
CN: **双线性（低秩）代价嵌入** —— 让 GPU 分离扫描对一大类族保持 `O(m n d)` 的 matmul 复杂度。

对形如 `c_ij(theta) = a_i(theta) + b_j(theta) - s * <u_i(theta), v_j(theta)>` 的代价，
`u_i, v_j` 是 r 维嵌入（`r` 通常 = 点维度 `d` 或因子数 `k`），扫描只需一次 matmul
`U @ V.T`，无需 materialize `(m, n, K)`。这就是 HELLO 的 `score_family="inner_product"` 形态。

    c_ij(theta) = a_i + b_j - s <u_i, v_j>
               <=>  score_ij = f_i + g_j - c_ij
                            = (f_i - a_i) + (g_j - b_j) + s <u_i, v_j>

因此扫描时把 `source_offset = f_i - a_i`、`target_offset = g_j - b_j`、`dot_scale = s`
传给 kernel 即可。

EN: **Bilinear (low-rank) cost embeddings** -- keep the GPU separation scan at `O(m n d)` matmul
complexity for a large class of families.

For a cost of the form `c_ij(theta) = a_i(theta) + b_j(theta) - s * <u_i(theta), v_j(theta)>` with
r-dimensional embeddings (r = point dimension or factor count), the scan is a single matmul
`U @ V.T` with no `(m, n, K)` tensor -- exactly HELLO's `score_family="inner_product"` shape.

用法 / EN: usage
    embed = bilinear_embedding(features, theta)      # -> BilinearEmbedding | None
    if embed is not None:  ... matmul scan ...
    else:                  ... generic O(m n K) scan ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.features import DiagonalWeightFeatures, FeatureMap


@dataclass(frozen=True)
class BilinearEmbedding:
    """
    CN: `c_ij(theta) = source_offset_i + target_offset_j - dot_scale * <source_i, target_j>` 的嵌入。
    EN: Embedding for `c_ij(theta) = source_offset_i + target_offset_j - dot_scale * <source_i, target_j>`.
    """

    source: np.ndarray      # (m, r)
    target: np.ndarray      # (n, r)
    source_offset: np.ndarray  # (m,)
    target_offset: np.ndarray  # (n,)
    dot_scale: float

    def __post_init__(self) -> None:
        source = np.ascontiguousarray(np.asarray(self.source, dtype=np.float64))
        target = np.ascontiguousarray(np.asarray(self.target, dtype=np.float64))
        source_offset = np.ascontiguousarray(np.asarray(self.source_offset, dtype=np.float64).reshape(-1))
        target_offset = np.ascontiguousarray(np.asarray(self.target_offset, dtype=np.float64).reshape(-1))
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("bilinear embeddings must be 2D")
        if source.shape[1] != target.shape[1]:
            raise ValueError("bilinear embeddings must share the embedding dimension r")
        if source_offset.size != source.shape[0] or target_offset.size != target.shape[0]:
            raise ValueError("offsets must match the embedding row counts")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "source_offset", source_offset)
        object.__setattr__(self, "target_offset", target_offset)
        object.__setattr__(self, "dot_scale", float(self.dot_scale))


@runtime_checkable
class BilinearEmbeddable(Protocol):
    """CN: 能提供双线性嵌入的特征族协议。EN: Protocol for families that expose a bilinear embedding."""

    def bilinear_embedding(self, theta: np.ndarray) -> BilinearEmbedding:
        """CN: 返回 `theta` 处的嵌入。EN: Return the embedding at `theta`."""
        ...


def features_support_bilinear(features: FeatureMap) -> bool:
    """CN: 该族是否支持 `O(m n d)` 的双线性扫描。EN: Whether the family supports the `O(m n d)` bilinear scan."""
    return isinstance(features, (DiagonalWeightFeatures, BilinearEmbeddable))


def _diagonal_embedding(features: DiagonalWeightFeatures, theta: np.ndarray) -> BilinearEmbedding:
    """CN: 对角族 `sum_k theta_k (x_ik-y_jk)^2 = a_i+b_j-2<sqrt(theta)x_i, sqrt(theta)y_j>`。
    EN: diagonal family `sum_k theta_k (x_ik-y_jk)^2 = a_i+b_j-2<sqrt(theta)x_i, sqrt(theta)y_j>`."""
    value = np.asarray(theta, dtype=np.float64).reshape(-1)
    root = np.sqrt(np.clip(value, 0.0, None))
    source = np.ascontiguousarray(features.source_points * root[None, :], dtype=np.float64)
    target = np.ascontiguousarray(features.target_points * root[None, :], dtype=np.float64)
    source_offset = np.ascontiguousarray((features.source_points * features.source_points) @ value)
    target_offset = np.ascontiguousarray((features.target_points * features.target_points) @ value)
    return BilinearEmbedding(source=source, target=target, source_offset=source_offset,
                             target_offset=target_offset, dot_scale=2.0)


def bilinear_embedding(features: FeatureMap, theta: np.ndarray) -> BilinearEmbedding | None:
    """
    CN: 若该族有双线性嵌入则返回之（`O(m n d)` 路径），否则返回 `None`（退回通用 `O(m n K)` 路径）。
    EN: Return the bilinear embedding when available (the `O(m n d)` path), else `None` (generic `O(m n K)`).
    """
    if isinstance(features, DiagonalWeightFeatures):
        return _diagonal_embedding(features, theta)
    if isinstance(features, BilinearEmbeddable):
        return features.bilinear_embedding(np.asarray(theta, dtype=np.float64).reshape(-1))
    return None


__all__ = [
    "BilinearEmbedding",
    "BilinearEmbeddable",
    "bilinear_embedding",
    "features_support_bilinear",
]
