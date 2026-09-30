"""
CN: 逆 OT 的特征张量 Phi 及其代价算子 `c_ij(theta) = sum_k theta_k Phi_ij^k`。
EN: Feature tensors Phi for inverse OT and their cost operator
    `c_ij(theta) = sum_k theta_k Phi_ij^k`.

CN: Phi 从不以 (m, n, K) 稠密形式保存；每个实现只按请求的边计算。
EN: Phi is never stored densely as (m, n, K); each implementation evaluates only the
    requested edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


def _index_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64).reshape(-1)
    if array.size < 1:
        raise ValueError(f"{name} must contain at least one index")
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return np.ascontiguousarray(array, dtype=np.int64)


def _linear_costs(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    CN: 对给定边的特征矩阵做 `features @ theta`，即线性代价。
    EN: Apply the linear cost operator `features @ theta` on the requested edges.
    """
    phi = np.asarray(features, dtype=np.float64)
    value = np.asarray(theta, dtype=np.float64).reshape(-1)
    if phi.ndim != 2:
        raise ValueError(f"edge features must be 2D; got shape={phi.shape}")
    if phi.shape[1] != value.size:
        raise ValueError(
            f"theta has length {value.size} but features have {phi.shape[1]} columns"
        )
    return np.ascontiguousarray(phi @ value, dtype=np.float64)


@runtime_checkable
class FeatureMap(Protocol):
    """
    CN: 线性特征代价 `c_ij(theta) = <Phi_ij, theta>` 的协议。
    EN: Protocol for a linear feature cost `c_ij(theta) = <Phi_ij, theta>`.
    """

    @property
    def n_source(self) -> int:
        """CN: 源点数 m。EN: Number of source points m."""
        ...

    @property
    def n_target(self) -> int:
        """CN: 目标点数 n。EN: Number of target points n."""
        ...

    @property
    def n_features(self) -> int:
        """CN: 特征数 K，即 theta 的维数。EN: Number of features K, i.e. len(theta)."""
        ...

    def features(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """
        CN: 返回请求边的特征，形状 (E, K)，E 为边数。
        EN: Return features for the requested edges with shape (E, K).
        """
        ...

    def costs(self, theta: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """CN: 返回请求边的 `c_ij(theta)`，形状 (E,)。EN: Return `c_ij(theta)` on edges, shape (E,)."""
        ...


@dataclass(frozen=True)
class DiagonalWeightFeatures:
    """
    CN: 逐坐标加权平方欧氏代价 `c_ij(theta) = sum_k theta_k (x_ik - y_jk)^2`。
    EN: Per-coordinate weighted squared Euclidean cost
        `c_ij(theta) = sum_k theta_k (x_ik - y_jk)^2`.

    CN: 这是 v1 与 HELLO-OT 双线性 runtime 精确对应的特征族，用作有真值实验。
    EN: This is the v1 feature family that matches the HELLO-OT bilinear runtime
        exactly, which makes it the natural ground-truth family.
    """

    source_points: np.ndarray
    target_points: np.ndarray

    def __post_init__(self) -> None:
        source = np.ascontiguousarray(np.asarray(self.source_points, dtype=np.float64))
        target = np.ascontiguousarray(np.asarray(self.target_points, dtype=np.float64))
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source_points and target_points must be 2D")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source_points and target_points must share a feature dimension")
        if source.shape[0] < 1 or target.shape[0] < 1:
            raise ValueError("source_points and target_points must be non-empty")
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "target_points", target)

    @property
    def n_source(self) -> int:
        return int(self.source_points.shape[0])

    @property
    def n_target(self) -> int:
        return int(self.target_points.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.source_points.shape[1])

    def features(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        row_index = _index_array(rows, name="rows")
        col_index = _index_array(cols, name="cols")
        difference = self.source_points[row_index] - self.target_points[col_index]
        return np.ascontiguousarray(difference * difference, dtype=np.float64)

    def costs(self, theta: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return _linear_costs(self.features(rows, cols), theta)


@dataclass(frozen=True)
class DenseFeatureTensor:
    """
    CN: 显式 (m, n, K) 特征张量，仅用于小规模测试与交叉验证。
    EN: Explicit (m, n, K) feature tensor, for small tests and cross-checks only.
    """

    tensor: np.ndarray

    def __post_init__(self) -> None:
        value = np.ascontiguousarray(np.asarray(self.tensor, dtype=np.float64))
        if value.ndim != 3:
            raise ValueError("feature tensor must be 3D with shape (m, n, K)")
        if value.shape[0] < 1 or value.shape[1] < 1 or value.shape[2] < 1:
            raise ValueError("feature tensor dimensions must be non-empty")
        object.__setattr__(self, "tensor", value)

    @property
    def n_source(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def n_target(self) -> int:
        return int(self.tensor.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.tensor.shape[2])

    def features(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        row_index = _index_array(rows, name="rows")
        col_index = _index_array(cols, name="cols")
        return np.ascontiguousarray(self.tensor[row_index, col_index, :], dtype=np.float64)

    def costs(self, theta: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return _linear_costs(self.features(rows, cols), theta)


@dataclass(frozen=True)
class BilinearPSDFeatures:
    """
    CN: 双线性-PSD 代价 `c_ij(theta) = -x_i^T A y_j`，`theta = vec(A)`，K = d * d。
    EN: Bilinear-PSD cost `c_ij(theta) = -x_i^T A y_j` with `theta = vec(A)`, K = d * d.

    CN: 特征张量 Phi_ij 是外积 `-x_i y_j^T` 的展平；代价对 theta 线性，但 Theta 是非线性的
        谱面体，因此需要 SDP master。
    EN: The feature row Phi_ij is the flattened outer product `-x_i y_j^T`; the cost is
        linear in theta, but Theta is the non-linear spectrahedron, so an SDP master is
        required.
    """

    source_points: np.ndarray
    target_points: np.ndarray

    def __post_init__(self) -> None:
        source = np.ascontiguousarray(np.asarray(self.source_points, dtype=np.float64))
        target = np.ascontiguousarray(np.asarray(self.target_points, dtype=np.float64))
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source_points and target_points must be 2D")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source_points and target_points must share a feature dimension")
        if source.shape[0] < 1 or target.shape[0] < 1:
            raise ValueError("source_points and target_points must be non-empty")
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "target_points", target)

    @property
    def point_dimension(self) -> int:
        return int(self.source_points.shape[1])

    @property
    def n_source(self) -> int:
        return int(self.source_points.shape[0])

    @property
    def n_target(self) -> int:
        return int(self.target_points.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.source_points.shape[1]) * int(self.source_points.shape[1])

    def features(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        row_index = _index_array(rows, name="rows")
        col_index = _index_array(cols, name="cols")
        source = self.source_points[row_index]
        target = self.target_points[col_index]
        outer = -np.einsum("ek,el->ekl", source, target, optimize=True)
        return np.ascontiguousarray(outer.reshape(source.shape[0], self.n_features), dtype=np.float64)

    def costs(self, theta: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        value = np.asarray(theta, dtype=np.float64).reshape(-1)
        dim = self.point_dimension
        if value.size != dim * dim:
            raise ValueError(f"theta must have length {dim * dim}; got {value.size}")
        row_index = _index_array(rows, name="rows")
        col_index = _index_array(cols, name="cols")
        source = self.source_points[row_index]
        target = self.target_points[col_index]
        matrix = value.reshape(dim, dim)
        return np.ascontiguousarray(
            -np.einsum("ek,kl,el->e", source, matrix, target, optimize=True), dtype=np.float64
        )


def dense_cost_matrix(
    feature_map: FeatureMap,
    theta: np.ndarray,
    *,
    max_elements: int = 50_000_000,
) -> np.ndarray:
    """
    CN: 为验证/小规模实验构造完整 (m, n) 代价矩阵，并设置元素上限防止失控。
    EN: Build the full (m, n) cost matrix for validation/small experiments, guarded by
        an element cap.
    """
    m = int(feature_map.n_source)
    n = int(feature_map.n_target)
    if m * n > int(max_elements):
        raise ValueError(
            f"dense cost matrix would have {m * n} elements, above max_elements={max_elements}"
        )
    rows = np.repeat(np.arange(m, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), m)
    return np.ascontiguousarray(feature_map.costs(theta, rows, cols).reshape(m, n), dtype=np.float64)


__all__ = [
    "BilinearPSDFeatures",
    "DenseFeatureTensor",
    "DiagonalWeightFeatures",
    "FeatureMap",
    "dense_cost_matrix",
]
