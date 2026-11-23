"""
CN: 低秩随机代价族：`c_ij(theta) = sum_k theta_k U_ik V_jk`，只存因子 U, V（O(nK)）。
    它实现 `FeatureMap` 协议，从不在内存里materialize (m, n, K) 张量。
EN: Low-rank random cost family: `c_ij(theta) = sum_k theta_k U_ik V_jk`, storing only the
    factors U, V (O(nK)). It implements the `FeatureMap` protocol and never materialises an
    (m, n, K) tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LowRankFeatures:
    """
    CN: 因子分解代价的特征映射：`Phi_ij^k = U_ik V_jk`。
    EN: Feature map of a factorised cost: `Phi_ij^k = U_ik V_jk`.
    """

    source_factors: np.ndarray
    target_factors: np.ndarray

    def __post_init__(self) -> None:
        source = np.ascontiguousarray(np.asarray(self.source_factors, dtype=np.float64))
        target = np.ascontiguousarray(np.asarray(self.target_factors, dtype=np.float64))
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("factors must be 2D (n_points, K)")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source and target factors must share K")
        if source.shape[0] < 1 or target.shape[0] < 1:
            raise ValueError("factors must be non-empty")
        object.__setattr__(self, "source_factors", source)
        object.__setattr__(self, "target_factors", target)

    @property
    def n_source(self) -> int:
        return int(self.source_factors.shape[0])

    @property
    def n_target(self) -> int:
        return int(self.target_factors.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.source_factors.shape[1])

    @property
    def stored_floats(self) -> int:
        """CN: 实际存储的浮点数（无 m*n*K 张量）。EN: Floats actually stored (no m*n*K tensor)."""
        return int(self.source_factors.size + self.target_factors.size)

    def features(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        row_index = np.asarray(rows, dtype=np.int64).reshape(-1)
        col_index = np.asarray(cols, dtype=np.int64).reshape(-1)
        return np.ascontiguousarray(
            self.source_factors[row_index] * self.target_factors[col_index], dtype=np.float64
        )

    def costs(self, theta: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        value = np.asarray(theta, dtype=np.float64).reshape(-1)
        return np.ascontiguousarray(self.features(rows, cols) @ value, dtype=np.float64)

    def bilinear_embedding(self, theta: np.ndarray):  # noqa: ANN201 - BilinearEmbedding (avoid import cycle)
        """
        CN: `sum_k theta_k U_ik V_jk = <sqrt(theta)<U_i, sqrt(theta)<V_j>`，`O(m n k)` matmul。
        EN: `sum_k theta_k U_ik V_jk = <sqrt(theta)<U_i, sqrt(theta)<V_j>`, an `O(m n k)` matmul.
        """
        from iot.backends.bilinear import BilinearEmbedding

        value = np.asarray(theta, dtype=np.float64).reshape(-1)
        root = np.sqrt(np.clip(value, 0.0, None))
        source = np.ascontiguousarray(self.source_factors * root[None, :], dtype=np.float64)
        target = np.ascontiguousarray(self.target_factors * root[None, :], dtype=np.float64)
        zeros_s = np.zeros(source.shape[0], dtype=np.float64)
        zeros_t = np.zeros(target.shape[0], dtype=np.float64)
        # CN: c_ij = <u_i, v_j> = 0 + 0 - (-1) * <u_i, v_j>，故 dot_scale = -1。
        # EN: c_ij = <u_i, v_j> = 0 + 0 - (-1) * <u_i, v_j>, hence dot_scale = -1.
        return BilinearEmbedding(source=source, target=target, source_offset=zeros_s,
                                 target_offset=zeros_t, dot_scale=-1.0)


__all__ = ["LowRankFeatures"]
