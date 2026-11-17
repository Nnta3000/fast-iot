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
from jaxtyping import Float, Int


def _index_array(value: Int[np.ndarray, "edges"], *, name: str) -> Int[np.ndarray, "edges"]:
    array = np.asarray(value, dtype=np.int64).reshape(-1)
    if array.size < 1:
        raise ValueError(f"{name} must contain at least one index")
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return np.ascontiguousarray(array, dtype=np.int64)


def _linear_costs(features: Float[np.ndarray, "edges k"], theta: Float[np.ndarray, "k"]
                  ) -> Float[np.ndarray, "edges"]:
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

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        """
        CN: 返回请求边的特征，形状 (E, K)，E 为边数。
        EN: Return features for the requested edges with shape (E, K).
        """
        ...

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
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

    source_points: Float[np.ndarray, "m d"]
    target_points: Float[np.ndarray, "n d"]

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

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        row_index = _index_array(rows, name="rows")
        col_index = _index_array(cols, name="cols")
        difference = self.source_points[row_index] - self.target_points[col_index]
        np.square(difference, out=difference)
        return np.ascontiguousarray(difference, dtype=np.float64)

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
        return _linear_costs(self.features(rows, cols), theta)

@dataclass(frozen=True)
class SquaredDifferenceFeatures:
    """
    CN: 逐坐标加权平方差 —— 两个**点集**（不是同一组点）之间的特征族：

        `c_ij(theta) = sum_k theta_k (a_ik - b_jk)^2`,  `Phi_ij,k = (a_ik - b_jk)^2`

        CN: 与 `DiagonalWeightFeatures` 的**唯一**区别：源点与目标点是两组不同的点（描述子），
            而不是同一组点的自配对。它存在的理由是**内存**：特征按需计算 `(E, K)`，从不物化
            `(m, n, K)`。对角族在完整网格上需要 `m*n*K*8` 字节（6890 顶点、K=13 时是 4.9 GB），
            而这个类在任何规模下只物化被请求的边（`|S|` 或 `|F|`，几千条）。

    EN: Coordinate-wise weighted squared difference between two **different point sets**:

        `c_ij(theta) = sum_k theta_k (a_ik - b_jk)^2`,  `Phi_ij,k = (a_ik - b_jk)^2`

        The only difference from `DiagonalWeightFeatures` is that source and target are *two different*
        descriptor sets rather than one set paired with itself. It exists for **memory**: features are
        computed on demand as `(E, K)` and `(m, n, K)` is never materialised. The full grid costs
        `m*n*K*8` bytes (4.9 GB at 6890 vertices with K = 13); this class materialises only the requested
        edges (`|S|` or `|F|`, a few thousand) at any scale.
    """

    source_points: Float[np.ndarray, "m d"]
    target_points: Float[np.ndarray, "n d"]

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

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        row_index = _index_array(rows, name="rows")
        col_index = _index_array(cols, name="cols")
        difference = self.source_points[row_index] - self.target_points[col_index]
        return np.ascontiguousarray(difference * difference, dtype=np.float64)

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
        return _linear_costs(self.features(rows, cols), theta)

    def bilinear_embedding(self, theta: Float[np.ndarray, "k"]) -> "object":
        """
        EN: The bilinear embedding of the weighted squared difference, so the GPU scan and the full-grid
            slack use one matmul instead of materialising `(edges, K)` features:

            `sum_k theta_k (a_ik - b_jk)^2 = |sqrt(theta) * a_i|^2 + |sqrt(theta) * b_j|^2
                                            - 2 <sqrt(theta) * a_i, sqrt(theta) * b_j>`

            Without this method `bilinear_embedding(...)` returns `None` for this family and the torch scan
            silently fell back to computing costs in **NumPy** per tile and copying them to the device —
            measured at 3.3 s of a 36 s fit, all of it host work plus transfers.

        CN: 加权平方差的双线性嵌入：`sum_k θ_k (a-b)^2 = |√θ⊙a|² + |√θ⊙b|² − 2⟨√θ⊙a, √θ⊙b⟩`。
            没有这个方法时，torch 扫描会**静默**退回"逐块用 NumPy 算代价再拷到设备"，实测占 36 s 中的 3.3 s。
        """
        from ..backends.bilinear import BilinearEmbedding  # noqa: PLC0415 - avoids a circular import

        root = np.sqrt(np.maximum(np.asarray(theta, dtype=np.float64).reshape(-1), 0.0))
        source = np.ascontiguousarray(self.source_points * root)
        target = np.ascontiguousarray(self.target_points * root)
        return BilinearEmbedding(
            source=source,
            target=target,
            source_offset=(source * source).sum(axis=1),
            target_offset=(target * target).sum(axis=1),
            dot_scale=2.0,
        )


@dataclass(frozen=True)
class DenseFeatureTensor:
    """
    CN: 显式 (m, n, K) 特征张量，仅用于小规模测试与交叉验证。
    EN: Explicit (m, n, K) feature tensor, for small tests and cross-checks only.
    """

    tensor: Float[np.ndarray, "m n k"]

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

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        row_index = _index_array(rows, name="rows")
        col_index = _index_array(cols, name="cols")
        return np.ascontiguousarray(self.tensor[row_index, col_index, :], dtype=np.float64)

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
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

    source_points: Float[np.ndarray, "m d"]
    target_points: Float[np.ndarray, "n d"]

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

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        row_index = _index_array(rows, name="rows")
        col_index = _index_array(cols, name="cols")
        source = self.source_points[row_index]
        target = self.target_points[col_index]
        outer = -np.einsum("ek,el->ekl", source, target, optimize=True)
        return np.ascontiguousarray(outer.reshape(source.shape[0], self.n_features), dtype=np.float64)

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
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


@dataclass(frozen=True)
class QuadraticFeatures:
    """
    CN: 二次交叉项代价 `c_ij(theta) = sum_{a<=b} theta_ab (x_ia - y_ja)(x_ib - y_jb)`，
        K = d(d+1)/2。分块计算，从不 materialize (m, n, K) 张量。
    EN: Quadratic cross-term cost
        `c_ij(theta) = sum_{a<=b} theta_ab (x_ia - y_ja)(x_ib - y_jb)`, K = d(d+1)/2.
        Computed blockwise; never materialises an (m, n, K) tensor.
    """

    source_points: Float[np.ndarray, "m d"]
    target_points: Float[np.ndarray, "n d"]
    chunk_elements: int = 1 << 20

    def __post_init__(self) -> None:
        source = np.ascontiguousarray(np.asarray(self.source_points, dtype=np.float64))
        target = np.ascontiguousarray(np.asarray(self.target_points, dtype=np.float64))
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source_points and target_points must be 2D")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source_points and target_points must share the dimension")
        if source.shape[0] < 1 or target.shape[0] < 1:
            raise ValueError("source_points and target_points must be non-empty")
        if int(self.chunk_elements) < 1:
            raise ValueError("chunk_elements must be >= 1")
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "target_points", target)

    @property
    def point_dimension(self) -> int:
        """CN: 点维度 d。EN: Point dimension d."""
        return int(self.source_points.shape[1])

    @property
    def n_source(self) -> int:
        return int(self.source_points.shape[0])

    @property
    def n_target(self) -> int:
        return int(self.target_points.shape[0])

    @property
    def n_features(self) -> int:
        d = int(self.point_dimension)
        return d * (d + 1) // 2

    def pairs(self) -> list[tuple[int, int]]:
        """CN: 特征列顺序 (a, b), a <= b。EN: Feature column order (a, b), a <= b."""
        d = int(self.point_dimension)
        return [(a, b) for a in range(d) for b in range(a, d)]

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        row_index = _index_array(rows, name="rows")
        col_index = _index_array(cols, name="cols")
        difference = self.source_points[row_index] - self.target_points[col_index]
        columns = [difference[:, a] * difference[:, b] for a, b in self.pairs()]
        return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float64)

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
        value = np.asarray(theta, dtype=np.float64).reshape(-1)
        d = int(self.point_dimension)
        if value.size != self.n_features:
            raise ValueError(
                f"theta has length {value.size} but quadratic features have {self.n_features}"
            )
        # CN: c = diff^T M diff，M_aa = theta_aa，M_ab = theta_ab / 2（a != b）。
        # EN: c = diff^T M diff with M_aa = theta_aa and M_ab = theta_ab / 2 (a != b).
        matrix = np.zeros((d, d), dtype=np.float64)
        for index, (a, b) in enumerate(self.pairs()):
            if a == b:
                matrix[a, a] = value[index]
            else:
                matrix[a, b] = 0.5 * value[index]
                matrix[b, a] = 0.5 * value[index]
        row_index = np.asarray(rows, dtype=np.int64).reshape(-1)
        col_index = np.asarray(cols, dtype=np.int64).reshape(-1)
        out = np.empty(row_index.size, dtype=np.float64)
        step = max(1, int(self.chunk_elements))
        for start in range(0, row_index.size, step):
            stop = min(row_index.size, start + step)
            difference = self.source_points[row_index[start:stop]] - self.target_points[col_index[start:stop]]
            out[start:stop] = np.einsum("ea,ab,eb->e", difference, matrix, difference, optimize=True)
        return np.ascontiguousarray(out, dtype=np.float64)

    def bilinear_embedding(  # noqa: ANN201 - BilinearEmbedding (avoid import cycle)
            self, theta: Float[np.ndarray, "k"]):
        """
        CN: `O(m n d)` 快速路径：`c = (x-y)^T M (x-y) = a_i + b_j - 2<M x_i, y_j>`，
            其中 `M` 由 theta 给出，`a_i = x_i^T M x_i`，`b_j = y_j^T M y_j`。
        EN: `O(m n d)` fast path: `c = (x-y)^T M (x-y) = a_i + b_j - 2<M x_i, y_j>`, with `M` built from
            theta, `a_i = x_i^T M x_i`, `b_j = y_j^T M y_j`.
        """
        from ..backends.bilinear import BilinearEmbedding

        value = np.asarray(theta, dtype=np.float64).reshape(-1)
        if value.size != self.n_features:
            raise ValueError(
                f"theta has length {value.size} but quadratic features have {self.n_features}"
            )
        d = int(self.point_dimension)
        matrix = np.zeros((d, d), dtype=np.float64)
        for index, (a, b) in enumerate(self.pairs()):
            if a == b:
                matrix[a, a] = value[index]
            else:
                matrix[a, b] = 0.5 * value[index]
                matrix[b, a] = 0.5 * value[index]
        source = np.ascontiguousarray(self.source_points @ matrix, dtype=np.float64)   # M x_i (symmetric)
        target = np.ascontiguousarray(self.target_points, dtype=np.float64)            # y_j
        source_offset = np.ascontiguousarray(
            np.einsum("ia,ab,ib->i", self.source_points, matrix, self.source_points, optimize=True)
        )
        target_offset = np.ascontiguousarray(
            np.einsum("ja,ab,jb->j", self.target_points, matrix, self.target_points, optimize=True)
        )
        return BilinearEmbedding(source=source, target=target, source_offset=source_offset,
                                 target_offset=target_offset, dot_scale=2.0)


def dense_cost_matrix(
    feature_map: FeatureMap,
    theta: Float[np.ndarray, "k"],
    *,
    max_elements: int = 50_000_000,
) -> Float[np.ndarray, "m n"]:
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
    "ConcatFeatures",
    "SquaredDifferenceFeatures",
    "DenseFeatureTensor",
    "DiagonalWeightFeatures",
    "FeatureMap",
    "KernelMixtureFeatures",
    "QuadraticFeatures",
    "RadialProfileFeatures",
    "dense_cost_matrix",
]


# ---------------------------------------------------------------------------------------------------------
# EN: **kernel / dictionary axis.** Both families below add expressivity to the COST ONLY: they are ordinary
#     `FeatureMap`s, so the solver, scan, loop and master are untouched.
#
#     The shared idea: the cost is linear in `theta`, so any finite dictionary of pairwise functions is
#     admissible. For a PSD kernel the squared feature-space distance is
#     `k(x,x) + k(y,y) - 2 k(x,y)`; the first two terms depend on ONE point each and are absorbed by the OT
#     potentials (adding `u_i + v_j` leaves the optimal plan unchanged under fixed marginals), so the
#     pair-dependent part of a kernel cost is `-k(x_i, y_j)`. That is what `KernelMixtureFeatures` encodes.
#
# CN: **核 / 字典方向。** 以下两个族只增强**代价**：它们就是普通的 `FeatureMap`，求解器、扫描、循环与
#     master 完全不动。代价对 `theta` 线性，所以任何有限的成对函数字典都可用。对 PSD 核，特征空间平方距离为
#     `k(x,x)+k(y,y)-2k(x,y)`，前两项只依赖单个点、被 OT 势函数吸收，故成对部分就是 `-k(x_i,y_j)`。
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelMixtureFeatures:
    """
    EN: cost as a mixture of **pairwise kernels**: `c_ij(theta) = sum_r theta_r * (-k_r(x_i, y_j))`.

        Supported kernels, all normalised so their diagonal is 1 where that is meaningful:

          `"gaussian:sigma1,sigma2,..."`  `k = exp(-||x-y||^2 / (2 sigma^2))`, sigma in *data* units
          `"cosine"`                      `k = <x,y> / (||x|| ||y||)`, in [-1, 1]
          `"linear"`                      `k = <x,y>`
          `"polynomial:d,c"`              `k = (<x,y> + c)^d`

        A mixture of Gaussians at several bandwidths is the "learned radial profile of the distance" case: it
        can express costs that saturate or have a flat basin, which a single squared Euclidean cannot.

        Because the kernel is *positive semi-definite*, the cost is the pair-dependent part of a squared
        distance in the kernel's feature space; `theta` then weights those distances. **Nothing here requires
        the kernel to be PSD in practice** — the solver only ever sees a linear cost — but only the PSD case
        has the distance interpretation, and any non-PSD entry would make the "cost" a similarity rather than a
        distance. The constructors below are all PSD.

    CN: 代价是若干**成对核**的混合 `c_ij(theta) = sum_r theta_r * (-k_r(x_i,y_j))`。支持高斯（多个带宽）、
        余弦、线性、多项式。混合多个带宽可以表达饱和或平底的代价，这是平方欧氏做不到的。
    """

    source_points: Float[np.ndarray, "m d"]
    target_points: Float[np.ndarray, "n d"]
    kernel_specs: tuple[str, ...]

    def __post_init__(self) -> None:
        source = np.ascontiguousarray(np.asarray(self.source_points, dtype=np.float64))
        target = np.ascontiguousarray(np.asarray(self.target_points, dtype=np.float64))
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source_points and target_points must be 2D")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source_points and target_points must share a feature dimension")
        specs = _expand_kernel_specs(tuple(str(spec) for spec in self.kernel_specs))
        if not specs:
            raise ValueError("kernel_specs must name at least one kernel")
        for spec in specs:
            _parse_kernel_spec(spec)
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "target_points", target)
        object.__setattr__(self, "kernel_specs", specs)

    @property
    def n_source(self) -> int:
        return int(self.source_points.shape[0])

    @property
    def n_target(self) -> int:
        return int(self.target_points.shape[0])

    @property
    def n_features(self) -> int:
        return len(self.kernel_specs)

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        """
        EN: `phi_ij,r = -k_r(x_i, y_j)` for each requested edge. CN: 每条请求边的 `-k_r(x_i,y_j)`。
        """
        row_index = _index_array(rows, name="rows")
        col_index = _index_array(cols, name="cols")
        source = self.source_points[row_index]
        target = self.target_points[col_index]
        out = np.empty((source.shape[0], len(self.kernel_specs)), dtype=np.float64)
        for position, spec in enumerate(self.kernel_specs):
            out[:, position] = -_kernel_values(spec, source, target)
        return np.ascontiguousarray(out, dtype=np.float64)

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
        return _linear_costs(self.features(rows, cols), theta)


@dataclass(frozen=True)
class RadialProfileFeatures:
    """
    EN: **additive per-coordinate radial profiles** — a hat-function (linear spline) basis on |a_ik - b_jk|.

        Features are `phi_ij,(k,l) = h_l(|a_ik - b_jk|)`, where `h_l` is the `l`-th hat function on a grid of
        knots spanning `[0, extent]`, and `extent` is a per-coordinate scale (default: the 99th percentile of
        the absolute difference over a deterministic subsample, so the basis is not swamped by outliers).

        Why this is strictly more expressive than `DiagonalWeightFeatures`: the squared difference is the single
        function `h(t) = t^2`, so a weighted sum of squares is a **fixed** shape with a free weight. A hat basis
        lets the cost saturate (robust to outliers), stay flat (tolerant within a band), or be non-monotone —
        none of which a squared Euclidean can do.

        `n_features = dim * basis_size`, so with 13 descriptors and 8 basis functions this is 104 parameters.
        A large basis dilutes every coordinate in the simplex, so `theta -> uniform` is the failure mode to
        watch; report `L1(theta, uniform)` with every fit.

    CN: **逐坐标可加径向剖面** —— 在 `|a_ik - b_jk|` 上的帽函数（线性样条）基。`phi_ij,(k,l) = h_l(|a_ik-b_jk|)`。
        比 `DiagonalWeightFeatures`（只有 `t^2` 一种形状）表达力更强：可以饱和（抗离群）、可以平底（带内容差）、
        可以非单调。特征数为 `维度 × 基函数数`；基太大时 `theta` 趋于均匀，必须同时报告 `L1(theta, uniform)`。
    """

    source_points: Float[np.ndarray, "m d"]
    target_points: Float[np.ndarray, "n d"]
    basis_size: int = 8
    extent: Float[np.ndarray, "d"] | float | None = None
    sample_size: int = 4096
    seed: int = 0
    include_ramp: bool = True

    def __post_init__(self) -> None:
        source = np.ascontiguousarray(np.asarray(self.source_points, dtype=np.float64))
        target = np.ascontiguousarray(np.asarray(self.target_points, dtype=np.float64))
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source_points and target_points must be 2D")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source_points and target_points must share a feature dimension")
        if int(self.basis_size) < 2:
            raise ValueError("basis_size must be at least 2 (a single hat cannot span a range)")
        dimension = int(source.shape[1])
        extent = self.extent
        if extent is None:
            scale = _default_extent(source, target, sample_size=int(self.sample_size), seed=int(self.seed))
        else:
            scale = np.asarray(extent, dtype=np.float64).reshape(-1)
            if scale.size == 1:
                scale = np.full(dimension, float(scale[0]), dtype=np.float64)
            if scale.size != dimension:
                raise ValueError(f"extent must be scalar or length {dimension}")
            if np.any(scale <= 0.0):
                raise ValueError("extent entries must be positive")
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "target_points", target)
        object.__setattr__(self, "basis_size", int(self.basis_size))
        object.__setattr__(self, "extent", np.ascontiguousarray(scale, dtype=np.float64))

    @property
    def n_source(self) -> int:
        return int(self.source_points.shape[0])

    @property
    def n_target(self) -> int:
        return int(self.target_points.shape[0])

    @property
    def n_features(self) -> int:
        # EN: `+1` for the linear ramp when enabled -- see `__post_init__`.
        # CN: 启用线性斜坡时多一个特征。
        per_coordinate = int(self.basis_size) + (1 if self.include_ramp else 0)
        return int(self.source_points.shape[1]) * per_coordinate

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        """
        EN: hat-basis values of the normalised absolute difference, flattened as `(dim, basis)` -- plus an
            optional **linear ramp** `t` per coordinate.

            The ramp is not decoration. Hats form a partition of unity (`sum_l h_l(t) = 1` for every `t`), so with
            a **uniform** `theta` the cost collapses to the constant `dim / K` for every candidate pair and the
            parameter is unidentifiable at the very point the fit starts from. Measured before the fix:
            `cost_std = 1.5e-17` at uniform `theta`, and the fit never moved (`L1(uniform) = 0.0`). Adding the
            ramp makes the uniform point non-degenerate.

        CN: 帽函数取值加可选的**线性斜坡** `t`。斜坡不是装饰：帽函数构成单位分解，`theta` 均匀时代价对每对候选
            恒为 `dim/K`，参数在拟合起点就不可辨识（实测 `cost_std=1.5e-17`、`theta` 不动）。加斜坡后均匀点非退化。
        """
        row_index = _index_array(rows, name="rows")
        col_index = _index_array(cols, name="cols")
        difference = np.abs(self.source_points[row_index] - self.target_points[col_index])
        normalised = difference / np.asarray(self.extent, dtype=np.float64)[None, :]
        np.clip(normalised, 0.0, 1.0, out=normalised)
        basis = int(self.basis_size)
        per_coordinate = basis + (1 if self.include_ramp else 0)
        # EN: knot spacing `1/(basis-1)`; hat `l` peaks at knot `l` and vanishes at its neighbours.
        # CN: 节点间距 `1/(basis-1)`；第 `l` 个帽函数在节点 `l` 取 1，在相邻节点为 0。
        spacing = 1.0 / float(basis - 1)
        position = normalised / spacing
        lower = np.floor(position)
        fraction = position - lower
        out = np.zeros((difference.shape[0], difference.shape[1] * per_coordinate), dtype=np.float64)
        for coordinate in range(difference.shape[1]):
            base = coordinate * per_coordinate
            index = lower[:, coordinate].astype(np.int64)
            index = np.clip(index, 0, basis - 2)
            frac = fraction[:, coordinate]
            # EN: the two hats that are nonzero at this position.
            # CN: 该位置处非零的两个帽函数。
            out[np.arange(out.shape[0]), base + index] = 1.0 - frac
            out[np.arange(out.shape[0]), base + index + 1] = frac
            # EN: exact 1.0 at the last knot belongs to the last hat alone.
            # CN: 恰在最后一个节点时只属于最后一个帽函数。
            at_end = (lower[:, coordinate] >= basis - 1)
            if np.any(at_end):
                out[at_end, base + basis - 1] = 1.0
                out[at_end, base + basis - 2] = 0.0
            if self.include_ramp:
                # EN: the ramp breaks the partition-of-unity degeneracy at uniform theta.
                # CN: 斜坡打破均匀 theta 处的单位分解退化。
                out[:, base + basis] = normalised[:, coordinate]
        return np.ascontiguousarray(out, dtype=np.float64)

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
        return _linear_costs(self.features(rows, cols), theta)


def _expand_kernel_specs(specs: tuple[str, ...]) -> tuple[str, ...]:
    """
    EN: one spec per FEATURE. `"gaussian:0.5,1.0"` becomes `("gaussian:0.5", "gaussian:1.0")`.

        This is the fix for a real bug in the first version: a multi-sigma spec was counted as one feature by
        `n_features` while the evaluator silently used only the first sigma, so the cost and the simplex
        dimension disagreed.

    CN: 一个 spec 对应一个**特征**。`"gaussian:0.5,1.0"` 展开为两个。早期版本把它算作一个特征却只用第一个
        sigma，导致代价与单纯形维数不一致。
    """
    expanded: list[str] = []
    for spec in specs:
        name, arguments = _parse_kernel_spec(spec)
        if name == "gaussian" and len(arguments) > 1:
            expanded.extend(f"gaussian:{value:g}" for value in arguments)
        else:
            expanded.append(str(spec).strip())
    return tuple(expanded)


def _parse_kernel_spec(spec: str) -> tuple[str, tuple[float, ...]]:
    """
    EN: parse `"name:arg,arg"` into a name and its numeric arguments. CN: 解析 `"名字:参数,参数"`。
    """
    text = str(spec).strip()
    if ":" in text:
        name, raw = text.split(":", 1)
        arguments = tuple(float(part) for part in raw.split(",") if part.strip() != "")
    else:
        name, arguments = text, ()
    name = name.strip().lower()
    if name == "gaussian":
        if not arguments:
            raise ValueError("gaussian needs at least one sigma, e.g. 'gaussian:0.5,1.0'")
        if any(value <= 0.0 for value in arguments):
            raise ValueError("gaussian sigmas must be positive")
    elif name == "polynomial":
        if len(arguments) not in (1, 2):
            raise ValueError("polynomial needs 'degree' or 'degree,offset'")
        if arguments[0] < 1.0:
            raise ValueError("polynomial degree must be >= 1")
    elif name in ("cosine", "linear"):
        if arguments:
            raise ValueError(f"{name} takes no arguments")
    else:
        raise ValueError(f"unknown kernel {name!r}; known: gaussian, cosine, linear, polynomial")
    return name, arguments


def _kernel_values(spec: str, source: Float[np.ndarray, "edges d"],
                   target: Float[np.ndarray, "edges d"]) -> Float[np.ndarray, "edges"]:
    """
    EN: evaluate one kernel on aligned rows of `source` and `target`. CN: 在成对的行上求一个核的值。
    """
    name, arguments = _parse_kernel_spec(spec)
    if name == "gaussian":
        # EN: exactly one sigma per spec; `_expand_kernel_specs` guarantees it.
        # CN: 每个 spec 恰好一个 sigma，由 `_expand_kernel_specs` 保证。
        if len(arguments) != 1:
            raise ValueError(f"gaussian spec must carry exactly one sigma, got {spec!r}")
        difference = source - target
        squared = np.einsum("ij,ij->i", difference, difference)
        sigma = arguments[0]
        return np.exp(-squared / (2.0 * sigma * sigma))
    if name in ("cosine", "linear"):
        dot = np.einsum("ij,ij->i", source, target)
        if name == "linear":
            return dot
        source_norm = np.linalg.norm(source, axis=1)
        target_norm = np.linalg.norm(target, axis=1)
        denominator = source_norm * target_norm
        return np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0.0)
    degree = arguments[0]
    offset = arguments[1] if len(arguments) == 2 else 0.0
    dot = np.einsum("ij,ij->i", source, target)
    return np.power(dot + offset, degree)


def _default_extent(source: Float[np.ndarray, "m d"], target: Float[np.ndarray, "n d"], *,
                    sample_size: int, seed: int) -> Float[np.ndarray, "d"]:
    """
    EN: per-coordinate 99th percentile of |a - b| over a deterministic random subsample of pairs.

        Using the full `m*n` product would be `O(mn*d)` before anything else; the subsample keeps it cheap and
        the percentile keeps the basis from being stretched by a handful of extreme differences.

    CN: 在确定性抽样的成对差上取逐坐标 99 分位，作为该坐标的尺度；避免用全 `m*n` 乘积的开销和离群拉伸。
    """
    rng = np.random.default_rng(int(seed))
    rows = rng.integers(0, source.shape[0], size=int(sample_size))
    cols = rng.integers(0, target.shape[0], size=int(sample_size))
    difference = np.abs(source[rows] - target[cols])
    percentile = np.percentile(difference, 99.0, axis=0)
    # EN: a degenerate coordinate (all equal) would divide by zero; floor it.
    # CN: 退化坐标（全相等）会除零，加下限。
    return np.maximum(percentile, 1e-12)


@dataclass(frozen=True)
class ConcatFeatures:
    """
    EN: concatenate several `FeatureMap`s that share a source/target cardinality.

        This is the composition that makes a *cross-distance* mixture possible: a Gaussian kernel on `xyz`, another
        on local area, another on HKS, and so on. It matters because a mixture of kernels on the **same** distance
        is a monotone transform of that distance and therefore induces the same ranking of candidates — measured
        directly: fitting `theta` over five bandwidths of one distance left `theta` exactly uniform
        (`L1(uniform) = 0.0`), i.e. unidentifiable. Kernels on *different* signals are not rank-equivalent, so
        `theta` can express a real preference.

        `c_ij(theta) = sum_g <phi_ij^(g), theta^(g)>`, with `theta` concatenated in the same order.

    CN: 连接若干共享源/目标规模的 `FeatureMap`。这是**跨距离**混合的前提：同一距离上的多个核互为单调变换，
        排序相同，因此 `theta` 不可辨识（实测 `L1(uniform)=0.0`）。不同信号上的核不是秩等价的，`theta` 才能
        表达真实偏好。
    """

    maps: tuple[FeatureMap, ...]

    def __post_init__(self) -> None:
        maps = tuple(self.maps)
        if not maps:
            raise ValueError("ConcatFeatures needs at least one map")
        first = maps[0]
        for other in maps[1:]:
            if int(other.n_source) != int(first.n_source) or int(other.n_target) != int(first.n_target):
                raise ValueError("all maps must share n_source and n_target")
        object.__setattr__(self, "maps", maps)

    @property
    def n_source(self) -> int:
        return int(self.maps[0].n_source)

    @property
    def n_target(self) -> int:
        return int(self.maps[0].n_target)

    @property
    def n_features(self) -> int:
        return int(sum(int(item.n_features) for item in self.maps))

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        """EN: block-horizontal concatenation of the children. CN: 子特征的横向拼接。"""
        return np.ascontiguousarray(
            np.concatenate([item.features(rows, cols) for item in self.maps], axis=1), dtype=np.float64)

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
        return _linear_costs(self.features(rows, cols), theta)
