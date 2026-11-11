"""
EN: **FAUST mesh I/O** -- a minimal reader for the registration PLY files, and the per-vertex quantities the
correspondence study needs.

    No mesh library is installed (`trimesh`, `potpourri3d`, `robust_laplacian`, `igl`, `open3d` are all
    absent), so this module parses the one format the MPI-FAUST archive actually uses and **refuses**
    anything else rather than pretending to be general:

        format binary_little_endian 1.0
        element vertex 6890   property float x / y / z
        element face 13776    property list uchar int vertex_indices

    It gives `read_ply`, `training_registrations`, `vertex_areas`, `one_ring` and `edge_lengths`. The
    quantities are deliberately cheap (no eigenbasis): they exist so the *data* can be checked before any
    descriptor or solver work depends on it.

CN: **FAUST 网格读写** —— 只用标准库解析 MPI-FAUST 的注册 PLY（本机没有装任何网格库），格式不符时**直接报错**
    而不是假装通用。提供 `read_ply`、`training_registrations`、`vertex_areas`、`one_ring`、`edge_lengths`。
"""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int


class FaustError(RuntimeError):
    """CN: FAUST 数据不可用或格式不符。EN: FAUST data missing or in an unsupported format."""


@dataclasses.dataclass(frozen=True)
class FaustMesh:
    """
    CN: 一个注册网格：顶点 `(n,3)` 与三角面 `(f,3)`，顶点顺序即对应关系。
    EN: one registration: vertices `(n,3)` and triangles `(f,3)`. **Vertex order is the correspondence.**
    """

    vertices: Float[np.ndarray, "vertices 3"]
    faces: Int[np.ndarray, "faces 3"]
    path: Path

    @property
    def n_vertices(self) -> int:
        """CN: 顶点数。EN: number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        """CN: 面数。EN: number of faces."""
        return int(self.faces.shape[0])


def read_ply(path: str | Path) -> FaustMesh:
    """
    CN: 读一个 MPI-FAUST 注册 PLY。只支持 `binary_little_endian` + `float x/y/z` + `list uchar int`；
        其他格式**抛错**，不做静默回退。
    EN: read one MPI-FAUST registration PLY. Only `binary_little_endian` with `float x/y/z` and a
        `list uchar int` face list is supported; anything else **raises**, never silently falls back.
    """
    p = Path(path)
    if not p.is_file():
        raise FaustError(f"missing mesh: {p}")
    with p.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise FaustError(f"{p}: header ended before 'end_header'")
            text = line.decode("ascii", errors="replace").strip()
            header.append(text)
            if text == "end_header":
                break
        blob = handle.read()
    return _parse(header, blob, p)


def _parse(header: list[str], blob: bytes, path: Path) -> FaustMesh:
    """CN: 头部与二进制的实际解析。EN: the actual header/binary parse."""
    if not header or header[0] != "ply":
        raise FaustError(f"{path}: not a PLY file")
    if not any(h.startswith("format binary_little_endian") for h in header):
        raise FaustError(f"{path}: only binary_little_endian is supported")
    n_vertices = n_faces = -1
    for h in header:
        if h.startswith("element vertex"):
            n_vertices = int(h.split()[-1])
        elif h.startswith("element face"):
            n_faces = int(h.split()[-1])
    if n_vertices <= 0 or n_faces <= 0:
        raise FaustError(f"{path}: header declares vertex={n_vertices} face={n_faces}")
    vertex_bytes = n_vertices * 3 * 4
    if len(blob) < vertex_bytes + n_faces * 13:
        raise FaustError(f"{path}: body is shorter than the header declares")
    vertices = np.frombuffer(blob, dtype="<f4", count=n_vertices * 3,
                             offset=0).reshape(n_vertices, 3).astype(np.float64)
    # CN: 每个面 = 1 字节计数（=3）+ 3 个 int32。EN: each face = 1 count byte (3) + three int32.
    offset = vertex_bytes
    faces = np.empty((n_faces, 3), dtype=np.int64)
    for i in range(n_faces):
        count = blob[offset]
        if count != 3:
            raise FaustError(f"{path}: face {i} has {count} vertices; only triangles are supported")
        faces[i] = struct.unpack_from("<3i", blob, offset + 1)
        offset += 1 + 12
    if faces.min() < 0 or faces.max() >= n_vertices:
        raise FaustError(f"{path}: face indices out of range")
    return FaustMesh(vertices=np.ascontiguousarray(vertices),
                     faces=np.ascontiguousarray(faces), path=path)


def training_registrations(root: str | Path) -> list[FaustMesh]:
    """
    CN: 读全部训练注册，按编号排序（`tr_reg_000.ply` …）。EN: all training registrations, ordered by id.
    """
    folder = Path(root)
    files = sorted(folder.glob("tr_reg_*.ply"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FaustError(f"no tr_reg_*.ply under {folder}")
    return [read_ply(f) for f in files]


def vertex_areas(vertices: Float[np.ndarray, "vertices 3"], faces: Int[np.ndarray, "faces 3"]
                 ) -> Float[np.ndarray, "vertices"]:
    """
    CN: 每顶点的 1-ring 面积（相邻三角面积之和除以 ?）。用作形状指纹与面积加权的权重。
    EN: per-vertex 1-ring area (sum of incident triangle areas). Used as a shape fingerprint and as an
        area weight.
    """
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    out = np.zeros(int(vertices.shape[0]), dtype=np.float64)
    for k in range(3):
        np.add.at(out, np.asarray(faces)[:, k], area)
    return out


def edge_lengths(vertices: Float[np.ndarray, "vertices 3"], faces: Int[np.ndarray, "faces 3"]
                 ) -> Float[np.ndarray, "edges"]:
    """CN: 网格全部无向边的长度（去重）。EN: lengths of the unique undirected edges."""
    f = np.asarray(faces, dtype=np.int64)
    pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    pairs.sort(axis=1)
    unique = np.unique(pairs, axis=0)
    v = np.asarray(vertices, dtype=np.float64)
    return np.asarray(np.linalg.norm(v[unique[:, 0]] - v[unique[:, 1]], axis=1), dtype=np.float64)


def one_ring(faces: Int[np.ndarray, "faces 3"], n_vertices: int
             ) -> tuple[Int[np.ndarray, "edges 2"], Int[np.ndarray, "vertices 1"],
                        Int[np.ndarray, "edges"]]:
    """
    CN: CSR 形式的 1-ring 邻接（`indptr`, `indices`），供局部描述子使用。
    EN: 1-ring adjacency in CSR form (`indptr`, `indices`), for local descriptors.
    """
    f = np.asarray(faces, dtype=np.int64)
    pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    pairs = np.sort(pairs, axis=1)
    symmetric = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    order = np.lexsort((symmetric[:, 1], symmetric[:, 0]))
    symmetric = symmetric[order]
    keep = np.ones(len(symmetric), dtype=bool)
    keep[1:] = np.any(symmetric[1:] != symmetric[:-1], axis=1)
    symmetric = symmetric[keep]
    counts = np.bincount(symmetric[:, 0], minlength=int(n_vertices))
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return symmetric, indptr.reshape(-1, 1), np.asarray(symmetric[:, 1], dtype=np.int64)


def spectral_basis(laplacian: "object", k: int = 60) -> tuple[Float[np.ndarray, "modes"],
                                                             Float[np.ndarray, "vertices modes"]]:
    """
    CN: 拉普拉斯的前 `k` 个特征对（升序）。用 `scipy.sparse.linalg.eigsh`（对称、稀疏）。
    EN: the first `k` eigenpairs of the Laplacian (ascending), via `scipy.sparse.linalg.eigsh`.
    """
    import scipy.sparse.linalg as sla  # noqa: PLC0415

    lap = laplacian
    n = int(lap.shape[0])
    kk = int(min(k, n - 2))
    # CN: 移位反演（`sigma` 略小于 0 以避开常数零空间），`which="LM"` 是移位后的最大幅值——
    #     用 `which="SM"` 配 `sigma` 会退化成极慢的 Lanczos（实测卡住）。
    # EN: shift-invert with `sigma` just below 0 to dodge the constant null space, and `which="LM"` for
    #     magnitude in the shifted operator. Pairing `which="SM"` with `sigma` degrades into a very slow
    #     Lanczos (measured: it hangs).
    values, vectors = sla.eigsh(lap, k=kk, sigma=-1e-4, which="LM")
    order = np.argsort(values)
    return (np.asarray(values[order], dtype=np.float64),
            np.asarray(vectors[:, order], dtype=np.float64))


def heat_kernel_signature(vertices: Float[np.ndarray, "vertices 3"], faces: Int[np.ndarray, "faces 3"],
                          *, times: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
                          modes: int = 60) -> Float[np.ndarray, "vertices times"]:
    """
    CN: 热核签名 `hks_t(x) = sum_m exp(-lambda_m t) phi_m(x)^2`（对数时间）。姿态不变，但左右对称不可分。
    EN: the heat kernel signature `hks_t(x) = sum_m exp(-lambda_m t) phi_m(x)^2` at log-spaced times.
        Pose-robust; blind to left/right symmetry (see the guideline §3.1).
    """
    lap = cotangent_laplacian(vertices, faces)
    values, vectors = spectral_basis(lap, k=int(modes))
    values = np.maximum(values, 0.0)
    out = np.empty((int(vectors.shape[0]), len(times)), dtype=np.float64)
    for j, t in enumerate(times):
        weights = np.exp(-values * float(t))
        out[:, j] = (vectors * vectors) @ weights
    # CN: 对数化以压缩动态范围，再交给标准化。EN: log to compress the range before standardisation.
    return np.log1p(np.maximum(out, 0.0) / max(float(np.median(out) if out.size else 1.0), 1e-12))




def mirror_indices(vertices: Float[np.ndarray, "vertices 3"], index: Int[np.ndarray, "sources"]
                   ) -> tuple[Int[np.ndarray, "sources"], Float[np.ndarray, "sources"]]:
    """
    CN: **镜像对应**：把点关于矢状面（x 的均值平面）反射，再在 `index` 子集里找最近邻。
        FAUST 的姿态并不左右对称（手臂高度不同），所以这不是精确的等距，只是一个用于**统计左右翻转**的代理。
        反射后最近邻与自身的欧氏距离也一并返回，便于设置"这个点是镜像点"的容差。
    EN: **mirror correspondence**: reflect points across the sagittal plane (the plane `x = mean(x)`) and take
        the nearest neighbour within `index`. FAUST poses are NOT left-right symmetric (arms differ), so this is
        not an exact isometry; it is a proxy used only to **count left/right flips**. Returns the mirror index
        per sampled vertex plus the reflection residual, so a tolerance can be set.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    v = np.asarray(vertices, dtype=np.float64)
    idx = np.asarray(index, dtype=np.int64)
    pts = v[idx]
    centre_x = float(v[:, 0].mean())
    reflected = pts.copy()
    reflected[:, 0] = 2.0 * centre_x - reflected[:, 0]
    tree = cKDTree(pts)
    residual, nearest = tree.query(reflected)
    return np.asarray(nearest, dtype=np.int64), np.asarray(residual, dtype=np.float64)




def geodesic_matrix(vertices: Float[np.ndarray, "vertices 3"], faces: Int[np.ndarray, "faces 3"],
                    index: Int[np.ndarray, "sources"]) -> Float[np.ndarray, "sources sources"]:
    """
    CN: 测地距离矩阵 `G[a, b]`：网格上从 `index[a]` 到 `index[b]` 的最短曲面距离（热方法，`potpourri3d`）。
    EN: geodesic distance matrix: shortest path **along the surface** from `index[a]` to `index[b]`, by the
        heat method (`potpourri3d`).
    """
    import potpourri3d as pp3d  # noqa: PLC0415

    idx = np.asarray(index, dtype=np.int64)
    solver = pp3d.MeshHeatMethodDistanceSolver(np.asarray(vertices, dtype=np.float64),
                                               np.asarray(faces, dtype=np.int64))
    out = np.empty((idx.size, idx.size), dtype=np.float64)
    for a, source in enumerate(idx):
        out[a] = solver.compute_distance(int(source))[idx]
    return out


def surface_area(vertices: Float[np.ndarray, "vertices 3"], faces: Int[np.ndarray, "faces 3"]) -> float:
    """CN: 网格总面积。EN: total surface area."""
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def cotangent_laplacian(vertices: Float[np.ndarray, "vertices 3"], faces: Int[np.ndarray, "faces 3"]
                        ) -> "object":
    """
    CN: 余切拉普拉斯（稀疏、对称、行和为 0）；有 `robust_laplacian` 时优先用它，否则退回本地实现。
    EN: the cotangent Laplacian (sparse, symmetric, zero row sums); prefers `robust_laplacian` when installed.
    """
    try:
        import robust_laplacian  # noqa: PLC0415

        value, _mass = robust_laplacian.mesh_laplacian(np.asarray(vertices, dtype=np.float64),
                                                       np.asarray(faces, dtype=np.int64))
        return value
    except ImportError:  # pragma: no cover - fallback for machines without the package
        pass
    import scipy.sparse as sparse  # noqa: PLC0415

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = int(v.shape[0])
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        i, j, k = f[:, a], f[:, b], f[:, c]
        e1 = v[j] - v[i]
        e2 = v[k] - v[i]
        cross = np.linalg.norm(np.cross(e1, e2), axis=1)
        cot = np.einsum("ij,ij->i", e1, e2) / np.maximum(cross, 1e-12)
        rows.append(j); cols.append(k); data.append(0.5 * cot)
        rows.append(k); cols.append(j); data.append(0.5 * cot)
    w = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)).tocsr()
    w = 0.5 * (w + w.T)
    degree = np.asarray(w.sum(axis=1)).reshape(-1)
    return (sparse.diags(degree) - w).tocsr()


def curvature(vertices: Float[np.ndarray, "vertices 3"], faces: Int[np.ndarray, "faces 3"]
              ) -> Float[np.ndarray, "vertices"]:
    """
    CN: 曲率代理 = 余切拉普拉斯作用在坐标上的向量长度。EN: curvature proxy = |L x| per vertex.
    """
    lap = cotangent_laplacian(vertices, faces)
    return np.asarray(np.linalg.norm(lap @ np.asarray(vertices, dtype=np.float64), axis=1),
                      dtype=np.float64)


__all__ = [
    "FaustError",
    "FaustMesh",
    "cotangent_laplacian",
    "curvature",
    "edge_lengths",
    "geodesic_matrix",
    "heat_kernel_signature",
    "mirror_indices",
    "one_ring",
    "read_ply",
    "spectral_basis",
    "surface_area",
    "training_registrations",
    "vertex_areas",
]
