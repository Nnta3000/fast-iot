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
import os
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int


class FaustError(RuntimeError):
    """CN: FAUST 数据不可用或格式不符。EN: FAUST data missing or in an unsupported format."""


# EN: the repository root, derived from this file rather than written down. `parents[2]` because this module is
#     `experiments/harness/faust.py`.
# CN: 仓库根目录，由本文件位置推出而不是写死；`parents[2]` 是因为本模块在 `experiments/harness/faust.py`。
REPO_ROOT = Path(__file__).resolve().parents[2]


def default_registrations_root() -> Path:
    """
    EN: where the MPI-FAUST registration PLYs live, resolved **portably**.

        The old code wrote the absolute path of one particular machine
        (`/home/agent/Documents/fast-iot/data/faust_extract/...`) into every FAUST driver. On any other
        checkout -- a different user, a server, a `git worktree` (where the git-ignored `data/` is absent by
        design) -- that path simply does not exist, and the failure surfaced as
        `FaustError: missing mesh: /home/agent/...`, which reads like corrupt data rather than a wrong
        default.

        Resolution order:

          1. `FAUST_ROOT` if set -- one `export` fixes every driver and every queued job at once, without
             touching command lines;
          2. otherwise `<repo root>/data/faust_extract/MPI-FAUST/training/registrations`, which is where
             `data/README.md`'s drop point puts an extracted archive.

        This only chooses a **default**; every driver keeps its `--root` flag, and an explicit `--root` always
        wins. The directory is not checked here: `read_ply` names the path it tried and tells you to pass
        `--root` / set `FAUST_ROOT` when the whole root is absent.

    CN: MPI-FAUST 注册 PLY 的**可移植**默认位置。旧代码把某台机器的绝对路径写进每个 FAUST 驱动；换用户、换服务器或
        在 `git worktree` 里（被 git 忽略的 `data/` 本来就不存在）该路径不存在，报错却是 `missing mesh:
        /home/agent/...`，看起来像数据坏了。解析顺序：`FAUST_ROOT` > `<repo>/data/faust_extract/...`。这里只决定
        **默认值**，各驱动的 `--root` 仍然优先。
    """
    override = os.environ.get("FAUST_ROOT")
    if override:
        return Path(override).expanduser()
    return REPO_ROOT / "data" / "faust_extract" / "MPI-FAUST" / "training" / "registrations"


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
        # EN: separate "the whole root is wrong" from "this one file is missing". The first is a config error
        #     (a hardcoded default, a fresh worktree, a server) and saying "missing mesh" for it sends the reader
        #     looking for corrupt data.
        # CN: 区分"整个根目录不对"与"单个文件缺失"。前者是配置错误（写死的默认值、新的 worktree、别的机器），
        #     只报 "missing mesh" 会让人去查数据是否损坏。
        if not p.parent.is_dir():
            raise FaustError(
                f"registrations root does not exist: {p.parent}\n"
                f"  looked for: {p.name}\n"
                f"  fix: pass --root <dir> or set FAUST_ROOT=<dir>; the default is "
                f"{default_registrations_root()} (repo-relative, overridable)")
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


@dataclasses.dataclass(frozen=True)
class DescriptorScaler:
    """
    EN: descriptor standardisation fitted on a **fixed set of shapes** (the training shapes) and applied unchanged
        everywhere else.

        Why this exists: the pairwise proxy used `standardise(a, b)`, which pools the statistics of both sides of
        the *evaluated* pair. On a symmetric pair that is harmless; under a one-way protocol it means the target's
        statistics enter the source's feature map -- a transductive leak. Fitting once on training shapes removes
        it and makes `theta` transferable in the only sense that matters: the same transform on both sides.

    CN: 在**固定形状集合**（训练形状）上拟合的描述子标准化，其余地方原样应用。早先的成对代理用 `standardise(a,b)`
        把评估对两侧的统计合并；在对称配对下无害，但在单向协议下等于让测试侧统计进入源侧特征（传导式泄漏）。
        只在训练形状上拟合一次即可消除，并使 θ 在同一变换下真正可迁移。
    """

    mean: Float[np.ndarray, "k"]
    std: Float[np.ndarray, "k"]

    @classmethod
    def fit(cls, arrays: Sequence[Float[np.ndarray, "n k"]]) -> "DescriptorScaler":
        """
        EN: pooled mean/std over the supplied descriptor matrices. A zero-variance column gets `std = 1`, so the
            transform is always well defined (and that column becomes constant, which is visible, not silent).

        CN: 在给定描述子矩阵上取合并均值/标准差；零方差列取 `std = 1`，变换恒有定义（该列变为常数，可见而非静默）。
        """
        stacked = np.concatenate([np.asarray(array, dtype=np.float64) for array in arrays], axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std = np.where(std > 0.0, std, 1.0)
        return cls(mean=np.ascontiguousarray(mean), std=np.ascontiguousarray(std))

    def transform(self, desc: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "n k"]:
        """EN: `(desc - mean) / std`. CN: `(desc - mean) / std`。"""
        value = np.asarray(desc, dtype=np.float64)
        if value.shape[1] != self.mean.size:
            raise ValueError(f"expected {self.mean.size} descriptor columns, got {value.shape[1]}")
        return np.ascontiguousarray((value - self.mean) / self.std, dtype=np.float64)


def diameter(vertices: Float[np.ndarray, "vertices d"], *, chunk: int = 2048) -> float:
    """
    EN: the mesh **diameter** = the largest Euclidean distance between any two vertices, computed in row blocks so
        an `n x n` distance matrix is never materialised.

        This is the normaliser FAUST's own evaluation uses (geodesic error divided by the shape's diameter,
        reported x100). The study's original choice, `sqrt(surface area)`, is a different scale and must not be
        mixed with it in one table -- drivers report both.

    CN: 网格**直径** = 任意两顶点的最大欧氏距离；按行分块计算，不物化 `n x n` 距离矩阵。这是 FAUST 自身评估使用的
        归一化尺度（测地误差除以直径，再乘 100）。研究早先使用的 `sqrt(表面积)` 是另一个尺度，两者不能放进同一张表；
        driver 会同时报告两者。
    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("vertices must be a 2D array with at least two rows")
    best = 0.0
    for start in range(0, points.shape[0], int(chunk)):
        block = points[start:start + int(chunk)]
        difference = block[:, None, :] - points[None, :, :]
        squared = np.einsum("ijd,ijd->ij", difference, difference)
        best = max(best, float(squared.max()))
    return float(np.sqrt(best))


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


# CN: HKS 列使用的时间尺度。放在 harness 是因为描述子定义（下面）也要用它，而描述子是**一个**定义，
#     不再由 `correspondence.py` 和 `gates.py` 各写一遍。
# EN: the HKS time scales. It lives in the harness because the descriptor definition below needs it, and the
#     descriptor is now defined **once** instead of separately in `correspondence.py` and `gates.py`.
HKS_TIMES: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def descriptors(mesh: FaustMesh, *, with_hks: bool = True) -> Float[np.ndarray, "vertices k"]:
    """
    EN: per-vertex descriptors: centred/scaled xyz, log 1-ring area, log curvature, and (optionally) the 8
        HKS columns. `with_hks=False` gives the 5-column family, which the pre-flight gate uses to show that
        position alone fails.

        Why one function: `correspondence.py` and `gates.py` had grown separate copies, and the gate copy took a
        flag -- so the two could silently drift apart while claiming to describe the same vertex.

    CN: 顶点描述子：中心化缩放的 xyz、log 面积、log 曲率、可选 8 列 HKS。`with_hks=False` 是 5 列族（预检门
        用它证明只用位置会失败）。统一一份定义，避免两个模块各写一遍后悄悄漂移。
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    centre = 0.5 * (v.min(axis=0) + v.max(axis=0))
    scale = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    xyz = (v - centre) / max(scale, 1e-12)
    area = np.log1p(vertex_areas(mesh.vertices, mesh.faces))
    curv = np.log1p(curvature(mesh.vertices, mesh.faces))
    parts = [xyz, area[:, None], curv[:, None]]
    if with_hks:
        parts.append(heat_kernel_signature(mesh.vertices, mesh.faces, times=HKS_TIMES))
    return np.ascontiguousarray(np.concatenate(parts, axis=1))


def standardise(a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"]
                ) -> tuple[Float[np.ndarray, "n k"], Float[np.ndarray, "n k"]]:
    """
    EN: standardise with the mean/std of the **pooled** sample, the same constants for both shapes. Pooling
        matters: the simplex budget is scale-dependent, so per-shape standardisation would change the problem.

    CN: 用**合并样本**的均值/标准差标准化，两个形状用同一组常数（单纯形预算与尺度有关）。
    """
    pooled = np.concatenate([np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)], axis=0)
    mu = pooled.mean(axis=0)
    sd = pooled.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (np.asarray(a, dtype=np.float64) - mu) / sd, (np.asarray(b, dtype=np.float64) - mu) / sd


def pair_geometry(a_mesh: FaustMesh, b_mesh: FaustMesh, index: Int[np.ndarray, "n"]) -> tuple[
        Float[np.ndarray, "n k"], Float[np.ndarray, "n k"], Float[np.ndarray, "n n"], float,
        Int[np.ndarray, "n"]]:
    """
    EN: the four things every evaluation needs from one mesh pair: the standardised descriptors of both
        shapes, the geodesic matrix among the subsampled target vertices, the normalising scale, and the
        mirror map used for the left/right flip statistic.

        Returned together because the geodesic matrix is the expensive part (an `n`-source Dijkstra) and must
        be computed **once per pair**, then reused for every theta. Drivers that loop over thetas and recompute
        it per theta were paying that cost repeatedly.

    CN: 一对网格评估所需的四样东西：两侧标准化描述子、子采样目标顶点间的测地矩阵、归一化尺度、镜像映射。
        测地矩阵是最贵的部分（n 个源点的 Dijkstra），必须**每对算一次**、对所有 θ 复用。
    """
    desc_a = descriptors(a_mesh)[index]
    desc_b = descriptors(b_mesh)[index]
    da, db = standardise(desc_a, desc_b)
    geo = geodesic_matrix(b_mesh.vertices, b_mesh.faces, index)
    norm = float(np.sqrt(surface_area(b_mesh.vertices, b_mesh.faces)))
    mirror, _residual = mirror_indices(b_mesh.vertices, index)
    return da, db, geo, norm, np.asarray(mirror, dtype=np.int64)


def shared_index(mesh: FaustMesh, count: int, seed: int) -> Int[np.ndarray, "n"]:
    """
    EN: the same index subset on both shapes (sorted, so the bijection and the vertex order are preserved).
        `seed` selects which draw, which is how the study gets several vertex subsamples from one mesh pair.

    CN: 两个形状共享的下标子集（排序后返回，双射与顶点顺序得以保留）；`seed` 决定抽哪一组。
    """
    order = np.random.default_rng(int(seed)).permutation(int(mesh.n_vertices))[: int(count)]
    return np.sort(order)


__all__ = [
    "HKS_TIMES",
    "FaustError",
    "DescriptorScaler",
    "FaustMesh",
    "cotangent_laplacian",
    "diameter",
    "curvature",
    "descriptors",
    "edge_lengths",
    "geodesic_matrix",
    "heat_kernel_signature",
    "mirror_indices",
    "one_ring",
    "pair_geometry",
    "read_ply",
    "shared_index",
    "spectral_basis",
    "standardise",
    "surface_area",
    "training_registrations",
    "vertex_areas",
]
