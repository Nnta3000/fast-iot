"""
EN: **Tests for the FAUST loader, the mesh descriptors and the left/right flip counter.**

    Why these exist: the FAUST work introduced a hand-written PLY parser, a hand-written cotangent Laplacian
    and a hand-written mirror map, none of which any other test touches. Each is the kind of code that fails
    quietly -- a PLY parser that reads the wrong offset produces a plausible-looking mesh, and a mirror map
    against the wrong axis still returns *some* permutation, so a flip count would be wrong without ever
    raising. Every check below is designed so that the failure mode it guards against is observable.

    Conventions:
      - the synthetic mesh is closed and **exactly** symmetric about `x = 0`, so the mirror is known;
      - geometry comparisons use tolerances tied to the mesh scale, never relative error near zero.

CN: **FAUST 读取器、网格描述子与左右翻转计数器的测试**。手写的 PLY 解析、余切拉普拉斯与镜像映射都不被其他测试
    覆盖，而它们的失败是**静默**的（读错偏移量仍然给出一个像样的网格；镜像轴弄错仍然返回某个置换）。
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from iot.core.features import SquaredDifferenceFeatures
from iot_experiments.harness.faust import (
    cotangent_laplacian,
    curvature,
    geodesic_matrix,
    mirror_indices,
    read_ply,
    surface_area,
    vertex_areas,
)


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """CN: 写一个 binary_little_endian 的三角网格 PLY（与 FAUST 相同格式）。EN: write a small PLY."""
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(np.asarray(vertices, dtype="<f4").tobytes())
        for face in np.asarray(faces, dtype=np.int64):
            handle.write(struct.pack("<B3i", 3, int(face[0]), int(face[1]), int(face[2])))


def _octahedron() -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 正八面体闭网格。关于 x=0 **精确**对称，所以镜像映射有已知答案。
    EN: a closed octahedron, **exactly** symmetric about `x = 0`, so the mirror map has a known answer.
    """
    vertices = np.asarray([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    faces = np.asarray([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ], dtype=np.int64)
    return vertices, faces


def test_read_ply_round_trips_vertices_and_faces(tmp_path: Path) -> None:
    """
    CN: 写→读必须逐位一致（顶点按 float32 存储，故比较用 float32 精度）。面必须是整数精确相等。
    EN: write -> read must reproduce the vertices (stored as float32, so compared at float32 precision) and
        the faces exactly.
    """
    vertices, faces = _octahedron()
    path = tmp_path / "octa.ply"
    _write_ply(path, vertices, faces)

    mesh = read_ply(path)

    assert mesh.n_vertices == len(vertices)
    assert mesh.n_faces == len(faces)
    assert np.array_equal(mesh.faces, faces)
    assert np.allclose(mesh.vertices, vertices.astype(np.float32).astype(np.float64), atol=1e-6)


def test_read_ply_rejects_a_truncated_body(tmp_path: Path) -> None:
    """
    CN: 一个**静默**的失败模式：文件被截断时，解析器不能"尽量读出一些顶点"就算成功。
    EN: a silent failure mode: a truncated file must not "read what it can" and succeed.
    """
    vertices, faces = _octahedron()
    path = tmp_path / "short.ply"
    _write_ply(path, vertices, faces)
    blob = path.read_bytes()
    path.write_bytes(blob[: len(blob) // 2])
    with pytest.raises(Exception):
        read_ply(path)


def test_surface_area_and_vertex_areas_agree_with_hand_computation() -> None:
    """
    CN: 正八面体边长 `sqrt(2)`，8 个面 ⇒ 总面积 `4*sqrt(3)`；每个顶点由 4 个面共享。
    EN: the octahedron has edge `sqrt(2)` and 8 faces, so the area is `4*sqrt(3)`; each vertex is shared by
        four faces.
    """
    vertices, faces = _octahedron()
    assert surface_area(vertices, faces) == pytest.approx(4.0 * np.sqrt(3.0), rel=1e-12)
    areas = vertex_areas(vertices, faces)
    assert areas.sum() == pytest.approx(3.0 * surface_area(vertices, faces), rel=1e-12)
    assert np.allclose(areas, areas[0], atol=1e-12)


def test_cotangent_laplacian_is_symmetric_with_zero_row_sums() -> None:
    """
    CN: 余切拉普拉斯必须对称、行和为 0，且常数向量在核里。这是 HKS 正确性的前提。
    EN: the cotangent Laplacian must be symmetric with zero row sums, and constants must be in its kernel.
        That is the precondition for HKS being meaningful.
    """
    vertices, faces = _octahedron()
    lap = cotangent_laplacian(vertices, faces)
    dense = np.asarray(lap.todense())
    assert np.allclose(dense, dense.T, atol=1e-12)
    assert np.allclose(dense.sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(dense @ np.ones(dense.shape[0]), 0.0, atol=1e-12)


def test_curvature_is_scale_free_and_finite() -> None:
    """
    CN: 曲率的量级必须随网格尺度缩放，且在闭合网格上有限（没有除零）。
    EN: curvature must scale with the mesh and stay finite on a closed mesh (no division by zero).
    """
    vertices, faces = _octahedron()
    base = curvature(vertices, faces)
    scaled = curvature(vertices * 2.0, faces)
    assert np.all(np.isfinite(base)) and np.all(base >= 0.0)
    assert np.allclose(scaled, 2.0 * base, rtol=1e-9, atol=1e-12)


def test_geodesic_matrix_is_symmetric_with_zero_diagonal() -> None:
    """
    CN: 测地矩阵对角为 0；对称性只到热方法的数值精度（实测非严格对称，这里用网格尺度做容差）。
    EN: the geodesic matrix has a zero diagonal; symmetry holds only to the heat method's numerical accuracy
        (measured: not exact), so the tolerance is tied to the mesh scale.
    """
    vertices, faces = _octahedron()
    index = np.arange(len(vertices), dtype=np.int64)
    geo = geodesic_matrix(vertices, faces, index)
    assert np.allclose(np.diag(geo), 0.0, atol=1e-8)
    assert np.allclose(geo, geo.T, atol=0.05 * float(geo.max()))
    # CN: 不用解析值断言（6 顶点的网格太粗，热方法有可见误差——实测 2.449 vs 沿边的 2.828）。改用两条
    #     **必然成立**的不变量：曲面距离不短于直线距离，且不长于沿边的最短路。
    # EN: do not assert an analytic value (a 6-vertex mesh is too coarse and the heat method has visible error:
    #     measured 2.449 against 2.828 along the edges). Use two **invariants** instead: a surface path is
    #     never shorter than the straight line, and never longer than the shortest edge path.
    straight = float(np.linalg.norm(vertices[0] - vertices[1]))
    along_edges = 2.0 * float(np.linalg.norm(vertices[0] - vertices[2]))
    assert geo[0, 1] >= straight - 1e-6
    assert geo[0, 1] <= along_edges + 0.05 * along_edges


def test_mirror_map_finds_the_known_symmetry() -> None:
    """
    CN: 在关于 x=0 对称的网格上，轴上的顶点（x=0）镜像到自己，`x=±1` 的两个顶点互换。
    EN: on a mesh symmetric about `x = 0`, the two vertices at `x = ±1` swap and the four on the plane map to
        themselves.
    """
    vertices, faces = _octahedron()
    index = np.arange(len(vertices), dtype=np.int64)
    mirror, residual = mirror_indices(vertices, index)
    assert mirror[0] == 1 and mirror[1] == 0
    for on_plane in (2, 3, 4, 5):
        assert mirror[on_plane] == on_plane
    assert np.all(residual < 1e-6)


def test_mirror_map_returns_indices_into_the_subsample() -> None:
    """
    CN: **最危险的静默错误**：若镜像返回的是全网格下标而不是子样本下标，翻转统计会悄悄算错。
    EN: **the most dangerous silent error**: if the mirror returns full-mesh indices instead of positions in
        the subsample, the flip count is quietly wrong. This pins the convention.
    """
    vertices, faces = _octahedron()
    index = np.asarray([0, 2, 3, 4], dtype=np.int64)
    mirror, _residual = mirror_indices(vertices, index)
    assert mirror.shape == index.shape
    assert mirror.min() >= 0 and mirror.max() < index.size
    # CN: 子样本里 0 号是 x=+1，它的镜像 x=-1 **不在**子样本里，所以只能落到平面上最近的顶点。
    # EN: vertex 0 (x=+1) is in the subsample but its mirror (x=-1) is not, so it must land on a plane vertex.
    assert mirror[0] == 1  # index[1] is vertex 2, on the plane


def test_squared_difference_features_match_the_dense_formula() -> None:
    """
    CN: `SquaredDifferenceFeatures` 必须与"显式 `(m, n, K)` 张量再切边"**逐位一致**——它是那个张量的替代品，
        引入它只是为了不物化张量。若两者有差异，"省内存"就变成了"改了结果"。
    EN: `SquaredDifferenceFeatures` must agree **exactly** with "build the explicit `(m, n, K)` tensor and
        slice it". It exists only to avoid materialising that tensor; any difference would turn "saves memory"
        into "changes the result".
    """
    rng = np.random.default_rng(0)
    a = rng.random((7, 4))
    b = rng.random((5, 4))
    dense = (a[:, None, :] - b[None, :, :]) ** 2

    features = SquaredDifferenceFeatures(a, b)
    assert (features.n_source, features.n_target, features.n_features) == (7, 5, 4)

    rows = np.array([0, 3, 6, 1])
    cols = np.array([4, 0, 2, 2])
    assert np.array_equal(features.features(rows, cols), dense[rows, cols])

    theta = rng.random(4)
    assert np.allclose(features.costs(theta, rows, cols),
                       np.einsum("ijk,k->ij", dense, theta)[rows, cols], rtol=0, atol=1e-12)


def test_squared_difference_features_reject_mismatched_dimensions() -> None:
    """CN: 维度不符必须早失败，不能广播。EN: mismatched dimensions must fail early, never broadcast."""
    with pytest.raises(ValueError):
        SquaredDifferenceFeatures(np.zeros((3, 4)), np.zeros((3, 5)))
    with pytest.raises(ValueError):
        SquaredDifferenceFeatures(np.zeros(3), np.zeros(3))
