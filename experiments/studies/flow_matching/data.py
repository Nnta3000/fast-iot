"""
EN: **Toy distributions for the flow-matching coupling study, now with a dimension knob.**

    `eight_gaussians` (d = 2), `two_moons` (d = 2), and `mixture_d8` / `mixture_d32`: eight equal-weight isotropic
    Gaussians whose centres are eight orthonormal directions scaled to radius 2. The d = 2 mixture uses the eight
    angles of the circle, so it matches the v1 instance exactly; d > 2 uses a seeded QR basis, so the instance is
    reproducible and the modes are orthogonal in the ambient space.

CN: **flow-matching 耦合研究的玩具分布，带维度旋钮。** `eight_gaussians`（d=2）、`two_moons`（d=2）与
    `mixture_d8`/`mixture_d32`：八个等权各向同性高斯，中心为八个正交方向缩放到半径 2。d=2 用圆周八个角，与 v1 完全一致；
    d>2 用带种子的 QR 基，实例可复现且模态在环境空间中正交。
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float

from .seeds import seed as seed_from

__all__ = ["DATASETS", "DIMENSIONS", "dataset", "eight_gaussians", "mode_centres", "two_moons"]

DATASETS: tuple[str, ...] = ("eight_gaussians", "two_moons", "mixture_d8", "mixture_d32")
DIMENSIONS: dict[str, int] = {"eight_gaussians": 2, "two_moons": 2, "mixture_d8": 8, "mixture_d32": 32}
_RADIUS = 2.0
_SIGMA = 0.1


def mode_centres(dim: int, modes: int = 8, seed: int = 0) -> Float[np.ndarray, "modes dim"]:
    """EN: eight mode centres: the circle for `dim = 2`, a seeded orthonormal basis otherwise. CN: 八个模态中心：`dim=2` 用圆，其余用带种子的正交基。"""
    if int(dim) == 2:
        angles = np.arange(int(modes), dtype=np.float64) * (2.0 * np.pi / int(modes))
        return np.ascontiguousarray(np.stack([_RADIUS * np.cos(angles), _RADIUS * np.sin(angles)], axis=1))
    rng = np.random.default_rng(seed_from("mixture-basis", int(dim), int(modes), int(seed)))
    basis, _ = np.linalg.qr(rng.normal(size=(int(dim), int(modes))))
    return np.ascontiguousarray(_RADIUS * basis.T)


def _mixture(n: int, rng: np.random.Generator, dim: int) -> Float[np.ndarray, "n dim"]:
    """EN: the Gaussian mixture in `dim` dimensions. CN: `dim` 维高斯混合。"""
    centres = mode_centres(int(dim))
    labels = rng.integers(0, len(centres), size=int(n))
    return np.ascontiguousarray(centres[labels] + _SIGMA * rng.normal(size=(int(n), int(dim))))


def eight_gaussians(n: int, rng: np.random.Generator) -> Float[np.ndarray, "n 2"]:
    """EN: the 2-D eight-mode mixture (kept for v1 comparability). CN: 2-D 八模态混合（保持与 v1 可比）。"""
    return _mixture(n, rng, 2)


def two_moons(n: int, rng: np.random.Generator, noise: float = 0.05) -> Float[np.ndarray, "n 2"]:
    """EN: the standard two-moons target with Gaussian noise. CN: 带高斯噪声的标准双月目标。"""
    half = int(n) // 2
    theta = rng.uniform(0.0, np.pi, size=half)
    outer = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    inner = np.stack([1.0 - np.cos(theta), 0.5 - np.sin(theta)], axis=1)
    points = np.concatenate([outer, inner], axis=0)
    points = points + float(noise) * rng.normal(size=points.shape)
    if points.shape[0] < int(n):
        points = np.concatenate([points, points[: int(n) - points.shape[0]]], axis=0)
    return np.ascontiguousarray(points[: int(n)])


def dataset(name: str, n: int, seed: int) -> Float[np.ndarray, "n dim"]:
    """EN: dispatch by name; the RNG is derived from `(name, seed)`. CN: 按名称分派；RNG 由 `(name, seed)` 派生。"""
    rng = np.random.default_rng(seed_from(str(name), int(seed)))
    if name == "eight_gaussians":
        return eight_gaussians(n, rng)
    if name == "two_moons":
        return two_moons(n, rng)
    if name in ("mixture_d8", "mixture_d32"):
        return _mixture(n, rng, DIMENSIONS[name])
    raise ValueError(f"unknown dataset: {name!r}")
