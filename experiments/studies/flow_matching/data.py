"""
EN: **Toy distributions for the flow-matching coupling study.**

    Two 2-D targets with different failure modes for minibatch OT: an eight-mode Gaussian mixture, where a pool of
    `B` points holds only `B/8` points per mode and the mode counts fluctuate, and two moons, where the target has
    thin curved arms and a pool of `B` points under-represents parts of each arm.

    All generators are pure NumPy and take an explicit `Generator`, so a run is reproducible from `(name, seed)`.

CN: **flow-matching 耦合研究的玩具分布。** 两个 2-D 目标，对 minibatch OT 的失败模式不同：八模态高斯混合（规模为
    `B` 的池每个模态只有 `B/8` 个点，模态计数会波动）与双月（目标为细弯臂，池会低估臂的某些部分）。全部用 NumPy，
    显式接收 `Generator`，故一次运行可由 `(name, seed)` 复现。
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float

from .seeds import seed as seed_from

__all__ = ["DATASETS", "dataset", "eight_gaussians", "two_moons", "mode_centres"]

DATASETS: tuple[str, ...] = ("eight_gaussians", "two_moons")
_RADIUS = 2.0
_SIGMA = 0.1


def mode_centres() -> Float[np.ndarray, "8 2"]:
    """EN: the eight mode centres, at angles `2πk/8` on a radius-2 circle. CN: 八个模态中心，半径 2 的圆上 `2πk/8`。"""
    angles = np.arange(8, dtype=np.float64) * (2.0 * np.pi / 8.0)
    return np.ascontiguousarray(np.stack([_RADIUS * np.cos(angles), _RADIUS * np.sin(angles)], axis=1))


def eight_gaussians(n: int, rng: np.random.Generator) -> Float[np.ndarray, "n 2"]:
    """EN: `n` points from an equal-weight mixture of eight isotropic Gaussians (`σ = 0.1`). CN: 八个等权各向同性高斯（`σ=0.1`）混合的 `n` 个点。"""
    centres = mode_centres()
    labels = rng.integers(0, 8, size=int(n))
    return np.ascontiguousarray(centres[labels] + _SIGMA * rng.normal(size=(int(n), 2)))


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


def dataset(name: str, n: int, seed: int) -> Float[np.ndarray, "n 2"]:
    """EN: dispatch by name; the RNG is derived from `(name, seed)` so draws do not collide across targets. CN: 按名称分派；RNG 由 `(name, seed)` 派生，目标之间不冲突。"""
    rng = np.random.default_rng(seed_from(str(name), int(seed)))
    if name == "eight_gaussians":
        return eight_gaussians(n, rng)
    if name == "two_moons":
        return two_moons(n, rng)
    raise ValueError(f"unknown dataset: {name!r}")
