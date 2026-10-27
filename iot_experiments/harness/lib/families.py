"""
CN: 实验的「代价族」轴：diagonal / quadratic / lowrank / bilinear-psd 的特征与参数集合。
EN: The "cost family" axis of the experiments: features and parameter sets for diagonal /
    quadratic / lowrank / bilinear-psd.

CN: 每个族只提供 (features, theta_set, theta_star)；具体求解由 `core.runner` 负责。
    `bilinear-signed` 不在其中——0∈Θ 使可行性平凡，已被有意排除。
EN: Each family only provides (features, theta_set, theta_star); `core.runner` does the solve.
    `bilinear-signed` is deliberately absent -- 0 in Theta makes feasibility trivial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iot import (
    BilinearPSDFeatures,
    BilinearPSDTheta,
    DiagonalWeightFeatures,
    FeatureMap,
    QuadraticFeatures,
    SimplexTheta,
    ThetaSet,
)
from iot_experiments.benchmarks.lowrank import LowRankFeatures

from .marginals import MarginalSpec

LP_FAMILIES: tuple[str, ...] = ("diagonal", "quadratic", "lowrank")
ALL_FAMILIES: tuple[str, ...] = ("diagonal", "quadratic", "lowrank", "bilinear-psd")


@dataclass(frozen=True)
class FamilySpec:
    """
    CN: 一个代价族的规格。`point_dimension` 用于 diagonal/quadratic/bilinear-psd；
        `lowrank_k` 用于 lowrank（其特征数 K 与点维度无关）。
    EN: One cost-family spec. `point_dimension` is used by diagonal/quadratic/bilinear-psd;
        `lowrank_k` is used by lowrank (whose feature count K is independent of a dimension).
    """

    name: str
    point_dimension: int = 4
    lowrank_k: int = 8

    def __post_init__(self) -> None:
        if self.name not in ALL_FAMILIES:
            raise ValueError(f"unknown family {self.name!r}; known: {ALL_FAMILIES}")
        if self.name != "lowrank" and int(self.point_dimension) < 1:
            raise ValueError("point_dimension must be >= 1")
        if self.name == "lowrank" and int(self.lowrank_k) < 1:
            raise ValueError("lowrank_k must be >= 1")

    @property
    def is_sdp(self) -> bool:
        """CN: 是否用谱面体 Theta（需要 SDP master）。EN: Whether Theta is a spectrahedron (needs SDP master)."""
        return self.name == "bilinear-psd"

    def n_features(self) -> int:
        """CN: 特征数 K（theta 的维数）。EN: Feature count K (length of theta)."""
        if self.name == "diagonal":
            return int(self.point_dimension)
        if self.name == "quadratic":
            d = int(self.point_dimension)
            return d * (d + 1) // 2
        if self.name == "lowrank":
            return int(self.lowrank_k)
        d = int(self.point_dimension)
        return d * d


def build_family(
    spec: FamilySpec, rng: np.random.Generator, m: int, n: int, *, marginal: MarginalSpec | None = None
) -> tuple[FeatureMap, ThetaSet, np.ndarray]:
    """
    CN: 抽样一个族的 (features, theta_set, theta_star)；theta_star 先抽，再抽点云/因子。
        `marginal` 可选：若给出且其 `kind=="gmm"`，点云由该混合的 `sample_points` 生成（几何承载混合），
        否则点云为 i.i.d. N(0, I)（向后兼容）。
    EN: Draw one family's (features, theta_set, theta_star); theta_star first, then points/factors.
        `marginal` is optional: when given and `marginal.kind == "gmm"`, the cloud is drawn from that
        mixture's `sample_points` (geometry carries the mixture); otherwise the cloud is i.i.d. N(0, I)
        (backward compatible).
    """
    d = int(spec.point_dimension)

    def _cloud(size: int) -> np.ndarray:
        if marginal is not None and marginal.kind == "gmm":
            return marginal.sample_points(size, rng, dimension=d)
        return np.ascontiguousarray(rng.normal(size=(int(size), d)), dtype=np.float64)

    if spec.name == "diagonal":
        theta_set = SimplexTheta(d)
        theta_star = theta_set.sample(rng)
        source = _cloud(int(m))
        target = _cloud(int(n))
        return DiagonalWeightFeatures(source, target), theta_set, theta_star
    if spec.name == "quadratic":
        theta_set = SimplexTheta(spec.n_features())
        theta_star = theta_set.sample(rng)
        source = _cloud(int(m))
        target = _cloud(int(n))
        return QuadraticFeatures(source, target), theta_set, theta_star
    if spec.name == "lowrank":
        theta_set = SimplexTheta(int(spec.lowrank_k))
        theta_star = theta_set.sample(rng)
        source_factors = rng.normal(size=(int(m), int(spec.lowrank_k)))
        target_factors = rng.normal(size=(int(n), int(spec.lowrank_k)))
        return LowRankFeatures(source_factors, target_factors), theta_set, theta_star
    theta_set = BilinearPSDTheta(d)
    theta_star = theta_set.sample(rng)
    source = _cloud(int(m))
    target = _cloud(int(n))
    return BilinearPSDFeatures(source, target), theta_set, theta_star


__all__ = ["ALL_FAMILIES", "LP_FAMILIES", "FamilySpec", "build_family"]
