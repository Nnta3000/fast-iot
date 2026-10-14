"""
CN: 实验的「边际」轴：把源/目标质量向量参数化为 uniform / gaussian / gmm。
EN: The "marginal" axis of the experiments: parameterise the source/target mass vectors as
    uniform / gaussian / gmm.

CN: 与既有 GMM 实验一致的定义：给每个样本一维潜坐标 z_i = (i + 0.5) / n（样本生成顺序），
    质量向量取该潜轴上的均匀/高斯/高斯混合密度并归一化。
EN: Same definition as the earlier GMM experiments: each sample gets a 1-D latent coordinate
    z_i = (i + 0.5) / n (generation order); the mass vector is a normalised uniform / Gaussian
    / Gaussian-mixture density on that latent axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _latent_axis(n: int) -> np.ndarray:
    """CN: 潜轴 z_i = (i + 0.5)/n。EN: The latent axis z_i = (i + 0.5)/n."""
    return (np.arange(int(n), dtype=np.float64) + 0.5) / float(n)


@dataclass(frozen=True)
class MarginalSpec:
    """
    CN: 一个边际族的规格；`kind` 为 uniform / gaussian / gmm。
    EN: One marginal-family spec; `kind` is uniform / gaussian / gmm.

    CN: gaussian 用 (center, width)；gmm 用 `components`（每个是 (权重, 中心, 宽度)）。
    EN: gaussian uses (center, width); gmm uses `components` (each (weight, center, width)).
    """

    kind: str
    center: float = 0.5
    width: float = 0.15
    components: tuple[tuple[float, float, float], ...] = ()
    floor_frac: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"uniform", "gaussian", "gmm"}:
            raise ValueError(f"unknown marginal kind {self.kind!r}")
        if self.kind == "gaussian" and not float(self.width) > 0.0:
            raise ValueError("gaussian marginal width must be positive")
        if self.kind == "gmm":
            if not self.components:
                raise ValueError("gmm marginal requires at least one component")
            for weight, _center, width in self.components:
                if float(weight) <= 0.0 or not float(width) > 0.0:
                    raise ValueError("gmm components need positive weight and width")
        if not 0.0 <= float(self.floor_frac) < 1.0:
            raise ValueError("floor_frac must be in [0, 1)")

    @property
    def name(self) -> str:
        """CN: 稳定名字（含参数与 floor）。EN: Stable name (parameters and floor included)."""
        if self.kind == "uniform":
            base = "uniform"
        elif self.kind == "gaussian":
            base = f"gaussian(c={self.center:g},w={self.width:g})"
        else:
            parts = ",".join(f"{w:g}@{c:g}/{s:g}" for w, c, s in self.components)
            base = f"gmm({parts})"
        return base if self.floor_frac <= 0.0 else f"{base}[floor={self.floor_frac:g}]"

    def weights(self, n: int) -> np.ndarray:
        """CN: 长度为 n、和为 1 的质量向量。EN: Mass vector of length n summing to 1."""
        size = int(n)
        if size < 1:
            raise ValueError("n must be >= 1")
        if self.kind == "uniform":
            weights = np.ones(size, dtype=np.float64)
        elif self.kind == "gaussian":
            z = _latent_axis(size)
            weights = np.exp(-0.5 * ((z - float(self.center)) / float(self.width)) ** 2)
        else:
            z = _latent_axis(size)
            weights = np.zeros(size, dtype=np.float64)
            for weight, center, width in self.components:
                weights += float(weight) * np.exp(
                    -0.5 * ((z - float(center)) / float(width)) ** 2
                )
        total = float(np.sum(weights, dtype=np.float64))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("marginal weights must have positive finite total mass")
        normalized = weights / total
        if self.floor_frac > 0.0:
            # CN: 与均匀混合，保证最小质量 ~ floor_frac/n，避免极小质量被稀疏 OT 丢行/列。
            # EN: Mix with uniform so the minimum mass is ~floor_frac/n; avoids the sparse OT
            #     solver dropping tiny-mass rows/columns.
            normalized = (1.0 - self.floor_frac) * normalized + self.floor_frac / float(size)
        return np.ascontiguousarray(normalized, dtype=np.float64)


# CN: 命名预设。`gaussian` 是「单个高斯」，`gmm` 是「二分量高斯混合」。
# EN: Named presets. `gaussian` is a single Gaussian; `gmm` is a two-component mixture.
PRESETS: dict[str, MarginalSpec] = {
    "uniform": MarginalSpec("uniform"),
    "gaussian": MarginalSpec("gaussian", center=0.5, width=0.15),
    "gmm": MarginalSpec("gmm", components=((0.5, 0.30, 0.07), (0.5, 0.72, 0.09))),
    # CN: 与均匀混合 1%，保证最小质量 ~1e-2/n（GPU/HELLO 真值放大规模时必需）。
    # EN: 1% uniform mixing so the minimum mass is ~1e-2/n (needed for the GPU/HELLO path at scale).
    "gmm_floored": MarginalSpec(
        "gmm", components=((0.5, 0.30, 0.07), (0.5, 0.72, 0.09)), floor_frac=0.01
    ),
}


def marginal_spec(name: str) -> MarginalSpec:
    """
    CN: 按名字取边际：预设名，或参数化的 `gaussian:<center>:<width>`（用于尖锐度扫描）。
    EN: Look up a marginal by name: a preset, or a parameterised `gaussian:<center>:<width>`
        (for the peakedness sweep).
    """
    if name in PRESETS:
        return PRESETS[name]
    parts = name.split(":")
    if len(parts) == 3 and parts[0] == "gaussian":
        return MarginalSpec("gaussian", center=float(parts[1]), width=float(parts[2]))
    raise ValueError(
        f"unknown marginal {name!r}; known presets: {sorted(PRESETS)}, or 'gaussian:<center>:<width>'"
    )


__all__ = ["MarginalSpec", "PRESETS", "marginal_spec"]
