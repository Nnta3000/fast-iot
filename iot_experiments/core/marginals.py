"""
CN: 实验的「边际」轴：把源/目标质量向量与**点云**参数化为 uniform / gaussian / gmm。
EN: The "marginal" axis of the experiments: parameterise the source/target mass vectors **and the
    point clouds** as uniform / gaussian / gmm.

CN: **设计（重要）**。质量与点云都由本模块产生，二者必须一致：
  - `uniform`：质量恒为 1（不归一化），点云 ~ N(0, I)。
  - `gaussian`：点云 ~ N(0, I)；质量取该点在潜轴 z_i=(i+0.5)/n 上的一维高斯密度，再**归一化为和 1**。
  - `gmm`：**点云来自 d 维高斯混合**（每个分量等权 1/C，位置 μ_c、协方差 σ_c² I）；质量恒为 1。
    这是对旧实现的修正：旧版点云是 i.i.d. N(0,I)，只把混合密度套在**生成顺序**的潜轴上，与几何无关，
    会产生与任何空间混合都无关的极小质量尾部（例如 n=3200 时 node 0 的质量 ~8e-8），使 HELLO 丢弃
    该行/列 → 破坏 "每行每列质量为正" 的不变量。

EN: **Design (important).** Both the masses and the point clouds come from this module and must agree:
  - `uniform`: masses are identically 1 (NOT normalised); points ~ N(0, I).
  - `gaussian`: points ~ N(0, I); masses are a 1-D Gaussian density on the latent axis z_i=(i+0.5)/n,
    **normalised to sum 1**.
  - `gmm`: **points come from a d-dimensional Gaussian mixture** (each component equally weighted 1/C,
    position μ_c, covariance σ_c² I); masses are identically 1. This fixes the old construction, where
    the clouds were i.i.d. N(0,I) and the mixture density was applied along the *generation order*, which
    is geometrically meaningless and produced tiny mass tails unrelated to any spatial mixture (e.g. mass
    ~8e-8 at node 0 for n=3200). HELLO then dropped that row/column, breaking the invariant
    "every row and column must have positive mass".

CN: **质量是否归一化？** 不。质量只要求源/目标**总和相等**（`InverseOTProblem` 检查这一点），不要求总和为 1。
    因此 `uniform`/`gmm` 保持权为 1（更喜欢这个：避免概率测度的额外约束与归一化误差）。
    `gaussian` 仍按密度的定义归一化（见 `weights`）。`floor_frac` 仅对 `gaussian` 有意义。
EN: **Are masses normalised?** No. Masses only need equal *source/target totals* (`InverseOTProblem` checks
    that), not a total of 1. So `uniform`/`gmm` keep weight 1 (preferred: avoids probability-measure
    bookkeeping and normalisation error). `gaussian` is still normalised, per the density definition
    (see `weights`). `floor_frac` only applies to `gaussian`.
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

    CN: `gaussian` 用 (center, width) 且点云为 N(0,I)；`gmm` 用 `components`（每个是 (权重, 中心, 宽度)），
        其**点云**由这些分量在 `point_dimension` 维空间中采样（分量按权重；默认权重等分）。
    EN: `gaussian` uses (center, width) with an N(0,I) cloud; `gmm` uses `components`
        (each (weight, centre, width)), whose **point cloud** is drawn from them in `point_dimension`
        dimensions (components sampled by weight; default weights are equal).
    """

    kind: str
    center: float = 0.5
    width: float = 0.15
    components: tuple[tuple[float, float, float], ...] = ()
    floor_frac: float = 0.0
    point_dimension: int = 8

    def __post_init__(self) -> None:
        if self.kind not in {"uniform", "gaussian", "gmm"}:
            raise ValueError(f"unknown marginal kind {self.kind!r}")
        if int(self.point_dimension) < 1:
            raise ValueError("point_dimension must be >= 1")
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
        """
        CN: 长度为 n 的质量向量。`uniform`/`gmm` 恒为 1（**不**归一化）；`gaussian` 为归一化密度。
        EN: Mass vector of length n. `uniform`/`gmm` are identically 1 (**not** normalised);
            `gaussian` is a normalised density.
        """
        size = int(n)
        if size < 1:
            raise ValueError("n must be >= 1")
        if self.kind == "uniform":
            weights = np.ones(size, dtype=np.float64)
        elif self.kind == "gmm":
            # CN: 质量恒为 1（不归一化）：gmm 的形状由**点云**承载，而非质量。
            # EN: masses are identically 1 (not normalised): gmm's shape lives in the *cloud*, not the mass.
            weights = np.ones(size, dtype=np.float64)
        else:
            z = _latent_axis(size)
            weights = np.exp(-0.5 * ((z - float(self.center)) / float(self.width)) ** 2)
        total = float(np.sum(weights, dtype=np.float64))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("marginal weights must have positive finite total mass")
        if self.kind == "gaussian":
            # CN: 只有 gaussian 归一化（它本质是概率密度）。EN: only gaussian is normalised (it is a density).
            weights = weights / total
            if self.floor_frac > 0.0:
                # CN: 与均匀混合，保证最小质量 ~ floor_frac/n。EN: mix with uniform, min mass ~ floor_frac/n.
                weights = (1.0 - self.floor_frac) * weights + self.floor_frac / float(size)
        return np.ascontiguousarray(weights, dtype=np.float64)

    def sample_points(self, n: int, rng: np.random.Generator, dimension: int | None = None) -> np.ndarray:
        """
        CN: 生成长度为 n、维度为 `dimension`（默认 `self.point_dimension`）的点云。
             - `uniform`/`gaussian`：i.i.d. N(0, I)。
             - `gmm`：d 维高斯混合，分量按自身权重选取（默认等权 1/C），每维标准差为该分量的 width。
        EN: Draw an (n, dimension) point cloud (`dimension` defaults to `self.point_dimension`).
             - `uniform`/`gaussian`: i.i.d. N(0, I).
             - `gmm`: a d-dimensional Gaussian mixture; components chosen by their weight (default 1/C each),
               each dimension having standard deviation equal to that component's width.

        CN: 这样混合形状由**几何**决定，与样本生成顺序无关；质量恒为 1，因此不会有极小质量尾部。
        EN: The mixture shape is now carried by the *geometry*, independent of generation order; masses are
            identically 1, so no tiny-mass tails arise.
        """
        size = int(n)
        dim = int(self.point_dimension) if dimension is None else int(dimension)
        if dim < 1:
            raise ValueError("dimension must be >= 1")
        if self.kind != "gmm":
            return np.ascontiguousarray(rng.normal(size=(size, dim)), dtype=np.float64)
        weights = np.array([float(w) for w, _c, _s in self.components], dtype=np.float64)
        centers = np.array([float(c) for _w, c, _s in self.components], dtype=np.float64)
        widths = np.array([float(s) for _w, _c, s in self.components], dtype=np.float64)
        probabilities = weights / float(np.sum(weights))
        labels = rng.choice(len(self.components), size=size, p=probabilities)
        # CN: 中心只放在第一维（其余维为 0）：与本族"沿某个方向分离"的直觉一致，且保持各维独立。
        # EN: centres live on coordinate 0 only (others are 0): matches the "separated along one direction"
        #     intuition of this family and keeps coordinates independent.
        points = rng.normal(loc=0.0, scale=1.0, size=(size, dim))
        points *= widths[labels][:, None]
        points[:, 0] += centers[labels]
        return np.ascontiguousarray(points, dtype=np.float64)


# CN: 命名预设。`gaussian` 是「单个高斯」；`gmm` 是「二分量高斯混合」，其**点云**为 2 维高斯混合。
# EN: Named presets. `gaussian` is a single Gaussian; `gmm` is a two-component mixture whose **cloud**
#     is a 2-D Gaussian mixture.
PRESETS: dict[str, MarginalSpec] = {
    "uniform": MarginalSpec("uniform"),
    "gaussian": MarginalSpec("gaussian", center=0.5, width=0.15),
    # CN: 点云 = 0.5·N(μ0=(-2,0,..), σ=0.7) + 0.5·N(μ1=(+2,0,..), σ=0.9)（分量等权 1/2）。
    # EN: cloud = 0.5·N(mu0=(-2,0,..), s=0.7) + 0.5·N(mu1=(+2,0,..), s=0.9) (components weighted 1/2 each).
    "gmm": MarginalSpec(
        "gmm", components=((1.0, -2.0, 0.7), (1.0, 2.0, 0.9)), point_dimension=8
    ),
    # CN: 旧的 1% floor 预设保留，但**只对 gaussian 生效**；gmm 现在不需要 floor（质量已远离 0）。
    # EN: the old 1% floor preset is kept but **only affects gaussian**; gmm no longer needs a floor
    #     (masses are bounded away from 0).
    "gmm_floored": MarginalSpec(
        "gmm", components=((1.0, -2.0, 0.7), (1.0, 2.0, 0.9)), floor_frac=0.01, point_dimension=8
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


__all__ = ["MarginalSpec", "marginal_spec", "PRESETS"]
