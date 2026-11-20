"""
EN: **On-the-fly point-distance features, so the feature tensor is never stored.**

    The memory-linear machinery in this subpackage needs a `FeatureMap` whose cost can be evaluated for a *block* of
    pairs from `O((m + n) d)` stored coordinates. `DiagonalWeightFeatures` qualifies; a feature built from a stored
    distance matrix does not (that matrix is `Theta(m n)`). This module provides the point-based one used by the
    synthetic studies, with the same gauge conventions as the dense reference:

      * profiles are `s, s^2, log(1 + s), s^3` with `s = ||x_i - y_j|| / scale`, `width` selecting a prefix;
      * each column is divided by a **scalar** normaliser estimated once from a deterministic subsample of pairs
        (unit-norm in expectation), which is the gauge that stops the fit from preferring the column with the
        smallest entries — the `Theta(1)` version of the dense reference's unit-Frobenius columns;
      * no constant column and no same-zone-only column, the two directions measured to make the regret trivially
        small.

    Storage: the two point clouds only. No `m x n` array, at any point.

CN: **即时计算的点距离特征，特征张量从不存储。** 本子包的内存线性机制需要一种能按**块**求代价的 `FeatureMap`，只存
    `O((m+n)d)` 个坐标。`DiagonalWeightFeatures` 满足；由存储的距离矩阵构造的特征不满足（那个矩阵是 `Theta(mn)`）。
    本模块提供综合实验所用的点式特征，规范与稠密参考一致：剖面为 `s, s², log(1+s), s³`（`s = ||x_i-y_j||/scale`）；
    每列除以一个由确定性抽样一次性估计的**标量**归一化常数（期望意义下单位范数），这是阻止拟合偏好"元素最小列"的规范，
    是稠密参考中单位 Frobenius 列的 `Theta(1)` 版本；没有常数列、没有同 zone 专列 —— 这两个方向都已被实测为使 regret
    平凡变小。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int

__all__ = ["PointDistanceFeatures", "PROFILE_NAMES"]

PROFILE_NAMES: tuple[str, ...] = ("s", "s2", "log1p", "s3")


@dataclass(frozen=True)
class PointDistanceFeatures:
    """
    EN: `phi_ij = [s, s^2, log(1+s), s^3][:width] / normaliser`, `s = ||x_i - y_j||_2 / scale`.

        `scale` is a scalar chosen by the caller (e.g. 1 for the unit square) and must be the **same for every
        window** in a multi-observation fit, otherwise θ mixes geometries.

    CN: `phi_ij = [s, s², log(1+s), s³][:width] / 归一化常数`，`s = ||x_i-y_j||/scale`。`scale` 是调用方给的一个标量
        （例如单位方格取 1），在同一个多观测拟合中**各窗必须相同**，否则 θ 混合了不同几何。
    """

    source_points: Float[np.ndarray, "m d"]
    target_points: Float[np.ndarray, "n d"]
    scale: float = 1.0
    width: int = 4
    sample_size: int = 100_000
    seed: int = 0
    normalisers: tuple[float, ...] = dataclasses.field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        """EN: validate and estimate the scalar normalisers. CN: 校验并估计标量归一化常数。"""
        source = np.ascontiguousarray(np.asarray(self.source_points, dtype=np.float64))
        target = np.ascontiguousarray(np.asarray(self.target_points, dtype=np.float64))
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source_points and target_points must be 2D")
        if source.shape[1] != target.shape[1]:
            raise ValueError("the two point clouds must share a dimension")
        if source.shape[0] < 1 or target.shape[0] < 1:
            raise ValueError("point clouds must be non-empty")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("point coordinates must be finite")
        if not float(self.scale) > 0.0:
            raise ValueError("scale must be positive")
        if int(self.width) < 1 or int(self.width) > len(PROFILE_NAMES):
            raise ValueError(f"width must be in 1..{len(PROFILE_NAMES)}")
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "target_points", target)
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "normalisers", self._normalisers())

    @property
    def n_source(self) -> int:
        """EN: `m`. CN: 源点数。"""
        return int(self.source_points.shape[0])

    @property
    def n_target(self) -> int:
        """EN: `n`. CN: 目标点数。"""
        return int(self.target_points.shape[0])

    @property
    def n_features(self) -> int:
        """EN: `K = width`. CN: 特征数。"""
        return int(self.width)

    def _profiles(self, spacing: Float[np.ndarray, "edges"]) -> Float[np.ndarray, "edges k"]:
        """EN: the raw profile values before normalisation. CN: 归一化前的剖面取值。"""
        columns = [spacing, spacing ** 2, np.log1p(spacing), spacing ** 3]
        return np.ascontiguousarray(np.stack(columns[: int(self.width)], axis=1), dtype=np.float64)

    def _normalisers(self) -> tuple[float, ...]:
        """
        EN: root-mean-square of each profile over a deterministic subsample of pairs (a scalar per column).

            This is the `Theta(1)` stand-in for the dense reference's `||phi_k||_F`: it uses `sample_size` pairs,
            never `m n`, and it fixes the gauge so that a column's *magnitude* is not what the simplex selects.

        CN: 每列在确定性抽样对上的均方根（每列一个标量）。这是稠密参考 `||phi_k||_F` 的 `Theta(1)` 替代：只用
            `sample_size` 对，从不用 `m n`，并固定规范，使单纯形不会因为列幅度而选中它。
        """
        rng = np.random.default_rng(int(self.seed))
        count = min(int(self.sample_size), self.n_source * self.n_target)
        rows = rng.integers(0, self.n_source, size=count)
        cols = rng.integers(0, self.n_target, size=count)
        spacing = self._spacing(rows, cols)
        profiles = self._profiles(spacing)
        norms = np.sqrt(np.mean(profiles ** 2, axis=0))
        norms = np.maximum(norms, 1e-12)
        return tuple(float(value) for value in norms)

    def _spacing(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges"]:
        """EN: `||x_i - y_j||_2 / scale` for the requested pairs. CN: 请求对上的归一化欧氏距离。"""
        difference = self.source_points[rows] - self.target_points[cols]
        return np.sqrt(np.einsum("ed,ed->e", difference, difference)) / float(self.scale)

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        """
        EN: normalised profile values for the requested pairs; `O(|edges| d)` time, no stored grid.
        CN: 请求对的归一化剖面值；`O(|edges| d)` 时间，不存网格。
        """
        row_index = np.asarray(rows, dtype=np.int64).reshape(-1)
        col_index = np.asarray(cols, dtype=np.int64).reshape(-1)
        if row_index.size != col_index.size:
            raise ValueError("rows and cols must have the same length")
        profiles = self._profiles(self._spacing(row_index, col_index))
        return np.ascontiguousarray(profiles / np.asarray(self.normalisers, dtype=np.float64)[None, :],
                                   dtype=np.float64)

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
        """EN: `phi @ theta` on the requested pairs. CN: 请求对上的 `phi @ theta`。"""
        return np.ascontiguousarray(self.features(rows, cols) @ np.asarray(theta, dtype=np.float64).reshape(-1),
                                   dtype=np.float64)


@dataclass(frozen=True)
class DistanceProfileFeatures:
    """
    EN: **a tunable-width radial profile of the distance** — `bumps` Gaussian functions of `s = ||x-y||/scale`.

        `phi_k = exp(-((s - c_k) / w)^2)` with centres `c_k` evenly spaced on `[0, 1]` and width `w = 1.5 / bumps`,
        each column divided by its subsample-RMS. `bumps` is the knob the arms study sweeps against `n`: the planted
        cost is expressed in **the same basis**, so the family always contains the truth and `bumps` controls only
        how many parameters must be pinned, not whether the truth is representable.

        The columns are **not** monotone transforms of one another (each bump peaks at its own `s`), unlike kernels
        `exp(-s^2 / sigma^2)` at different bandwidths, which are rank-equivalent and leave θ unidentifiable.
        No constant column and no same-zone column.

    CN: **宽度可调的径向剖面** —— `bumps` 个关于 `s = ||x-y||/scale` 的高斯函数。`phi_k = exp(-((s-c_k)/w)²)`，中心
        `c_k` 在 `[0,1]` 等距，宽度 `w = 1.5/bumps`，每列除以抽样 RMS。`bumps` 是 arms 研究相对 `n` 扫描的旋钮：植入代价
        用**同一基**表示，故族总含真值，`bumps` 只控制需要钉住的参数个数。各列**不是**彼此的单调变换（每个鼓包在自己
        的 `s` 处取峰），不同于不同带宽的 `exp(-s²/sigma²)`（秩等价，θ 不可辨识）。无常数列、无同 zone 列。
    """

    source_points: Float[np.ndarray, "m d"]
    target_points: Float[np.ndarray, "n d"]
    bumps: int = 8
    scale: float = 1.0
    sample_size: int = 100_000
    seed: int = 0
    centres: Float[np.ndarray, "k"] = dataclasses.field(default=None, init=False, repr=False)  # type: ignore[assignment]
    width: float = dataclasses.field(default=0.0, init=False, repr=False)
    normals: tuple[float, ...] = dataclasses.field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        """EN: validate, place the centres, estimate the column scales. CN: 校验、放置中心、估计列尺度。"""
        source = np.ascontiguousarray(np.asarray(self.source_points, dtype=np.float64))
        target = np.ascontiguousarray(np.asarray(self.target_points, dtype=np.float64))
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
            raise ValueError("two point clouds of the same dimension are required")
        if np.isfinite(source).sum() != source.size or np.isfinite(target).sum() != target.size:
            raise ValueError("point coordinates must be finite")
        if int(self.bumps) < 1:
            raise ValueError("bumps must be at least 1")
        if not float(self.scale) > 0.0:
            raise ValueError("scale must be positive")
        centres = np.linspace(0.0, 1.0, int(self.bumps))
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "target_points", target)
        object.__setattr__(self, "bumps", int(self.bumps))
        object.__setattr__(self, "centres", np.ascontiguousarray(centres))
        object.__setattr__(self, "width", 1.5 / float(self.bumps))
        object.__setattr__(self, "normals", self._normals())

    @property
    def n_source(self) -> int:
        """EN: `m`. CN: 源点数。"""
        return int(self.source_points.shape[0])

    @property
    def n_target(self) -> int:
        """EN: `n`. CN: 目标点数。"""
        return int(self.target_points.shape[0])

    @property
    def n_features(self) -> int:
        """EN: `K = bumps`. CN: 特征数。"""
        return int(self.bumps)

    def _spacing(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges"]:
        """EN: normalised Euclidean distance for the requested pairs. CN: 请求对的归一化欧氏距离。"""
        difference = self.source_points[rows] - self.target_points[cols]
        return np.sqrt(np.einsum("ed,ed->e", difference, difference)) / float(self.scale)

    def _raw(self, spacing: Float[np.ndarray, "edges"]) -> Float[np.ndarray, "edges k"]:
        """EN: the bump values before normalisation. CN: 归一化前的鼓包取值。"""
        centres = np.asarray(self.centres, dtype=np.float64)
        width = float(self.width)
        return np.exp(-((spacing[:, None] - centres[None, :]) / width) ** 2)

    def _normals(self) -> tuple[float, ...]:
        """EN: subsample-RMS per column, the `Theta(1)` gauge. CN: 每列的抽样 RMS，即 `Theta(1)` 规范。"""
        rng = np.random.default_rng(int(self.seed))
        count = min(int(self.sample_size), self.n_source * self.n_target)
        rows = rng.integers(0, self.n_source, size=count)
        cols = rng.integers(0, self.n_target, size=count)
        raw = self._raw(self._spacing(rows, cols))
        return tuple(float(max(value, 1e-12)) for value in np.sqrt(np.mean(raw ** 2, axis=0)))

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        """EN: normalised bump values for the requested pairs. CN: 请求对的归一化鼓包取值。"""
        row_index = np.asarray(rows, dtype=np.int64).reshape(-1)
        col_index = np.asarray(cols, dtype=np.int64).reshape(-1)
        raw = self._raw(self._spacing(row_index, col_index))
        return np.ascontiguousarray(raw / np.asarray(self.normals, dtype=np.float64)[None, :], dtype=np.float64)

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
        """EN: `phi @ theta`. CN: `phi @ theta`。"""
        return np.ascontiguousarray(self.features(rows, cols) @ np.asarray(theta, dtype=np.float64).reshape(-1),
                                   dtype=np.float64)


@dataclass(frozen=True)
class RampFeatures:
    """
    EN: **a cumulative (hinge) basis**: `phi_k = max(0, s - t_k)`, so every feature is nondecreasing in the distance.

        Thresholds `t_k` are evenly spaced between the 1st and 99th percentile of the distance over a deterministic
        subsample of pairs — the "observed distance range". A cost `sum_k theta_k phi_k` with `theta >= 0` is
        therefore **nondecreasing in `s`**, which is the structural assumption a distance cost should have; the bump
        basis does not enforce it.

        One hole remains by construction and is measured, not hidden: a single ramp with a high threshold is zero on
        every shorter pair, so a fit can still choose weights that vanish on the observed support. `diagnose.py`
        reports the observed cost relative to the uniform plan's cost on the same edges.

    CN: **累积（hinge）基** `phi_k = max(0, s - t_k)`，每个特征对距离非减。阈值 `t_k` 在抽样对距离的 1%–99% 分位间等距
        —— 即"观测距离范围"。`theta>=0` 时代价对 `s` **非减**，这是距离代价应有的结构；鼓包基不强制它。仍然存在一个
        构造性的洞并会被测量而非隐藏：高阈值的单个 ramp 在所有更短的配对上为 0，故拟合仍可选到在观测支撑上消失的权重；
        `diagnose.py` 报告观测代价相对均匀计划在同一批边上的代价。
    """

    source_points: Float[np.ndarray, "m d"]
    target_points: Float[np.ndarray, "n d"]
    ramps: int = 8
    scale: float = 1.0
    sample_size: int = 100_000
    seed: int = 0
    thresholds: tuple[float, ...] = dataclasses.field(default=(), init=True, repr=False)
    threshold_rule: str = "range"
    differences: bool = False
    normals: tuple[float, ...] = dataclasses.field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        """EN: validate and fix the thresholds on the observed range. CN: 校验并在观测范围内固定阈值。"""
        source = np.ascontiguousarray(np.asarray(self.source_points, dtype=np.float64))
        target = np.ascontiguousarray(np.asarray(self.target_points, dtype=np.float64))
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
            raise ValueError("two point clouds of the same dimension are required")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("point coordinates must be finite")
        if int(self.ramps) < 1:
            raise ValueError("ramps must be at least 1")
        if not float(self.scale) > 0.0:
            raise ValueError("scale must be positive")
        if self.threshold_rule not in ("range", "quantile"):
            raise ValueError("threshold_rule must be 'range' or 'quantile'")
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "target_points", target)
        object.__setattr__(self, "ramps", int(self.ramps))
        object.__setattr__(self, "differences", bool(self.differences))
        if len(self.thresholds) == int(self.ramps):
            thresholds = np.ascontiguousarray(np.asarray(self.thresholds, dtype=np.float64))
        else:
            thresholds = self._default_thresholds()
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "normals", self._normals())

    @property
    def n_source(self) -> int:
        """EN: `m`. CN: 源点数。"""
        return int(self.source_points.shape[0])

    @property
    def n_target(self) -> int:
        """EN: `n`. CN: 目标点数。"""
        return int(self.target_points.shape[0])

    @property
    def n_features(self) -> int:
        """EN: `K = ramps`. CN: 特征数。"""
        return int(self.ramps)

    def _spacing(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges"]:
        """EN: normalised distance. CN: 归一化距离。"""
        difference = self.source_points[rows] - self.target_points[cols]
        return np.sqrt(np.einsum("ed,ed->e", difference, difference)) / float(self.scale)

    def _default_thresholds(self) -> np.ndarray:
        """EN: thresholds on the sampled distance range (evenly spaced) or at its quantiles. CN: 阈值在抽样距离范围上等距，或取其分位。"""
        rng = np.random.default_rng(int(self.seed))
        count = min(int(self.sample_size), self.n_source * self.n_target)
        rows = rng.integers(0, self.n_source, size=count)
        cols = rng.integers(0, self.n_target, size=count)
        spacing = self._spacing(rows, cols)
        if self.threshold_rule == "quantile":
            # EN: t_k at the (k+1)/(K+1) quantiles, so every hinge is positive on a known share of pairs.
            # CN: t_k 取 (k+1)/(K+1) 分位，故每个 hinge 在已知比例的配对上为正。
            levels = np.arange(1, int(self.ramps) + 1, dtype=np.float64) / (int(self.ramps) + 1.0)
            return np.ascontiguousarray(np.quantile(spacing, levels))
        low, high = np.percentile(spacing, 1.0), np.percentile(spacing, 99.0)
        return np.ascontiguousarray(np.linspace(float(low), float(high), int(self.ramps)))

    def _raw(self, spacing: Float[np.ndarray, "edges"]) -> Float[np.ndarray, "edges k"]:
        """
        EN: `max(0, s - t_k)`, or its first difference `raw_k - raw_{k+1}` (with `raw_K = 0`) if `differences`.

            A difference column is zero below `t_k`, rises on `(t_k, t_{k+1})` and saturates at `t_{k+1} - t_k`; the
            columns are much less collinear than nested hinges.

        CN: `max(0, s - t_k)`；若 `differences` 则为它的一阶差分 `raw_k - raw_{k+1}`（取 `raw_K = 0`）。差分列在
            `t_k` 以下为零、在 `(t_k, t_{k+1})` 上升、在 `t_{k+1}-t_k` 饱和，列间共线性远小于嵌套 hinge。
        """
        hinges = np.maximum(0.0, spacing[:, None] - np.asarray(self.thresholds, dtype=np.float64)[None, :])
        if not bool(self.differences):
            return hinges
        later = np.concatenate([hinges[:, 1:], np.zeros((hinges.shape[0], 1), dtype=np.float64)], axis=1)
        return np.ascontiguousarray(hinges - later)

    def _normals(self) -> tuple[float, ...]:
        """EN: subsample-RMS per column. CN: 每列抽样 RMS。"""
        rng = np.random.default_rng(int(self.seed) + 1)
        count = min(int(self.sample_size), self.n_source * self.n_target)
        rows = rng.integers(0, self.n_source, size=count)
        cols = rng.integers(0, self.n_target, size=count)
        raw = self._raw(self._spacing(rows, cols))
        return tuple(float(max(value, 1e-12)) for value in np.sqrt(np.mean(raw ** 2, axis=0)))

    def features(self, rows: Int[np.ndarray, "edges"], cols: Int[np.ndarray, "edges"]
                 ) -> Float[np.ndarray, "edges k"]:
        """EN: normalised ramp values. CN: 归一化 hinge 取值。"""
        raw = self._raw(self._spacing(np.asarray(rows, dtype=np.int64).reshape(-1),
                                      np.asarray(cols, dtype=np.int64).reshape(-1)))
        return np.ascontiguousarray(raw / np.asarray(self.normals, dtype=np.float64)[None, :], dtype=np.float64)

    def costs(self, theta: Float[np.ndarray, "k"], rows: Int[np.ndarray, "edges"],
              cols: Int[np.ndarray, "edges"]) -> Float[np.ndarray, "edges"]:
        """EN: `phi @ theta`. CN: `phi @ theta`。"""
        return np.ascontiguousarray(self.features(rows, cols) @ np.asarray(theta, dtype=np.float64).reshape(-1),
                                   dtype=np.float64)
