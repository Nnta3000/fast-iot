"""
CN: **特征与几何注册表** —— 预处理链**唯一**的所在地。

    CN: 为什么需要它：之前 `dinov2_pool4` / `l2_normalise` / `standardise_labelled` 这条链在
        `norm_ablation`、`cos_geometry`、`geometry_sweep`、`pool_variants`、`full_lp_sweep` 里被**复制了五份**，
        于是「cosine」在不同文件里成了两件事（L2-only → R@1 0.270；先标准化再 L2 → 0.768）。
        现在特征与几何各按**名字**取用，且 `uniform θ ≡ cosine` 在每次取用时被断言。

    CN: 几何定义（PCA 之后，只做后处理；PCA 基在全部未标注 T1+T2 上拟合一次，池化均值）：

      | 名字 | 变换 | 说明 |
      |---|---|---|
      | `G0` | 逐行 L2 归一化 | 上一轮扫描的版本 |
      | `G1` | 减去**本模态**均值 → L2 | 每模态各自中心化 |
      | `G2` | 池化标准化（全部未标注 T1+T2 的均值/标准差）→ L2 | 0.768 那条流水线 |
      | `G3` | **按模态**标准化（各自均值/标准差）→ L2 | |

    EN: **Feature and geometry registry** -- the **only** place the preprocessing chain lives.

    Why it exists: `dinov2_pool4` / `l2_normalise` / `standardise_labelled` had been copied into five drivers,
    so "cosine" came to mean two different things (L2-only, R@1 0.270, versus standardise-then-L2, 0.768).
    Features and geometries are now selected **by name**, and `uniform theta ≡ cosine` is asserted on every
    retrieval. Geometry definitions (post-PCA only; one PCA basis fitted on all unlabelled T1+T2, pooled mean):
    `G0` row-wise L2; `G1` subtract each modality's mean then L2; `G2` pooled standardisation (label-free over
    all unlabelled T1+T2) then L2; `G3` per-modality standardisation then L2.

用法 / EN: usage
    from iot_experiments.harness.registry import features, geometry, assert_uniform_equals_cosine
    x, y = features("dinov2-pool4", dim=256)
    geo = geometry(x, y, "G2", labelled=lab, unlabelled=unl)   # returns a GeometryResult
    geo.x_label, geo.y_label, geo.x_unlabelled, geo.y_unlabelled
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float, Int, Shaped

FeatureName = Literal["dinov2-pool4", "dinov2-cls", "block"]
GeometryName = Literal["G0", "G1", "G2", "G3"]

_KNOWN_FEATURES: frozenset[str] = frozenset({"dinov2-pool4", "dinov2-cls", "block"})
_KNOWN_GEOMETRIES: frozenset[str] = frozenset({"G0", "G1", "G2", "G3"})
_EPS: float = 1e-12


class RegistryError(RuntimeError):
    """CN: 未知的特征/几何名，或几何断言失败。EN: unknown feature/geometry name, or a failed assertion."""


def l2_normalise(a: Shaped[np.ndarray, "m d"]) -> Float[np.ndarray, "m d"]:
    """
    CN: 逐行 L2 归一化（分母加 `eps` 防止零行）。**只应在这里出现一次**。
    EN: Row-wise L2 normalisation (with `eps` against zero rows). **Should appear only here.**
    """
    values: Float[np.ndarray, "m d"] = np.asarray(a, dtype=np.float64)
    norms: Float[np.ndarray, "m"] = np.linalg.norm(values, axis=1)
    return values / (norms[:, None] + _EPS)


def _standardise(a: Float[np.ndarray, "m d"], mean: Float[np.ndarray, "d"],
                 std: Float[np.ndarray, "d"]) -> Float[np.ndarray, "m d"]:
    """CN: 标准化（std 为 0 时取 1）。EN: standardise (unit std where std is 0)."""
    safe_std: Float[np.ndarray, "d"] = np.where(std > _EPS, std, 1.0)
    return (a - mean[None, :]) / safe_std[None, :]


def assert_uniform_equals_cosine(x: Float[np.ndarray, "m d"], y: Float[np.ndarray, "n d"]) -> None:
    """
    CN: 断言 `uniform theta` 的对角 L2² 与余弦给出**相同排序**。
        依据：`||x|| = ||y|| = 1` 时 `0.5 * d * (1 - cos) = 0.5 * ||x-y||²`，故 `cos = 0.5 * k * uni`。
        只在**并列处**允许排序不同；否则抛错。
    EN: Assert that `uniform theta` diagonal L2^2 and cosine induce the **same ordering**. With unit-norm
        rows, `cos = 0.5 * k * uni`; differing orderings are tolerated only at ties.
    """
    d = int(x.shape[1])
    cosine: Float[np.ndarray, "m n"] = 1.0 - x @ y.T
    uniform: Float[np.ndarray, "m n"] = (
        (x ** 2).sum(axis=1)[:, None] - 2.0 * (x @ y.T) + (y ** 2).sum(axis=1)[None, :]
    ) / float(d)
    gap: Float[np.ndarray, "m n"] = np.abs(cosine - 0.5 * float(d) * uniform)
    if float(gap.max()) > 1e-9:
        raise RegistryError(
            f"cosine and uniform-theta cost disagree by {float(gap.max()):.3e}; geometry is not L2-normalised")
    for row in range(int(x.shape[0])):
        if not np.array_equal(np.argsort(cosine[row], kind="stable"),
                              np.argsort(uniform[row], kind="stable")):
            if float(gap[row].max()) > 1e-9:      # pragma: no cover - defensive
                raise RegistryError(f"ranking mismatch beyond ties at row {row}")


@dataclass(frozen=True)
class GeometryResult:
    """
    CN: 一个几何下切分好的四个视图：`labelled` 的 T1/T2、`unlabelled` 的 T1/T2，全部已 L2 归一化。
    EN: The four split views under one geometry: labelled T1/T2 and unlabelled T1/T2, all L2-normalised.
    """

    x_label: Float[np.ndarray, "m d"]
    y_label: Float[np.ndarray, "m d"]
    x_unlabelled: Float[np.ndarray, "n d"]
    y_unlabelled: Float[np.ndarray, "n d"]
    name: str


def geometry(
    x: Float[np.ndarray, "subjects d"],
    y: Float[np.ndarray, "subjects d"],
    name: str,
    *,
    labelled: Int[np.ndarray, "m"],
    unlabelled: Int[np.ndarray, "n"],
) -> GeometryResult:
    """
    CN: 按名字施加**PCA 之后**的变换，并用**未标注**图像拟合所有统计量（无标签）。
        返回四个已 L2 归一化的视图；随后会断言 uniform θ 与余弦一致。
    EN: Apply the **post-PCA** transform named by `name`, fitting every statistic on **unlabelled** images
        only (label-free). Returns four L2-normalised views and then asserts uniform theta == cosine.
    """
    if str(name) not in _KNOWN_GEOMETRIES:
        raise RegistryError(f"unknown geometry {name!r}; known: {sorted(_KNOWN_GEOMETRIES)}")
    x_u: Float[np.ndarray, "n d"] = np.asarray(x, dtype=np.float64)[unlabelled]
    y_u: Float[np.ndarray, "n d"] = np.asarray(y, dtype=np.float64)[unlabelled]
    x_l: Float[np.ndarray, "m d"] = np.asarray(x, dtype=np.float64)[labelled]
    y_l: Float[np.ndarray, "m d"] = np.asarray(y, dtype=np.float64)[labelled]

    if name == "G0":
        xl, yl, xu, yu = l2_normalise(x_l), l2_normalise(y_l), l2_normalise(x_u), l2_normalise(y_u)
    elif name == "G1":
        mean_x: Float[np.ndarray, "d"] = x_u.mean(axis=0)
        mean_y: Float[np.ndarray, "d"] = y_u.mean(axis=0)
        xl, yl = l2_normalise(x_l - mean_x), l2_normalise(y_l - mean_y)
        xu, yu = l2_normalise(x_u - mean_x), l2_normalise(y_u - mean_y)
    elif name == "G2":
        pooled: Float[np.ndarray, "pooled d"] = np.concatenate([x_u, y_u], axis=0)
        mean: Float[np.ndarray, "d"] = pooled.mean(axis=0)
        std: Float[np.ndarray, "d"] = pooled.std(axis=0)
        xl, yl = l2_normalise(_standardise(x_l, mean, std)), l2_normalise(_standardise(y_l, mean, std))
        xu, yu = l2_normalise(_standardise(x_u, mean, std)), l2_normalise(_standardise(y_u, mean, std))
    else:  # G3
        mean_x, std_x = x_u.mean(axis=0), x_u.std(axis=0)
        mean_y, std_y = y_u.mean(axis=0), y_u.std(axis=0)
        xl, yl = l2_normalise(_standardise(x_l, mean_x, std_x)), l2_normalise(_standardise(y_l, mean_y, std_y))
        xu, yu = l2_normalise(_standardise(x_u, mean_x, std_x)), l2_normalise(_standardise(y_u, mean_y, std_y))

    assert_uniform_equals_cosine(xu, yu)
    return GeometryResult(x_label=xl, y_label=yl, x_unlabelled=xu, y_unlabelled=yu, name=str(name))


def features(name: str, *, dim: int | None = None, grid: int = 16,
             dpm_root: str | None = None) -> tuple[Float[np.ndarray, "subjects d"],
                                                   Float[np.ndarray, "subjects d"]]:
    """
    CN: 按名字取特征，返回 `(T1, T2)` 两个 `(subjects, d)` 数组，**未做任何后处理**。
        `dinov2-pool4` / `dinov2-cls` 需要 DPM 侧编码器（首次会下载权重）；`block` 是本地块特征。
    EN: Feature arrays `(T1, T2)` of shape `(subjects, d)`, with **no post-processing**. `dinov2-*` needs the
        DPM-side encoders (weights downloaded on first use); `block` is the local block-mean feature.
    """
    if str(name) not in _KNOWN_FEATURES:
        raise RegistryError(f"unknown feature {name!r}; known: {sorted(_KNOWN_FEATURES)}")
    if name == "block":
        from iot_experiments.harness.data import block_features_all

        x, y = block_features_all(grid=int(grid))
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    # CN: DINOv2 走 DPM 包（与项目其它部分同一实现）。
    # EN: DINOv2 goes through the DPM package (same implementation as the rest of the project).
    import sys

    root = str(dpm_root or "/home/agent/Documents/dpm-inverse-ot")
    if root not in sys.path:
        sys.path.insert(0, root)
    from DPM.embeddings.encoders import load_encoder_latents  # noqa: PLC0415

    if name == "dinov2-cls":
        x, _ids = load_encoder_latents("dinov2", "T1", split="train")
        y, _ids_y = load_encoder_latents("dinov2", "T2", split="train")
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    # CN: pool4 + PCA，基在全部未标注 T1+T2 上拟合一次。
    # EN: pool4 + PCA, basis fitted once on all unlabelled T1+T2.
    from iot_experiments.harness.data import dinov2_pool4_pca  # noqa: PLC0415

    x, y = dinov2_pool4_pca(int(dim or 256))
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


__all__ = [
    "FeatureName",
    "GeometryName",
    "GeometryResult",
    "RegistryError",
    "assert_uniform_equals_cosine",
    "features",
    "geometry",
    "l2_normalise",
]
