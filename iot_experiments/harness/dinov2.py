"""
CN: **DINOv2 池化特征（通用版）** —— 输入就是一组 192×192 的切片数组，输出每个切片的特征向量。

    CN: 为什么需要通用版：`data.dinov2_pool4_pca` 写死了 T1/T2 的 PNG 目录，SynthRAD 的 MRI/CT 是 NIfTI 数组，
        无法复用。这里把"编码器"与"数据来源"分开——输入是数组，所以任何数据集都能用同一个编码器，
        这也正是实验中"两模态共享一个 encoder"的要求。

    CN: 预处理与 T1/T2 一致：缩放到 224、灰度复制成 3 通道、ImageNet 均值方差、
        `vit_small_patch14_dinov2`、patch token 池化到 4×4（16×384 = 6144 维）。

    CN: **PCA 在这里单独提供**（`pca_fit` / `pca_apply`），因为 PCA 基必须在**两模态全部池切片**上拟合
        （无标签），再由调用方按几何标准化。把 PCA 与编码器分开，才能保证"PCA 不偷看标签"。

EN: **DINOv2 pooled features (generic)** -- takes an array of 192x192 slices and returns one feature vector
    per slice.

    Why: `data.dinov2_pool4_pca` is hard-wired to the T1/T2 PNG folders, so it cannot be reused for
    SynthRAD's NIfTI arrays. Here the encoder is separated from the data source: the input is an array, so one
    encoder serves any dataset -- which is also the "one shared encoder for both modalities" the experiments
    require.

    Preprocessing matches the T1/T2 path: resize to 224, replicate grey to 3 channels, ImageNet mean/std,
    `vit_small_patch14_dinov2`, patch tokens pooled to 4x4 (16*384 = 6144 dims).

    **PCA is separate** (`pca_fit` / `pca_apply`) because the basis must be fitted on **all pool slices of both
    modalities** (label-free) and the caller then applies the geometry. Keeping PCA out of the encoder is what
    makes "PCA never sees the labels" auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

DINOV2_INPUT: int = 224
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class EncoderError(RuntimeError):
    """CN: 编码器前置条件失败。EN: an encoder precondition failed."""


@dataclass(frozen=True)
class PooledFeatures:
    """
    CN: 一个批次的池化特征：`(batch, 6144)` 的 float32，以及每切片的分块形状（4×4 网格 × 384 通道）。
    EN: one batch's pooled features: `(batch, 6144)` float32, from a 4x4 grid of 384-dim patch tokens.
    """

    pooled: Float[np.ndarray, "batch features"]
    grid: int = 4

    def __post_init__(self) -> None:
        """CN: 形状必须是 `(batch, grid*grid*384)`。EN: shape must be `(batch, grid*grid*384)`."""
        expected = int(self.grid) * int(self.grid) * 384
        if self.pooled.shape[1] != expected:
            raise EncoderError(f"expected {expected} features, got {self.pooled.shape[1]}")


_CACHE: dict[str, object] = {}


def _model(device: str) -> object:
    """
    CN: 懒加载并缓存 DINOv2（ViT-S/14，224 输入）。同一进程内只加载一次，节省显存与时间。
    EN: lazily load and cache DINOv2 (ViT-S/14 at 224). Loaded once per process to save memory and time.
    """
    if device in _CACHE:
        return _CACHE[device]
    import timm  # noqa: PLC0415

    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                              img_size=DINOV2_INPUT).eval().to(device)
    _CACHE[device] = model
    return model


def _prepare(images: Float[np.ndarray, "batch side side"]) -> "object":
    """
    CN: `(B, S, S)` in [0,1] → 张量 `(B, 3, 224, 224)`，已做 ImageNet 标准化。
    EN: `(B, S, S)` in [0,1] -> a `(B, 3, 224, 224)` tensor, ImageNet-normalised.
    """
    import torch  # noqa: PLC0415

    from scipy.ndimage import zoom  # noqa: PLC0415

    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim != 3:
        raise EncoderError(f"expected (batch, side, side), got {arr.shape}")
    side = int(arr.shape[1])
    if side != DINOV2_INPUT:
        factor = float(DINOV2_INPUT) / float(side)
        arr = np.stack([zoom(arr[i], factor, order=1) for i in range(arr.shape[0])]).astype(np.float32)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[None, :, None, None]
    rgb = np.repeat(arr[:, None, :, :], 3, axis=1)
    return torch.from_numpy(((rgb - mean) / std).astype(np.float32))


def encode_pool4(images: Float[np.ndarray, "batch side side"], *, device: str = "cuda",
                 grid: int = 4, batch_size: int = 32) -> PooledFeatures:
    """
    CN: 把一个批次的切片编码成 DINOv2 pool4 特征。`grid=4` 时是 16 个格 × 384 = 6144 维。
        **不在这里做 PCA**（见模块说明）。
    EN: encode a batch of slices into DINOv2 pool4 features; `grid=4` gives 16 cells x 384 = 6144 dims.
        **No PCA here** (see the module docstring).
    """
    import torch  # noqa: PLC0415

    model = _model(str(device))
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(images.shape[0]), int(batch_size)):
            chunk = _prepare(np.asarray(images[start:start + int(batch_size)], dtype=np.float32))
            chunk = chunk.to(str(device))
            feats = model.forward_features(chunk)
            tokens = feats[:, 1:, :]
            side = int(grid)
            patches_per_side = int(round(float(tokens.shape[1]) ** 0.5))
            reshaped = tokens.reshape(tokens.shape[0], side, patches_per_side // side,
                                      side, patches_per_side // side, tokens.shape[2])
            pooled = reshaped.mean(dim=(2, 4))
            out.append(pooled.reshape(pooled.shape[0], -1).cpu().numpy().astype(np.float32))
    stacked: Float[np.ndarray, "batch features"] = np.concatenate(out, axis=0)
    return PooledFeatures(pooled=stacked, grid=int(grid))


@dataclass(frozen=True)
class PcaBasis:
    """
    CN: 一个 PCA 基：均值 + 前 k 个方向。拟合时必须只用**无标签**的池数据。
    EN: a PCA basis: the mean plus the first k directions. Fitting must use **unlabelled** pool data only.
    """

    mean: Float[np.ndarray, "features"]
    components: Float[np.ndarray, "features k"]

    @property
    def k(self) -> int:
        """CN: 保留的主成分数。EN: the number of components kept."""
        return int(self.components.shape[1])


def pca_fit(pooled: Float[np.ndarray, "n features"], *, k: int) -> PcaBasis:
    """
    CN: 在**两模态全部池切片**的拼接上拟合 PCA（无标签）。用 SVD；
    EN: fit PCA on the concatenation of **all pool slices of both modalities** (label-free), via SVD.
    """
    arr = np.asarray(pooled, dtype=np.float64)
    if arr.shape[0] < 2:
        raise EncoderError("need at least two slices to fit PCA")
    mean = arr.mean(axis=0)
    centred = arr - mean[None, :]
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    kk = int(min(k, vt.shape[0]))
    return PcaBasis(mean=mean, components=vt[:kk].T.copy())


def pca_apply(pooled: Float[np.ndarray, "n features"], basis: PcaBasis
              ) -> Float[np.ndarray, "n k"]:
    """CN: 投影到 PCA 基。EN: project onto the PCA basis."""
    arr = np.asarray(pooled, dtype=np.float64)
    return (arr - basis.mean[None, :]) @ basis.components


def cache_root() -> Path:
    """
    CN: 特征缓存的根目录（与切片缓存一致）。EN: root of the feature cache (same tree as the slice cache).
    """
    return Path(__file__).resolve().parents[2] / "cache" / "features"


__all__ = [
    "DINOV2_INPUT",
    "EncoderError",
    "PcaBasis",
    "PooledFeatures",
    "cache_root",
    "encode_pool4",
    "pca_apply",
    "pca_fit",
]
