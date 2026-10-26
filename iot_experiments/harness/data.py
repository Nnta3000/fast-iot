"""
CN: **数据加载** —— 解码 MRI 切片、生成块特征、DINOv2 pool4 + PCA。**无标签、无划分逻辑**。

    CN: 这里只做"从磁盘到数组"，不做任何标准化/中心化（那些在 `registry`）。这样"特征"与"几何"
        两个概念在代码里彻底分开，也就不可能再出现两个叫 cosine 的不同流水线。

EN: **Data loading** -- decode MRI slices, build block features, and produce DINOv2 pool4 coordinates with a
    PCA basis. **No labels, no splitting logic.**

    This module only goes from disk to arrays; all standardisation/centring lives in `registry`. Keeping
    "features" and "geometry" apart in the code is what makes two different pipelines both called "cosine"
    impossible to reintroduce.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from jaxtyping import Float, Shaped

IMG_SIZE: int = 192
_DEFAULT_DATA_ROOT: Path = Path("/home/agent/Documents/dpm-inverse-ot/data/Task_002_MRI_T1_T2/Train")
_DEFAULT_LATENT_ROOT: Path = Path("/home/agent/Documents/dpm-inverse-ot/DPM/embeddings")


def data_root(path: str | Path | None = None) -> Path:
    """CN: 数据集根（含 `T1/`、`T2/`）。EN: dataset root (containing `T1/`, `T2/`)."""
    return Path(path) if path is not None else _DEFAULT_DATA_ROOT


def image_files(modality: str, *, root: str | Path | None = None) -> list[Path]:
    """CN: 某模态的图像文件，按 id 升序。EN: image files of one modality, sorted by id."""
    folder = data_root(root) / modality.upper()
    return sorted(folder.glob("mri_*.png"), key=lambda p: int(p.stem.split("_")[1]))


def load_gray(path: Path, *, size: int = IMG_SIZE) -> Float[np.ndarray, "img img"]:
    """CN: 用 ImageMagick 读灰度 PNG 并缩放到 `size`。EN: read a grayscale PNG via ImageMagick."""
    raw = subprocess.run(
        ["magick", str(path), "-colorspace", "Gray", "-resize", f"{size}x{size}!",
         "-depth", "8", "gray:-"],
        check=True, capture_output=True,
    ).stdout
    # CN: ImageMagick 输出 uint8 灰度；`Int[np.ndarray, ...]` 不接受无符号类型，故中间量用 `Shaped`。
    # EN: ImageMagick emits uint8 grey; `Int[np.ndarray, ...]` rejects unsigned dtypes, so the intermediate
    #     is annotated `Shaped`.
    arr: Shaped[np.ndarray, "img img"] = np.frombuffer(raw, dtype=np.uint8).reshape(size, size)
    return arr.astype(np.float64) / 255.0


def block_features(image: Float[np.ndarray, "img img"], grid: int = 16
                   ) -> Float[np.ndarray, "blocks"]:
    """
    CN: `grid x grid` 块均值特征；`blocks = grid*grid`。
    EN: `grid x grid` block-mean features; `blocks = grid*grid`.
    """
    step = image.shape[0] // int(grid)
    return image.reshape(int(grid), step, int(grid), step).mean(axis=(1, 3)).reshape(-1)


def block_features_all(grid: int = 16, *, root: str | Path | None = None
                       ) -> tuple[Float[np.ndarray, "subjects blocks"],
                                  Float[np.ndarray, "subjects blocks"]]:
    """CN: 全部 T1/T2 的块特征。EN: block features for all T1/T2 slices."""
    out: dict[str, Float[np.ndarray, "subjects blocks"]] = {}
    for mod in ("T1", "T2"):
        files = image_files(mod, root=root)
        out[mod] = np.stack([block_features(load_gray(f), grid) for f in files]).astype(np.float64)
    return out["T1"], out["T2"]


def dinov2_pool4_pca(dim: int, *, root: str | Path | None = None,
                     latent_root: str | Path | None = None
                     ) -> tuple[Float[np.ndarray, "subjects K"], Float[np.ndarray, "subjects K"]]:
    """
    CN: DINOv2 patch token 池化到 4×4（16×384=6144 维），PCA 到 `dim` 维。
        **PCA 基在全部（标注+未标注）T1+T2 图像上拟合一次**（池化均值，`np.linalg.svd`）：
        只用图像本身、**不使用任何配对信息**，因此不是标签泄漏（与 DPM 项目的做法一致）。
    EN: Pool DINOv2 patch tokens to a 4x4 grid (16*384 = 6144 dims) and PCA-reduce to `dim`. The **PCA basis
        is fitted once on all T1+T2 images, labelled and unlabelled** (pooled mean, `np.linalg.svd`), using
        only the images and **no pairing information**, so it is not label leakage.
    """
    # CN: 只有**显式传入**非默认值时才拒绝（DPM 侧的编码器加载器接受固定路径）。
    # EN: refuse only when a **non-default** root is passed explicitly (the DPM encoder loader takes fixed
    #     paths).
    if root is not None or latent_root is not None:  # pragma: no cover - cache paths
        raise NotImplementedError("custom roots are not supported for the DINOv2 path yet")
    root_default = Path("/home/agent/Documents/dpm-inverse-ot")
    import sys

    if str(root_default) not in sys.path:
        sys.path.insert(0, str(root_default))

    import timm  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                              img_size=224).eval().to("cuda")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    pooled: dict[str, Float[np.ndarray, "subjects blocks384"]] = {}
    for mod in ("T1", "T2"):
        files = image_files(mod)
        acc: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(files), 32):
                arrs = []
                for p in files[start:start + 32]:
                    im = Image.open(p).convert("L").resize((224, 224), Image.BICUBIC)
                    g = np.asarray(im, dtype=np.float32) / 255.0
                    arrs.append(np.repeat(g[None], 3, axis=0))
                x = torch.from_numpy(((np.stack(arrs) - mean) / std).astype(np.float32)).to("cuda")
                f = model.forward_features(x)
                p4 = f[:, 1:].reshape(f.shape[0], 4, 4, 4, 4, -1).mean(dim=(2, 4))
                acc.append(p4.reshape(f.shape[0], -1).cpu().numpy())
        pooled[mod] = np.concatenate(acc, axis=0).astype(np.float64)
    stacked: Float[np.ndarray, "pooled blocks384"] = np.concatenate([pooled["T1"], pooled["T2"]], axis=0)
    mu: Float[np.ndarray, "blocks384"] = stacked.mean(axis=0)
    _u, _s, vt = np.linalg.svd(stacked - mu[None, :], full_matrices=False)
    basis: Float[np.ndarray, "blocks384 K"] = vt[: int(dim)].T
    return ((pooled["T1"] - mu[None, :]) @ basis).astype(np.float64), \
           ((pooled["T2"] - mu[None, :]) @ basis).astype(np.float64)


__all__ = [
    "IMG_SIZE",
    "block_features",
    "block_features_all",
    "data_root",
    "dinov2_pool4_pca",
    "image_files",
    "load_gray",
]
