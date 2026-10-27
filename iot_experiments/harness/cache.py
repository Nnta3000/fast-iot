"""
CN: **实验缓存** —— 预处理切片与池化特征的磁盘缓存。**可删除、可重建**，从不入版本库。

    CN: 为什么需要：从 NIfTI 读一次 192×192 切片约 1 s/患者，DINOv2 前向更慢；探针与门禁要反复读同一批
        切片，重算代价远超缓存本身。缓存只存**已经固定预处理**的结果，因此读缓存与直接预处理在语义上
        完全相同（同样的常量、同样的掩膜规则）。

    CN: **缓存键**包含全部影响内容的参数（数据集根、解剖、患者、尺寸、CT 窗、MR 百分位、掩膜阈值），
        任何参数变化都会得到不同的文件，**不会**静默复用旧数据。

    EN: **Experiment caches** -- on-disk cache for preprocessed slices and pooled features. **Rebuildable and
        deletable**, never committed.

        Reading a 192x192 slice from NIfTI costs about 1 s per patient and DINOv2 is slower still; the probe
        and the gate re-read the same slices many times. The cache stores only the result of the **fixed
        preprocessing**, so a cache read and a fresh preprocess are semantically identical.

        The **cache key** includes every parameter that changes the content (dataset root, anatomy, patient,
        size, CT window, MR percentiles, mask threshold), so changing any of them yields a different file
        instead of silently reusing stale data.

用法 / EN: usage
    from iot_experiments.harness.cache import load_slices_cached
    slices = load_slices_cached(paths)          # numpy arrays, cached
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT: Path = _REPO_ROOT / "cache"


class CacheError(RuntimeError):
    """CN: 缓存读写失败。EN: a cache read or write failed."""


def _key(*parts: object) -> str:
    """
    CN: 由参数生成稳定的短键。参数按 `repr` 拼接后取 sha1 前 16 位——只用于命名，不用于安全。
    EN: build a stable short key from the parameters: `repr` them, join, take the first 16 hex of sha1. Used
        for naming only, not for security.
    """
    joined = "|".join(repr(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def slices_path(patient: str, *, size: int, ct_window: tuple[float, float],
                mr_percentiles: tuple[float, float], min_mask_area: float, anatomy: str) -> Path:
    """
    CN: 某个患者预处理切片的缓存路径。键包含所有影响内容的参数。
    EN: cache path for one patient's preprocessed slices; the key covers every content-affecting parameter.
    """
    key = _key(patient, anatomy, size, tuple(ct_window), tuple(mr_percentiles), float(min_mask_area))
    return CACHE_ROOT / "slices" / f"{anatomy}_{patient}_{key}.npz"


def save_slices(path: Path, *, mr: Float[np.ndarray, "slices pixels"],
                ct: Float[np.ndarray, "slices pixels"], mask: Bool[np.ndarray, "slices pixels"],
                z_normalised: Float[np.ndarray, "slices"], z_index: Int[np.ndarray, "slices"],
                patient: str, centre: str, meta: dict[str, object]) -> None:
    """
    CN: 存为 **uint8**（crop 在 [0,1]，乘 255 后无损到 1/255；掩膜另存为 bool），体积约为 float64 的 1/8。
    EN: store as **uint8** (values in [0,1] scaled by 255, lossless to 1/255; the mask stored as bool),
        about 1/8 the size of float64.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mr_u8=np.round(np.clip(mr, 0.0, 1.0) * 255.0).astype(np.uint8),
        ct_u8=np.round(np.clip(ct, 0.0, 1.0) * 255.0).astype(np.uint8),
        mask=np.asarray(mask, dtype=bool),
        z_normalised=np.asarray(z_normalised, dtype=np.float32),
        z_index=np.asarray(z_index, dtype=np.int32),
        meta=np.asarray(json.dumps({"patient": patient, "centre": centre, **meta})),
    )


def read_slices(path: Path) -> dict[str, object]:
    """CN: 读缓存，uint8 → float64/255；返回数组与元数据。EN: read the cache, uint8 -> float64/255."""
    if not path.is_file():
        raise CacheError(f"missing slice cache: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {
            "mr": data["mr_u8"].astype(np.float64) / 255.0,
            "ct": data["ct_u8"].astype(np.float64) / 255.0,
            "mask": data["mask"].astype(bool),
            "z_normalised": data["z_normalised"].astype(np.float64),
            "z_index": data["z_index"].astype(np.int64),
            "meta": json.loads(str(data["meta"])),
        }


def features_path(patient: str, *, encoder: str, size: int, anatomy: str) -> Path:
    """
    CN: 池化特征的缓存路径。**按切片**存（每片一行），float16。
    EN: cache path for pooled features, **per slice** (one row each), float16.
    """
    key = _key(patient, anatomy, encoder, int(size))
    return CACHE_ROOT / "features" / f"{anatomy}_{patient}_{encoder}_{key}.npz"


def save_features(path: Path, *, modality: str, pooled: Float[np.ndarray, "slices dims"],
                  z_index: Int[np.ndarray, "slices"]) -> None:
    """
    CN: 保存某患者某模态的池化特征（float16，省一半空间）。`z_index` 用于与切片对齐。
    EN: save one patient's pooled features for one modality (float16, half the space); `z_index` keeps them
        aligned with the slices.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, np.ndarray] = {}
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            existing = {k: data[k] for k in data.files}
    existing[f"{modality}_pooled"] = np.asarray(pooled, dtype=np.float16)
    existing[f"{modality}_z"] = np.asarray(z_index, dtype=np.int32)
    np.savez_compressed(path, **existing)


def cache_size_gb() -> float:
    """CN: 缓存占用（GB），用于在报告里说明成本。EN: cache size in GB, for reporting the cost."""
    total = sum(p.stat().st_size for p in CACHE_ROOT.rglob("*") if p.is_file()) if CACHE_ROOT.is_dir() else 0
    return float(total) / 1e9


__all__ = [
    "CACHE_ROOT",
    "CacheError",
    "cache_size_gb",
    "features_path",
    "read_slices",
    "save_features",
    "save_slices",
    "slices_path",
]
