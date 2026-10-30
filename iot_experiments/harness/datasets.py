"""
CN: **SynthRAD 数据工具** —— 切片加载、对齐表、特征流、降维。原本住在
    `studies/real/synthrad/v2_run.py` 里，导致**一个实验脚本同时充当库**（5 个脚本 import 它）。
    现搬到 harness：实验脚本只做实验，共享工具只有一份。

EN: **SynthRAD data helpers** -- slice loading, the aligned table, feature streaming and reduction. These
    used to live inside `studies/real/synthrad/v2_run.py`, so a *study script doubled as a library* (five
    scripts imported it). They live in the harness now: a study script only does the study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from iot_experiments.harness.cache import read_slices, save_slices, slices_path
from iot_experiments.harness.dinov2 import encode_pool4, pca_apply, pca_fit
from iot_experiments.harness.registry import apply_geometry, fit_geometry
from iot_experiments.harness.synthrad import (
    IMAGE_SIZE,
    MIN_MASK_AREA,
    MR_PERCENTILES,
    PatientPaths,
    SynthRadSlices,
    preprocess_patient,
)

class DatasetError(RuntimeError):
    """CN: 数据装载的前置条件失败（原 `v2_run.V2Error`）。EN: a data-loading precondition failed."""


__all__ = [
    "DatasetError", "IMAGE_SIZE", "MIN_MASK_AREA", "MR_PERCENTILES", "SynthRadSlices", "Table",
    "build_table", "cosine_cost", "features", "load_slices", "preprocess_patient",
    "read_slices", "reduce_features", "save_slices", "slices_path", "stream_features",
]

def load_slices(paths: PatientPaths, *, window: tuple[float, float]) -> SynthRadSlices:
    """CN: 读缓存或预处理（键含 CT 窗）。EN: cached read, else preprocess (the key includes the CT window)."""
    target = slices_path(paths.patient, size=IMAGE_SIZE, ct_window=tuple(window),
                         mr_percentiles=MR_PERCENTILES, min_mask_area=MIN_MASK_AREA,
                         anatomy=paths.anatomy.lower())
    if target.is_file():
        data = read_slices(target)
        return SynthRadSlices(patient=str(data["meta"]["patient"]), centre=str(data["meta"]["centre"]),
                              mr=np.asarray(data["mr"]), ct=np.asarray(data["ct"]),
                              mask=np.asarray(data["mask"]), z_normalised=np.asarray(data["z_normalised"]),
                              z_index=np.asarray(data["z_index"]), meta=dict(data["meta"]))
    slices = preprocess_patient(paths, ct_window=tuple(window))
    save_slices(target, mr=slices.mr, ct=slices.ct, mask=slices.mask,
                z_normalised=slices.z_normalised, z_index=slices.z_index,
                patient=slices.patient, centre=slices.centre, meta=slices.meta)
    return slices

@dataclass(frozen=True)
class Table:
    """CN: 对齐的切片表：第 i 行的 MR 与第 i 行的 CT 是**同一患者同一层**。EN: aligned table (row i = one slice)."""

    mr: Float[np.ndarray, "n pixels"]
    ct: Float[np.ndarray, "n pixels"]
    mask: Bool[np.ndarray, "n pixels"]
    patient: list[str]
    centre: list[str]
    z_index: Int[np.ndarray, "n"]
    z: Float[np.ndarray, "n"]

    def __len__(self) -> int:
        """CN: 切片数。EN: number of slices."""
        return int(self.mr.shape[0])

def build_table(patients: list[PatientPaths], *, window: tuple[float, float], max_per_patient: int | None,
                min_gap: int, seed: int) -> Table:
    """
    CN: `max_per_patient=None` 表示**全部切片**（用于拟合）。EN: `None` means **all slices** (for fitting).
    """
    rng = np.random.default_rng(int(seed))
    mr, ct, mask, pid, ctr, zidx, z = [], [], [], [], [], [], []
    for paths in patients:
        slices = load_slices(paths, window=window)
        n = len(slices)
        if max_per_patient is None:
            picked = np.arange(n, dtype=np.int64)
        else:
            order = rng.permutation(n)
            chosen: list[int] = []
            for idx in order:
                if all(abs(int(idx) - c) >= int(min_gap) for c in chosen):
                    chosen.append(int(idx))
                    if len(chosen) >= int(max_per_patient):
                        break
            picked = np.sort(np.asarray(chosen, dtype=np.int64))
        for i in picked:
            mr.append(slices.mr[i]); ct.append(slices.ct[i]); mask.append(slices.mask[i])
            pid.append(slices.patient); ctr.append(slices.centre)
            zidx.append(int(slices.z_index[i])); z.append(float(slices.z_normalised[i]))
    if not mr:
        raise DatasetError("empty table")
    return Table(mr=np.stack(mr), ct=np.stack(ct), mask=np.stack(mask), patient=pid, centre=ctr,
                 z_index=np.asarray(zidx, dtype=np.int64), z=np.asarray(z, dtype=np.float64))


# ---------------------------------------------------------------------------------------
# CN: 特征：DINOv2 pool4 -> label-free PCA -> 几何 -> 逐行 L2。`whiten=True` 时 PCA 后按各主成分标准差白化。
# EN: features: DINOv2 pool4 -> label-free PCA -> geometry -> row L2. `whiten=True` divides each component by
#     its standard deviation before the geometry.
# ---------------------------------------------------------------------------------------

def features(mr: Float[np.ndarray, "n pixels"], ct: Float[np.ndarray, "n pixels"], *, k: int,
             geometry_name: str, whiten: bool) -> tuple[Float[np.ndarray, "n k"], Float[np.ndarray, "n k"]]:
    """CN: 两模态共享编码器与 PCA 基。EN: one shared encoder and PCA basis for both modalities."""
    side = int(round(float(np.asarray(mr).shape[1]) ** 0.5))
    fm = encode_pool4(np.asarray(mr).reshape(-1, side, side), device="cuda").pooled.astype(np.float64)
    fc = encode_pool4(np.asarray(ct).reshape(-1, side, side), device="cuda").pooled.astype(np.float64)
    basis = pca_fit(np.concatenate([fm, fc], axis=0), k=int(k))
    proj_mr, proj_ct = pca_apply(fm, basis), pca_apply(fc, basis)
    if whiten:
        pooled = np.concatenate([proj_mr, proj_ct], axis=0)
        sd = pooled.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        proj_mr, proj_ct = proj_mr / sd, proj_ct / sd
    stats = fit_geometry(proj_mr, proj_ct, str(geometry_name))
    out_mr, out_ct = apply_geometry(proj_mr, proj_ct, stats)
    return np.asarray(out_mr, dtype=np.float64), np.asarray(out_ct, dtype=np.float64)

def stream_features(patients: list[PatientPaths], *, window: tuple[float, float], max_per_patient: int | None,
                    min_gap: int, seed: int) -> tuple[Float[np.ndarray, "n features"],
                                                      Float[np.ndarray, "n features"], list[str],
                                                      Int[np.ndarray, "n"], list[str]]:
    """
    CN: **逐患者**编码，只保留 DINOv2 特征，**从不**把所有像素同时留在内存里。
        60 个患者约 11,400 片；像素级 float64 需要 ~6.7 GB（上次因此 OOM），特征只需 11,400×6144×4 B ≈ 0.28 GB。
        返回 `(pooled_mr, pooled_ct, patients, z_index, centres)`。
    EN: encode **patient by patient**, keeping only DINOv2 features and **never** holding all pixels at once.
        60 patients is ~11,400 slices; float64 pixels need ~6.7 GB (this OOM-killed the first attempt), while
        features need 11,400 x 6144 x 4 B ~ 0.28 GB.
    """
    rng = np.random.default_rng(int(seed))
    fm: list[np.ndarray] = []
    fc: list[np.ndarray] = []
    pid: list[str] = []
    ctr: list[str] = []
    zidx: list[int] = []
    for paths in patients:
        slices = load_slices(paths, window=window)
        n = len(slices)
        if max_per_patient is None:
            picked = np.arange(n, dtype=np.int64)
        else:
            order = rng.permutation(n)
            chosen: list[int] = []
            for idx in order:
                if all(abs(int(idx) - c) >= int(min_gap) for c in chosen):
                    chosen.append(int(idx))
                    if len(chosen) >= int(max_per_patient):
                        break
            picked = np.sort(np.asarray(chosen, dtype=np.int64))
        side = IMAGE_SIZE
        # CN: **分块编码并立即释放**：每个患者约 190 片 × 36,864 float64 × 2 模态 ≈ 112 MB 常驻，
        #     60 个患者连日累积会 OOM（实测 15 GB 机器上 exit 137）。这里按 32 片一块处理。
        # EN: **encode in chunks and release immediately**: one patient is ~190 slices x 36,864 float64 x 2
        #     modalities ~ 112 MB resident, and 60 patients accumulate to an OOM on a 15 GB machine (measured
        #     exit 137). Process in blocks of 32 slices.
        block = 32
        for start in range(0, len(picked), block):
            idx = picked[start:start + block]
            mr_imgs = np.asarray(slices.mr[idx], dtype=np.float32).reshape(-1, side, side)
            fm.append(encode_pool4(mr_imgs, device="cuda").pooled.astype(np.float32))
            del mr_imgs
            ct_imgs = np.asarray(slices.ct[idx], dtype=np.float32).reshape(-1, side, side)
            fc.append(encode_pool4(ct_imgs, device="cuda").pooled.astype(np.float32))
            del ct_imgs
        for i in picked:
            pid.append(slices.patient); ctr.append(slices.centre); zidx.append(int(slices.z_index[i]))
        del slices
        import gc as _gc  # noqa: PLC0415
        _gc.collect()
    if not fm:
        raise DatasetError("no slices encoded")
    return (np.concatenate(fm, axis=0).astype(np.float64), np.concatenate(fc, axis=0).astype(np.float64),
            pid, np.asarray(zidx, dtype=np.int64), ctr)

def reduce_features(pooled_mr: Float[np.ndarray, "n features"], pooled_ct: Float[np.ndarray, "n features"],
                    *, k: int, geometry_name: str, whiten: bool
                    ) -> tuple[Float[np.ndarray, "n k"], Float[np.ndarray, "n k"]]:
    """
    CN: 在**已池化**的特征上做 label-free PCA 与几何变换（不再重新跑编码器）。
    EN: label-free PCA and geometry on **already pooled** features (no second encoder pass).
    """
    # CN: **PCA 只在行的样本上拟合**。全部 22,526 行 × 6144 维做 SVD 需要 1.1 GB 输入 + 约 2.2 GB
    #     LAPACK 工作区，在 15 GB 机器上（ComfyUI 等占用后只剩 7 GB）会被 OOM 杀掉——实测 exit 137。
    #     行数是几万的量级，用 4,000 行的随机子集拟合主方向，统计上无差别，内存降一个数量级。
    # EN: **fit PCA on a SAMPLE of rows.** A full SVD on 22,526 x 6144 needs 1.1 GB of input plus ~2.2 GB of
    #     LAPACK workspace and gets OOM-killed on this 15 GB machine (7 GB free; measured exit 137). With tens
    #     of thousands of rows a random 4,000-row subset gives the same principal directions and an order of
    #     magnitude less memory.
    stacked = np.concatenate([pooled_mr, pooled_ct], axis=0)
    limit = 4000
    if stacked.shape[0] > limit:
        take = np.random.default_rng(424_242).permutation(stacked.shape[0])[:limit]
        stacked_sample = stacked[np.sort(take)]
    else:
        stacked_sample = stacked
    basis = pca_fit(stacked_sample, k=int(k))
    del stacked, stacked_sample
    proj_mr, proj_ct = pca_apply(pooled_mr, basis), pca_apply(pooled_ct, basis)
    if whiten:
        sd = np.concatenate([proj_mr, proj_ct], axis=0).std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        proj_mr, proj_ct = proj_mr / sd, proj_ct / sd
    stats = fit_geometry(proj_mr, proj_ct, str(geometry_name))
    out_mr, out_ct = apply_geometry(proj_mr, proj_ct, stats)
    return np.asarray(out_mr, dtype=np.float64), np.asarray(out_ct, dtype=np.float64)

def cosine_cost(a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "n n"]:
    """CN: 余弦距离。EN: cosine distance."""
    return 1.0 - np.asarray(a) @ np.asarray(b).T


# ---------------------------------------------------------------------------------------
# CN: 学到的代价：在 FIT 的**真实配对上**解逆 OT（盒 ∩ 单纯形）；`rel` 在 FIT 的验证折上选。
# EN: learned costs: inverse OT on the FIT **true pairs** (box ∩ simplex); `rel` chosen on a FIT validation fold.
# ---------------------------------------------------------------------------------------
