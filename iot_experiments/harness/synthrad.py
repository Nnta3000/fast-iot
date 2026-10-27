"""
CN: **SynthRAD2023 数据加载** —— 只读 NIfTI（自带最小解析器，不引入新依赖）与切片登记。

    CN: 本模块**不依赖任何下载**：`load_volume` 只读文件，测试用合成 NIfTI 覆盖全部路径。
        真实数据到位后无需改动代码，只改路径。

    CN: **固定的预处理**（每个臂完全相同，报告里也必须照抄这一段）：

      | 步骤 | 规则 |
      |---|---|
      | 切片 | 只取**轴位**（沿第 3 轴），方向以 NIfTI 仿射为准，不翻转 |
      | 尺寸 | 缩放到 **192×192**（双线性，抗锯齿） |
      | 丢弃 | 身体掩膜面积 < **5%** 的切片 |
      | CT 窗 | 固定窗 **-200…300 HU**（软组织窗），线性映射到 [0,1]，随后裁剪 |
      | MR 归一 | **按体数据**百分位归一化：取 1% 与 99% 分位，截断后线性映射到 [0,1] |
      | 掩膜 | 最近邻缩放到 192×192，阈值 0.5 |
      | `z_normalised` | 切片序号 / (有效切片数 - 1)，范围 [0,1] |

    CN: 为什么 MR 用**体数据**百分位而不是逐切片：逐切片会把"这一层没什么组织"的切片拉到满量程，
        破坏层与层之间的可比性；体数据归一化是同一患者的整体对比度，符合配对任务。

    CN: **登记对是"同患者对齐"的**（MR 与 CT 来自同一患者、同一层位），所以"隐藏配对"测试对像素方法
        会**饱和**（见 `reports/NMI_PAIRING_REPORT.md`）。这个数据集只能把隐藏配对当**健全性检查**，
        不能当头条结果。

EN: **SynthRAD2023 data loading** -- NIfTI reading (a minimal self-contained parser; no new dependency) and
    slice registration.

    This module **needs no download**: `load_volume` only reads files, and the tests cover every path with
    synthetic NIfTI. When the real data arrives only the path changes.

    **Fixed preprocessing** (identical for every arm; quote this block in any report):
    axial slices only (3rd axis, no flip), resized to 192x192 (bilinear, antialiased), slices with body-mask
    area < 5% dropped, CT clipped to a **fixed -200..300 HU** soft-tissue window mapped to [0,1], MR
    percentile-normalised **per volume** (1st/99th percentile), mask nearest-neighbour at 0.5,
    `z_normalised = index / (n_valid - 1)` in [0,1].

    MR is normalised per **volume**, not per slice: per-slice normalisation would stretch near-empty slices
    to full range and destroy slice-to-slice comparability.

    **Registered pairs are same-patient and aligned**, so the hidden-partner test is **saturated** for pixel
    methods (see `reports/NMI_PAIRING_REPORT.md`). On this dataset it is a **sanity check only**, never a
    headline result.
"""

from __future__ import annotations

import gzip
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int, Shaped

# CN: 固定的预处理常量（改动必须同时改报告，否则表格无法复现）。
# EN: fixed preprocessing constants; a change here must change the report too, or tables stop reproducing.
IMAGE_SIZE: int = 192
CT_WINDOW: tuple[float, float] = (-200.0, 300.0)
MR_PERCENTILES: tuple[float, float] = (1.0, 99.0)
MIN_MASK_AREA: float = 0.05
MASK_THRESHOLD: float = 0.5


class SynthRadError(RuntimeError):
    """CN: 数据布局或 NIfTI 内容不符合预期。EN: the layout or NIfTI content is not as expected."""


@dataclass(frozen=True)
class NiftiVolume:
    """
    CN: 一个已解析的 NIfTI 体积：三维数据 + 说明方向的仿射 + 体素间距。
    EN: one parsed NIfTI volume: 3-D data plus the affine that encodes orientation and the voxel spacing.
    """

    data: Float[np.ndarray, "x y z"]
    affine: Float[np.ndarray, "4 4"]
    spacing: Float[np.ndarray, "3"]

    def is_axial_first(self) -> bool:
        """
        CN: 第 3 轴（索引 2）是否是**头足方向**。是则轴位切片沿该轴取；否则本模块拒绝处理，
            而不是悄悄旋转（旋转会改变与报告中的预处理不符的像素内容）。
        EN: whether axis 2 (index) is the **superior-inferior** direction. If not, this module refuses rather
            than silently rotating, because a rotation would change pixel content away from the documented
            preprocessing.
        """
        column: Float[np.ndarray, "3"] = np.asarray(self.affine, dtype=np.float64)[:3, 2]
        dominant = int(np.argmax(np.abs(column)))
        return dominant == 2


def _read_nifti_bytes(raw: bytes) -> NiftiVolume:
    """
    CN: 解析 NIfTI-1 头部（348 字节）与体素数据。支持常见 dtype 与小端/大端。
        只实现本数据集需要的子集：三维、scl_slope/scl_inter 缩放。
    EN: parse a NIfTI-1 header (348 bytes) and the voxel data. Common dtypes and both endiannesses are
        supported; only the subset this dataset needs (3-D, `scl_slope`/`scl_inter` scaling) is implemented.
    """
    if len(raw) < 348:
        raise SynthRadError(f"file too short for a NIfTI-1 header: {len(raw)} bytes")
    sizeof_hdr: int = struct.unpack("<i", raw[:4])[0]
    endian = "<"
    if sizeof_hdr != 348:
        sizeof_hdr_be: int = struct.unpack(">i", raw[:4])[0]
        if sizeof_hdr_be != 348:
            raise SynthRadError(f"not a NIfTI-1 file (sizeof_hdr={sizeof_hdr})")
        endian = ">"
    dims: tuple[int, ...] = struct.unpack(endian + "8h", raw[40:56])
    datatype: int = struct.unpack(endian + "h", raw[70:72])[0]
    pixdim: tuple[float, ...] = struct.unpack(endian + "8f", raw[76:108])
    vox_offset: float = struct.unpack(endian + "f", raw[108:112])[0]
    slope: float = struct.unpack(endian + "f", raw[112:116])[0]
    inter: float = struct.unpack(endian + "f", raw[116:120])[0]
    qform_code: int = struct.unpack(endian + "h", raw[252:254])[0]
    sform_code: int = struct.unpack(endian + "h", raw[254:256])[0]
    srow: tuple[float, ...] = struct.unpack(endian + "12f", raw[280:328])

    ndim: int = int(dims[0])
    if ndim < 3:
        raise SynthRadError(f"expected at least 3 dimensions, got {ndim}")
    shape: tuple[int, int, int] = (int(dims[1]), int(dims[2]), int(dims[3]))

    dtypes: dict[int, str] = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2",
                              768: "u4"}
    if datatype not in dtypes:
        raise SynthRadError(f"unsupported NIfTI datatype code {datatype}")
    code = dtypes[datatype]
    if code[0] == "i" and endian == "<":
        code = "<" + code
    elif code[0] == "i" and endian == ">":
        code = ">" + code
    elif code[0] == "f":
        code = endian + code

    offset: int = int(vox_offset) if vox_offset > 0 else 352
    count: int = int(np.prod(shape))
    itemsize: int = np.dtype(code).itemsize
    need: int = offset + count * itemsize
    if len(raw) < need:
        raise SynthRadError(f"truncated voxel data: need {need} bytes, have {len(raw)}")
    # CN: NIfTI 的体素按 **Fortran 顺序**存储（x 变化最快），因此必须用 order="F" 重排；
    #     否则每个体素的坐标会被转置（数值不丢，位置全错）。
    # EN: NIfTI voxels are stored in **Fortran order** (x varies fastest), so reshape with order="F";
    #     otherwise every voxel's coordinates are transposed (values survive, positions do not).
    data: Shaped[np.ndarray, "x y z"] = np.frombuffer(raw, dtype=np.dtype(code), count=count,
                                                      offset=offset).reshape(shape, order="F"
                                                                             ).astype(np.float64)

    # CN: scl_slope == 0 表示"没有缩放"（NIfTI 规范）。
    # EN: scl_slope == 0 means "no scaling" per the NIfTI spec.
    if slope != 0.0 and not np.isnan(slope):
        data = data * float(slope) + float(inter)

    spacing: Float[np.ndarray, "3"] = np.abs(np.asarray(pixdim[1:4], dtype=np.float64))
    spacing = np.where(spacing > 0.0, spacing, 1.0)

    if sform_code > 0:
        affine: Float[np.ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        affine[:3, :] = np.asarray(srow, dtype=np.float64).reshape(3, 4)
    elif qform_code > 0:
        # CN: 无 sform 时退化为对角仿射（本数据集都带 sform；这是保守兜底）。
        # EN: without sform fall back to a diagonal affine; this dataset ships sform, so it is a guard.
        affine = np.eye(4, dtype=np.float64)
        affine[0, 0], affine[1, 1], affine[2, 2] = spacing[0], spacing[1], spacing[2]
    else:
        affine = np.eye(4, dtype=np.float64)
        affine[0, 0], affine[1, 1], affine[2, 2] = spacing[0], spacing[1], spacing[2]

    return NiftiVolume(data=data, affine=affine, spacing=spacing)


def load_volume(path: str | Path) -> NiftiVolume:
    """
    CN: 读取 `.nii` 或 `.nii.gz`。gzip 由标准库处理，不依赖 nibabel。
    EN: read `.nii` or `.nii.gz`. gzip comes from the standard library; nibabel is not required.
    """
    p = Path(path)
    if not p.is_file():
        raise SynthRadError(f"missing file: {p}")
    raw: bytes = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    return _read_nifti_bytes(raw)


def write_nifti(path: str | Path, data: Float[np.ndarray, "x y z"], *,
                spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Path:
    """
    CN: 写一个最小 NIfTI-1（float32、带 sform 的对角仿射）。仅用于**测试**与合成数据。
    EN: write a minimal NIfTI-1 (float32, diagonal sform affine). Intended for **tests** and synthetic data.
    """
    arr: Float[np.ndarray, "x y z"] = np.asarray(data, dtype=np.float32)
    shape = arr.shape
    hdr = bytearray(352)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, shape[0], shape[1], shape[2], 1, 1, 1, 1)
    struct.pack_into("<h", hdr, 70, 16)                      # datatype float32
    struct.pack_into("<h", hdr, 72, 32)                      # bitpix
    struct.pack_into("<8f", hdr, 76, 1.0, spacing[0], spacing[1], spacing[2], 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", hdr, 108, 352.0)                  # vox_offset
    struct.pack_into("<f", hdr, 112, 1.0)                    # scl_slope
    struct.pack_into("<h", hdr, 252, 0)                      # qform_code
    struct.pack_into("<h", hdr, 254, 1)                      # sform_code = scanner
    struct.pack_into("<12f", hdr, 280, spacing[0], 0.0, 0.0, 0.0,
                     0.0, spacing[1], 0.0, 0.0,
                     0.0, 0.0, spacing[2], 0.0)
    hdr[344:348] = b"n+1\x00"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = bytes(hdr) + arr.tobytes(order="F")
    if p.suffix == ".gz":
        p.write_bytes(gzip.compress(payload))
    else:
        p.write_bytes(payload)
    return p


@dataclass(frozen=True)
class PatientPaths:
    """
    CN: 一个患者的三个文件。目录名编码 `<task 数字><anatomy 字母><centre 字母><编号>`：
        `1B` = Task 1 脑，`1P` = Task 1 骨盆，第三位是中心。例：`1BA001`（Task1 脑 A 中心），
        `1PC001`（Task1 骨盆 C 中心）。**第 2 位是解剖、第 3 位是中心**，不要颠倒。
    EN: one patient's three files. The folder name encodes `<numeric task><anatomy letter><centre letter>
        <index>`: `1B` = Task 1 brain, `1P` = Task 1 pelvis, and the third character is the centre. E.g.
        `1BA001` (task 1, brain, centre A) and `1PC001` (task 1, pelvis, centre C). **Position 2 is the
        anatomy and position 3 is the centre** -- do not swap them.
    """

    patient: str
    anatomy: str
    centre: str
    mr: Path
    ct: Path
    mask: Path

    @classmethod
    def from_dir(cls, folder: str | Path) -> "PatientPaths":
        """CN: 从患者目录推断 id，并检查三个文件齐全。EN: infer the id from the folder and check all three files."""
        d = Path(folder)
        pid = d.name
        if len(pid) < 4 or not pid[0].isdigit():
            raise SynthRadError(f"unexpected patient folder name {pid!r}; expected e.g. 1PB123")
        anatomy, centre = pid[1].upper(), pid[2].upper()
        paths = {k: d / f"{k}.nii.gz" for k in ("mr", "ct", "mask")}
        missing = [str(v) for v in paths.values() if not v.is_file()]
        if missing:
            raise SynthRadError(f"patient {pid} is missing {missing}")
        return cls(patient=pid, anatomy=anatomy, centre=centre,
                   mr=paths["mr"], ct=paths["ct"], mask=paths["mask"])


def discover_patients(root: str | Path, *, anatomy: str) -> list[PatientPaths]:
    """
    CN: 列出某解剖结构下的全部患者（按 id 排序）。**两层布局**：`root` 下先是解剖目录（`brain/`、`pelvis/`），患者目录在其内部
        （`data/Task1/brain/1BA001/`）。因此这里先按解剖取子目录，再在其内部找以 `1<letter>` 开头的患者目录。
        （早先的实现只在 `root` 的**直接**子目录里找患者，真实布局下会一个也找不到。）
    EN: **two-level layout**: `root` holds anatomy folders (`brain/`, `pelvis/`) and patient folders live
        inside them (`data/Task1/brain/1BA001/`). So take the anatomy subdirectory first, then look for
        `1<letter>` patient folders inside it. (The earlier version scanned only `root`'s **immediate**
        children, which found nothing in the real layout.)
    """
    base = Path(root)
    if not base.is_dir():
        raise SynthRadError(f"missing dataset root: {base}")
    letter = {"brain": "B", "pelvis": "P"}.get(str(anatomy).lower())
    if letter is None:
        raise SynthRadError(f"unknown anatomy {anatomy!r}; expected 'brain' or 'pelvis'")
    out: list[PatientPaths] = []
    # CN: 先找 `root/<anatomy>`；若不存在，则退回"`root` 直接含患者目录"的扁平布局（测试用）。
    # EN: prefer `root/<anatomy>`; fall back to the flat layout where `root` holds patient folders directly
    #     (used by the tests).
    anatomy_dir = base / str(anatomy).lower()
    search_dirs = [anatomy_dir] if anatomy_dir.is_dir() else [base]
    for parent in search_dirs:
        for folder in sorted(p for p in parent.iterdir() if p.is_dir()):
            if folder.name[:1] == "1" and folder.name[1:2].upper() == letter:
                out.append(PatientPaths.from_dir(folder))
    if not out:
        raise SynthRadError(f"no {anatomy} patients under {base}")
    return out


@dataclass(frozen=True)
class SynthRadSlices:
    """
    CN: 一个病例经**固定预处理**后的切片集合。每个切片是一行 192×192 的图（展平）。
    EN: one case after the **fixed preprocessing**. Each slice is one row of a flattened 192x192 image.
    """

    patient: str
    centre: str
    mr: Float[np.ndarray, "slices pixels"]
    ct: Float[np.ndarray, "slices pixels"]
    mask: Bool[np.ndarray, "slices pixels"]
    z_normalised: Float[np.ndarray, "slices"]
    z_index: Int[np.ndarray, "slices"]
    meta: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        """CN: 有效切片数。EN: number of valid slices."""
        return int(self.mr.shape[0])


def _resize(image: Float[np.ndarray, "x y"], *, size: int, order: int) -> Float[np.ndarray, "s s"]:
    """
    CN: 双线性（order=1）或最近邻（order=0）缩放。用 `scipy.ndimage.zoom`，并显式给出输出尺寸。
    EN: bilinear (`order=1`) or nearest (`order=0`) resize via `scipy.ndimage.zoom`, forced to the exact size.
    """
    from scipy.ndimage import zoom  # noqa: PLC0415

    src: Float[np.ndarray, "x y"] = np.asarray(image, dtype=np.float64)
    factors = (float(size) / float(src.shape[0]), float(size) / float(src.shape[1]))
    out: Float[np.ndarray, "s s"] = zoom(src, factors, order=int(order))
    # CN: zoom 会有取整误差；必要时裁剪/补齐到精确尺寸。
    # EN: zoom can round; crop/pad to the exact size.
    if out.shape != (size, size):
        fixed = np.zeros((size, size), dtype=np.float64)
        h, w = min(size, out.shape[0]), min(size, out.shape[1])
        fixed[:h, :w] = out[:h, :w]
        out = fixed
    return out


def _window_ct(volume: Float[np.ndarray, "x y z"], *, window: tuple[float, float]) -> Float[np.ndarray, "x y z"]:
    """
    CN: 固定 HU 窗 → [0,1]。窗是**常量**，因此不同患者之间可比（不按患者自适应）。
    EN: fixed HU window -> [0,1]. The window is a **constant**, so patients are comparable (no per-patient
        adaptation).
    """
    lo, hi = float(window[0]), float(window[1])
    clipped: Float[np.ndarray, "x y z"] = np.clip(volume, lo, hi)
    return (clipped - lo) / (hi - lo)


def _normalise_mr(volume: Float[np.ndarray, "x y z"], *, percentiles: tuple[float, float]
                  ) -> Float[np.ndarray, "x y z"]:
    """
    CN: **按体数据**百分位归一化（1%/99%），截断后映射到 [0,1]。非有限值先按 0 处理。
    EN: **per-volume** percentile normalisation (1st/99th), clipped and mapped to [0,1]; non-finite values
        are treated as 0 first.
    """
    finite: Float[np.ndarray, "x y z"] = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(finite, [float(percentiles[0]), float(percentiles[1])])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(finite)
    return np.clip((finite - lo) / (hi - lo), 0.0, 1.0)


def preprocess_patient(paths: PatientPaths, *, size: int = IMAGE_SIZE,
                       ct_window: tuple[float, float] = CT_WINDOW,
                       mr_percentiles: tuple[float, float] = MR_PERCENTILES,
                       min_mask_area: float = MIN_MASK_AREA) -> SynthRadSlices:
    """
    CN: 对一个病例施加**固定的**预处理，返回**纳入的**切片（原始网格上身体掩膜面积 >= `min_mask_area`）。
        任何一步不符合（例如第 3 轴不是头足方向）都**报错**，而不是静默旋转。
    EN: apply the **fixed** preprocessing to one case and return the valid slices (mask area >=
        `min_mask_area`, evaluated on the **native** grid). Any mismatch -- e.g. axis 2 not being
        superior-inferior -- **raises** rather than silently rotating.
    """
    mr = load_volume(paths.mr)
    ct = load_volume(paths.ct)
    mask = load_volume(paths.mask)
    for name, vol in (("mr", mr), ("ct", ct), ("mask", mask)):
        if not vol.is_axial_first():
            raise SynthRadError(
                f"{paths.patient}/{name}: axis 2 is not superior-inferior; refusing to rotate silently")
    if mr.data.shape != ct.data.shape or mr.data.shape != mask.data.shape:
        raise SynthRadError(
            f"{paths.patient}: shape mismatch mr{mtuple(mr.data.shape)} ct{mtuple(ct.data.shape)} "
            f"mask{mtuple(mask.data.shape)}")

    mr_n: Float[np.ndarray, "x y z"] = _normalise_mr(mr.data, percentiles=mr_percentiles)
    ct_n: Float[np.ndarray, "x y z"] = _window_ct(ct.data, window=ct_window)
    mask_b: Bool[np.ndarray, "x y z"] = np.asarray(mask.data, dtype=np.float64) >= MASK_THRESHOLD

    n_slices: int = int(mr.data.shape[2])
    mr_rows: list[Float[np.ndarray, "pixels"]] = []
    ct_rows: list[Float[np.ndarray, "pixels"]] = []
    mask_rows: list[Bool[np.ndarray, "pixels"]] = []
    z_kept: list[int] = []
    for z in range(n_slices):
        m_slice: Bool[np.ndarray, "x y"] = mask_b[:, :, z]
        # CN: **纳入判据在原始网格上**（不是 192 缩放后）：判据是解剖学性质，不应取决于插值。
        #     缩放后的面积会与原始面积略有不同（实测 0.0501 -> 0.0492），但阈值只按原始网格判定。
        # EN: the **inclusion rule is evaluated on the NATIVE grid**, not after the 192 resize: the criterion
        #     is an anatomical property and must not depend on interpolation. The resized area differs
        #     slightly (measured 0.0501 -> 0.0492) but only the native value is compared to the threshold.
        if float(m_slice.mean()) < float(min_mask_area):
            continue
        mr_rows.append(_resize(mr_n[:, :, z], size=int(size), order=1).ravel())
        ct_rows.append(_resize(ct_n[:, :, z], size=int(size), order=1).ravel())
        mask_rows.append(_resize(m_slice.astype(np.float64), size=int(size), order=0).ravel() >= 0.5)
        z_kept.append(z)
    if not mr_rows:
        raise SynthRadError(f"{paths.patient}: every slice was dropped by the mask-area rule")

    kept: Int[np.ndarray, "slices"] = np.asarray(z_kept, dtype=np.int64)
    # CN: `z_normalised` 定义在**保留的**切片上（0 = 第一层有效切片，1 = 最后一层有效切片），
    #     而非原始层号：否则丢掉顶部/底部空层后范围就不在 [0,1]，跨患者也不可比。
    # EN: `z_normalised` is defined over the **retained** slices (0 = first valid, 1 = last valid), not the
    #     original index: otherwise dropping empty top/bottom slices would leave the range outside [0,1] and
    #     across patients it would not be comparable.
    denom: float = float(kept.size - 1) if kept.size > 1 else 1.0
    z_norm: Float[np.ndarray, "slices"] = np.arange(kept.size, dtype=np.float64) / denom
    return SynthRadSlices(
        patient=paths.patient, centre=paths.centre,
        mr=np.stack(mr_rows, axis=0), ct=np.stack(ct_rows, axis=0),
        mask=np.stack(mask_rows, axis=0), z_normalised=z_norm, z_index=kept,
        meta={"anatomy": paths.anatomy, "n_total_slices": n_slices, "size": int(size),
              "ct_window": list(ct_window), "mr_percentiles": list(mr_percentiles),
              "min_mask_area": float(min_mask_area)})


def mtuple(shape: Sequence[int]) -> tuple[int, ...]:
    """CN: 便于打印的形状元组。EN: a shape tuple for messages."""
    return tuple(int(s) for s in shape)


__all__ = [
    "CT_WINDOW",
    "IMAGE_SIZE",
    "MASK_THRESHOLD",
    "MIN_MASK_AREA",
    "MR_PERCENTILES",
    "NiftiVolume",
    "PatientPaths",
    "SynthRadError",
    "SynthRadSlices",
    "discover_patients",
    "load_volume",
    "preprocess_patient",
    "write_nifti",
]
