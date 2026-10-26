"""
CN: `harness/synthrad.py` 的断言。全部用**合成 NIfTI**，不需要真实数据（真实归档到位前不跑任何真实读数）。

    CN: 覆盖：NIfTI 读写往返（含 dtype/缩放）、固定预处理的确切规则（轴向、尺寸、掩膜阈值、CT 窗、
        MR 体数据百分位、z_normalised 范围）、以及**必须报错**的情形（形状不一致、非轴向、文件缺失）。

EN: Assertions for `harness/synthrad.py`, entirely on **synthetic NIfTI**; no real data is needed (and no
    real read-out runs before the archive arrives). Covers the read/write round trip, the exact fixed
    preprocessing rules, and the cases that **must raise**.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float

from iot_experiments.harness.synthrad import (
    CT_WINDOW,
    IMAGE_SIZE,
    SynthRadError,
    PatientPaths,
    discover_patients,
    load_volume,
    preprocess_patient,
    write_nifti,
)


def make_case(root: Path, patient: str, *, slices: int = 12, size: int = 32,
              empty_first: int = 0) -> Path:
    """
    CN: 造一个患者目录：MR/CT 有解剖结构，掩膜是一个圆盘，前 `empty_first` 层掩膜为空（用于测试丢弃）。
    EN: build a patient folder: MR/CT carry structure, the mask is a disc, and the first `empty_first`
        slices have an empty mask (to exercise the drop rule).
    """
    folder = root / patient
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(patient)) % 2 ** 31)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = ((yy - size // 2) ** 2 + (xx - size // 2) ** 2) < (size // 3) ** 2
    mr: Float[np.ndarray, "x y z"] = rng.normal(400.0, 40.0, (size, size, slices))
    mr = mr + disc[..., None] * 600.0
    ct: Float[np.ndarray, "x y z"] = np.full((size, size, slices), -1000.0)
    ct[disc, :] = 40.0 + 200.0 * rng.random((int(disc.sum()), slices))
    mask: Float[np.ndarray, "x y z"] = np.zeros((size, size, slices))
    mask[disc, :] = 1.0
    mask[:, :, : int(empty_first)] = 0.0
    write_nifti(folder / "mr.nii.gz", mr)
    write_nifti(folder / "ct.nii.gz", ct)
    write_nifti(folder / "mask.nii.gz", mask)
    return folder


def test_nifti_round_trip_preserves_values(tmp_path: Path) -> None:
    """CN: 写入再读出必须逐元素相等（float32 精度内），且轴向判定为真。"""
    data: Float[np.ndarray, "x y z"] = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    path = write_nifti(tmp_path / "v.nii.gz", data, spacing=(1.0, 1.0, 2.5))
    vol = load_volume(path)
    assert vol.data.shape == (4, 5, 6)
    assert np.allclose(vol.data, data, atol=1e-5)
    assert vol.is_axial_first()
    assert np.isclose(vol.spacing[2], 2.5)


def test_plain_nii_without_gzip(tmp_path: Path) -> None:
    """CN: 未压缩 `.nii` 也要能读。EN: an uncompressed `.nii` must also load."""
    data: Float[np.ndarray, "x y z"] = np.ones((3, 3, 3))
    path = write_nifti(tmp_path / "plain.nii", data)
    assert np.allclose(load_volume(path).data, 1.0, atol=1e-6)


def test_preprocessing_rules_exactly(tmp_path: Path) -> None:
    """CN: 固定预处理的每条规则都要可测：尺寸、掩膜阈值、CT 窗、MR 体数据归一、z 范围。"""
    folder = make_case(tmp_path, "1PA001", slices=12, size=32, empty_first=2)
    out = preprocess_patient(PatientPaths.from_dir(folder))
    # CN: 2 层空掩膜被丢弃 → 10 层；每层 192*192。
    assert len(out) == 10
    assert out.mr.shape == (10, IMAGE_SIZE * IMAGE_SIZE)
    assert out.ct.shape == out.mr.shape
    assert out.mask.dtype == np.bool_
    # CN: MR 按体数据归一 → [0,1]；CT 窗 → [0,1]（-1000 HU 被截到窗下端 0）。
    assert 0.0 <= float(out.mr.min()) <= float(out.mr.max()) <= 1.0
    assert 0.0 <= float(out.ct.min()) <= float(out.ct.max()) <= 1.0
    # CN: z_normalised 恰在 [0,1]，起点 0、终点 1，单调。
    assert float(out.z_normalised[0]) == 0.0
    assert float(out.z_normalised[-1]) == 1.0
    assert np.all(np.diff(out.z_normalised) > 0.0)
    assert out.meta["ct_window"] == list(CT_WINDOW)


def test_ct_window_maps_hu_to_endpoints(tmp_path: Path) -> None:
    """CN: 窗端点 -200/300 HU 必须映到 0/1，窗外被截断。EN: window endpoints map to 0/1 and clip."""
    folder = tmp_path / "1PA002"
    folder.mkdir()
    size, slices = 16, 4
    ct: Float[np.ndarray, "x y z"] = np.zeros((size, size, slices))
    ct[:, :, :] = -200.0
    ct[:8, :, :] = 300.0
    ct[8:12, :, :] = 5000.0
    ct[12:, :, :] = -5000.0
    write_nifti(folder / "ct.nii.gz", ct)
    write_nifti(folder / "mr.nii.gz", np.ones((size, size, slices)))
    make_mask = np.ones((size, size, slices))
    write_nifti(folder / "mask.nii.gz", make_mask)
    out = preprocess_patient(PatientPaths.from_dir(folder))
    values = np.unique(np.round(out.ct, 6))
    assert 0.0 in values
    assert 1.0 in values
    assert float(out.ct.max()) <= 1.0 + 1e-9


def test_mr_normalised_per_volume_not_per_slice(tmp_path: Path) -> None:
    """
    CN: 体数据归一化必须跨层一致：把某一层整体调暗后，它的取值应显著低于未调暗的层，
        而不是被逐层拉伸回满量程（那会是逐切片归一化）。
    EN: per-volume normalisation must be slice-consistent: dimming one slice must leave it markedly darker,
        rather than being stretched back to full range (which would be per-slice normalisation).
    """
    folder = tmp_path / "1PA003"
    folder.mkdir()
    size, slices = 16, 4
    mr: Float[np.ndarray, "x y z"] = np.full((size, size, slices), 1000.0)
    mr[:, :, 1] = 100.0                       # one dark slice
    write_nifti(folder / "mr.nii.gz", mr)
    write_nifti(folder / "ct.nii.gz", np.zeros((size, size, slices)))
    write_nifti(folder / "mask.nii.gz", np.ones((size, size, slices)))
    out = preprocess_patient(PatientPaths.from_dir(folder))
    means = [float(out.mr[i].mean()) for i in range(len(out))]
    assert means[1] < 0.5 * means[0], means


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    """CN: MR/CT/掩膜形状不一致必须报错。EN: mismatched shapes must raise."""
    folder = make_case(tmp_path, "1PA004", slices=6, size=16)
    write_nifti(folder / "ct.nii.gz", np.zeros((16, 16, 5)))
    with pytest.raises(SynthRadError):
        preprocess_patient(PatientPaths.from_dir(folder))


def test_missing_file_raises(tmp_path: Path) -> None:
    """CN: 缺文件必须报错（不允许静默跳过）。EN: a missing file must raise, never be skipped."""
    folder = make_case(tmp_path, "1PA005", slices=4, size=16)
    (folder / "mask.nii.gz").unlink()
    with pytest.raises(SynthRadError):
        PatientPaths.from_dir(folder)


def test_discover_filters_by_anatomy(tmp_path: Path) -> None:
    """CN: 按解剖字母筛选患者（1PB=脑，1PP=骨盆）。EN: filter patients by anatomy letter (1PB brain, 1PP pelvis)."""
    make_case(tmp_path, "1BA001", slices=4, size=16)
    make_case(tmp_path, "1BB002", slices=4, size=16)
    make_case(tmp_path, "1PC001", slices=4, size=16)
    brain = discover_patients(tmp_path, anatomy="brain")
    pelvis = discover_patients(tmp_path, anatomy="pelvis")
    assert [p.patient for p in brain] == ["1BA001", "1BB002"]
    assert [p.patient for p in pelvis] == ["1PC001"]
    assert brain[0].anatomy == "B" and brain[0].centre == "A"
    assert brain[1].centre == "B" and pelvis[0].anatomy == "P" and pelvis[0].centre == "C"


def test_all_slices_dropped_raises(tmp_path: Path) -> None:
    """CN: 若掩膜规则丢掉所有切片，必须报错而不是返回空数组。"""
    folder = tmp_path / "1PA006"
    folder.mkdir()
    write_nifti(folder / "mr.nii.gz", np.ones((8, 8, 3)))
    write_nifti(folder / "ct.nii.gz", np.zeros((8, 8, 3)))
    write_nifti(folder / "mask.nii.gz", np.zeros((8, 8, 3)))
    with pytest.raises(SynthRadError):
        preprocess_patient(PatientPaths.from_dir(folder))
