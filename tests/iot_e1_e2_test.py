"""
CN: E1 / E2 的断言。全部在**合成特征与合成切片**上，不需要真实数据。

    CN: 固定住的三条不变量：
      (1) E2 的 Hungarian 与 EMD 在**同一代价**上必须给出相同的总目标值（单位一致，见 E2 的注释）；
      (2) `discrimination_check` 对 E1 的**字典式样本**必须可用（样本类型不是数组也必须工作）；
      (3) NMI 的**样本/格子**必须被报告，且 128×128/32 箱是 16 样本/格子——这是样本饥饿的判据。

EN: Assertions for E1 / E2, entirely on synthetic features and slices; no real data required. Three pinned
    invariants: E2's Hungarian and EMD agree on the same cost (consistent units); the discrimination check
    works with E1's **dict-shaped** samples; and NMI's **samples per cell** is reported, with 128x128/32 bins
    giving 16 -- the starvation criterion.
"""

from __future__ import annotations

import numpy as np
import pytest
from jaxtyping import Bool, Float

from iot_experiments.harness.discriminates import format_check
from iot_experiments.harness.e1_metrics import (
    E1Error,
    SlicePool,
    mask_iou,
    nmi,
    run_e1,
    ssim,
    z_gap,
)
from iot_experiments.harness.e2_scale import cosine_cost, report_lines, run_emd, run_hungarian, samples_per_cell, sweep
from iot_experiments.harness.nmi_helpers import mutual_information_detail


def synthetic_pool(n: int = 60, *, pixels: int = 64 * 64, seed: int = 0,
                   centres: tuple[int, int] = (30, 30)) -> SlicePool:
    """
    CN: 造一个有**内容信号**的切片池：同一患者 MR 与 CT 共享底图，不同患者底图独立。
    EN: build a slice pool with a **content signal**: a patient's MR and CT share a base image, patients do
        not.
    """
    rng = np.random.default_rng(seed)
    base: Float[np.ndarray, "n pixels"] = rng.random((n, pixels))
    mr: Float[np.ndarray, "n pixels"] = np.clip(base + rng.normal(scale=0.05, size=(n, pixels)), 0, 1)
    ct: Float[np.ndarray, "n pixels"] = np.clip(base + rng.normal(scale=0.05, size=(n, pixels)), 0, 1)
    yy, xx = np.mgrid[0:8, 0:8]
    inside = (((yy - 4) ** 2 + (xx - 4) ** 2) < 9).ravel()
    mask: Bool[np.ndarray, "n pixels"] = np.tile(inside, (n, 1))[: n, : int(pixels)] if pixels == 64 else \
        np.ones((n, pixels), dtype=bool)
    if mask.shape != (n, pixels):
        mask = np.ones((n, pixels), dtype=bool)
    z: Float[np.ndarray, "n"] = np.linspace(0.0, 1.0, n)
    centre = [str(0)] * int(centres[0]) + [str(1)] * int(centres[1])
    return SlicePool(mr=mr, ct=ct, mask=mask, z_normalised=z,
                     patient=[f"p{i:03d}" for i in range(n)], centre=centre[:n])


def test_e2_hungarian_and_emd_agree_in_units() -> None:
    """CN: 同一代价上两种精确求解必须给出相同总目标值，残差为 0。"""
    rng = np.random.default_rng(1)
    cost: Float[np.ndarray, "n n"] = rng.random((80, 80))
    h_obj, h_res = run_hungarian(cost)
    e_obj, e_res = run_emd(cost)
    assert h_obj is not None and e_obj is not None
    assert abs(h_obj - e_obj) < 1e-9, (h_obj, e_obj)
    assert h_res is not None and h_res < 1e-12
    assert e_res is not None and e_res < 1e-9


def test_e2_oracle_cost_zero_on_true_pairs() -> None:
    """CN: 代价矩阵在真实配对上为 0 时，两种求解都必须选出恒等置换（目标值 0）。"""
    n = 40
    cost: Float[np.ndarray, "n n"] = np.ones((n, n))
    np.fill_diagonal(cost, 0.0)
    h_obj, _ = run_hungarian(cost)
    e_obj, _ = run_emd(cost)
    assert h_obj == pytest.approx(0.0)
    assert e_obj == pytest.approx(0.0)


def test_e2_sweep_records_time_and_memory() -> None:
    """CN: 扫描必须给出时间与峰值内存，且每个 n×arm 一行。"""
    rng = np.random.default_rng(2)

    def make(n: int) -> tuple[Float[np.ndarray, "n k"], Float[np.ndarray, "n k"]]:
        z = rng.normal(size=(n, 8))
        return z + rng.normal(scale=0.3, size=(n, 8)), z + rng.normal(scale=0.3, size=(n, 8))

    rows = sweep([20, 40], make_features=make, arms=("hungarian", "emd"), time_limit_s=60.0)
    assert len(rows) == 4
    assert all(r.status == "ok" for r in rows)
    assert all(r.seconds >= 0.0 for r in rows)
    text = "\n".join(report_lines(rows))
    assert "hungarian" in text and "seconds" in text


def test_e2_nmi_matrix_reports_starvation() -> None:
    """CN: 128×128/32 箱 = 16 样本/格子；饥饿阈值是 ~5，因此这一配置**不**饥饿，512 像素才饥饿。"""
    assert samples_per_cell(128 * 128) == pytest.approx(16.0)
    assert samples_per_cell(64 * 64) == pytest.approx(4.0)
    assert samples_per_cell(128 * 128) > 5.0
    assert samples_per_cell(64 * 64) < 5.0


def test_nmi_direction_and_identity() -> None:
    """CN: 相同向量 MI = log(bins)；独立向量明显更低。"""
    rng = np.random.default_rng(3)
    a: Float[np.ndarray, "samples"] = rng.random(4096)
    same = mutual_information_detail(a, a, bins=32)
    diff = mutual_information_detail(a, rng.random(4096), bins=32)
    assert same.nats == pytest.approx(float(np.log(32)), abs=1e-9)
    assert diff.nats < 0.5 * same.nats
    assert same.samples_per_cell == pytest.approx(4096.0 / 1024.0)


def test_e1_metrics_basic_behaviour() -> None:
    """CN: IoU/SSIM 自身为 1；z 差为 0 时返回 1；掩膜形状不符要报错。"""
    rng = np.random.default_rng(4)
    img: Float[np.ndarray, "pixels"] = rng.random(256)
    mask: Bool[np.ndarray, "pixels"] = img > 0.5
    assert mask_iou(mask, mask) == pytest.approx(1.0)
    assert ssim(img, img) == pytest.approx(1.0)
    assert float(z_gap(np.asarray([0.3]), np.asarray([0.3]))[0]) == pytest.approx(1.0)
    assert float(z_gap(np.asarray([0.0]), np.asarray([1.0]))[0]) == pytest.approx(0.0)
    with pytest.raises(E1Error):
        mask_iou(np.ones(4, dtype=bool), np.ones(5, dtype=bool))


def test_e1_accepts_content_metric_and_runs_with_dict_samples() -> None:
    """CN: E1 在字典样本上可用，且内容度量（nmi/ssim）应被判为有判别力。"""
    pool = synthetic_pool()
    result = run_e1(pool, draws=1, seed=0)
    table = format_check(result)
    assert "nmi" in table and "usable" in table
    assert "nmi" in result.usable_metrics(), table
    assert "ssim" in result.usable_metrics(), table


def test_e1_nmi_separates_true_from_cross() -> None:
    """CN: 有内容信号时，真配对的 MI 必须高于跨患者配对（方向性检查）。"""
    rng = np.random.default_rng(5)
    base: Float[np.ndarray, "pixels"] = rng.random(128 * 128)
    mr: Float[np.ndarray, "pixels"] = np.clip(base + rng.normal(scale=0.02, size=base.size), 0, 1)
    other: Float[np.ndarray, "pixels"] = np.clip(rng.random(base.size), 0, 1)
    true_signal = nmi(mr, np.clip(base + rng.normal(scale=0.02, size=base.size), 0, 1))
    cross_signal = nmi(mr, other)
    assert true_signal > cross_signal, (true_signal, cross_signal)


def test_cosine_cost_is_zero_on_identical_rows() -> None:
    """CN: 相同行的余弦代价为 0（oracle 性质）。EN: identical rows give zero cosine cost."""
    rng = np.random.default_rng(6)
    x: Float[np.ndarray, "n k"] = rng.normal(size=(12, 5))
    cost = cosine_cost(x, x)
    assert np.allclose(np.diag(cost), 0.0, atol=1e-12)
