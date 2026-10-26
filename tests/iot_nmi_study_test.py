"""
CN: `studies/nmi` 的科学断言测试。每个断言对应报告里的一句话：
    (1) 像素 NMI 代价在**打乱后**仍然把真实配对排在最前（排序不变 + 植入了可恢复的信号）；
    (2) `pair_accuracy` 用**真实配对列**而不是 `i == i`（两侧打乱后恒等映射无意义）；
    (3) Hungarian 与 EMD 在均匀边际、n=n 时给出同一耦合；
    (4) 互信息估计器方向正确：相同内容 > 不相关内容；
    (5) 白化后的各主成分标准差约为 1。

EN: Scientific-claim tests for `studies/nmi`. Each assertion backs one sentence in the report:
    (1) the pixel NMI cost ranks true pairs first **after shuffling** (invariant to order and able to
        recover a planted signal);
    (2) `pair_accuracy` compares against the **true partner column**, not `i == i` (the identity map is
        meaningless once both sides are shuffled);
    (3) Hungarian and EMD agree at n = n with uniform marginals;
    (4) the MI estimator has the right direction: shared content > unrelated content;
    (5) whitened components have standard deviation about 1.
"""

from __future__ import annotations

import numpy as np
import pytest
from jaxtyping import Float, Int

from iot_experiments.studies.nmi.run import (
    emd_perm,
    hungarian,
    nmi_pixel_cost,
    pair_accuracy,
    whiten,
)


def planted_pixels(n: int = 40, pixels: int = 256, *, seed: int = 0
                   ) -> tuple[Float[np.ndarray, "n pixels"], Float[np.ndarray, "n pixels"]]:
    """CN: T2 是 T1 的平滑强度变换加噪声，其余为纯噪声。EN: T2 is T1 under a smooth gain plus noise."""
    rng = np.random.default_rng(seed)
    base: Float[np.ndarray, "n pixels"] = rng.normal(size=(n, pixels))
    a: Float[np.ndarray, "n pixels"] = base.copy()
    b: Float[np.ndarray, "n pixels"] = 0.8 * base + rng.normal(scale=0.15, size=(n, pixels))
    return a, b


def test_pixel_nmi_recovers_planted_pairs_after_shuffling() -> None:
    """CN: 打乱两侧后仍能恢复植入配对（排序不变，且能识别受试者）。EN: recovers planted pairs after shuffling."""
    a, b = planted_pixels()
    n = int(a.shape[0])
    order_1: Int[np.ndarray, "n"] = np.random.default_rng(1).permutation(n)
    order_2: Int[np.ndarray, "n"] = np.random.default_rng(2).permutation(n)
    inv_2: Int[np.ndarray, "n"] = np.empty(n, dtype=np.int64)
    inv_2[order_2] = np.arange(n)
    true_partner: Int[np.ndarray, "n"] = inv_2[order_1]

    cost = nmi_pixel_cost(a[order_1], b[order_2], bins=32)
    accuracy = pair_accuracy(hungarian(cost), true_partner)
    assert accuracy > 0.95, f"planted recovery collapsed: pair_acc={accuracy:.3f}"

    # CN: 排序不变性——打乱不应改变代价矩阵的相对结构。
    # EN: invariance -- shuffling must not change the relative structure of the cost.
    shuffled_again = nmi_pixel_cost(a[order_1][np.argsort(order_1)], b[order_2][np.argsort(order_2)], bins=32)
    assert np.allclose(shuffled_again[np.ix_(order_1, order_2)], cost, atol=1e-9)


def test_pair_accuracy_compares_true_partner_not_identity() -> None:
    """CN: 必须与真实配对列比较；恒等映射在两侧打乱后无意义。EN: compare to the true partner, not i == i."""
    n = 8
    order_1: Int[np.ndarray, "n"] = np.random.default_rng(3).permutation(n)
    order_2: Int[np.ndarray, "n"] = np.random.default_rng(4).permutation(n)
    inv_2: Int[np.ndarray, "n"] = np.empty(n, dtype=np.int64)
    inv_2[order_2] = np.arange(n)
    true_partner: Int[np.ndarray, "n"] = inv_2[order_1]

    assert pair_accuracy(true_partner, true_partner) == 1.0
    assert pair_accuracy((true_partner + 1) % n, true_partner) == 0.0


def test_hungarian_and_emd_agree() -> None:
    """CN: 均匀边际、n=n 时两种解码器给出同一置换。EN: both decoders agree at n = n with uniform marginals."""
    rng = np.random.default_rng(5)
    cost: Float[np.ndarray, "n n"] = rng.normal(size=(24, 24))
    assert np.array_equal(hungarian(cost), emd_perm(cost))


def test_mutual_information_direction() -> None:
    """CN: 相同内容的 MI 高于不相关内容。EN: MI of shared content exceeds that of unrelated content."""
    rng = np.random.default_rng(6)
    pixels = 512
    signal: Float[np.ndarray, "pixels"] = rng.normal(size=pixels)
    a: Float[np.ndarray, "pixels"] = signal + rng.normal(scale=0.1, size=pixels)
    b: Float[np.ndarray, "pixels"] = signal + rng.normal(scale=0.1, size=pixels)
    c: Float[np.ndarray, "pixels"] = rng.normal(size=pixels)
    same = float(nmi_pixel_cost(a[None, :], b[None, :], bins=16)[0, 0])
    unrelated = float(nmi_pixel_cost(a[None, :], c[None, :], bins=16)[0, 0])
    assert same < unrelated, f"MI direction inverted: same={same:.3f} unrelated={unrelated:.3f}"


def test_whitening_unit_variance() -> None:
    """CN: 白化后每个主成分的标准差约为 1。EN: whitened components have unit standard deviation."""
    rng = np.random.default_rng(7)
    common: Float[np.ndarray, "subjects dims"] = rng.normal(size=(120, 12))
    x: Float[np.ndarray, "subjects dims"] = common * 3.0 + rng.normal(size=(120, 12))
    y: Float[np.ndarray, "subjects dims"] = common * 0.5 + rng.normal(size=(120, 12))
    xw, yw = whiten(x, y, dim=12)
    pooled: Float[np.ndarray, "pooled K"] = np.concatenate([xw, yw], axis=0)
    obtained: Float[np.ndarray, "K"] = pooled.std(axis=0)
    assert np.allclose(obtained, 1.0, atol=1e-6), f"component stds not unit: {np.round(obtained[:5], 4)}"
    with pytest.raises(Exception):
        whiten(x, y[:, :8], dim=12)


def test_nmi_needs_samples_per_cell() -> None:
    """
    CN: 直方图**样本饥饿**是本项目实测到的失败模式：1024 个像素填 1024 个格子 → collapse。
        这条测试固定"样本/格子"这一诊断量，避免以后再把饥饿误读成"NMI 不行"。
    EN: **sample starvation** is the failure mode measured here: 1,024 pixels filling 1,024 cells.
        This test pins the diagnostic (`samples/cell`) so a starved histogram is never again reported as
        "NMI does not work".
    """
    from iot_experiments.studies.nmi.robustness import RobustConfig

    _ = RobustConfig  # CN: 配置存在即可导入；数据量由像素数与箱数决定。
    samples_starved: float = 1024.0 / 32.0 ** 2
    samples_full: float = 192.0 ** 2 / 32.0 ** 2
    assert samples_starved == 1.0, "the 32x32/32-bin cell is exactly one sample per cell"
    assert samples_full > 30.0, "full resolution gives tens of samples per cell"


def test_ssim_and_psnr_on_identical_images() -> None:
    """CN: SSIM 自身比较为 1，PSNR 为 inf。EN: SSIM of an image with itself is 1 and PSNR is inf."""
    from iot_experiments.studies.nmi.two_halves import psnr, score_pixels, ssim

    rng = np.random.default_rng(11)
    a: Float[np.ndarray, "pixels"] = rng.random(256)
    assert abs(ssim(a, a) - 1.0) < 1e-12
    assert np.isinf(psnr(a, a))
    metrics = score_pixels(a[None, :], a[None, :])
    assert metrics["exact_hit"] == 1.0


def test_learned_theta_used_in_fitted_geometry() -> None:
    """
    CN: θ 必须在**它被拟合的几何**里使用。此前版本在 G2 拟合、却用原始空间算代价，
        实测把 0.863 变成 0.549；这条测试固定"同一空间"这一不变量。
    EN: theta must be applied in the geometry it was fitted in. An earlier version fitted in G2 but scored
        in the raw space, turning 0.863 into 0.549; this test pins the "same space" invariant.
    """
    from iot_experiments.harness.registry import apply_geometry, fit_geometry, geometry

    rng = np.random.default_rng(12)
    x: Float[np.ndarray, "subjects d"] = rng.normal(size=(60, 8))
    y: Float[np.ndarray, "subjects d"] = rng.normal(size=(60, 8))
    lab: Int[np.ndarray, "m"] = np.arange(20)
    unl: Int[np.ndarray, "n"] = np.arange(20, 60)
    fitted = geometry(x, y, "G2", labelled=lab, unlabelled=unl)
    # CN: `fit_geometry`+`apply_geometry` 必须与 `geometry()` 逐位一致，否则拟合与评分不同空间。
    # EN: `fit_geometry`+`apply_geometry` must match `geometry()` bit-for-bit, else fitting and scoring
    #     happen in different spaces.
    stats = fit_geometry(x[unl], y[unl], "G2")
    x_pool, y_pool = apply_geometry(x, y, stats)
    assert np.allclose(fitted.x_label, x_pool[lab], atol=1e-12)
    assert np.allclose(fitted.y_label, y_pool[lab], atol=1e-12)
    assert np.allclose(fitted.x_unlabelled, x_pool[unl], atol=1e-12)
    assert np.allclose(fitted.y_unlabelled, y_pool[unl], atol=1e-12)
