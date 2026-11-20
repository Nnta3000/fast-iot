"""
EN: tests for the ramp basis (`iot.sparse.features.RampFeatures`) used by the multi-observation study.

    Three claims, each measured: quantile thresholds put a known share of pairs above each hinge; first-difference
    columns are non-negative and telescope back to the first hinge; and the difference construction is far better
    conditioned than nested hinges at the same K (that is why it is the basis of the discriminating rerun).

CN: 多观测研究使用的 ramp 基（`iot.sparse.features.RampFeatures`）的测试。三项各自被测量的断言：分位阈值使每条 hinge
    在已知比例的配对上为正；一阶差分列非负且可望远镜式求和回到第一条 hinge；同一 K 下差分构造的条件数远好于嵌套
    hinge（这正是判别性重跑选用它的原因）。
"""

from __future__ import annotations

import numpy as np

from iot.sparse.features import RampFeatures

N = 40


def _grid(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EN: two point clouds on the unit square and the full edge grid. CN: 单位方格上两片点云与全边网格。"""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(N, dtype=np.int64), N)
    cols = np.tile(np.arange(N, dtype=np.int64), N)
    return rng.random((N, 2)), rng.random((N, 2)), rows, cols


def test_quantile_thresholds_give_a_known_positivity_share() -> None:
    """EN: hinge k (threshold at the (k+1)/(K+1) quantile) is positive on about `1 - (k+1)/(K+1)` of pairs. CN: 阈值在 (k+1)/(K+1) 分位的 hinge 约在 `1-(k+1)/(K+1)` 的配对上为正。"""
    source, target, rows, cols = _grid()
    k = 10
    features = RampFeatures(source, target, ramps=k, threshold_rule="quantile")
    positive = (features.features(rows, cols) > 0.0).mean(axis=0)
    expected = 1.0 - np.arange(1, k + 1) / (k + 1.0)
    assert np.max(np.abs(positive - expected)) < 0.05


def test_difference_columns_are_nonnegative_and_telescope() -> None:
    """EN: `phi_k = raw_k - raw_{k+1}` is non-negative and `Σ_k raw_k = raw_0` before normalisation. CN: `phi_k=raw_k-raw_{k+1}` 非负，归一化前 `Σ_k raw_k = raw_0`。"""
    source, target, rows, cols = _grid(1)
    features = RampFeatures(source, target, ramps=8, threshold_rule="quantile", differences=True)
    spacing = features._spacing(rows, cols)  # noqa: SLF001 - the raw form is what the claim is about
    raw = features._raw(spacing)  # noqa: SLF001
    assert raw.min() >= -1e-12
    first_hinge = np.maximum(0.0, spacing - float(features.thresholds[0]))
    np.testing.assert_allclose(raw.sum(axis=1), first_hinge, atol=1e-10)


def test_difference_basis_is_better_conditioned_than_nested_hinges() -> None:
    """EN: at K=20 the difference basis has a larger smallest correlation eigenvalue and a smaller max |corr|. CN: K=20 时差分基的最小相关特征值更大、max|corr| 更小。"""
    source, target, rows, cols = _grid(2)
    hinge = RampFeatures(source, target, ramps=20, threshold_rule="quantile")
    difference = RampFeatures(source, target, ramps=20, threshold_rule="quantile", differences=True)
    hinge_corr = np.corrcoef(hinge.features(rows, cols).T)
    difference_corr = np.corrcoef(difference.features(rows, cols).T)
    assert np.linalg.eigvalsh(difference_corr).min() > 100.0 * np.linalg.eigvalsh(hinge_corr).min()
    assert np.abs(difference_corr - np.eye(20)).max() < np.abs(hinge_corr - np.eye(20)).max()
