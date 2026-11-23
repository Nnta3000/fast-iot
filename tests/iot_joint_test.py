"""
EN: **Tests for the shared-theta diagnostics.** Two things are easy to get silently wrong here and both would
    invalidate the experiment:

    1. the blocked cost must equal the full-grid cost (a block boundary bug would corrupt every score while the
       shapes still agree);
    2. the flatness diagnostic must actually detect a flat cost, because the entire interpretation of the
       shared-theta result -- "it satisfied every plan by tying them" -- rests on that measurement.

CN: **共享 θ 诊断的测试**。两处容易静默出错、且都会让实验失效：分块代价值必须等于全网格代价；平坦度诊断必须
    真的能发现平坦代价（"通过打平满足所有方案"这一解读完全依赖它）。
"""

from __future__ import annotations

import numpy as np

from iot import DiagonalWeightFeatures, HingeProfileFeatures
from experiments.studies.iot.real.faust.joint import _cost_rows, _flatness


def test_blocked_cost_equals_full_grid_cost() -> None:
    """
    EN: `_cost_rows` exists so that `(n*n, K)` is never materialised (600 MB at K=117, n=800). The block size
        must not change the answer.
    CN: `_cost_rows` 存在是为了不物化 `(n*n, K)`；块大小不能改变结果。
    """
    rng = np.random.default_rng(0)
    source = rng.normal(size=(19, 4))
    target = rng.normal(size=(19, 4))
    theta = rng.dirichlet(np.ones(4))
    features = DiagonalWeightFeatures(source, target)
    rows = np.repeat(np.arange(19, dtype=np.int64), 19)
    cols = np.tile(np.arange(19, dtype=np.int64), 19)
    expected = features.costs(theta, rows, cols).reshape(19, 19)
    for block in (1, 5, 19, 64):
        np.testing.assert_allclose(_cost_rows(features, theta, 19, block=block), expected,
                                   rtol=0, atol=1e-12)


def test_blocked_cost_works_for_a_rich_family() -> None:
    """
    EN: the rich families are the reason the helper exists; they must go through the same path.
    CN: 丰富族正是该助手存在的理由，必须走同一路径。
    """
    rng = np.random.default_rng(1)
    source = rng.normal(size=(11, 3))
    target = rng.normal(size=(11, 3))
    features = HingeProfileFeatures(source, target, basis_size=4, extent=np.ones(3))
    theta = rng.dirichlet(np.ones(features.n_features))
    rows = np.repeat(np.arange(11, dtype=np.int64), 11)
    cols = np.tile(np.arange(11, dtype=np.int64), 11)
    expected = features.costs(theta, rows, cols).reshape(11, 11)
    np.testing.assert_allclose(_cost_rows(features, theta, 11, block=4), expected, rtol=0, atol=1e-12)


def test_flatness_detects_a_constant_cost() -> None:
    """
    EN: a constant cost ties every candidate, so every plan is optimal and the decoded map is arbitrary. That is
        the failure mode measured on FAUST (`tied_fraction = 1.0`, `miss@5 % = 0.96`), and this is the check that
        reports it.
    CN: 常数代价使所有候选打平，任何方案都最优，解码任意。这就是 FAUST 上测到的失效模式。
    """
    flat = np.zeros((9, 9))
    report = _flatness(flat)
    assert report["tied_fraction_1e6"] == 1.0
    assert abs(report["row_gap_median"]) < 1e-15


def test_flatness_is_zero_when_the_diagonal_is_strictly_better() -> None:
    """
    EN: the complementary case, so the diagnostic is not simply always firing. A strictly better diagonal is
        gap = -1 (negative), which must NOT be counted as tied.
    CN: 互补情形。严格更优的对角 gap = -1（负），绝不能算成打平。
    """
    separated = np.ones((9, 9))
    np.fill_diagonal(separated, 0.0)
    report = _flatness(separated)
    assert report["tied_fraction_1e6"] == 0.0
    assert report["row_gap_median"] < -0.9
