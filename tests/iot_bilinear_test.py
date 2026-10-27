"""
CN: 双线性嵌入的**正确性**测试：对每个族，`a_i + b_j - s<u_i,v_j>` 必须逐边等于 `features.costs`。
    这是 `O(m n d)` 快速扫描的数学基石；任何族的嵌入写错都会在这里立刻暴露。
EN: **Correctness** tests for the bilinear embeddings: for each family, `a_i + b_j - s<u_i,v_j>` must
    equal `features.costs` edge-for-edge. This underpins the `O(m n d)` fast scan; a wrong embedding
    for any family is caught here immediately.
"""

from __future__ import annotations

import numpy as np

from iot import DiagonalWeightFeatures, QuadraticFeatures, bilinear_embedding
from iot_experiments.benchmarks.lowrank import LowRankFeatures


def _embedding_error(features, theta: np.ndarray) -> float:
    embed = bilinear_embedding(features, theta)
    assert embed is not None
    n = int(features.n_source)
    rows = np.repeat(np.arange(n, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), n)
    from_definition = np.asarray(features.costs(theta, rows, cols), dtype=np.float64)
    from_embedding = (
        embed.source_offset[rows]
        + embed.target_offset[cols]
        - embed.dot_scale * np.sum(embed.source[rows] * embed.target[cols], axis=1)
    )
    return float(np.max(np.abs(from_definition - from_embedding)))


def test_diagonal_bilinear_embedding_matches_costs() -> None:
    rng = np.random.default_rng(1)
    d, n = 4, 30
    features = DiagonalWeightFeatures(rng.normal(size=(n, d)), rng.normal(size=(n, d)))
    assert _embedding_error(features, rng.dirichlet(np.ones(d))) < 1e-12


def test_quadratic_bilinear_embedding_matches_costs() -> None:
    rng = np.random.default_rng(2)
    d, n = 4, 30
    features = QuadraticFeatures(rng.normal(size=(n, d)), rng.normal(size=(n, d)))
    k = d * (d + 1) // 2
    assert _embedding_error(features, rng.dirichlet(np.ones(k))) < 1e-12


def test_lowrank_bilinear_embedding_matches_costs() -> None:
    rng = np.random.default_rng(3)
    n, k = 30, 3
    features = LowRankFeatures(rng.normal(size=(n, k)), rng.normal(size=(n, k)))
    assert _embedding_error(features, rng.dirichlet(np.ones(k))) < 1e-12


def test_families_report_bilinear_support() -> None:
    from iot import features_support_bilinear

    rng = np.random.default_rng(4)
    n, d = 10, 3
    assert features_support_bilinear(DiagonalWeightFeatures(rng.normal(size=(n, d)), rng.normal(size=(n, d))))
    assert features_support_bilinear(QuadraticFeatures(rng.normal(size=(n, d)), rng.normal(size=(n, d))))
    assert features_support_bilinear(LowRankFeatures(rng.normal(size=(n, 2)), rng.normal(size=(n, 2))))
    # CN: 纯 DenseFeatureTensor 没有就地下标双线性结构 -> 无嵌入（走通用 O(m n K) 路径）。
    # EN: a plain dense feature tensor has no in-place low-rank structure -> no embedding (generic path).
    from iot import DenseFeatureTensor

    dense = DenseFeatureTensor(np.ones((n, n, 2), dtype=np.float64))
    assert not features_support_bilinear(dense)
