from __future__ import annotations

import numpy as np
import pytest

from iot import (
    ConcatFeatures,
    DenseFeatureTensor,
    DiagonalWeightFeatures,
    KernelMixtureFeatures,
    QuadraticFeatures,
    RadialProfileFeatures,
    dense_cost_matrix,
)


def _edge_grid(n_source: int, n_target: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.repeat(np.arange(n_source, dtype=np.int64), n_target)
    cols = np.tile(np.arange(n_target, dtype=np.int64), n_source)
    return rows, cols


def test_diagonal_weight_features_are_linear_in_theta() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(5, 3))
    target = rng.normal(size=(4, 3))
    features = DiagonalWeightFeatures(source, target)
    rows, cols = _edge_grid(features.n_source, features.n_target)
    phi = features.features(rows, cols)
    theta = rng.dirichlet(np.ones(3))
    np.testing.assert_allclose(features.costs(theta, rows, cols), phi @ theta, rtol=1e-12, atol=1e-12)


def test_dense_tensor_and_diagonal_features_agree() -> None:
    rng = np.random.default_rng(1)
    source = rng.normal(size=(4, 3))
    target = rng.normal(size=(3, 3))
    diagonal = DiagonalWeightFeatures(source, target)
    rows, cols = _edge_grid(diagonal.n_source, diagonal.n_target)
    tensor = diagonal.features(rows, cols).reshape(4, 3, 3)
    dense = DenseFeatureTensor(tensor)
    theta = rng.dirichlet(np.ones(3))
    np.testing.assert_allclose(
        diagonal.costs(theta, rows, cols), dense.costs(theta, rows, cols), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        dense_cost_matrix(diagonal, theta), dense_cost_matrix(dense, theta), rtol=1e-12, atol=1e-12
    )


def test_dense_cost_matrix_respects_element_cap() -> None:
    dense = DenseFeatureTensor(np.ones((3, 4, 2)))
    with pytest.raises(ValueError, match="max_elements"):
        dense_cost_matrix(dense, np.full(2, 0.5), max_elements=10)


def test_feature_costs_reject_wrong_theta_length() -> None:
    features = DenseFeatureTensor(np.ones((2, 2, 2)))
    rows = np.array([0])
    cols = np.array([0])
    with pytest.raises(ValueError, match="theta has length"):
        features.costs(np.ones(3), rows, cols)


def test_feature_index_validation_rejects_negative() -> None:
    features = DenseFeatureTensor(np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="nonnegative"):
        features.features(np.array([-1]), np.array([0]))


def test_quadratic_features_match_direct_cross_terms() -> None:
    rng = np.random.default_rng(2)
    source = rng.normal(size=(5, 3))
    target = rng.normal(size=(4, 3))
    features = QuadraticFeatures(source, target)
    assert features.n_features == 3 * 4 // 2
    rows, cols = _edge_grid(features.n_source, features.n_target)
    difference = source[rows] - target[cols]
    expected = np.stack(
        [difference[:, a] * difference[:, b] for a in range(3) for b in range(a, 3)], axis=1
    )
    np.testing.assert_allclose(features.features(rows, cols), expected, rtol=0.0, atol=0.0)
    theta = rng.dirichlet(np.ones(features.n_features))
    np.testing.assert_allclose(
        features.costs(theta, rows, cols), expected @ theta, rtol=1e-12, atol=1e-12
    )


def test_quadratic_features_agree_with_dense_tensor() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(6, 4))
    target = rng.normal(size=(5, 4))
    quadratic = QuadraticFeatures(source, target)
    rows, cols = _edge_grid(quadratic.n_source, quadratic.n_target)
    tensor = quadratic.features(rows, cols).reshape(6, 5, quadratic.n_features)
    dense = DenseFeatureTensor(tensor)
    theta = rng.dirichlet(np.ones(quadratic.n_features))
    np.testing.assert_allclose(
        dense_cost_matrix(quadratic, theta), dense_cost_matrix(dense, theta), rtol=1e-12, atol=1e-12
    )


def test_quadratic_features_chunking_is_invariant() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(8, 3))
    target = rng.normal(size=(7, 3))
    rows, cols = _edge_grid(8, 7)
    theta = rng.dirichlet(np.ones(6))
    reference = QuadraticFeatures(source, target, chunk_elements=1 << 20).costs(theta, rows, cols)
    chunked = QuadraticFeatures(source, target, chunk_elements=5).costs(theta, rows, cols)
    np.testing.assert_array_equal(reference, chunked)


# --- kernel / dictionary axis -------------------------------------------------------------------------


def test_kernel_mixture_features_are_minus_the_kernel() -> None:
    """
    EN: `phi_ij,r = -k_r(x_i, y_j)` and `n_features` equals the number of *expanded* kernels.

        The expansion matters: `"gaussian:0.5,1.0"` must become TWO features, not one. The first version of this
        family counted it as one while evaluating only the first sigma, so the cost and the simplex dimension
        disagreed — a silent corruption of every fit.
    CN: `phi=-k`，且 `n_features` 等于**展开后**的核数；`"gaussian:0.5,1.0"` 必须是两个特征，否则代价与单纯形维数
        不一致。
    """
    rng = np.random.default_rng(0)
    source = rng.normal(size=(5, 3))
    target = rng.normal(size=(4, 3))
    features = KernelMixtureFeatures(source, target, ("gaussian:0.5,2.0", "cosine"))
    assert features.n_features == 3
    rows, cols = _edge_grid(features.n_source, features.n_target)
    phi = features.features(rows, cols)
    assert phi.shape == (rows.size, 3)
    # EN: the Gaussian columns are minus a Gaussian in (0, 1], so they lie in [-1, 0).
    # CN: 高斯列是负的高斯值，落在 [-1,0)。
    difference = source[rows] - target[cols]
    squared = (difference * difference).sum(axis=1)
    expected_0 = -np.exp(-squared / (2.0 * 0.5 * 0.5))
    expected_1 = -np.exp(-squared / (2.0 * 2.0 * 2.0))
    np.testing.assert_allclose(phi[:, 0], expected_0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(phi[:, 1], expected_1, rtol=0, atol=1e-12)
    # EN: cost is linear in theta, as for every FeatureMap. CN: 代价对 theta 线性。
    theta = np.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(features.costs(theta, rows, cols), phi @ theta, rtol=1e-12, atol=1e-12)


def test_radial_profile_features_are_not_constant_at_uniform_theta() -> None:
    """
    EN: regression for a degenerate parameterisation. A partition-of-unity hat basis satisfies `sum_l h_l = 1`
        for every pair, so with a uniform `theta` the cost is the constant `dim/K` and the fit cannot move.
        Measured before the fix: `cost_std = 1.5e-17` and `theta` stayed exactly uniform.

        The linear ramp is what breaks the degeneracy, so this test asserts the cost actually varies at the
        uniform point -- the thing that was wrong.
    CN: 退化参数化的回归测试。单位分解的帽基使均匀 `theta` 下代价恒为 `dim/K`，拟合无法移动（实测
        `cost_std=1.5e-17`）。线性斜坡打破退化，本测试断言均匀点处代价确实有变化。
    """
    rng = np.random.default_rng(1)
    source = rng.normal(size=(6, 3))
    target = rng.normal(size=(5, 3))
    features = RadialProfileFeatures(source, target, basis_size=4)
    rows, cols = _edge_grid(features.n_source, features.n_target)
    theta = np.full(features.n_features, 1.0 / features.n_features)
    cost = features.costs(theta, rows, cols)
    assert float(cost.std()) > 1e-6, f"cost is (nearly) constant at uniform theta: std={cost.std():.3e}"
    # EN: K = dim * (basis + ramp). CN: K = 维度 x (帽基 + 斜坡)。
    assert features.n_features == 3 * (4 + 1)


def test_radial_profile_hats_plus_ramp_partition_the_range() -> None:
    """
    EN: the hat block alone is a partition of unity per coordinate (`sum_l h_l = 1`), which is the property that
        made the uniform point degenerate; the ramp column is separate and is asserted to equal the normalised
        absolute difference itself.
    CN: 仅帽基部分在每坐标上构成单位分解（这正是均匀点退化的原因）；斜坡列单独存在，等于归一化绝对差本身。
    """
    rng = np.random.default_rng(2)
    source = rng.normal(size=(4, 2))
    target = rng.normal(size=(3, 2))
    extent = np.array([1.0, 1.0])
    features = RadialProfileFeatures(source, target, basis_size=3, extent=extent)
    rows, cols = _edge_grid(features.n_source, features.n_target)
    phi = features.features(rows, cols)
    hats = phi.reshape(-1, 2, 4)[:, :, :3]
    np.testing.assert_allclose(hats.sum(axis=2), 1.0, rtol=0, atol=1e-12)
    normalised = np.clip(np.abs(source[rows] - target[cols]), 0.0, 1.0)
    np.testing.assert_allclose(phi.reshape(-1, 2, 4)[:, :, 3], normalised, rtol=0, atol=1e-12)


def test_concat_features_concatenate_in_order_and_validate_shapes() -> None:
    """
    EN: `ConcatFeatures` is what makes a CROSS-distance mixture possible. It must concatenate in order, sum the
        feature counts, and refuse children that disagree on the cardinalities.
    CN: `ConcatFeatures` 使**跨距离**混合成为可能；必须按顺序拼接、特征数相加，并拒绝规模不一致的子项。
    """
    rng = np.random.default_rng(3)
    source = rng.normal(size=(5, 4))
    target = rng.normal(size=(4, 4))
    left = DiagonalWeightFeatures(source[:, :2], target[:, :2])
    right = KernelMixtureFeatures(source[:, 2:], target[:, 2:], ("gaussian:1.0",))
    both = ConcatFeatures((left, right))
    assert both.n_features == left.n_features + right.n_features
    rows, cols = _edge_grid(5, 4)
    phi = both.features(rows, cols)
    expected = np.concatenate([left.features(rows, cols), right.features(rows, cols)], axis=1)
    np.testing.assert_allclose(phi, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        ConcatFeatures((left, DiagonalWeightFeatures(source[:3, :2], target[:, :2])))
