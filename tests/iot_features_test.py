from __future__ import annotations

import numpy as np
import pytest

from iot import DenseFeatureTensor, DiagonalWeightFeatures, dense_cost_matrix


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
