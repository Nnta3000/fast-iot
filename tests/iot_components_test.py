from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from iot import (
    DenseFeatureTensor,
    DiagonalWeightFeatures,
    InverseOTError,
    InverseOTProblem,
    SimplexTheta,
    connector_edges,
    mandatory_edges,
    support_components,
)


def _permutation_problem(size: int) -> InverseOTProblem:
    rng = np.random.default_rng(0)
    features = DenseFeatureTensor(rng.normal(size=(size, size, 2)))
    plan = sparse.csr_matrix(np.eye(size) / size)
    return InverseOTProblem(features=features, theta_set=SimplexTheta(2), plan=plan)


def test_permutation_support_has_one_component_per_pair() -> None:
    problem = _permutation_problem(4)
    partition = support_components(problem)
    assert partition.n_components == 4
    assert partition.component_sources == (0, 1, 2, 3)


def test_connector_links_components_in_a_cycle() -> None:
    problem = _permutation_problem(4)
    partition = support_components(problem)
    connectors = connector_edges(problem, partition)
    assert connectors.size == 4
    # CN: 连接边必须跨分量，即不在对角支撑上。
    # EN: Connector edges must cross components, i.e. leave the diagonal support.
    assert not bool(np.all(connectors.rows == connectors.cols))


def test_connected_support_uses_one_edge_outside_s() -> None:
    source = np.array([[0.0], [1.0], [2.0]])
    features = DiagonalWeightFeatures(source, source)
    plan = sparse.csr_matrix(np.diag([0.2, 0.3, 0.5]))
    problem = InverseOTProblem(features=features, theta_set=SimplexTheta(1), plan=plan)
    partition = support_components(problem)
    assert partition.n_components == 3  # diagonal-only support is disconnected
    connectors = connector_edges(problem, partition)
    assert connectors.size == 3


def test_mandatory_set_contains_support_and_connectors() -> None:
    problem = _permutation_problem(5)
    mandatory, partition = mandatory_edges(problem)
    assert partition.n_components == 5
    assert mandatory.size == 5 + 5  # S plus the connector cycle
    support = problem.support_keys()
    assert bool(np.all(np.isin(support, mandatory.keys)))


def test_full_support_is_rejected() -> None:
    features = DenseFeatureTensor(np.ones((2, 2, 2)))
    plan = sparse.csr_matrix(np.full((2, 2), 0.25))
    problem = InverseOTProblem(features=features, theta_set=SimplexTheta(2), plan=plan)
    with pytest.raises(InverseOTError, match="full edge set"):
        mandatory_edges(problem)
