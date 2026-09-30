from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from iot import DenseFeatureTensor, InverseOTProblem, SimplexTheta


def _diagonal_problem() -> InverseOTProblem:
    rng = np.random.default_rng(0)
    features = DenseFeatureTensor(rng.normal(size=(3, 3, 2)))
    plan = sparse.csr_matrix(np.diag([0.2, 0.3, 0.5]))
    return InverseOTProblem(features=features, theta_set=SimplexTheta(2), plan=plan)


def test_problem_extracts_support_and_marginals() -> None:
    problem = _diagonal_problem()
    assert (problem.n_source, problem.n_target, problem.n_features) == (3, 3, 2)
    assert problem.support_size == 3
    np.testing.assert_allclose(problem.resolved_source_mass, [0.2, 0.3, 0.5])
    np.testing.assert_allclose(problem.resolved_target_mass, [0.2, 0.3, 0.5])
    rows, cols, values = problem.support()
    np.testing.assert_array_equal(rows, [0, 1, 2])
    np.testing.assert_array_equal(cols, [0, 1, 2])
    np.testing.assert_allclose(values, [0.2, 0.3, 0.5])
    np.testing.assert_array_equal(problem.support_keys(), [0, 4, 8])


def test_problem_rejects_feature_dimension_mismatch() -> None:
    features = DenseFeatureTensor(np.ones((2, 2, 3)))
    with pytest.raises(ValueError, match="dimension must equal"):
        InverseOTProblem(features=features, theta_set=SimplexTheta(2), plan=sparse.csr_matrix(np.eye(2) * 0.5))


def test_problem_rejects_plan_shape_mismatch() -> None:
    features = DenseFeatureTensor(np.ones((2, 3, 2)))
    with pytest.raises(ValueError, match="does not match"):
        InverseOTProblem(features=features, theta_set=SimplexTheta(2), plan=sparse.csr_matrix(np.eye(2) * 0.5))


def test_problem_rejects_mass_mismatch() -> None:
    features = DenseFeatureTensor(np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="row marginals"):
        InverseOTProblem(
            features=features,
            theta_set=SimplexTheta(2),
            plan=sparse.csr_matrix(np.eye(2) * 0.5),
            source_mass=np.array([0.4, 0.6]),
        )


def test_problem_rejects_zero_marginal() -> None:
    features = DenseFeatureTensor(np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="positive mass"):
        InverseOTProblem(features=features, theta_set=SimplexTheta(2), plan=sparse.csr_matrix([[0.5, 0.5], [0.0, 0.0]]))
