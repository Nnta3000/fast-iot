from __future__ import annotations

import numpy as np
import ot

from iot import (
    fit_psd_moment_entropic,
    make_bilinear_psd_grounded_instance,
    moment_from_plan,
    project_psd,
    scale_invariant_error,
    sinkhorn_plan,
)
from iot.eval.evaluation import l0_gap


def test_project_psd_clips_negative_eigenvalues() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    projected = project_psd(matrix)
    np.testing.assert_allclose(projected, projected.T, atol=1e-12)
    assert float(np.linalg.eigvalsh(projected).min()) >= -1e-12


def test_sinkhorn_marginals_match_and_approach_exact_ot() -> None:
    rng = np.random.default_rng(0)
    cost = rng.normal(size=(6, 6))
    mass = np.full(6, 1.0 / 6.0)
    plan = sinkhorn_plan(cost, mass, mass, epsilon=0.005, max_iterations=5000, tolerance=1e-12)
    np.testing.assert_allclose(plan.sum(axis=1), mass, atol=1e-4)
    np.testing.assert_allclose(plan.sum(axis=0), mass, atol=1e-4)
    exact = ot.emd(mass, mass, cost)
    assert float(np.max(np.abs(plan - exact))) < 0.05


def test_moment_from_plan_matches_weighted_outer_products() -> None:
    rng = np.random.default_rng(1)
    source = rng.normal(size=(4, 2))
    target = rng.normal(size=(3, 2))
    plan = rng.random((4, 3))
    plan /= plan.sum()
    expected = sum(
        plan[i, j] * np.outer(source[i], target[j]) for i in range(4) for j in range(3)
    )
    np.testing.assert_allclose(moment_from_plan(plan, source, target), expected, atol=1e-12)


def test_fit_reduces_exact_l0_gap_and_stays_psd_trace_one() -> None:
    rng = np.random.default_rng(2)
    instance = make_bilinear_psd_grounded_instance(
        n_source=40, n_target=40, point_dimension=2, rng=rng
    )
    problem = instance.problem
    source = problem.features.source_points
    target = problem.features.target_points
    plan = problem.plan.toarray()
    source_mass = problem.resolved_source_mass
    target_mass = problem.resolved_target_mass
    observed = moment_from_plan(plan, source, target)
    initial_matrix = np.eye(2) / 2.0

    result = fit_psd_moment_entropic(
        source_points=source,
        target_points=target,
        source_mass=source_mass,
        target_mass=target_mass,
        observed_moment=observed,
        initial_matrix=initial_matrix,
        num_outer=40,
    )
    assert l0_gap(problem, result.matrix.reshape(-1)) < l0_gap(problem, initial_matrix.reshape(-1))
    assert abs(float(np.trace(result.matrix)) - 1.0) < 1e-9
    assert float(np.linalg.eigvalsh(result.matrix).min()) >= -1e-10
    true_matrix = instance.theta_star.reshape(2, 2)
    assert scale_invariant_error(true_matrix, result.matrix) < scale_invariant_error(
        true_matrix, initial_matrix
    )
