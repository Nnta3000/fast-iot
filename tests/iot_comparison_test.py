from __future__ import annotations

import numpy as np
import ot
import pytest

from iot import BilinearPSDTheta, dense_cost_matrix, make_bilinear_psd_grounded_instance
from iot.eval.evaluation import evaluate_inverse_ot, l0_gap
from iot.generators.grounded import GroundedInstance
from iot.exact.loop import solve_inverse_ot
from iot.exact.sdp import SdpMasterSolver
from iot.alternatives.sinkhorn_theta import fit_theta_sinkhorn


def test_shared_l0_gap_matches_direct_pot_computation() -> None:
    # EN: After consolidating the two duplicate l0_gap/_exact_gap helpers onto
    #     evaluation.l0_gap, this pins the shared metric to the raw POT formula on five
    #     random PSD instances with a generic PSD theta (so the gap is nonzero).
    theta_set = BilinearPSDTheta(2)
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        instance = make_bilinear_psd_grounded_instance(
            n_source=12, n_target=12, point_dimension=2, rng=rng
        )
        problem = instance.problem
        plan = problem.plan.toarray()
        theta = theta_set.sample(rng)
        cost = dense_cost_matrix(problem.features, theta)
        source_mass = problem.resolved_source_mass
        target_mass = problem.resolved_target_mass
        direct_gap = float(
            np.sum(plan * cost) - np.sum(ot.emd(source_mass, target_mass, cost) * cost)
        )
        assert abs(direct_gap - l0_gap(problem, theta)) <= 1e-12


def _assert_metrics_finite(metrics) -> None:
    for value in (
        metrics.l0_gap,
        metrics.moment_error,
        metrics.dist_to_theta_star,
        metrics.wall_time,
        metrics.peak_memory_mb,
    ):
        assert np.isfinite(value)


def test_constraint_generation_certifies_on_simplex(simplex_instance: GroundedInstance) -> None:
    problem = simplex_instance.problem
    result = solve_inverse_ot(problem, tie_break="lex", max_iterations=60)
    assert result.is_optimal is True
    assert problem.theta_set.contains(result.theta, tol=1e-6)
    metrics = evaluate_inverse_ot(
        problem, result.theta, theta_star=simplex_instance.theta_star, iterations=result.n_iterations
    )
    assert metrics.is_optimal
    _assert_metrics_finite(metrics)


def test_sinkhorn_theta_reduces_gap_on_simplex(simplex_instance: GroundedInstance) -> None:
    problem = simplex_instance.problem
    initial = np.full(problem.n_features, 1.0 / problem.n_features)
    initial_gap = l0_gap(problem, initial)
    result = fit_theta_sinkhorn(problem, num_outer=20, sinkhorn_iterations=50)
    assert l0_gap(problem, result.theta) < initial_gap
    assert problem.theta_set.contains(result.theta, tol=1e-9)
    _assert_metrics_finite(
        evaluate_inverse_ot(problem, result.theta, theta_star=simplex_instance.theta_star)
    )


def test_entropic_stops_once_no_longer_improving(simplex_instance: GroundedInstance) -> None:
    # EN: The entropic plan is piecewise constant in theta, so L_eps plateaus and never reaches
    #     the exact optimum. The runner must stop instead of burning the whole outer budget.
    problem = simplex_instance.problem
    result = fit_theta_sinkhorn(
        problem, num_outer=500, epsilon_decay=0.9, sinkhorn_iterations=100
    )
    assert result.stopped_reason == "no_improvement"
    assert 0 < len(result.iterations) < 500
    assert result.stopped_iteration == len(result.iterations) - 1


def test_constraint_generation_certifies_on_psd(psd_instance: GroundedInstance) -> None:
    pytest.importorskip("cvxpy")
    problem = psd_instance.problem
    result = solve_inverse_ot(
        problem, solver=SdpMasterSolver(), tie_break="lex", max_iterations=200, prune=False
    )
    assert result.is_optimal is True
    assert problem.theta_set.contains(result.theta, tol=1e-5)


def test_sinkhorn_theta_reduces_gap_on_psd(psd_instance: GroundedInstance) -> None:
    problem = psd_instance.problem
    dim = int(problem.features.point_dimension)
    # CN: 用秩一 diag(1,0,...,0) 作为起点；I/d 在该小实例上已经最优（gap=0），无法严格下降。
    # EN: Start from the rank-one diag(1,0,...,0); I/d is already optimal on this small
    #     instance (gap=0) and cannot strictly decrease.
    initial = np.zeros(dim * dim)
    initial[0] = 1.0
    initial_gap = l0_gap(problem, initial)
    assert initial_gap > 0.0
    result = fit_theta_sinkhorn(problem, initial_theta=initial, num_outer=20, sinkhorn_iterations=50)
    assert l0_gap(problem, result.theta) < initial_gap
    assert problem.theta_set.contains(result.theta, tol=1e-9)
    _assert_metrics_finite(
        evaluate_inverse_ot(problem, result.theta, theta_star=psd_instance.theta_star)
    )
