"""
CN: `iot.subgradient` 的不变量测试：次梯度正确性（凸性不等式）、观测矩一致、三种步长下的
    收敛/投影不变量与确定性。
EN: Invariant tests for `iot.subgradient`: subgradient correctness (convexity inequality),
    observed-moment consistency, and convergence/projection/determinism invariants across the
    step rules.
"""

from __future__ import annotations

import numpy as np
import ot
import pytest
from scipy import sparse

from iot import (
    DiagonalWeightFeatures,
    InverseOTProblem,
    SimplexTheta,
    dense_cost_matrix,
    is_optimal_plan,
)
from iot.evaluation import feature_moment
from iot.subgradient import (
    SubgradientConfig,
    fit_theta_subgradient,
    make_pot_solver,
    observed_feature_moment,
    subgradient_from_plan,
    support_moment,
)


def _problem(n: int = 24, k: int = 3, seed: int = 0) -> tuple[InverseOTProblem, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    theta_set = SimplexTheta(int(k))
    theta_star = theta_set.sample(rng)
    source = rng.normal(size=(int(n), int(k)))
    target = rng.normal(size=(int(n), int(k)))
    features = DiagonalWeightFeatures(source, target)
    cost = dense_cost_matrix(features, theta_star)
    mass = np.full(int(n), 1.0 / float(n))
    plan = ot.emd(mass, mass, cost, numItermax=2_000_000)
    return InverseOTProblem(features, theta_set, sparse.csr_matrix(plan)), theta_star


def test_observed_moment_matches_dense_feature_moment() -> None:
    problem, _ = _problem()
    expected = feature_moment(problem, problem.plan.toarray())
    np.testing.assert_allclose(observed_feature_moment(problem), expected, atol=1e-12)


def test_subgradient_is_moment_difference_and_gap_is_theta_dot_gradient() -> None:
    problem, _ = _problem()
    theta = np.full(problem.n_features, 1.0 / problem.n_features)
    solver = make_pot_solver(problem)
    plan = solver(theta)
    gradient, gap = subgradient_from_plan(problem, theta, plan)
    expected = observed_feature_moment(problem) - support_moment(problem.features, plan)
    np.testing.assert_allclose(gradient, expected, atol=1e-12)
    assert gap == pytest.approx(float(theta @ gradient))
    assert gap >= -1e-12


def test_subgradient_satisfies_convexity_inequality() -> None:
    # g must be a valid subgradient: L0(theta') >= L0(theta) + <g, theta' - theta> for feasible theta'.
    problem, _ = _problem(n=20, k=3, seed=1)
    theta_set = problem.theta_set
    theta0 = np.full(problem.n_features, 1.0 / problem.n_features)
    solver = make_pot_solver(problem)
    gradient, gap0 = subgradient_from_plan(problem, theta0, solver(theta0))

    rng = np.random.default_rng(0)
    for _ in range(20):
        perturbation = rng.normal(scale=0.05, size=problem.n_features)
        theta_prime = theta0 + perturbation
        # Project onto the simplex via the same routine the learner uses.
        from iot.sinkhorn_theta import project_theta

        theta_prime = project_theta(theta_prime, theta_set)
        _, gap_prime = subgradient_from_plan(problem, theta_prime, solver(theta_prime))
        assert gap_prime >= gap0 + float(gradient @ (theta_prime - theta0)) - 1e-9


def test_constant_rule_reaches_optimal_on_clean_instance() -> None:
    problem, _ = _problem(n=24, k=3, seed=0)
    result = fit_theta_subgradient(
        problem,
        config=SubgradientConfig(step_rule="constant", learning_rate=0.05, max_iterations=3000,
                                 tol=1e-7),
    )
    assert result.converged
    assert result.final_gap <= 1e-7
    assert problem.theta_set.contains(result.theta, tol=1e-9)
    assert is_optimal_plan(problem, result.theta).is_optimal


def test_polyak_rule_reaches_small_gap_on_clean_instance() -> None:
    problem, _ = _problem(n=24, k=3, seed=0)
    result = fit_theta_subgradient(
        problem,
        config=SubgradientConfig(step_rule="polyak", learning_rate=1.0, max_iterations=2000,
                                 tol=1e-7),
    )
    assert result.final_gap <= 1e-6
    assert problem.theta_set.contains(result.theta, tol=1e-9)


def test_theta_stays_in_simplex_for_every_rule() -> None:
    problem, _ = _problem(n=20, k=4, seed=2)
    for rule in ("constant", "inv_sqrt", "polyak"):
        result = fit_theta_subgradient(
            problem,
            config=SubgradientConfig(step_rule=rule, learning_rate=0.05, max_iterations=50),
        )
        assert problem.theta_set.contains(result.theta, tol=1e-9)
        assert np.all(np.isfinite(result.theta))


def test_learner_is_deterministic() -> None:
    problem, _ = _problem(n=20, k=3, seed=3)
    config = SubgradientConfig(step_rule="constant", learning_rate=0.05, max_iterations=200)
    first = fit_theta_subgradient(problem, config=config)
    second = fit_theta_subgradient(problem, config=config)
    np.testing.assert_array_equal(first.theta, second.theta)
    assert [item.gap for item in first.iterations] == [item.gap for item in second.iterations]


def test_injected_solver_is_used() -> None:
    problem, _ = _problem(n=16, k=3, seed=4)
    calls: list[int] = []

    def solver(theta: np.ndarray) -> sparse.spmatrix:
        calls.append(int(theta.size))
        import ot as _ot

        cost = dense_cost_matrix(problem.features, np.asarray(theta, dtype=np.float64))
        return sparse.csr_matrix(
            _ot.emd(
                problem.resolved_source_mass,
                problem.resolved_target_mass,
                cost,
                numItermax=2_000_000,
            )
        )

    result = fit_theta_subgradient(
        problem,
        config=SubgradientConfig(
            step_rule="constant", learning_rate=0.05, max_iterations=5, tol=-1.0
        ),
        exact_plan_solver=solver,
    )
    assert len(calls) == len(result.iterations) == 5


def test_bad_step_rule_raises() -> None:
    with pytest.raises(ValueError):
        SubgradientConfig(step_rule="nope")  # type: ignore[arg-type]
