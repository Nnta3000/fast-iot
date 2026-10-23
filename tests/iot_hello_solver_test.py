"""
CN: `iot.hello_solver` 的测试：HELLO-OT 内层求解器与 POT 给出一致的最优代价，且能把次梯度法
    跑到收敛。缺少 hello_ot 时跳过。
EN: Tests for `iot.hello_solver`: the HELLO-OT inner solver matches POT's optimal cost and drives
    the subgradient method to convergence. Skipped when `hello_ot` is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from iot import (
    DiagonalWeightFeatures,
    dense_cost_matrix,
    hello_solver_supported,
    make_hello_solver,
    make_diagonal_grounded_instance,
    mandatory_edges,
)
from iot.alternatives.subgradient import (
    SubgradientConfig,
    fit_theta_subgradient,
    make_pot_solver,
)


def _instance(n: int = 30, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(int(seed))
    return make_diagonal_grounded_instance(
        n_source=int(n), n_target=int(n), n_features=int(k), rng=rng
    )


def test_hello_solver_supported_only_for_diagonal() -> None:
    from iot import BilinearPSDFeatures

    assert hello_solver_supported(DiagonalWeightFeatures(np.zeros((3, 2)), np.zeros((3, 2))))
    assert not hello_solver_supported(BilinearPSDFeatures(np.zeros((3, 2)), np.zeros((3, 2))))


def test_hello_solver_matches_pot_cost() -> None:
    pytest.importorskip("hello_ot")
    instance = _instance()
    problem = instance.problem
    theta = np.full(int(problem.n_features), 1.0 / int(problem.n_features))
    plan_hello = make_hello_solver(problem, device="auto")(theta)
    plan_pot = make_pot_solver(problem)(theta)
    cost = dense_cost_matrix(problem.features, theta)
    hello_cost = float(np.sum(cost * plan_hello.toarray()))
    pot_cost = float(np.sum(cost * plan_pot.toarray()))
    # both plans are optimal for the same cost; their costs agree to solver tolerance
    assert abs(hello_cost - pot_cost) < 1e-4 * (1.0 + abs(pot_cost))


def test_subgradient_with_hello_solver_reaches_optimal() -> None:
    pytest.importorskip("hello_ot")
    instance = _instance(n=30, k=3, seed=1)
    result = fit_theta_subgradient(
        instance.problem,
        config=SubgradientConfig(step_rule="inv_sqrt", learning_rate=0.5, max_iterations=300, tol=1e-7),
        exact_plan_solver=make_hello_solver(instance.problem, device="auto"),
    )
    assert result.best_gap < 1e-4
