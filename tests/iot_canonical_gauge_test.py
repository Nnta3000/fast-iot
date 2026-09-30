from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from iot import (
    CanonicalGaugeMasterSolver,
    EdgeSet,
    LinearMasterSolver,
    RegularizedMasterSolver,
    canonical_potentials,
    make_diagonal_grounded_instance,
    mandatory_edges,
)


def _instance(seed: int = 3):
    rng = np.random.default_rng(seed)
    return make_diagonal_grounded_instance(
        n_source=8, n_target=8, n_features=3, rng=rng, mixing_strength=0.5
    )


def _face_lp(problem, edges: EdgeSet, *, delta: float, theta: np.ndarray) -> tuple[float, np.ndarray]:
    """CN: 固定 (delta,theta) 最大化 sum_i f_i 的 LP。EN: LP that maximises sum_i f_i at fixed
    (delta, theta)."""
    m = int(problem.n_source)
    n = int(problem.n_target)
    support_rows, support_cols, _ = problem.support()
    match_source = np.empty(m, dtype=np.int64)
    match_source[support_rows] = support_cols
    cost_by_source = np.empty(m, dtype=np.float64)
    cost_by_source[support_rows] = problem.features.costs(theta, support_rows, support_cols)

    a_eq = np.zeros((m + 1, m + n), dtype=np.float64)
    b_eq = np.zeros(m + 1, dtype=np.float64)
    for i in range(m):
        a_eq[i, i] = 1.0
        a_eq[i, m + int(match_source[i])] = 1.0
        b_eq[i] = cost_by_source[i]
    a_eq[m, m] = 1.0  # gauge g_0 = 0

    in_sup = np.isin(edges.keys, problem.support_keys())
    off_rows = edges.rows[~in_sup]
    off_cols = edges.cols[~in_sup]
    off_costs = problem.features.costs(theta, off_rows, off_cols)
    a_ub = np.zeros((off_rows.size, m + n), dtype=np.float64)
    b_ub = np.zeros(off_rows.size, dtype=np.float64)
    for position in range(off_rows.size):
        a_ub[position, int(off_rows[position])] = 1.0
        a_ub[position, m + int(off_cols[position])] = 1.0
        b_ub[position] = float(off_costs[position]) - float(delta)

    objective = np.zeros(m + n, dtype=np.float64)
    objective[:m] = 1.0
    result = linprog(-objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                     bounds=[(None, None)] * (m + n), method="highs")
    assert result.success
    return -float(result.fun), np.asarray(result.x, dtype=np.float64)


def test_canonical_potentials_are_feasible_and_gauged() -> None:
    instance = _instance()
    problem = instance.problem
    assert problem.support_size == int(problem.n_source)
    mandatory, _ = mandatory_edges(problem)
    base = RegularizedMasterSolver(epsilon=1e-4, random_seed=0).solve(problem, mandatory, tie_break="none")
    f, g = canonical_potentials(
        problem, mandatory, delta=base.delta, theta=base.theta, reference_source_dual=base.source_dual
    )
    # CN: 规范固定 g_0 = 0。EN: the gauge fixes g_0 = 0.
    assert abs(float(g[0])) < 1e-12
    in_support = np.isin(mandatory.keys, problem.support_keys())
    costs = problem.features.costs(base.theta, mandatory.rows, mandatory.cols)
    residual = f[mandatory.rows] + g[mandatory.cols] - costs
    np.testing.assert_allclose(residual[in_support], 0.0, atol=1e-9)
    assert bool(np.all(residual[~in_support] + base.delta <= 1e-9))


def test_canonical_potentials_are_pointwise_maximal() -> None:
    instance = _instance(seed=4)
    problem = instance.problem
    mandatory, _ = mandatory_edges(problem)
    base = RegularizedMasterSolver(epsilon=1e-4, random_seed=0).solve(problem, mandatory, tie_break="none")
    f, g = canonical_potentials(
        problem, mandatory, delta=base.delta, theta=base.theta, reference_source_dual=base.source_dual
    )
    best_sum, x_lp = _face_lp(problem, mandatory, delta=base.delta, theta=base.theta)
    m = int(problem.n_source)
    np.testing.assert_allclose(float(np.sum(f)), best_sum, atol=1e-7)
    np.testing.assert_allclose(f, x_lp[:m], atol=1e-7)
    np.testing.assert_allclose(g, x_lp[m:], atol=1e-7)


def test_canonical_gauge_wrapper_is_master_agnostic() -> None:
    instance = _instance(seed=5)
    problem = instance.problem
    mandatory, _ = mandatory_edges(problem)
    direct = RegularizedMasterSolver(epsilon=1e-4, random_seed=0)
    wrapped = CanonicalGaugeMasterSolver(RegularizedMasterSolver(epsilon=1e-4, random_seed=0))
    base = direct.solve(problem, mandatory, tie_break="none")
    canonical = wrapped.solve(problem, mandatory, tie_break="none")
    # CN: 包装器只重算 (f,g)，不改 (delta,theta)。EN: the wrapper only recomputes (f,g).
    np.testing.assert_allclose(canonical.delta, base.delta, atol=1e-12)
    np.testing.assert_allclose(canonical.theta, base.theta, atol=1e-12)
    f, g = canonical_potentials(
        problem, mandatory, delta=base.delta, theta=base.theta, reference_source_dual=base.source_dual
    )
    np.testing.assert_allclose(canonical.source_dual, f, atol=1e-12)
    np.testing.assert_allclose(canonical.target_dual, g, atol=1e-12)


def test_canonical_gauge_matches_across_references() -> None:
    """CN: 相同 (delta,theta) 下，不同参考势给出相同规范势。EN: same (delta,theta), same result."""
    instance = _instance(seed=6)
    problem = instance.problem
    mandatory, _ = mandatory_edges(problem)
    solution = LinearMasterSolver(random_seed=0).solve(problem, mandatory, tie_break="none")
    f0, g0 = canonical_potentials(
        problem, mandatory, delta=solution.delta, theta=solution.theta,
        reference_source_dual=solution.source_dual,
    )
    perturbed = solution.source_dual + 0.5  # a different but still feasible reference
    f1, g1 = canonical_potentials(
        problem, mandatory, delta=solution.delta, theta=solution.theta,
        reference_source_dual=perturbed,
    )
    np.testing.assert_allclose(f1, f0, atol=1e-9)
    np.testing.assert_allclose(g1, g0, atol=1e-9)
