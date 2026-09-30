"""
CN: `lex_generic`（证明第 6 节的两步 tie-break）的正确性测试。
EN: Correctness tests for `lex_generic` (the two-solve tie-break of proof section 6).

CN: 关键事实：`lex_generic` 只保证与 `lex` 在 delta 与 theta 上一致；在 delta 最优面上若
    (f,g) 仍有自由度（P_1 非单点），两者可以取到不同的对偶势。两者都必须给出有效的
    max-margin 证书。
EN: Key fact: `lex_generic` agrees with `lex` in delta and theta; on a positive-dimensional
    delta-optimal face (P_1 not a singleton) the two may pick different dual potentials.
    Both must still give a valid max-margin certificate.
"""

from __future__ import annotations

import numpy as np
import pytest

from iot import (
    LinearMasterSolver,
    draw_generic_direction,
    is_optimal_plan,
    make_diagonal_grounded_instance,
    mandatory_edges,
    solve_inverse_ot,
)


def test_draw_generic_direction_is_seeded_and_sized() -> None:
    first = draw_generic_direction(random_seed=5, n_features=3, n_source=4, n_target=6)
    again = draw_generic_direction(random_seed=5, n_features=3, n_source=4, n_target=6)
    other = draw_generic_direction(random_seed=6, n_features=3, n_source=4, n_target=6)
    np.testing.assert_array_equal(first, again)
    assert first.size == 3 + 4 + 6
    assert not np.array_equal(first, other)


def test_lex_generic_matches_lex_on_the_mandatory_set() -> None:
    # CN: 循环的基边集 B 上 delta 最优面是单点，两种 tie-break 必须给出同一点。
    # EN: On the loop's base set B the delta-optimal face is a singleton, so both
    #     tie-breaks must return the same point.
    rng = np.random.default_rng(11)
    instance = make_diagonal_grounded_instance(n_source=12, n_target=12, n_features=3, rng=rng)
    mandatory, _ = mandatory_edges(instance.problem)
    lex = LinearMasterSolver().solve(instance.problem, mandatory, tie_break="lex")
    generic = LinearMasterSolver(random_seed=0).solve(
        instance.problem, mandatory, tie_break="lex_generic"
    )
    assert abs(lex.delta - generic.delta) < 1e-9
    np.testing.assert_allclose(lex.theta, generic.theta, atol=1e-9)
    np.testing.assert_allclose(lex.source_dual, generic.source_dual, atol=1e-8)
    np.testing.assert_allclose(lex.target_dual, generic.target_dual, atol=1e-8)


def test_explicit_generic_direction_matches_internal_draw() -> None:
    rng = np.random.default_rng(13)
    instance = make_diagonal_grounded_instance(n_source=10, n_target=10, n_features=3, rng=rng)
    mandatory, _ = mandatory_edges(instance.problem)
    direction = draw_generic_direction(
        random_seed=0,
        n_features=int(instance.problem.n_features),
        n_source=int(instance.problem.n_source),
        n_target=int(instance.problem.n_target),
    )
    implicit = LinearMasterSolver(random_seed=0).solve(
        instance.problem, mandatory, tie_break="lex_generic"
    )
    explicit = LinearMasterSolver(
        random_seed=0, generic_direction=tuple(float(value) for value in direction)
    ).solve(instance.problem, mandatory, tie_break="lex_generic")
    assert abs(implicit.delta - explicit.delta) < 1e-12
    np.testing.assert_allclose(implicit.theta, explicit.theta, atol=1e-12)
    np.testing.assert_allclose(implicit.source_dual, explicit.source_dual, atol=1e-12)
    np.testing.assert_allclose(implicit.target_dual, explicit.target_dual, atol=1e-12)


def test_lex_generic_rejects_direction_of_wrong_length() -> None:
    rng = np.random.default_rng(14)
    instance = make_diagonal_grounded_instance(n_source=8, n_target=8, n_features=2, rng=rng)
    mandatory, _ = mandatory_edges(instance.problem)
    solver = LinearMasterSolver(generic_direction=(0.0, 1.0))
    with pytest.raises(ValueError, match="generic_direction"):
        solver.solve(instance.problem, mandatory, tie_break="lex_generic")


def test_lex_generic_loop_converges_and_agrees_with_lex_in_theta() -> None:
    rng = np.random.default_rng(12)
    instance = make_diagonal_grounded_instance(n_source=12, n_target=12, n_features=3, rng=rng)
    lex = solve_inverse_ot(
        instance.problem, tie_break="lex", max_iterations=150, prune_hysteresis=False
    )
    generic = solve_inverse_ot(
        instance.problem, tie_break="lex_generic", random_seed=0, max_iterations=150,
        prune_hysteresis=False,
    )
    assert lex.converged and generic.converged
    assert generic.is_optimal is True
    assert abs(lex.delta - generic.delta) < 1e-9
    # CN: theta 是可辨识参数；delta 最优面上不同的对偶势都可以。
    # EN: theta is the identifiable parameter; different duals on the delta-optimal face
    #     are both acceptable.
    np.testing.assert_allclose(lex.theta, generic.theta, atol=1e-9)
    assert is_optimal_plan(instance.problem, generic.theta).is_optimal
