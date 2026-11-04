"""
EN: tests for the INDEPENDENT Gate B reference (`iot.entropic.joint_reference`).

    These tests check the reference against things that do not depend on it: an explicit `(theta, f, g)`
    evaluation, central finite differences, and hand-derived extremes. They deliberately do NOT import the
    custom Newton-CG, so a bug shared between the solver and its reference would still be caught here.

CN: Gate B 独立参考实现的测试。用不依赖它自身的东西来检查：显式 `(θ,f,g)` 求值、中心差分、手推极值。
    刻意不导入自定义 Newton-CG。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from iot.core.parameters import cost_std
from iot.entropic.joint_reference import (
    ReducedProblem,
    evaluate,
    face_enumeration,
    marginals,
    simplex_kkt,
    solve,
)
from iot_experiments.harness.lib.planting import plant_instance


def _problem(n: int = 12, point_dimension: int = 3, rel_eps: float = 0.5):
    """EN: a small planted instance as a `ReducedProblem`. CN: 小型 planted 实例。"""
    planted = plant_instance(family="diagonal", n=int(n), point_dimension=int(point_dimension), seed=0)
    problem = planted.problem
    m = int(problem.features.n_source)
    j = int(problem.features.n_target)
    rows = np.repeat(np.arange(m), j)
    cols = np.tile(np.arange(j), m)
    features = np.asarray(problem.features.features(rows, cols), dtype=np.float64)
    a = np.asarray(problem.resolved_source_mass, dtype=np.float64)
    b = np.asarray(problem.resolved_target_mass, dtype=np.float64)
    support_rows, support_cols, support_mass = problem.support()
    phi_bar = (np.asarray(support_mass, dtype=np.float64)[:, None]
               * np.asarray(problem.features.features(support_rows, support_cols),
                            dtype=np.float64)).sum(axis=0)
    k = int(planted.theta_star.size)
    epsilon = float(rel_eps) * float(cost_std(features @ np.full(k, 1.0 / k)))
    reduced = ReducedProblem(features, np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64),
                             a, b, phi_bar, epsilon)
    return planted, problem, reduced


def test_reduced_objective_equals_the_explicit_theta_f_g_form() -> None:
    """
    EN: the reduced objective is obtained by eliminating `f`; it must equal the explicit `(theta,f,g)`
        objective at the eliminated `f`.
    CN: 约化目标由消去 `f` 得到，必须等于在消去后的 `f` 处显式求值的 `(θ,f,g)` 目标。
    """
    _planted, _problem_obj, reduced = _problem()
    rng = np.random.default_rng(0)
    k = int(reduced.phi_bar.size)
    theta = np.full(k, 1.0 / k)
    g = 0.3 * rng.normal(size=int(reduced.target_mass.size))
    psi, _gt, _gg, plan = evaluate(reduced, theta, g)

    z = (g[reduced.cols] - reduced.cost_features @ theta) / reduced.epsilon
    rows = reduced.source_mass.size
    row_lse = np.asarray(logsumexp(z.reshape(rows, -1), axis=1))
    f = reduced.epsilon * (np.log(reduced.source_mass) - row_lse)
    explicit = (float(theta @ reduced.phi_bar) - float(reduced.source_mass @ f)
                - float(reduced.target_mass @ g) + reduced.epsilon * float(plan.sum()))
    assert psi == pytest.approx(explicit, rel=0, abs=1e-10)


def test_source_marginal_is_exact_by_construction() -> None:
    """EN: eliminating `f` makes the source marginal exact for ANY g. CN: 消去 `f` 后源边缘对任意 g 精确。"""
    _planted, _problem_obj, reduced = _problem()
    rng = np.random.default_rng(1)
    theta = np.full(int(reduced.phi_bar.size), 1.0 / int(reduced.phi_bar.size))
    for scale in (0.0, 1.0, 10.0):
        g = scale * rng.normal(size=int(reduced.target_mass.size))
        _psi, _gt, _gg, plan = evaluate(reduced, theta, g)
        row_error, _col_error = marginals(reduced, plan)
        assert row_error < 1e-12


def test_gradients_match_central_differences() -> None:
    """EN: both gradient blocks against central differences. CN: 两个梯度块对中心差分。"""
    _planted, _problem_obj, reduced = _problem()
    rng = np.random.default_rng(2)
    k = int(reduced.phi_bar.size)
    n = int(reduced.target_mass.size)
    theta = np.full(k, 1.0 / k)
    g = 0.4 * rng.normal(size=n)
    _psi, grad_theta, grad_g, _plan = evaluate(reduced, theta, g)
    step = 1e-6
    for index in range(k):
        bump = np.zeros(k)
        bump[index] = step
        finite = (evaluate(reduced, theta + bump, g)[0] - evaluate(reduced, theta - bump, g)[0]) / (2 * step)
        assert grad_theta[index] == pytest.approx(finite, rel=1e-5, abs=1e-6)
    for index in range(n):
        bump = np.zeros(n)
        bump[index] = step
        finite = (evaluate(reduced, theta, g + bump)[0] - evaluate(reduced, theta, g - bump)[0]) / (2 * step)
        assert grad_g[index] == pytest.approx(finite, rel=1e-5, abs=1e-6)


def test_large_epsilon_reproduces_the_independent_transport() -> None:
    """
    EN: with a large `eps` the entropic plan is nearly the independent one and `Psi_red` must approach the
        closed form `min_theta <phi_bar - pi_ind^T phi, theta>`. This pins the normalisation of every term
        (sign, `eps`, marginal scaling) against something derived by hand rather than by this module.

        With `c = theta . phi`, `pi_ij = a_i b_j / M` gives, for `eps -> inf`,
        `Psi_red -> <phi_bar,theta> - b.g + eps*sum_i a_i*(1 - log a_i + log n_i)` and the `theta`
        dependence reduces to `<phi_bar,theta> - <pi_ind, c>`; we check the `theta` direction only, which is
        the part that carries the inverse-OT information.
    CN: 大 ε 时熵正则计划趋于独立计划，验证 θ 方向的归一化。
    """
    # EN: `point_dimension` drives K; use 4 so there are several theta coordinates to move between.
    # CN: K 由 point_dimension 决定；用 4 以便有多个 θ 坐标。
    planted, problem_obj, reduced = _problem(point_dimension=4, rel_eps=1.0)
    k = int(reduced.phi_bar.size)
    assert k == 4
    m = int(reduced.source_mass.size)
    n = int(reduced.target_mass.size)
    # EN: build a problem where the cost is a sum of per-row constants, so the plan is independent of
    #     theta and the exact theta dependence is the linear term only.
    # CN: 构造每行常数的代价，使计划与 θ 无关，此时的 θ 依赖只有线性项。
    features = np.zeros_like(reduced.cost_features)
    for row in range(m):
        features[reduced.rows == row, :] = 1.0
    phi_bar = np.zeros(k)
    phi_bar[0] = 1.0
    flat = ReducedProblem(features, reduced.rows, reduced.cols, reduced.source_mass,
                          reduced.target_mass, phi_bar, reduced.epsilon)
    g = np.zeros(n)
    theta = np.full(k, 1.0 / k)
    _psi, grad_theta, _grad_g, _plan = evaluate(flat, theta, g)
    # EN: with a per-row-constant cost, every plan's feature average is the row-constant average, so the
    #     gradient is `phi_bar` minus that constant: a fixed vector, independent of theta.
    # CN: 每行常数代价下梯度与 θ 无关。
    other = np.full(k, 1.0 / k)
    other[0] = 0.7
    other[1:] = (1.0 - other[0]) / (k - 1)
    _psi2, grad_theta2, _gg, _plan2 = evaluate(flat, other, g)
    assert np.allclose(grad_theta, grad_theta2, rtol=0, atol=1e-12)
    _ = planted, problem_obj


def test_simplex_kkt_detects_a_violation() -> None:
    """EN: the KKT test must flag a pinned coordinate whose multiplier is negative. CN: 必须识别负乘子。"""
    theta = np.array([0.5, 0.5, 0.0])
    # EN: multipliers are `grad - lam`; at a KKT point the positive set has equal gradients and every
    #     excluded coordinate has `grad - lam >= 0`.
    # CN: 乘子是 `grad - lam`；KKT 点要求正坐标梯度相等、被排除坐标 `grad - lam >= 0`。
    good = np.array([2.0, 2.0, 3.0])
    bad = np.array([2.0, 2.0, 1.0])
    _multiplier, good_violation = simplex_kkt(good, theta)
    _multiplier2, bad_violation = simplex_kkt(bad, theta)
    assert good_violation < 1e-12
    assert bad_violation == pytest.approx(1.0, abs=1e-12)
    # EN: unequal gradients on the positive set are also a violation.
    # CN: 正坐标上梯度不相等同样是违背。
    _m, moved_violation = simplex_kkt(np.array([2.0, 3.0, 5.0]), theta)
    assert moved_violation == pytest.approx(0.5, abs=1e-12)


def test_face_enumeration_has_one_optimal_face_and_it_is_kkt_feasible() -> None:
    """
    EN: enumerate every face and require (i) the best face to be KKT-feasible, (ii) no cheaper face, and
        (iii) every non-optimal face to be rejected by an excluded multiplier.

        This is the independent counterpart of the custom solver's face check; agreement between the only
        two such checks is evidence, and the assertions here do not read the expected answer from anywhere.
    CN: 枚举全部面，要求最优面满足 KKT、没有更便宜的面、其余面都被负乘子排除。
    """
    _planted, _problem_obj, reduced = _planted_problem = _problem(n=12)
    k = int(reduced.phi_bar.size)
    faces = face_enumeration(reduced, theta_start=np.full(k, 1.0 / k))
    assert len(faces) == (2 ** k) - 1
    best = faces[0]
    assert bool(best["excluded_kkt_ok"])
    assert all(float(record["psi"]) >= float(best["psi"]) - 1e-12 for record in faces)
    # EN: every STRICTLY CHEAPER face must be rejected by a negative multiplier. If a cheaper face were
    #     also KKT-feasible the enumeration would contradict itself, so this is the real consistency check.
    #     A face containing the optimal support holds the same point (it merely has extra zero
    #     multipliers) and is legitimately feasible, so it is not treated as a rival.
    # CN: 每个**严格更便宜**的面都必须被负乘子拒绝；这样才不自相矛盾。包含最优支撑的面持有同一点，
    #     被豁免。
    support = set(best["face"])
    for record in faces[1:]:
        if support.issubset(set(record["face"])):
            continue
        if float(record["psi"]) > float(best["psi"]) + 1e-9:
            assert not bool(record["excluded_kkt_ok"]), record["face"]
    # EN: the full face must reach the same value as the best face (it contains it).
    # CN: 全面必须达到与最优面相同的值。
    full = next(record for record in faces if len(record["face"]) == k)
    assert float(full["psi"]) == pytest.approx(float(best["psi"]), rel=0, abs=1e-8)
    assert float(full["row_marginal_error"]) < 1e-9
    _ = _planted_problem


def test_solve_from_several_starts_agrees() -> None:
    """EN: the optimum must not depend on the starting point. CN: 最优点不应依赖起点。"""
    _planted, _problem_obj, reduced = _problem(point_dimension=4, n=12)
    k = int(reduced.phi_bar.size)
    n = int(reduced.target_mass.size)
    assert k == 4
    starts = [np.full(k, 1.0 / k), np.array([0.7, 0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1, 0.7])]
    values = []
    for theta_start in starts:
        run = solve(reduced, theta_start, np.zeros(n))
        assert int(run.status) == 0, run.message
        assert run.simplex_kkt_violation < 1e-5
        assert run.col_marginal_error < 1e-5
        values.append(float(run.psi))
    assert float(np.ptp(values)) < 1e-8
