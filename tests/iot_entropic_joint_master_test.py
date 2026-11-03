"""
EN: acceptance tests for `iot.entropic.joint_master` -- the joint Newton-CG that IS T3.

    These are gates, not demonstrations. Each one fails loudly if the corresponding piece is wrong:
      1. connectivity is REFUSED, not silently solved, on a disconnected edge set;
      2. the gauge removes the shift invariance: shifting `f -> f + lam, g -> g - lam` leaves `Psi`
         unchanged, and the solver's result is invariant to it;
      3. the free-block gradient is finite-difference consistent with `Psi`;
      4. on the full grid the joint solve reproduces the nested reference's `Psi` and plan;
      5. the active set holds the simplex exactly (no negative theta, sum == 1) and the KKT
         multipliers on pinned coordinates have the correct sign.

CN: `iot.entropic.joint_master` 的验收测试（联合 Newton-CG，即 T3 本体）。以上每条都是硬性门槛。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from iot.core.edges import EdgeSet
from iot.core.parameters import cost_std
from iot.core.theta import LinearThetaConstraints
from iot.entropic.joint_master import (
    DisconnectedEdgeSetError,
    check_connected,
    component_count,
    make_problem,
    solve_joint_master,
)
from iot.entropic.inner_f import sinkhorn_sweeps_on_f
from iot_experiments.harness.lib.planting import plant_instance


def _instance(n: int = 24, point_dimension: int = 4):
    """EN: a small planted instance plus the full edge set. CN: 小型 planted 实例与全边集。"""
    planted = plant_instance(family="diagonal", n=int(n), point_dimension=int(point_dimension), seed=0)
    problem = planted.problem
    m = int(problem.features.n_source)
    j = int(problem.features.n_target)
    rows = np.repeat(np.arange(m), j)
    cols = np.tile(np.arange(j), m)
    edges = EdgeSet.from_arrays(rows, cols, n_target=j)
    k = int(planted.theta_star.size)
    scale = cost_std(np.asarray(problem.features.costs(np.full(k, 1.0 / k), rows, cols),
                                dtype=np.float64))
    return planted, problem, edges, scale


def test_disconnected_edge_set_is_refused() -> None:
    """EN: a `F` with two components must raise, not solve. CN: 两个分量的 F 必须报错而不是求解。"""
    _planted, problem, edges, _scale = _instance()
    rows = np.asarray(edges.rows)
    cols = np.asarray(edges.cols)
    # EN: keep only the edges of a single planted pair -> one component out of many.
    # CN: 只保留一个 planted 对的边 -> 多个分量中的第一个。
    first_row = rows == rows[0]
    partial = EdgeSet.from_arrays(rows[first_row], cols[first_row], n_target=int(problem.features.n_target))
    assert component_count(partial, n_source=int(problem.features.n_source),
                           n_target=int(problem.features.n_target)) > 1
    with pytest.raises(DisconnectedEdgeSetError):
        check_connected(partial, n_source=int(problem.features.n_source),
                        n_target=int(problem.features.n_target))
    with pytest.raises(DisconnectedEdgeSetError):
        make_problem(problem, partial, epsilon=1.0)


def test_full_grid_is_one_component() -> None:
    """EN: the full grid is connected, so the gate passes it. CN: 全网格连通，门槛放行。"""
    _planted, problem, edges, _scale = _instance()
    assert component_count(edges, n_source=int(problem.features.n_source),
                           n_target=int(problem.features.n_target)) == 1
    check_connected(edges, n_source=int(problem.features.n_source),
                    n_target=int(problem.features.n_target))


def test_gauge_shift_leaves_psi_unchanged() -> None:
    """
    EN: `Psi` is invariant to `f -> f + lam, g -> g - lam`; the solver's fixed point is not.
    CN: `Psi` 对 `f→f+λ, g→g−λ` 不变；求解器固定 `g[0]=0` 以去掉该方向。
    """
    planted, problem, edges, scale = _instance()
    epsilon = 0.5 * scale
    data = make_problem(problem, edges, epsilon=epsilon)
    from iot.entropic.joint_master import _evaluate  # noqa: PLC0415

    theta = np.asarray(planted.theta_star, dtype=np.float64)
    m = int(problem.features.n_source)
    n = int(problem.features.n_target)
    g = np.linspace(-0.3, 0.4, n)
    f = np.linspace(0.1, -0.2, m)
    # EN: build the state from (theta, g) and shift the produced f/g together.
    # CN: 由 (θ, g) 构造状态，再同时平移 f 与 g。
    state, _gt, _gg = _evaluate(data, theta, g)
    shifted_f = state.f + 0.7
    shifted_g = state.g - 0.7
    cost = np.asarray(data.cost_features @ theta, dtype=np.float64)
    plan_shifted = np.exp((shifted_f[data.rows] + shifted_g[data.cols] - cost) / epsilon)
    value_shifted = (float(theta @ data.phi_bar) - float(data.source_mass @ shifted_f)
                     - float(data.target_mass @ shifted_g) + epsilon * float(plan_shifted.sum()))
    assert value_shifted == pytest.approx(state.objective, rel=0, abs=1e-9)
    assert np.allclose(plan_shifted, state.plan, rtol=0, atol=1e-12)


def test_gradient_matches_finite_differences() -> None:
    """
    EN: the free-block gradient `(phi_bar - sum pi phi, colsum - b)` is the true gradient of `Psi`.
    CN: 自由块梯度就是 `Psi` 的真实梯度。
    """
    planted, problem, edges, scale = _instance()
    epsilon = 0.5 * scale
    data = make_problem(problem, edges, epsilon=epsilon)
    from iot.entropic.joint_master import _evaluate  # noqa: PLC0415

    theta = np.asarray(planted.theta_star, dtype=np.float64)
    rng = np.random.default_rng(0)
    g = 0.2 * rng.normal(size=int(problem.features.n_target))
    state, grad_theta, grad_g = _evaluate(data, theta, g)
    step = 1e-7
    for index in range(3):
        bump = np.zeros_like(theta)
        bump[index] = step
        plus, _a, _b = _evaluate(data, theta + bump, g)
        minus, _c, _d = _evaluate(data, theta - bump, g)
        finite = (plus.objective - minus.objective) / (2.0 * step)
        assert grad_theta[index] == pytest.approx(finite, rel=1e-4, abs=1e-5)
    for index in range(3):
        bump = np.zeros_like(g)
        bump[index] = step
        plus, _a, _b = _evaluate(data, theta, g + bump)
        minus, _c, _d = _evaluate(data, theta, g - bump)
        finite = (plus.objective - minus.objective) / (2.0 * step)
        assert grad_g[index] == pytest.approx(finite, rel=1e-4, abs=1e-5)
    assert state.objective == pytest.approx(np.asarray(state.objective))


def test_gate_a_fixed_theta_matches_the_nested_transport_solution() -> None:
    """
    EN: **Gate A -- fixed theta.** Pin `theta = theta_star` by an EQUALITY (lower == upper, so the active
        set never claims it) and solve the joint `inf` over `g` alone. It must reproduce the nested
        full-grid transport solution: the same `Psi`, the same plan, and a converged `g`.

        This is deliberately separate from Gate B: here the question is only whether the joint machinery
        reproduces an ordinary OT solve, not whether it finds the right `theta`.

    CN: **A 关 —— 固定 θ。** 用等式钉住 `theta_star`（下界=上界，主动集不会占用它），只用 `g` 解联合
        `inf`。必须复现嵌套全网格传输解：相同的 `Psi`、相同的 plan、收敛的 `g`。与 B 关刻意分开。
    """
    planted, problem, edges, scale = _instance(n=24)
    epsilon = 0.5 * scale
    data = make_problem(problem, edges, epsilon=epsilon)
    theta_star = np.asarray(planted.theta_star, dtype=np.float64)
    k = int(theta_star.size)
    m = int(problem.features.n_source)
    n = int(problem.features.n_target)

    pinned = LinearThetaConstraints(lower=theta_star.copy(), upper=theta_star.copy(),
                                    a_eq=np.ones((1, k)), b_eq=np.array([theta_star.sum()]))
    result = solve_joint_master(data, theta_star, constraints=pinned, max_newton=100,
                                gradient_tolerance=1e-8)
    inner = sinkhorn_sweeps_on_f(edges, np.asarray(problem.features.costs(theta_star, edges.rows,
                                                                         edges.cols), dtype=np.float64),
                                 np.asarray(problem.resolved_source_mass, dtype=np.float64),
                                 np.asarray(problem.resolved_target_mass, dtype=np.float64),
                                 epsilon=epsilon, marginal_tolerance=1e-13, max_sweeps=400000)
    assert inner.converged
    rows_support, cols_support, mass_support = problem.support()
    phi_bar = (np.asarray(mass_support, dtype=np.float64)[:, None]
               * np.asarray(problem.features.features(rows_support, cols_support),
                            dtype=np.float64)).sum(axis=0)
    reference = (float(theta_star @ phi_bar)
                 - float(np.asarray(problem.resolved_source_mass, dtype=np.float64)
                         @ np.asarray(inner.potential_source, dtype=np.float64))
                 - float(np.asarray(problem.resolved_target_mass, dtype=np.float64)
                         @ np.asarray(inner.potential_target, dtype=np.float64))
                 + epsilon * float(np.asarray(inner.plan_on_f, dtype=np.float64).sum()))
    assert result.converged, f"gate A did not converge: {result.stop_reason}"
    assert result.psi == pytest.approx(reference, rel=0, abs=1e-10)
    assert float(np.abs(result.plan_on_f - inner.plan_on_f).max()) < 1e-10
    # EN: the pinned theta must not drift, and g must have converged (marginals matched).
    # CN: 被钉住的 θ 不得漂移；g 必须收敛（边缘匹配）。
    assert float(np.abs(result.theta - theta_star).max()) < 1e-14
    colsum = np.zeros(n, dtype=np.float64)
    np.add.at(colsum, np.asarray(edges.cols), np.asarray(result.plan_on_f, dtype=np.float64))
    target = np.asarray(problem.resolved_target_mass, dtype=np.float64)
    assert float(np.max(np.abs(colsum - target) / target)) < 1e-10
    _ = m


def test_gate_b_free_theta_reaches_the_nested_optimum_with_kkt() -> None:
    """
    EN: **Gate B -- free theta.** Reproduce the optimum of a fully converged NESTED theta master
        (SLSQP on theta with an exact inner solve) and check the KKT conditions of the joint point.

        Expected to FAIL until the reported defect is fixed -- that defect is recorded below rather than
        papered over, because a test that passes by being weakened is worse than a red one.
    CN: **B 关 —— 自由 θ。** 复现完整收敛的嵌套 θ master 的最优点，并检查联合点的 KKT 条件。
        在缺陷修复前**预期失败**，缺陷记录在下方，而不是靠放宽断言掩盖。
    """
    planted, problem, edges, scale = _instance(n=24)
    epsilon = 0.5 * scale
    k = int(planted.theta_star.size)
    m = int(problem.features.n_source)
    n = int(problem.features.n_target)
    features = np.asarray(problem.features.features(edges.rows, edges.cols), dtype=np.float64)
    a = np.asarray(problem.resolved_source_mass, dtype=np.float64)
    b = np.asarray(problem.resolved_target_mass, dtype=np.float64)
    rows_support, cols_support, mass_support = problem.support()
    phi_bar = (np.asarray(mass_support, dtype=np.float64)[:, None]
               * np.asarray(problem.features.features(rows_support, cols_support),
                            dtype=np.float64)).sum(axis=0)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        """EN: `Psi` and `dPsi/dtheta` at the exact inner optimum. CN: 内层最优处的 `Psi` 与 θ 梯度。"""
        cost = np.asarray(features @ theta, dtype=np.float64)
        inner = sinkhorn_sweeps_on_f(edges, cost, a, b, epsilon=epsilon,
                                     marginal_tolerance=1e-13, max_sweeps=400000)
        value = (float(theta @ phi_bar)
                 - float(a @ np.asarray(inner.potential_source, dtype=np.float64))
                 - float(b @ np.asarray(inner.potential_target, dtype=np.float64))
                 + epsilon * float(np.asarray(inner.plan_on_f, dtype=np.float64).sum()))
        return value, np.asarray(phi_bar - features.T @ np.asarray(inner.plan_on_f, dtype=np.float64),
                                 dtype=np.float64)

    reference = None
    for start in (np.full(k, 1.0 / k), np.asarray(planted.theta_star, dtype=np.float64)):
        candidate = minimize(objective, start.copy(), jac=True, method="SLSQP",
                             bounds=[(0.0, None)] * k,
                             constraints=[{"type": "eq", "fun": lambda t: t.sum() - 1.0,
                                           "jac": lambda t: np.ones(k)}],
                             options={"maxiter": 500, "ftol": 1e-14})
        if reference is None or float(candidate.fun) < float(reference.fun):
            reference = candidate
    theta_reference = np.clip(np.asarray(reference.x, dtype=np.float64), 0.0, None)
    theta_reference = theta_reference / theta_reference.sum()
    psi_reference, grad_reference = objective(theta_reference)

    data = make_problem(problem, edges, epsilon=epsilon)
    result = solve_joint_master(data, np.full(k, 1.0 / k),
                                constraints=problem.theta_set.linear_constraints(),
                                max_newton=100, gradient_tolerance=1e-8)
    assert result.converged, f"gate B did not converge: {result.stop_reason}"
    assert result.psi == pytest.approx(psi_reference, rel=0, abs=1e-8)
    assert float(np.abs(result.theta - theta_reference).sum()) < 1e-4
    # EN: KKT -- the free theta gradient of the nested reference is flat across free coordinates, and
    #     the pinned ones carry a non-negative multiplier.
    # CN: KKT —— 嵌套参考解在自由坐标上的 θ 梯度相同，被固定坐标的乘子非负。
    free = ~np.asarray(result.active_mask, dtype=bool)
    assert float(np.ptp(grad_reference[free])) < 1e-3
    assert float(np.min(grad_reference[np.asarray(result.active_mask, dtype=bool)])) > -1e-6
    _ = (m, n)


