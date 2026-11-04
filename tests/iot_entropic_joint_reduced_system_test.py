"""
EN: targeted tests for the CUSTOM joint solver's Newton system (`iot.entropic.joint_master`).

    These do not depend on Gate B converging. They test the machinery the task asked to be proven:
      A. the reduced HVP equals finite differences of the independently evaluated reduced gradient;
      B. the reduced HVP is symmetric (CG needs a symmetric operator);
      C. a dense reduced Hessian built by finite differences matches the matrix-free operator;
      D. curvature along feasible directions at an interior point.

    The reference side of every comparison is built from `iot.entropic.joint_reference`, never from the
    custom solver's own reduction, so a shared bug cannot be hidden.

CN: 自定义联合求解器牛顿系统的定向测试（不依赖 Gate B 收敛）。参考侧一律由 `joint_reference` 构造。
"""

from __future__ import annotations

import numpy as np

from iot.core.edges import EdgeSet
from iot.core.parameters import cost_std
from iot.entropic.inner_f import joint_hvp
from iot.entropic.joint_master import make_problem
from iot.entropic.joint_reference import ReducedProblem, evaluate
from iot_experiments.harness.lib.planting import plant_instance


def _setup(rel_eps: float = 0.5, n: int = 24):
    """EN: a planted instance, both problem views, and the epsilon. CN: 实例与两种问题视图。"""
    planted = plant_instance(family="diagonal", n=int(n), point_dimension=4, seed=0)
    problem = planted.problem
    m = int(problem.features.n_source)
    j = int(problem.features.n_target)
    k = int(planted.theta_star.size)
    rows = np.repeat(np.arange(m), j)
    cols = np.tile(np.arange(j), m)
    features = np.asarray(problem.features.features(rows, cols), dtype=np.float64)
    a = np.asarray(problem.resolved_source_mass, dtype=np.float64)
    b = np.asarray(problem.resolved_target_mass, dtype=np.float64)
    support_rows, support_cols, support_mass = problem.support()
    phi_bar = (np.asarray(support_mass, dtype=np.float64)[:, None]
               * np.asarray(problem.features.features(support_rows, support_cols),
                            dtype=np.float64)).sum(axis=0)
    epsilon = float(rel_eps) * float(cost_std(features @ np.full(k, 1.0 / k)))
    reference = ReducedProblem(features, np.asarray(rows, dtype=np.int64),
                               np.asarray(cols, dtype=np.int64), a, b, phi_bar, epsilon)
    edges = EdgeSet.from_arrays(rows, cols, n_target=j)
    solver = make_problem(problem, edges, epsilon=epsilon)
    return planted, problem, reference, solver, epsilon, k, j


def _reduced_gradient(reference: ReducedProblem, theta: np.ndarray, g: np.ndarray,
                      non_pivot: np.ndarray, pivot: int) -> np.ndarray:
    """
    EN: the reduced gradient built INDEPENDENTLY: `S^T grad` with `S` the pivot map.
        theta part: `grad_k - grad_pivot`; g part: `grad_g[1:]` (gauge coordinate 0 is fixed).
    CN: 独立构造的约化梯度：`S^T grad`。
    """
    _psi, grad_theta, grad_g, _plan = evaluate(reference, theta, g)
    return np.concatenate([grad_theta[non_pivot] - grad_theta[pivot], grad_g[1:]])


def _expand(v: np.ndarray, non_pivot: np.ndarray, pivot: int, k: int, n: int):
    """EN: reduced direction -> full `(d_theta, d_g)` with the pivot absorbing the equality."""
    d_theta = np.zeros(k, dtype=np.float64)
    d_theta[non_pivot] = v[:non_pivot.size]
    d_theta[pivot] = -float(np.sum(v[:non_pivot.size]))
    d_g = np.zeros(n, dtype=np.float64)
    d_g[1:] = v[non_pivot.size:]
    return d_theta, d_g


def _full_hvp(solver, theta: np.ndarray, g: np.ndarray, d_theta: np.ndarray,
              d_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    EN: the FULL-space product `H w` at `(theta, g)`, obtained from `joint_hvp` (which is left returning
        the full space by design; the pivot reduction belongs to the optimiser).
    CN: 全空间的 `H w`。
    """
    _state, _gt, _gg = solver_eval(solver, theta, g)
    plan = _plan(solver, theta, g)
    hv_theta, hv_g = joint_hvp(plan, solver.rows, solver.cols, solver.cost_features,
                               solver.source_mass, solver.epsilon, d_theta, d_g)
    return np.asarray(hv_theta, dtype=np.float64), np.asarray(hv_g, dtype=np.float64)


def solver_eval(solver, theta: np.ndarray, g: np.ndarray):
    """EN: the solver's own evaluation (plan and gradients). CN: 求解器自身的求值。"""
    from iot.entropic.joint_master import _evaluate  # noqa: PLC0415

    return _evaluate(solver, theta, g)


def _plan(solver, theta: np.ndarray, g: np.ndarray) -> np.ndarray:
    """EN: the plan at `(theta, g)` from the solver's evaluation. CN: 求解器求值得到的 plan。"""
    state, _gt, _gg = solver_eval(solver, theta, g)
    return np.asarray(state.plan, dtype=np.float64)


def _reduced_hvp(reference, solver, theta, g, non_pivot, pivot, v, k, n) -> np.ndarray:
    """
    EN: the reduced operator `S^T H S`, assembled by the TEST from the full product.
        (H_red v)_theta = (Hv)_theta_k - (a_k / a_p) (Hv)_pivot   and   (H_red v)_g = (Hv)_g[1:].
    CN: 测试侧组装 `S^T H S`。
    """
    d_theta, d_g = _expand(v, non_pivot, pivot, k, n)
    hv_theta, hv_g = _full_hvp(solver, theta, g, d_theta, d_g)
    return np.concatenate([hv_theta[non_pivot] - hv_theta[pivot], hv_g[1:]])


def _hf_feasible(theta: np.ndarray, v: np.ndarray, non_pivot: np.ndarray, pivot: int,
                 step: float = 1e-6) -> np.ndarray:
    """EN: perturb `theta` along the reduced theta direction, preserving `sum(theta)=1`."""
    d_theta = np.zeros_like(theta)
    d_theta[non_pivot] = v[:non_pivot.size]
    d_theta[pivot] = -float(np.sum(v[:non_pivot.size]))
    return theta + step * d_theta


def test_a_reduced_hvp_matches_finite_differences_of_the_reduced_gradient() -> None:
    """
    EN: Test A. `S^T H S v` against a central difference of the INDEPENDENTLY built reduced gradient, at
        several feasible points including the Gate B stalled point.
    CN: 测试 A：约化 HVP 与独立约化梯度的中心差分一致。
    """
    _planted, _problem, reference, solver, _epsilon, k, n = _setup()
    points = {
        "gate_b_stall": (np.array([0.493804, 0.506196, 0.013428, 0.0]), np.array([0, 2]), 1),
        "interior": (np.full(k, 0.25), np.array([0, 2, 3]), 1),
        "another": (np.array([0.55, 0.25, 0.15, 0.05]), np.array([0, 2]), 1),
    }
    rng = np.random.default_rng(0)
    for label, (theta0, non_pivot, pivot) in points.items():
        theta = np.asarray(theta0, dtype=np.float64)
        theta = theta / theta.sum()
        g = np.zeros(n, dtype=np.float64)
        dims = non_pivot.size + (n - 1)
        for _trial in range(3):
            v = rng.normal(size=dims)
            analytic = _reduced_hvp(reference, solver, theta, g, non_pivot, pivot, v, k, n)
            step = 1e-6
            plus = _reduced_gradient(reference, _hf_feasible(theta, v, non_pivot, pivot, step),
                                     g + step * np.concatenate([[0.0], v[non_pivot.size:]]),
                                     non_pivot, pivot)
            minus = _reduced_gradient(reference, _hf_feasible(theta, v, non_pivot, pivot, -step),
                                      g - step * np.concatenate([[0.0], v[non_pivot.size:]]),
                                      non_pivot, pivot)
            finite = (plus - minus) / (2 * step)
            error = float(np.max(np.abs(analytic - finite)))
            relative = error / max(1e-30, float(np.max(np.abs(finite))))
            assert error < 1e-4, f"{label}: reduced HVP differs from finite differences by {error:.3e}"
            assert relative < 1e-4, f"{label}: relative error {relative:.3e}"


def test_b_reduced_hvp_is_symmetric() -> None:
    """
    EN: Test B. `u^T H_red v == v^T H_red u` for independent random `u, v`. CG requires a symmetric
        operator, and a correct full-space HVP can still be wrongly reduced.
    CN: 测试 B：约化 HVP 对称（CG 必需）。
    """
    _planted, _problem, reference, solver, _epsilon, k, n = _setup()
    theta = np.array([0.55, 0.25, 0.15, 0.05])
    theta = theta / theta.sum()
    g = np.zeros(n, dtype=np.float64)
    non_pivot = np.array([0, 2])
    pivot = 1
    dims = non_pivot.size + (n - 1)
    rng = np.random.default_rng(1)
    for _trial in range(4):
        u = rng.normal(size=dims)
        v = rng.normal(size=dims)
        hu = _reduced_hvp(reference, solver, theta, g, non_pivot, pivot, u, k, n)
        hv = _reduced_hvp(reference, solver, theta, g, non_pivot, pivot, v, k, n)
        left = float(u @ hv)
        right = float(v @ hu)
        assert abs(left - right) <= 1e-8 * max(1.0, abs(left)), f"asymmetry {left - right:.3e}"


def test_c_dense_reduced_hessian_matches_the_matrix_free_operator() -> None:
    """
    EN: Test C. Build the whole `S^T H S` by central differences of the independent reduced gradient, then
        compare every column against the matrix-free operator.
    CN: 测试 C：用中心差分构造完整的 `S^T H S`，逐列与无矩阵算子比较。
    """
    _planted, _problem, reference, solver, _epsilon, k, n = _setup()
    theta = np.array([0.55, 0.25, 0.15, 0.05])
    theta = theta / theta.sum()
    g = np.zeros(n, dtype=np.float64)
    non_pivot = np.array([0, 2])
    pivot = 1
    dims = non_pivot.size + (n - 1)
    step = 1e-6
    dense = np.zeros((dims, dims))
    for column in range(dims):
        e = np.zeros(dims)
        e[column] = 1.0
        plus = _reduced_gradient(reference, _hf_feasible(theta, e, non_pivot, pivot, step),
                                 g + step * np.concatenate([[0.0], e[non_pivot.size:]]),
                                 non_pivot, pivot)
        minus = _reduced_gradient(reference, _hf_feasible(theta, e, non_pivot, pivot, -step),
                                  g - step * np.concatenate([[0.0], e[non_pivot.size:]]),
                                  non_pivot, pivot)
        dense[:, column] = (plus - minus) / (2 * step)
    rng = np.random.default_rng(2)
    for _trial in range(3):
        v = rng.normal(size=dims)
        matrix_free = _reduced_hvp(reference, solver, theta, g, non_pivot, pivot, v, k, n)
        assert float(np.max(np.abs(dense @ v - matrix_free))) < 1e-4
    # EN: the dense matrix must be symmetric on its own, independently of the operator under test.
    # CN: 稠密矩阵自身必须对称（不依赖被测算子）。
    assert float(np.max(np.abs(dense - dense.T))) < 1e-5


def test_d_positive_curvature_at_an_interior_point() -> None:
    """
    EN: Test D. The smallest eigenvalue of the explicitly constructed reduced Hessian at an interior point
        must be positive, and `v^T H_red v > 0` along random feasible directions.
    CN: 测试 D：内点处约化 Hessian 的最小特征值为正。
    """
    _planted, _problem, reference, solver, _epsilon, k, n = _setup()
    theta = np.array([0.55, 0.25, 0.15, 0.05])
    theta = theta / theta.sum()
    g = np.zeros(n, dtype=np.float64)
    non_pivot = np.array([0, 2])
    pivot = 1
    dims = non_pivot.size + (n - 1)
    step = 1e-6
    dense = np.zeros((dims, dims))
    for column in range(dims):
        e = np.zeros(dims)
        e[column] = 1.0
        plus = _reduced_gradient(reference, _hf_feasible(theta, e, non_pivot, pivot, step),
                                 g + step * np.concatenate([[0.0], e[non_pivot.size:]]),
                                 non_pivot, pivot)
        minus = _reduced_gradient(reference, _hf_feasible(theta, e, non_pivot, pivot, -step),
                                  g - step * np.concatenate([[0.0], e[non_pivot.size:]]),
                                  non_pivot, pivot)
        dense[:, column] = (plus - minus) / (2 * step)
    symmetric = 0.5 * (dense + dense.T)
    smallest = float(np.linalg.eigvalsh(symmetric)[0])
    assert smallest > 0.0, f"reduced Hessian is not positive definite at an interior point: {smallest:.3e}"
    rng = np.random.default_rng(3)
    for _trial in range(4):
        v = rng.normal(size=dims)
        hv = _reduced_hvp(reference, solver, theta, g, non_pivot, pivot, v, k, n)
        assert float(v @ hv) > 0.0
