"""
CN: **参考实现（测试专用）**：熵 OT 的稠密外层求解器。EOT 的验收基准，不是生产代码。

CN: **熵 OT 的外层求解器（可收敛）** —— 精确梯度 + SLSQP，把 `L_eps(θ)` 解到底。

    CN: 第 1 轮用的是"固定步长投影梯度 + 固定步数"，每一格都停在 `max_steps`，于是 `eps` 与提前停止
        混在一起。这里用 `scipy.optimize.minimize(method="SLSQP")`，`Θ` 由**仓库自己的**线性约束给出
        （`theta_set.linear_constraints()`：`A_eq θ = b_eq` 与 `lb <= θ <= ub`），梯度为闭式
        `∇L_eps = φ̄ − E_{π_ε(θ)}[φ]`。

    CN: **停止判据**（事先固定）：**Frank–Wolfe gap** `<= 1e-4·eps`，或外层 500 次。
        达到上限的拟合标记 `NOT CONVERGED`，**不参与 eps 选择与判定**，但仍被报告。

    CN: 内层必须收敛（相对边缘误差 <= 1e-6）才使用；失败即抛，绝不用部分方案。

EN: **Convergent outer solver for entropic OT** -- exact gradient plus SLSQP.

    Round 1 used a fixed-step projected gradient with a fixed budget; every cell stopped at `max_steps`,
    confounding `eps` with early stopping. Here `scipy.optimize.minimize(method="SLSQP")` is used, with `Theta`
    supplied by the **repository's own** linear constraints (`theta_set.linear_constraints()`:
    `A_eq theta = b_eq`, `lb <= theta <= ub`) and the closed-form gradient `grad L_eps = phi_bar - E[phi]`.

    **Stopping rule** (fixed in advance): the **Frank-Wolfe gap** `<= 1e-4 * eps`, or 500 outer
    iterations. A fit that hits the cap is marked `NOT CONVERGED` and is
    **excluded** from epsilon selection and the verdict, but is still reported.

    The inner solve must converge (relative marginal error <= 1e-6) before use; on failure this raises and a
    partial plan is never used.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
from jaxtyping import Float, Int

from iot.core.problem import InverseOTProblem
from .dense_newton import solve_semidual_newton
from iot.core.theta import frank_wolfe_gap

MAX_OUTER_ITERATIONS: int = 500
FRANK_WOLFE_RTOL: float = 1e-4
INNER_MARGINAL_TOLERANCE: float = 1e-10
"""CN: 内层目标容差。**但要在求解器的地板上取 max**：实测 `newton.py` 在 rel eps=0.05、m=100 时
    无论容差与迭代上限都停在 3.83e-10（精度地板），所以硬性 1e-10 会让该格永远 `FAILED`。
    实际使用 `max(requested, INNER_FLOOR)`，并把实际达到的值记进结果。
EN: the requested inner tolerance. It must be raised to the solver's **floor**: measured, `newton.py`
    plateaus at 3.83e-10 at rel eps=0.05, m=100 regardless of tolerance or iteration cap, so a hard 1e-10
    would make that cell permanently `FAILED`. The effective tolerance is `max(requested, INNER_FLOOR)` and the
    achieved value is recorded."""

INNER_FLOOR: float = 1e-9
"""CN: 求解器地板（实测 3.83e-10，留 2.6 倍余量）。EN: the solver floor (measured 3.83e-10, 2.6x margin)."""


@dataclasses.dataclass(frozen=True)
class ConvergedFit:
    """
    CN: 一次拟合的读数。`theta` 只在 `converged` 为真时可信。
    EN: one fit reading. `theta` is trustworthy only when `converged`.
    """

    theta: Float[np.ndarray, "k"]
    loss: float
    status: str
    converged: bool
    frank_wolfe_gap: float
    projected_gradient: float
    outer_iterations: int
    function_evaluations: int
    wall_seconds: float
    epsilon: float
    cost_std: float
    inner_marginal_error: float
    at_bound_fraction: float


def project_simplex(value: Float[np.ndarray, "k"]) -> Float[np.ndarray, "k"]:
    """
    CN: 精确欧氏投影到单纯形（排序法）。只用于停止判据里算**投影梯度**，不用于迭代。
    EN: exact Euclidean projection onto the simplex (sorting). Used only to form the **projected gradient** for
        the stopping test, never in the iteration itself.
    """
    u = np.sort(np.asarray(value, dtype=np.float64))[::-1]
    css = np.cumsum(u) - 1.0
    index = np.arange(1, u.size + 1)
    rho = np.nonzero(u - css / index > 0.0)[0][-1]
    tau = css[rho] / (rho + 1.0)
    return np.maximum(np.asarray(value, dtype=np.float64) - tau, 0.0)


def _moment_setup(problem: InverseOTProblem) -> tuple[Int[np.ndarray, "edges"],
                                                      Int[np.ndarray, "edges"],
                                                      Float[np.ndarray, "edges k"],
                                                      Float[np.ndarray, "k"]]:
    """CN: 全网格索引、特征行与 `φ̄`。EN: full-grid indices, feature rows and `phi_bar`."""
    m, n = int(problem.n_source), int(problem.n_target)
    rows: Int[np.ndarray, "edges"] = np.repeat(np.arange(m), n)
    cols: Int[np.ndarray, "edges"] = np.tile(np.arange(n), m)
    features: Float[np.ndarray, "edges k"] = np.asarray(problem.features.features(rows, cols),
                                                        dtype=np.float64)
    s_rows, s_cols, s_values = problem.support()
    phi_bar: Float[np.ndarray, "k"] = (
        np.asarray(s_values, dtype=np.float64)[:, None]
        * np.asarray(problem.features.features(s_rows, s_cols), dtype=np.float64)).sum(axis=0)
    return rows, cols, features, phi_bar


def _absolute_epsilon(problem: InverseOTProblem, relative_epsilon: float,
                      reference_theta: Float[np.ndarray, "k"]) -> tuple[float, float]:
    """
    CN: `eps = rel * std(c(θ_ref))`，`θ_ref` 在**一次拟合内固定**（否则 `L` 不是固定的函数）。
    EN: `eps = rel * std(c(theta_ref))`, with `theta_ref` **fixed within a fit** (otherwise `L` is not a
        fixed function of theta).
    """
    m, n = int(problem.n_source), int(problem.n_target)
    rows: Int[np.ndarray, "edges"] = np.repeat(np.arange(m), n)
    cols: Int[np.ndarray, "edges"] = np.tile(np.arange(n), m)
    cost_ref = np.asarray(problem.features.costs(reference_theta, rows, cols),
                          dtype=np.float64).reshape(m, n)
    scale = _cost_std(cost_ref)
    return float(relative_epsilon) * scale, scale


def _inner_plan(problem: InverseOTProblem, cost: Float[np.ndarray, "m n"], epsilon: float,
                inner_tolerance: float, max_newton_iterations: int,
                warm: Float[np.ndarray, "n"] | None
                ) -> tuple[Float[np.ndarray, "m n"], float, Float[np.ndarray, "n"],
                           Float[np.ndarray, "m"], Float[np.ndarray, "n"]]:
    """
    CN: 一次内层求解。不收敛即抛（绝不用部分方案）。返回 `(plan, 边缘误差, g)`，`g` 供热启动。
    EN: one inner solve. Raises when it does not converge (a partial plan is never used). Returns
        `(plan, marginal_error, g)` where `g` warm-starts the next evaluation.
    """
    a: Float[np.ndarray, "m"] = np.asarray(problem.resolved_source_mass, dtype=np.float64).reshape(-1)
    b: Float[np.ndarray, "n"] = np.asarray(problem.resolved_target_mass, dtype=np.float64).reshape(-1)
    solved = solve_semidual_newton(cost, a, b, epsilon=float(epsilon), g_init=warm,
                                   marginal_tolerance=float(inner_tolerance),
                                   max_iterations=int(max_newton_iterations))
    if not solved.converged:
        raise RuntimeError(
            f"inner solve did not converge (marginal error {solved.marginal_error:.2e} > "
            f"{inner_tolerance:.1e}, range/eps={(float(cost.max()) - float(cost.min())) / epsilon:.1f}); "
            "refusing to use a partial plan")
    return (np.asarray(solved.plan, dtype=np.float64), float(solved.marginal_error),
            np.asarray(solved.potential_target, dtype=np.float64),
            np.asarray(solved.potential_source, dtype=np.float64),
            np.asarray(solved.potential_target, dtype=np.float64))


def fit_entropic_converged(
    problem: InverseOTProblem,
    *,
    relative_epsilon: float,
    max_outer_iterations: int = MAX_OUTER_ITERATIONS,
    frank_wolfe_rtol: float = FRANK_WOLFE_RTOL,
    inner_tolerance: float = INNER_MARGINAL_TOLERANCE,
    max_newton_iterations: int = 400,
    reference_theta: Float[np.ndarray, "k"] | None = None,
    start_theta: Float[np.ndarray, "k"] | None = None,
) -> ConvergedFit:
    """
    CN: 在 `Θ` 上最小化 `L_eps(θ) = <π̂,c(θ)> - W_eps(θ)`。`loss` 用**值** `W_eps`（包络定理只对值成立）。
    EN: minimise `L_eps(theta) = <pi_hat, c(theta)> - W_eps(theta)` over Theta. `loss` uses the **value**
        `W_eps` (the envelope theorem holds only for the value).
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    started = time.perf_counter()
    k = int(problem.n_features)
    rows, cols, features, phi_bar = _moment_setup(problem)
    theta_ref = (np.full(k, 1.0 / k, dtype=np.float64) if reference_theta is None
                 else np.asarray(reference_theta, dtype=np.float64).reshape(-1))
    epsilon, scale = _absolute_epsilon(problem, relative_epsilon, theta_ref)
    # CN: 内层容差 = max(请求值, 求解器地板)。地板由实测决定，见 `INNER_FLOOR`。
    # EN: inner tolerance = max(requested, solver floor); the floor is measured, see `INNER_FLOOR`.
    effective_inner_tolerance: float = max(float(inner_tolerance), float(INNER_FLOOR))

    constraints = problem.theta_set.linear_constraints()
    lower = np.asarray(constraints.lower, dtype=np.float64)
    upper = np.asarray(constraints.upper, dtype=np.float64)
    bounds = [(None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi))
              for lo, hi in zip(lower, upper)]
    a_eq = np.asarray(constraints.a_eq, dtype=np.float64)
    b_eq = np.asarray(constraints.b_eq, dtype=np.float64)

    source_mass: Float[np.ndarray, "m"] = np.asarray(problem.resolved_source_mass,
                                                     dtype=np.float64).reshape(-1)
    target_mass: Float[np.ndarray, "n"] = np.asarray(problem.resolved_target_mass,
                                                     dtype=np.float64).reshape(-1)
    state: dict[str, object] = {"evaluations": 0, "inner_errors": [], "warm": None}

    def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        cost = np.asarray(problem.features.costs(theta, rows, cols), dtype=np.float64).reshape(
            int(problem.n_source), int(problem.n_target))
        plan, inner_err, warm, dual_source, dual_target = _inner_plan(
            problem, cost, epsilon, effective_inner_tolerance, max_newton_iterations,
            state["warm"])  # type: ignore[arg-type]
        state["warm"] = warm
        state["evaluations"] = int(state["evaluations"]) + 1  # type: ignore[arg-type]
        inner_errors = state["inner_errors"]
        assert isinstance(inner_errors, list)
        inner_errors.append(inner_err)
        model_moment = plan.reshape(-1) @ features
        # Dual value W_eps = a.f + b.g. Equal to the primal <pi,c> - eps*H, but its error is SECOND
        # order in the marginal error (measured 5e-7 against 3e-6), which is what SLSQP compares.
        value_eps = float(source_mass @ dual_source + target_mass @ dual_target)
        return float(theta @ phi_bar) - value_eps, np.asarray(phi_bar - model_moment, dtype=np.float64)

    start = np.full(k, 1.0 / k, dtype=np.float64) if start_theta is None else \
        np.asarray(start_theta, dtype=np.float64).reshape(-1)
    result = minimize(loss_and_grad, start, jac=True, method="SLSQP", bounds=bounds,
                      constraints=[{"type": "eq", "fun": lambda t: float(a_eq @ t - b_eq),
                                    "jac": lambda t: a_eq}],
                      options={"maxiter": int(max_outer_iterations), "ftol": 1e-16})
    theta: Float[np.ndarray, "k"] = np.asarray(result.x, dtype=np.float64)
    _loss, grad = loss_and_grad(theta)
    gap = frank_wolfe_gap(problem, theta, grad)
    projected_gradient = float(np.abs(theta - project_simplex(theta - grad)).max())
    threshold = float(frank_wolfe_rtol) * epsilon
    converged = bool(gap <= threshold)
    inner_errors = state["inner_errors"]
    assert isinstance(inner_errors, list)
    return ConvergedFit(
        theta=theta, loss=float(result.fun),
        status="ok" if converged else f"NOT CONVERGED: {result.message}",
        converged=converged, frank_wolfe_gap=gap, projected_gradient=projected_gradient,
        outer_iterations=int(getattr(result, "nit", -1)),
        function_evaluations=int(state["evaluations"]),
        wall_seconds=float(time.perf_counter() - started), epsilon=epsilon, cost_std=scale,
        inner_marginal_error=float(max(inner_errors)) if inner_errors else float("nan"),
        at_bound_fraction=float(np.mean(theta <= 1e-12)))


def entropic_plan_at(problem: InverseOTProblem, theta: Float[np.ndarray, "k"], *,
                     relative_epsilon: float, reference_theta: Float[np.ndarray, "k"] | None = None,
                     inner_tolerance: float = INNER_MARGINAL_TOLERANCE) -> Float[np.ndarray, "m n"]:
    """
    CN: 在给定 θ 下取熵方案 `P_ε(θ)`（用于 S2 评分）。`eps` 用与训练**同一个**参考 θ 定尺度。
    EN: the entropic plan `P_eps(theta)` (for the S2 score). `eps` is scaled from the **same** reference theta
        as training.
    """
    k = int(problem.n_features)
    theta_ref = (np.full(k, 1.0 / k, dtype=np.float64) if reference_theta is None
                 else np.asarray(reference_theta, dtype=np.float64).reshape(-1))
    epsilon, _scale = _absolute_epsilon(problem, relative_epsilon, theta_ref)
    rows, cols, _features, _phi_bar = _moment_setup(problem)
    cost = np.asarray(problem.features.costs(theta, rows, cols), dtype=np.float64).reshape(
        int(problem.n_source), int(problem.n_target))
    plan, _err, _warm = _inner_plan(problem, cost, epsilon, inner_tolerance, 4000, None)
    return plan


__all__ = ["ConvergedFit", "INNER_FLOOR", "entropic_plan_at", "fit_entropic_converged",
           "frank_wolfe_gap", "project_simplex"]
