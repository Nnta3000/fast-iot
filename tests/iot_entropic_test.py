"""
CN: **熵 OT 族的测试** —— 半对偶牛顿求解器。全部是纯数值，不读数据。

    CN: 关键约定（三次搞错之后才由实测确定）：
        - `f` 闭式：`f_i = eps*(log a_i - LSE_j((g_j - c_ij)/eps))`，因此行和**恰好**是 `a_i`；
        - `Hess D = -(1/eps)*(diag(colsum) - Pi^T diag(1/a) Pi)`，`1/eps` 不能漏；
        - `d grad / d g = -Hess D`，符号由有限差分确认。

EN: **Tests for the entropic family** -- the semi-dual Newton solver. Purely numerical; reads no data.

    The conventions that took three attempts to settle:
      - `f` in closed form (`f_i = eps*(log a_i - LSE_j(...))`), so rows sum to **exactly** `a_i`;
      - `Hess D = -(1/eps)*(diag(colsum) - Pi^T diag(1/a) Pi)`; the `1/eps` must not be dropped;
      - `d grad / d g = -Hess D`, sign confirmed by finite differences.
"""

from __future__ import annotations

import numpy as np
import pytest
from jaxtyping import Float

from iot.entropic import sinkhorn_plan_torch
from iot.entropic.newton import _evaluate, _hessian, solve_semidual_newton


def _tiny(m: int = 20, n: int = 20, seed: int = 0) -> tuple[Float[np.ndarray, "m n"],
                                                           Float[np.ndarray, "m"],
                                                           Float[np.ndarray, "n"]]:
    rng = np.random.default_rng(seed)
    cost = np.ascontiguousarray(rng.normal(size=(m, n)) * 1e-2)
    return cost, np.full(m, 1.0 / m), np.full(n, 1.0 / n)


@pytest.mark.parametrize("rel_eps", [0.5, 0.05])
def test_newton_hessian_matches_finite_differences(rel_eps: float) -> None:
    """
    CN: Hessian 的有限差分检查，**含 `1/eps`**。若漏掉该因子，相对误差会到 1e2。
    EN: finite-difference check of the Hessian, **including `1/eps`**. Dropping the factor puts the
        relative error at 1e2.
    """
    cost, a, b = _tiny()
    eps = rel_eps * float(cost.std())
    log_a = np.log(a)
    g = np.zeros(int(b.size))
    _f, pi, colsum, _value, _grad = _evaluate(cost, log_a, b, eps, g)
    core = _hessian(pi, colsum, a, eps)          # = -Hess D
    step = 1e-6
    numeric = np.zeros((int(b.size), int(b.size)))
    for k in range(int(b.size)):
        delta = np.zeros_like(g)
        delta[k] = step
        _f2, _p2, _c2, _v2, plus = _evaluate(cost, log_a, b, eps, g + delta)
        _f3, _p3, _c3, _v3, minus = _evaluate(cost, log_a, b, eps, g - delta)
        numeric[:, k] = (plus - minus) / (2.0 * step)
    # d grad / d g = -Hess D = -core
    assert np.abs(numeric + core).max() / np.abs(core).max() <= 1e-4
    # and the 1/eps factor is load-bearing
    assert np.abs(numeric + core * eps).max() / max(np.abs(core * eps).max(), 1e-300) > 1.0


def test_newton_matches_sinkhorn_where_sinkhorn_converges() -> None:
    """
    CN: 在 Sinkhorn 能收敛的 `eps` 上，两者方案一致到 ~1e-9。
    EN: where Sinkhorn converges, the two plans agree to ~1e-9.
    """
    cost, a, b = _tiny(m=30, n=30)
    for rel_eps in (1.0, 0.5):
        eps = rel_eps * float(cost.std())
        mine = solve_semidual_newton(cost, a, b, epsilon=eps)
        reference = sinkhorn_plan_torch(cost, a, b, relative_epsilon=rel_eps,
                                        max_iterations=200_000, tolerance=1e-12, device="cpu")
        assert float(np.abs(mine.plan - reference.plan).max()) <= 1e-8


@pytest.mark.parametrize("rel_eps", [0.25, 0.1, 0.05])
def test_newton_reaches_protocol_accuracy(rel_eps: float) -> None:
    """
    CN: 协议要求边缘误差 <= 1e-6（相对，逐边缘项的最大值）；L-BFGS 只到 ~5e-6，牛顿到 1e-8 以下。
    EN: the protocol requires a marginal error <= 1e-6 (relative, max per marginal entry); L-BFGS stalls
        at ~5e-6, Newton reaches below 1e-8.
    """
    cost, a, b = _tiny(m=60, n=60)
    result = solve_semidual_newton(cost, a, b, epsilon=rel_eps * float(cost.std()), max_iterations=400)
    assert result.marginal_error <= 1e-6
    assert result.converged
    row = np.abs((result.plan.sum(1) - a) / a).max()
    assert float(row) <= 1e-10          # rows are exact by construction


def test_newton_step_is_trust_region_limited() -> None:
    """
    CN: 小 eps 下牛顿步必须受限；不设限时实测列和误差从 4.5 冲到 59。
    EN: the Newton step must be bounded at small epsilon; unlimited, the column error was measured going
        from 4.5 to 59.
    """
    cost, a, b = _tiny(m=60, n=60)
    result = solve_semidual_newton(cost, a, b, epsilon=0.05 * float(cost.std()), max_iterations=400)
    assert result.marginal_error <= 1e-6


def test_gradient_uses_the_entropic_value_not_the_plan_cost() -> None:
    """
    CN: **Step 0 检查 (b)**：`grad_theta L_eps` 的有限差分检查（目标 1e-4）。

        关键：`L_eps = <C,pi_hat> - W_eps`，其中 `W_eps = <C,P_eps> - eps*H(P_eps)`。
        包络定理只对**值** `W_eps` 成立。用**方案代价** `<C,P_eps>` 会漏掉 `eps*dH/dtheta`；
        实测方向导数相对误差 2.4%（对照：用 `W_eps` 时 1e-10）。
    EN: **Step 0 check (b)**: finite-difference check of `grad_theta L_eps` (target 1e-4).

        The point: `L_eps = <C,pi_hat> - W_eps` with `W_eps = <C,P_eps> - eps*H(P_eps)`. The envelope
        theorem holds only for the **value** `W_eps`. Using the **plan cost** `<C,P_eps>` omits
        `eps*dH/dtheta`; measured, the directional derivative was off by 2.4% (against 1e-10 with `W_eps`).
    """
    import ot  # noqa: PLC0415
    import scipy.sparse as sparse  # noqa: PLC0415

    from iot import DiagonalWeightFeatures  # noqa: PLC0415
    from iot.core.problem import InverseOTProblem  # noqa: PLC0415
    from iot.core.theta import SimplexTheta  # noqa: PLC0415
    from iot.entropic.newton import solve_semidual_newton  # noqa: PLC0415
    from iot.entropic.sinkhorn import _cost_std  # noqa: PLC0415

    rng = np.random.default_rng(0)
    m = n = 20
    d = 3
    x = rng.normal(size=(m, d))
    y = rng.normal(size=(n, d))
    features = DiagonalWeightFeatures(x, y)
    rows = np.repeat(np.arange(m), n)
    cols = np.tile(np.arange(n), m)
    theta_star = np.abs(rng.normal(size=d))
    theta_star = theta_star / theta_star.sum()
    cost = np.asarray(features.costs(theta_star, rows, cols)).reshape(m, n)
    plan = ot.emd(np.full(m, 1 / m), np.full(n, 1 / n), cost, numItermax=10**7)
    problem = InverseOTProblem(features=features, theta_set=SimplexTheta(dimension=d),
                               plan=sparse.csr_matrix(plan))
    a = np.full(m, 1.0 / m)
    phi_hat = (problem.support_values[:, None] * features.features(*problem.support()[:2])).sum(axis=0)
    phi = np.asarray(features.features(rows, cols))
    theta = np.array([0.35, 0.30, 0.35])
    eps = 0.1 * _cost_std(np.asarray(features.costs(theta, rows, cols)).reshape(m, n))

    def value_and_grad(at: np.ndarray) -> tuple[float, np.ndarray, float]:
        cost_now = np.asarray(features.costs(at, rows, cols)).reshape(m, n)
        solved = solve_semidual_newton(cost_now, a, a, epsilon=eps, marginal_tolerance=1e-13)
        assert solved.converged, solved.marginal_error
        p = solved.plan
        entropy = float(-np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0)))
        value_eps = float(np.sum(p * cost_now)) - eps * entropy
        return float(at @ phi_hat) - value_eps, phi_hat - (p.reshape(-1) @ phi), solved.marginal_error

    h = 1e-6
    for direction in (np.array([1.0, 0.0, 0.0]), np.array([1.0, -1.0, 0.0])):
        _l0, grad, inner = value_and_grad(theta)
        assert inner <= 1e-10, "inner solve too loose for the quotient"
        numeric = (value_and_grad(theta + h * direction)[0]
                   - value_and_grad(theta - h * direction)[0]) / (2.0 * h)
        analytic = float(grad @ direction)
        assert abs(numeric - analytic) / max(abs(analytic), 1e-300) <= 1e-4
