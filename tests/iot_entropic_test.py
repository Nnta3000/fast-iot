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
