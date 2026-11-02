"""
CN: **半对偶求解器** —— 只在 `g` 上优化，`f` 用闭式消去。

    CN: 为什么要换掉交替 Sinkhorn：最坏情况下它需要约 `(代价幅度)/eps` 次扫描，所以 `eps` 小的时候
        达不到收敛；而对 `(f, g)` 联合做 L-BFGS 在 `eps` 很小时会在 `exp(shift/eps)` 上溢出。

    CN: 半对偶形式：给定 `g`，`f` 的最优值有闭式解

            f_i(g) = eps * log(a_i) - eps * LSE_j((g_j - c_ij) / eps)

        于是 `pi_ij = exp((f_i + g_j - c_ij)/eps)` 的行和**恰好**是 `a_i`，而且**每一个指数都在
        `logsumexp` 里面**，因此不可能溢出。剩余的唯一变量是 `g`，目标是把列和拉回 `b_j`：

            G(g) = sum_j b_j * (LSE_i(log a_i + (g_j - c_ij)/eps) - g_j/eps)

        （这是对偶的负值再缩放；最小化它等价于求 Sinkhorn 的不动点。）

    CN: **注意约定**：`f` 里的 `eps*log(a_i)` 是参考测度形式；它与计数测度的势只差这一项，
        而 `plan == exp(sigma/eps)` 在实测中精确成立（见 `sinkhorn.py` 的记录）。

EN: **Semi-dual solver** -- optimise over `g` only, eliminating `f` in closed form.

    Why replace alternating Sinkhorn: in the worst case it needs about (cost range)/epsilon sweeps, so it
    cannot converge at small epsilon; and joint L-BFGS on `(f, g)` overflows in `exp(shift/eps)` when epsilon
    is small.

    Semi-dual form: for given `g` the optimal `f` is available in closed form,

            f_i(g) = eps * log(a_i) - eps * LSE_j((g_j - c_ij) / eps)

    so `pi_ij = exp((f_i + g_j - c_ij)/eps)` has row sums **exactly** `a_i`, and **every exponential is
    inside a `logsumexp`**, so overflow is impossible. The only remaining variable is `g`; the objective
    drives the column sums back to `b_j`:

            G(g) = sum_j b_j * (LSE_i(log a_i + (g_j - c_ij)/eps) - g_j/eps)

    (the dual, negated and rescaled; minimising it is the same as solving the Sinkhorn fixed point.)

    **Convention note**: the `eps*log(a_i)` in `f` is the reference-measure form; it differs from the
    counting-measure potentials by exactly that term, and `plan == exp(sigma/eps)` holds exactly in
    measurement (see `sinkhorn.py`).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from jaxtyping import Float


@dataclasses.dataclass(frozen=True)
class SemiDualResult:
    """CN: 半对偶求解的读数。EN: a semi-dual solve reading."""

    potential_source: Float[np.ndarray, "m"]
    potential_target: Float[np.ndarray, "n"]
    plan: Float[np.ndarray, "m n"]
    marginal_error: float
    converged: bool
    iterations: int
    epsilon: float


def _lse_rows(value: Float[np.ndarray, "m n"]) -> Float[np.ndarray, "m"]:
    """CN: 逐行 `logsumexp`（稳定）。EN: row-wise `logsumexp` (stable)."""
    peak = value.max(axis=1, keepdims=True)
    return np.asarray(peak.reshape(-1) + np.log(np.exp(value - peak).sum(axis=1)), dtype=np.float64)


def solve_semidual(
    cost: Float[np.ndarray, "m n"],
    source_mass: Float[np.ndarray, "m"],
    target_mass: Float[np.ndarray, "n"],
    *,
    epsilon: float,
    g_init: Float[np.ndarray, "n"] | None = None,
    max_iterations: int = 4000,
    tolerance: float = 1e-10,
) -> SemiDualResult:
    """
    CN: 用 L-BFGS 在 `g` 上最小化 `G(g)`，`f` 每步闭式重算。返回收敛的 `(f, g, pi)`。
        不收敛时 `converged=False`，调用方必须自行决定（本项目的约定是**不使用未收敛的 plan**）。

    EN: L-BFGS on `g` minimising `G(g)`, with `f` recomputed in closed form at every evaluation. Returns
        the converged `(f, g, pi)`. On non-convergence `converged=False`; the caller must decide (this
        project's convention is: **never use a non-converged plan**).
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    m, n = int(cost.shape[0]), int(cost.shape[1])
    a = np.asarray(source_mass, dtype=np.float64).reshape(-1)
    b = np.asarray(target_mass, dtype=np.float64).reshape(-1)
    if a.size != m or b.size != n:
        raise ValueError(f"masses ({a.size}, {b.size}) do not match cost shape {(m, n)}")
    if float(epsilon) <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon!r}")
    if not np.all(np.isfinite(cost)):
        raise ValueError("cost contains non-finite entries")
    eps = float(epsilon)
    log_a = np.log(a)
    inv = 1.0 / eps

    def dual(g: np.ndarray) -> tuple[float, np.ndarray]:
        # CN: `log pi_ij = log a_i + (g_j - c_ij)/eps - LSE_i(...)`；行和恒为 a_i。
        #     对 g_j 的导数： `-b_j + sum_i pi_ij`。
        # EN: `log pi_ij = log a_i + (g_j - c_ij)/eps - LSE_i(...)`; row sums are exactly a_i. The derivative
        #     with respect to g_j is `-b_j + sum_i pi_ij`.
        # CN: `row_lse` 是 **j 方向** 的 LSE，不含 `log a`：
        #     `f_i = eps*(log a_i - LSE_j((g_j - C_ij)/eps))`，于是行和恰为 `a_i`。
        #     把 `log a` 放进 LSE 里会重复计数（实测行和变成 1 而不是 1/m）。
        # EN: `row_lse` is the LSE over **j**, without `log a`:
        #     `f_i = eps*(log a_i - LSE_j((g_j - C_ij)/eps))`, which makes the row sums exactly `a_i`.
        #     Putting `log a` inside the LSE double-counts it (measured: row sums became 1, not 1/m).
        base = (g[None, :] - cost) * inv
        row_lse = _lse_rows(base)                          # (m,)
        f = eps * (log_a - row_lse)
        log_pi = (f[:, None] + g[None, :] - cost) * inv    # log of pi
        col_sum = np.exp(log_pi).sum(axis=0)               # sum_i pi_ij  (exp of a stable log)
        # CN: 半对偶目标是 `max_{g} a.f(g) + b.g`，即 `min -(a.f + b.g)`。
        #     **不能**用 `-a.f - b.g + eps*sum(pi)`：因为 `f` 已被选成使行和恰好为 `a_i`，
        #     `sum(pi) = sum(a) = 1` 是常数，那一项对 `g` 没有任何曲率，优化器会立刻停住。
        # EN: the semi-dual objective is `max_g a.f(g) + b.g`, i.e. `min -(a.f + b.g)`.
        #     **Not** `-a.f - b.g + eps*sum(pi)`: since `f` is chosen to match the rows exactly,
        #     `sum(pi) = sum(a) = 1` is a constant, so that term contributes no curvature in `g` and the
        #     optimiser stops immediately.
        # CN: 最小化 `-a.f - b.g + eps*sum(pi)`；`f` 取闭式后 `sum(pi) = 1` 是常数，
        #     曲率来自 `-a.f`（`f` 随 `g` 变化）。
        # EN: minimise `-a.f - b.g + eps*sum(pi)`; with the closed-form `f`, `sum(pi) = 1` is a constant and
        #     the curvature comes from `-a.f` (which moves with `g`).
        value = float(-(a @ f) - (b @ g) + eps * float(np.exp(log_pi).sum()))
        grad = -b + col_sum
        return value, np.asarray(grad, dtype=np.float64)

    start = np.zeros(n, dtype=np.float64) if g_init is None else \
        np.asarray(g_init, dtype=np.float64).reshape(-1).copy()
    result = minimize(dual, start, jac=True, method="L-BFGS-B",
                      # CN: `ftol` 太紧会让线搜索立刻失败（实测 1e-18 时 1 次迭代就退出）。
                      #     `gtol` 与边缘质量容差挂钩：梯度是 `-b_j + sum_i pi_ij`，直接度量列和误差。
                      # EN: an over-tight `ftol` makes the line search fail immediately (measured: 1
                      #     iteration at 1e-18). `gtol` is tied to the marginal tolerance: the gradient is
                      #     `-b_j + sum_i pi_ij`, i.e. the column-sum error itself.
                      options={"maxiter": int(max_iterations), "maxcor": 20,
                               "ftol": 1e-15, "gtol": float(tolerance)})
    g = np.asarray(result.x, dtype=np.float64)
    base = (g[None, :] - cost) * inv
    row_lse = _lse_rows(base)
    f = eps * (log_a - row_lse)
    plan = np.exp((f[:, None] + g[None, :] - cost) * inv)
    err = float(max(np.abs((plan.sum(1) - a) / a).max(), np.abs((plan.sum(0) - b) / b).max()))
    return SemiDualResult(potential_source=f, potential_target=g, plan=plan, marginal_error=err,
                          converged=err <= 1e-6, iterations=int(result.nit), epsilon=eps)


__all__ = ["SemiDualResult", "solve_semidual"]
