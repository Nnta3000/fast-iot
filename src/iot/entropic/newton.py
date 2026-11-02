"""
CN: **半对偶的稠密牛顿求解器** —— 只在 `g` 上优化，`f` 闭式消去，用**精确 Hessian**。

    CN: 为什么需要它：L-BFGS 在列和上停在 ~5e-6（行和已是 1e-14，因为 `f` 是闭式的），
        这达不到协议要求的 1e-6。牛顿在这类问题上二次收敛。

    CN: 目标与导数（`pi_ij = a_i p_ij`，`p` 是对 `j` 的 softmax）：
            D(g)          = a·f(g) + b·g + const
            grad D_j      = b_j - sum_i pi_ij
            Hess D        = -(1/eps) * ( diag(colsum) - Pi^T diag(1/a) Pi )
        **`1/eps` 因子不能漏**：漏了牛顿步会短 `eps` 倍（在 `eps ~ 1e-3` 时等于爬行），
        而回溯只会缩短步长、永远不会纠正它。

    CN: 三处必须处理：
        1. **规范**：`1` 精确落在 Hessian 零空间（`Pi^T diag(1/a) Pi 1 = colsum`），
           所以固定一个坐标，或对特征值设阈值。`n = 100..600` 时两者都便宜。
        2. **近零特征值**：小 `eps` 下方案近分块，这些方向病态；梯度对其不敏感，因此不需要走远。
           截断相对阈值以下的特征值（或 Levenberg 项 `mu*I`，随梯度收缩）。
        3. **对称化**：float64 的舍入会让 Hessian 略不定，因子化前先对称化。

    CN: **接受判据用梯度范数**（merit function），不是目标值：目标值的线搜索在最优点附近
        分辨不出下降量，这正是 L-BFGS 失败的原因。范数已经很小（低于阈值）时直接取满牛顿步。
        停止判据用**边缘误差本身**。

EN: **Dense Newton solver for the semi-dual** -- optimise over `g` only, `f` eliminated, with the
    **exact Hessian**.

    Why: L-BFGS stalls at ~5e-6 on the column sums (rows are already 1e-14 because `f` is closed-form),
    which misses the protocol's 1e-6. Newton converges quadratically here.

    Objective and derivatives (`pi_ij = a_i p_ij`, `p` the softmax over `j`):
        D(g)     = a.f(g) + b.g + const
        grad D_j = b_j - sum_i pi_ij
        Hess D   = -(1/eps) * ( diag(colsum) - Pi^T diag(1/a) Pi )
    **The `1/eps` factor must not be dropped**: without it the Newton step is `eps` times too short
    (crawling at `eps ~ 1e-3`), and backtracking only shortens steps so it never corrects.

    Three things must be handled: the **gauge** (`1` is exactly in the Hessian's null space, since
    `Pi^T diag(1/a) Pi 1 = colsum`), **near-null directions** (at small eta the plan nearly splits into
    blocks; the gradient is insensitive to them so no distant move is needed -- truncate small eigenvalues
    or add a Levenberg term), and **symmetrisation** before factorising.

    The **acceptance test uses the gradient norm** (a merit function), not the objective: an objective
    line search cannot resolve decreases near the optimum, which is exactly why L-BFGS failed. When the norm
    is already below a threshold, take the full Newton step. **Stop on the marginal error itself.**
"""

from __future__ import annotations

import dataclasses

import numpy as np
from jaxtyping import Float


@dataclasses.dataclass(frozen=True)
class NewtonResult:
    """CN: 一次牛顿求解的读数。EN: a Newton solve reading."""

    potential_source: Float[np.ndarray, "m"]
    potential_target: Float[np.ndarray, "n"]
    plan: Float[np.ndarray, "m n"]
    marginal_error: float
    converged: bool
    iterations: int
    epsilon: float
    gauge_pinned: bool


def _row_lse(value: Float[np.ndarray, "m n"]) -> Float[np.ndarray, "m"]:
    """CN: 逐行 `logsumexp`（稳定）。EN: row-wise `logsumexp` (stable)."""
    peak = value.max(axis=1, keepdims=True)
    return np.asarray(peak.reshape(-1) + np.log(np.exp(value - peak).sum(axis=1)), dtype=np.float64)


def _evaluate(cost: Float[np.ndarray, "m n"], log_a: Float[np.ndarray, "m"],
              b: Float[np.ndarray, "n"], eps: float,
              g: Float[np.ndarray, "n"]) -> tuple[Float[np.ndarray, "m"],
                                                  Float[np.ndarray, "m n"],
                                                  Float[np.ndarray, "n"],
                                                  float, Float[np.ndarray, "n"]]:
    """
    CN: 由 `g` 算出 `(f, pi, colsum, D, grad)`。`f` 闭式，行和恰为 `a`。
    EN: from `g`, return `(f, pi, colsum, D, grad)`. `f` is closed form, so rows sum to `a` exactly.
    """
    inv = 1.0 / eps
    base = (g[None, :] - cost) * inv
    row = _row_lse(base)
    f = eps * (log_a - row)
    log_pi = (f[:, None] + g[None, :] - cost) * inv
    pi = np.exp(log_pi)
    colsum = pi.sum(axis=0)
    value = float(f @ np.exp(log_a) + b @ g)
    grad = b - colsum
    return f, pi, colsum, value, grad


def _hessian(pi: Float[np.ndarray, "m n"], colsum: Float[np.ndarray, "n"],
             a: Float[np.ndarray, "m"], eps: float) -> Float[np.ndarray, "n n"]:
    """
    CN: 返回 **`-Hess D`** 的 PSD 核，即 `(1/eps)*(diag(colsum) - Pi^T diag(1/a) Pi)`，**含 `1/eps`**。
        **符号已由有限差分确认**：`d grad / d g = -Hess D = +core`（实测比值 -1 是相对未取负的
        `Hess D` 而言）。`1/eps` 也由有限差分确认（去掉它相对误差会变成 2e2）。
    EN: returns the PSD core of **`-Hess D`**, i.e. `(1/eps)*(diag(colsum) - Pi^T diag(1/a) Pi)`,
        **with the `1/eps`**. **The sign is confirmed by finite differences**: `d grad / d g = -Hess D
        = +core` (the measured ratio -1 was against `Hess D` without the negation). The `1/eps` is also
        confirmed by finite differences (dropping it puts the relative error at 2e2).
    """
    inv_a = 1.0 / np.asarray(a, dtype=np.float64)
    core = np.diag(np.asarray(colsum, dtype=np.float64)) - pi.T @ (pi * inv_a[:, None])
    # CN: 因子化前的对称化（float64 舍入会让它略不定）。
    # EN: symmetrise before factorising (float64 rounding can make it slightly indefinite).
    core = 0.5 * (core + core.T)
    return core / float(eps)


def solve_semidual_newton(
    cost: Float[np.ndarray, "m n"],
    source_mass: Float[np.ndarray, "m"],
    target_mass: Float[np.ndarray, "n"],
    *,
    epsilon: float,
    g_init: Float[np.ndarray, "n"] | None = None,
    max_iterations: int = 60,
    marginal_tolerance: float = 1e-10,
    eigenvalue_rtol: float = 1e-10,
) -> NewtonResult:
    """
    CN: 阻尼牛顿，用**梯度范数**作 merit function，用**边缘误差**作停止判据。
        规范：固定 `g[0]`（把梯度与 Hessian 的第 0 行/列去掉，解完再回填 0）。
    EN: damped Newton with the **gradient norm** as the merit function and the **marginal error** as the
        stopping test. The gauge is fixed by pinning `g[0]` (drop its row/column, then leave it at 0).
    """
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
    g = np.zeros(n, dtype=np.float64) if g_init is None else \
        np.asarray(g_init, dtype=np.float64).reshape(-1).copy()

    f, pi, colsum, _value, grad = _evaluate(cost, log_a, b, eps, g)
    err = float(max(np.abs((pi.sum(1) - a) / a).max(), np.abs((colsum - b) / b).max()))
    if err <= float(marginal_tolerance):
        return NewtonResult(f, g, pi, err, True, 0, eps, True)

    pinned = True
    newton_steps = 0
    for _iteration in range(1, int(max_iterations) + 1):
        # CN: 固定 g[0]：去掉第 0 个坐标。`1` 在零空间里，去掉一个坐标即可消除奇异。
        # EN: pin g[0] by dropping coordinate 0; `1` spans the null space, so this removes the singularity.
        gi = grad[1:] if pinned else grad
        core = _hessian(pi, colsum, a, eps)[1:, 1:] if pinned else _hessian(pi, colsum, a, eps)
        # CN: 谱截断 —— 截掉相对阈值以下的方向（梯度对其不敏感，不需要走远）。
        # EN: spectral truncation -- drop directions below a relative threshold (the gradient is insensitive
        #     to them, so no distant move is needed).
        eigenvalues, eigenvectors = np.linalg.eigh(core)
        floor = float(eigenvalue_rtol) * max(float(eigenvalues.max()), 1e-300)
        keep = eigenvalues > floor
        if not np.any(keep):
            break
        # CN: `Hess D = -core`，因此牛顿方向是 `+/core^{-1} grad`（不是负号）。有限差分已确认。
        # EN: `Hess D = -core`, so the Newton direction is `+core^{-1} grad` (not negative). Confirmed by
        #     finite differences.
        # CN: 用矩阵乘而不是布尔掩码赋值（掩码在秩亏时会形状不匹配）。
        # EN: use a matrix product rather than boolean-mask assignment (the mask mismatches when rank-deficient).
        effective_inverse = eigenvectors @ np.diag(np.where(keep, 1.0 / np.where(keep, eigenvalues, 1.0), 0.0)) \
            @ eigenvectors.T
        step = effective_inverse @ gi
        g_direction = np.concatenate([[0.0], step]) if pinned else step
        merit = float(np.linalg.norm(grad))
        # CN: 接受判据是**梯度范数**下降，不是目标值下降；范数已很小则取满牛顿步。
        # EN: accept on a decrease in the **gradient norm**, not the objective; take the full step when the
        #     norm is already small.
        scale = 1.0
        accepted = False
        for _backtrack in range(40):
            trial = g + scale * g_direction
            f_t, pi_t, col_t, _v, grad_t = _evaluate(cost, log_a, b, eps, trial)
            err_t = float(max(np.abs((pi_t.sum(1) - a) / a).max(), np.abs((col_t - b) / b).max()))
            if err_t <= float(marginal_tolerance):
                g, f, pi, colsum, grad = trial, f_t, pi_t, col_t, grad_t
                err = err_t
                accepted = True
                newton_steps += 1
                break
            if float(np.linalg.norm(grad_t)) < merit or merit < 1e-6:
                g, f, pi, colsum, grad = trial, f_t, pi_t, col_t, grad_t
                err = err_t
                accepted = True
                newton_steps += 1
                break
            scale *= 0.5
        if not accepted:
            break
        if err <= float(marginal_tolerance):
            return NewtonResult(f, g, pi, err, True, newton_steps, eps, pinned)

    return NewtonResult(f, g, pi, err, err <= float(marginal_tolerance), newton_steps, eps, pinned)


__all__ = ["NewtonResult", "solve_semidual_newton"]
