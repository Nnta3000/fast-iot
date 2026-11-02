"""
CN: **θ 空间的熵 OT（逆 OT 的第三个目标函数族）**。

    CN: 与精确/松弛循环的区别，说清楚：
        - **没有 master、没有 δ、没有证书**。给定 θ 由 Sinkhorn 得到 `P_ε(θ)`，
          外层对 θ 做投影梯度，最小化 `L_ε(θ) = <C_θ, π̂> - <C_θ, P_ε(θ)>`。
          由包络定理 `∇L_ε(θ)_k = <Φ_k, π̂ - P_ε(θ)>`，即"观测特征矩 − 模型特征矩"。
        - 因此它**不会**证明 π̂ 最优。它给的是"在熵正则化模型下最可能的 θ"。
        - 它的价值在于**泛化**（软、稠密、条件数好的方案），不是精确性——它存在的意义正是
          精确 master 不可行的那些实例。

    CN: ε 用**相对量** `relative_epsilon`，内部乘以 `std(C_θ)`（见 `sinkhorn.sinkhorn_plan_torch`）。

EN: **Entropic OT in theta space** (the third objective family of inverse OT).

    How it differs from the exact/relaxed loops, stated plainly:
      - **no master, no delta, no certificate**. Given theta, Sinkhorn gives `P_eps(theta)`; an outer loop
        does projected gradient on theta, minimising `L_eps(theta) = <C_theta, pi_hat> - <C_theta, P_eps(theta)>`.
        By the envelope theorem `grad L_eps(theta)_k = <Phi_k, pi_hat - P_eps(theta)>`, the observed minus the
        model feature moment.
      - So it does **not** prove pi_hat optimal. It returns "the most likely theta under a regularised model".
      - Its value is **generalisation** (soft, dense, well-conditioned plans), not exactness -- it exists
        precisely for the instances where the exact master is infeasible.

    Epsilon is **relative**: `relative_epsilon`, multiplied internally by `std(C_theta)`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from jaxtyping import Float, Int

from ..core.problem import InverseOTProblem
from .newton import solve_semidual_newton
from .sinkhorn import SinkhornResult, _cost_std, sinkhorn_plan_torch


@dataclasses.dataclass(frozen=True)
class EntropicStep:
    """CN: 一步外层迭代读数。EN: one outer iteration's reading."""

    index: int
    regularization: float
    loss: float
    gradient_norm: float
    sinkhorn_iterations: int
    marginal_error: float


@dataclasses.dataclass(frozen=True)
class EntropicResult:
    """
    CN: 熵 OT 拟合的结果。**没有 delta、没有区间**：这不是证书。`stall_reason` 说明为什么停下。
    EN: entropic-fit result. **No delta, no bracket**: this is not a certificate. `stall_reason` says why it
        stopped.
    """

    theta: Float[np.ndarray, "k"]
    loss: float
    steps: int
    converged: bool
    stall_reason: str
    cost_std: float
    steps_record: tuple[EntropicStep, ...]


def fit_entropic(
    problem: InverseOTProblem,
    *,
    relative_epsilon: float = 0.25,
    learning_rate: float = 1.0,
    max_steps: int = 200,
    sinkhorn_iterations: int = 5000,
    sinkhorn_tolerance: float = 1e-9,
    gradient_tolerance: float = 1e-8,
    device: str = "auto",
    source_mass: Float[np.ndarray, "m"] | None = None,
    target_mass: Float[np.ndarray, "n"] | None = None,
    inner_solver: str = "sinkhorn",
    inner_tolerance: float = 1e-12,
) -> EntropicResult:
    """
    CN: 在 Θ 上最小化 `L_ε(θ)`。梯度是**观测特征矩 − 模型特征矩**；步长固定，
        每步之后投影回 Θ。

        停止条件：`‖∇L‖_∞ <= gradient_tolerance`（收敛）；或步数用尽（`stall_reason="max_steps"`）。

    CN: `source_mass`/`target_mass` 默认取方案的边缘；显式给出可覆盖（例如池的子集）。

    EN: minimise `L_eps(theta)` over Theta. The gradient is **observed feature moment minus model feature
        moment**; the step is fixed and theta is projected back onto Theta after each step.

        Stops when `||grad L||_inf <= gradient_tolerance` (converged) or on exhausting the step budget
        (`stall_reason="max_steps"`).

        `source_mass`/`target_mass` default to the plan's marginals and may be overridden.

    CN: `inner_solver`：
        - `"sinkhorn"`（默认）：torch 上的 log-domain 交替 Sinkhorn，**行为与之前完全一致**；
        - `"newton"`：半对偶稠密牛顿（`entropic/newton.py`）。`eps` 小的时候交替 Sinkhorn
          达不到精度（实测 `eps=0.05` 停在 5e-6），牛顿到 1e-8 以下。
        两种都**只**在收敛时使用方案（本项目的约定：绝不用未收敛的方案）。

    EN: `inner_solver`:
        - `"sinkhorn"` (default): torch log-domain alternating Sinkhorn, **behaviour identical to before**;
        - `"newton"`: the semi-dual dense Newton (`entropic/newton.py`). Alternating Sinkhorn misses the
          accuracy at small epsilon (measured: stalls at 5e-6 at `eps=0.05`); Newton reaches below 1e-8.
        Both use the plan **only** when converged (this project's rule: never use a non-converged plan).
    """
    from ..core.theta import SimplexTheta  # noqa: PLC0415 - local to keep the import graph shallow

    if not isinstance(problem.theta_set, SimplexTheta):
        raise ValueError("fit_entropic currently supports SimplexTheta only")
    # CN: 从**均匀 θ 起步**，而不是随机采样。均匀 θ 就是"不学任何代价"（cosine）的那个点，
    #     因此拟合只可能从它出发改进；随机 Dirichlet 起点会落在远离均匀处，梯度会把它推向一个
    #     集中的顶点——实测池上命中率因此从 0.5825 掉到 0.4325（cosine 是 0.5675）。
    # EN: start from **uniform theta**, not a random sample. Uniform theta is the "learn no cost yet"
    #     (cosine) point, so the fit can only improve from there; a random Dirichlet start sits far from
    #     uniform and the gradient drives it to a concentrated vertex -- measured pool hit fell from 0.5825
    #     to 0.4325 (cosine is 0.5675) purely from the starting point.
    k = int(problem.n_features)
    theta: Float[np.ndarray, "k"] = np.full(k, 1.0 / float(k), dtype=np.float64)
    a = problem.resolved_source_mass if source_mass is None else np.asarray(source_mass, dtype=np.float64)
    b = problem.resolved_target_mass if target_mass is None else np.asarray(target_mass, dtype=np.float64)
    m, n = int(problem.n_source), int(problem.n_target)
    # CN: 扁平的边索引，长度 `m·n`（不是二维）。轴名必须与形状一致，否则 beartype 会拒绝。
    # EN: flat edge indices of length `m*n` (not 2-D); the axis name must match the shape or beartype rejects.
    rows: Int[np.ndarray, "edges"] = np.repeat(np.arange(m), n)
    cols: Int[np.ndarray, "edges"] = np.tile(np.arange(n), m)
    phi_hat: Float[np.ndarray, "k"] = np.asarray(
        problem.support_values[:, None] * problem.features.features(*problem.support()[:2]),
        dtype=np.float64).sum(axis=0)

    records: list[EntropicStep] = []
    converged = False
    stall_reason = "max_steps"
    cost_std = float("nan")
    loss_value = float("inf")
    warm_g: Float[np.ndarray, "n"] | None = None
    warm_fg: tuple[np.ndarray, np.ndarray] | None = None
    gauss_err = float("nan")
    for step in range(1, int(max_steps) + 1):
        cost = np.asarray(problem.features.costs(theta, rows, cols), dtype=np.float64).reshape(m, n)
        if str(inner_solver) == "newton":
            # CN: 牛顿需要一个**绝对** eps；这里从代价尺度定一次，见 Step 1 的约定。
            # EN: Newton needs an **absolute** epsilon; fixed once from the cost scale, per Step 1.
            eps_absolute = float(relative_epsilon) * _cost_std(cost)
            newton = solve_semidual_newton(cost, a, b, epsilon=eps_absolute,
                                           g_init=warm_g, marginal_tolerance=float(inner_tolerance))
            if not newton.converged:
                raise RuntimeError(
                    f"Newton inner solve did not converge (marginal error {newton.marginal_error:.2e}); "
                    "refusing to use a non-converged plan")
            plan = newton.plan
            cost_std = eps_absolute / float(relative_epsilon)
            sinkhorn_iterations_used = newton.iterations
            gauss_err = float(newton.marginal_error)
            warm_g = np.asarray(newton.potential_target, dtype=np.float64)
        elif str(inner_solver) == "sinkhorn":
            sinkhorn: SinkhornResult = sinkhorn_plan_torch(
                cost, a, b, relative_epsilon=float(relative_epsilon),
                max_iterations=int(sinkhorn_iterations), tolerance=float(sinkhorn_tolerance),
                init=warm_fg, device=str(device))
            warm_fg = (sinkhorn.potential_source, sinkhorn.potential_target)
            plan = sinkhorn.plan
            cost_std = sinkhorn.cost_std
            sinkhorn_iterations_used = sinkhorn.iterations
            gauss_err = float(sinkhorn.marginal_error)
        else:
            raise ValueError(f"inner_solver must be 'sinkhorn' or 'newton', got {inner_solver!r}")
        features = np.asarray(problem.features.features(rows, cols), dtype=np.float64)
        model_moment = plan.reshape(-1) @ features
        observed_moment = phi_hat
        gradient = observed_moment - model_moment
        # CN: **损失必须是 `<C,pi_hat> - W_eps`，不是 `<C,pi_hat> - <C,P_eps>`**。
        #     `W_eps = <C,P_eps> - eps*H(P_eps)` 是熵 OT 的**值**；包络定理（`dW/dtheta = E[phi]`）
        #     只对**值**成立。用方案代价会漏掉 `eps*dH/dtheta`，实测方向导数因此差 2.4%（4.5e-2 对 1e-10）。
        # EN: **the loss must be `<C,pi_hat> - W_eps`, NOT `<C,pi_hat> - <C,P_eps>`**.
        #     `W_eps = <C,P_eps> - eps*H(P_eps)` is the entropic OT **value**, and the envelope theorem
        #     (`dW/dtheta = E[phi]`) holds only for the value. Using the plan cost omits `eps*dH/dtheta`;
        #     measured, the directional derivative was off by 2.4% (4.5e-2 against 1e-10).
        entropy = float(-np.sum(np.where(plan > 0.0, plan * np.log(np.maximum(plan, 1e-300)), 0.0)))
        plan_cost = float(theta @ model_moment)
        value_eps = plan_cost - float(relative_epsilon) * cost_std * entropy
        loss_value = float(theta @ observed_moment) - value_eps
        gradient_norm = float(np.max(np.abs(gradient)))
        records.append(EntropicStep(index=step, regularization=float(relative_epsilon) * cost_std,
                                    loss=loss_value, gradient_norm=gradient_norm,
                                    sinkhorn_iterations=int(sinkhorn_iterations_used),
                                    marginal_error=float(gauss_err)))
        if gradient_norm <= float(gradient_tolerance):
            converged = True
            stall_reason = "converged"
            break
        theta = _project_simplex(np.asarray(theta - float(learning_rate) * gradient, dtype=np.float64))
    return EntropicResult(theta=np.asarray(theta, dtype=np.float64), loss=loss_value, steps=len(records),
                          converged=converged, stall_reason=stall_reason, cost_std=cost_std,
                          steps_record=tuple(records))


def _project_simplex(value: Float[np.ndarray, "k"]) -> Float[np.ndarray, "k"]:
    """CN: 欧氏投影到单纯形（排序法，精确）。EN: exact Euclidean projection onto the simplex (sorting)."""
    u = np.sort(np.asarray(value, dtype=np.float64))[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, u.size + 1)
    rho = np.nonzero(u - css / ind > 0)[0][-1]
    tau = css[rho] / (rho + 1.0)
    return np.maximum(value - tau, 0.0)


__all__ = ["EntropicResult", "EntropicStep", "fit_entropic"]
