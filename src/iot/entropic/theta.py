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
from .sinkhorn import SinkhornResult, sinkhorn_plan_torch


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
    """
    from ..core.theta import SimplexTheta  # noqa: PLC0415 - local to keep the import graph shallow

    theta: Float[np.ndarray, "k"] = problem.theta_set.sample(np.random.default_rng(0))
    if not isinstance(problem.theta_set, SimplexTheta):
        raise ValueError("fit_entropic currently supports SimplexTheta only")
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
    warm: tuple[np.ndarray, np.ndarray] | None = None
    for step in range(1, int(max_steps) + 1):
        cost = np.asarray(problem.features.costs(theta, rows, cols), dtype=np.float64).reshape(m, n)
        sinkhorn: SinkhornResult = sinkhorn_plan_torch(
            cost, a, b, relative_epsilon=float(relative_epsilon),
            max_iterations=int(sinkhorn_iterations), tolerance=float(sinkhorn_tolerance),
            init=warm, device=str(device))
        warm = (sinkhorn.potential_source, sinkhorn.potential_target)
        cost_std = sinkhorn.cost_std
        features = np.asarray(problem.features.features(rows, cols), dtype=np.float64)
        model_moment = sinkhorn.plan.reshape(-1) @ features
        observed_moment = phi_hat
        gradient = observed_moment - model_moment
        loss_value = float(theta @ observed_moment - theta @ model_moment)
        gradient_norm = float(np.max(np.abs(gradient)))
        records.append(EntropicStep(index=step, regularization=sinkhorn.regularization,
                                    loss=loss_value, gradient_norm=gradient_norm,
                                    sinkhorn_iterations=sinkhorn.iterations,
                                    marginal_error=sinkhorn.marginal_error))
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
