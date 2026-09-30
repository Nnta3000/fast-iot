"""
CN: 逆 OT 方法的统一评估指标，供 loop、Sinkhorn 基线与实验共用。
EN: Unified evaluation metrics for inverse-OT methods, shared by the loop, the Sinkhorn
    baseline, and experiments.

CN: `l0_gap` 直接取自 `validation.is_optimal_plan` 的绝对间隙，避免再次实现同一公式。
EN: `l0_gap` is taken from `validation.is_optimal_plan`'s absolute gap, so the formula is
    not reimplemented.

CN: `optimality_tol` 固定为 1e-7：它与 loop 内部的判优容差一致（`loop.py` 调
    `is_optimal_plan(..., tol=1e-7)`），并留出 HiGHS/POT 在大规模实例上的数值残差；1e-9
    在这些规模上会把纯数值噪声误判为不可行。
EN: `optimality_tol` is fixed to 1e-7: it matches the loop's own optimality tolerance
    (`loop.py` calls `is_optimal_plan(..., tol=1e-7)`) and leaves room for HiGHS/POT
    numerical residuals at large sizes; 1e-9 misclassifies pure numerical noise there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot

from .features import dense_cost_matrix
from .problem import InverseOTProblem
from .validation import is_optimal_plan

DEFAULT_OPTIMALITY_TOL = 1e-7


@dataclass(frozen=True)
class InverseOTMetrics:
    """
    CN: 单个 `(problem, theta)` 的统一指标；`dist_to_theta_star` 只是诊断，不可当作断言。
    EN: Unified metrics for one `(problem, theta)`; `dist_to_theta_star` is a diagnostic
        only and must never be asserted.
    """

    is_optimal: bool
    l0_gap: float
    moment_error: float
    dist_to_theta_star: float
    wall_time: float
    peak_memory_mb: float
    iterations: int | None


def _all_edges(n_source: int, n_target: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.repeat(np.arange(int(n_source), dtype=np.int64), int(n_target))
    cols = np.tile(np.arange(int(n_target), dtype=np.int64), int(n_source))
    return rows, cols


def feature_moment(problem: InverseOTProblem, plan: np.ndarray) -> np.ndarray:
    """
    CN: 方案在特征张量下的矩 `sum_ij P_ij Phi_ij^k`，对 PSD 双线性族即 `-X^T P Y` 的展平。
    EN: Feature moment of a plan `sum_ij P_ij Phi_ij^k`; for the PSD bilinear family this
        is the flattening of `-X^T P Y`.

    CN: 这是 `moment_error` 的通用定义，对任意 `FeatureMap` 都成立，而不只是 PSD。
    EN: This is the general definition behind `moment_error`, valid for any `FeatureMap`,
        not only PSD.
    """
    plan_matrix = np.asarray(plan, dtype=np.float64)
    if plan_matrix.shape != (int(problem.n_source), int(problem.n_target)):
        raise ValueError("plan shape must match the problem")
    rows, cols = _all_edges(int(problem.n_source), int(problem.n_target))
    phi = problem.features.features(rows, cols)
    return np.ascontiguousarray(phi.T @ plan_matrix.reshape(-1), dtype=np.float64)


def moment_error(problem: InverseOTProblem, theta: np.ndarray) -> float:
    """
    CN: `||moment(pi_hat) - moment(P*(theta))||_2`，用 POT 求 `P*(theta)`。
    EN: `||moment(pi_hat) - moment(P*(theta))||_2` with `P*(theta)` solved by POT.
    """
    cost = dense_cost_matrix(problem.features, np.asarray(theta, dtype=np.float64))
    optimal_plan = ot.emd(problem.resolved_source_mass, problem.resolved_target_mass, cost)
    observed = feature_moment(problem, problem.plan.toarray())
    fitted = feature_moment(problem, optimal_plan)
    return float(np.linalg.norm(observed - fitted))


def l0_gap(
    problem: InverseOTProblem,
    theta: np.ndarray,
    *,
    optimality_tol: float = DEFAULT_OPTIMALITY_TOL,
) -> float:
    """
    CN: `langle C_theta, pi_hat rangle - min_P langle C_theta, P rangle`。
    EN: `langle C_theta, pi_hat rangle - min_P langle C_theta, P rangle`.

    CN: 唯一实现，来自 `validation.is_optimal_plan` 的绝对间隙。
    EN: The single implementation, taken from `validation.is_optimal_plan`'s absolute gap.
    """
    check = is_optimal_plan(problem, np.asarray(theta, dtype=np.float64), tol=float(optimality_tol))
    return float(check.absolute_gap)


def evaluate_inverse_ot(
    problem: InverseOTProblem,
    theta: np.ndarray,
    *,
    optimality_tol: float = DEFAULT_OPTIMALITY_TOL,
    theta_star: np.ndarray | None = None,
    wall_time: float = 0.0,
    peak_memory_mb: float = 0.0,
    iterations: int | None = None,
) -> InverseOTMetrics:
    """
    CN: 用统一容差评估一个逆 OT 解 `theta`，返回 `InverseOTMetrics`。
    EN: Evaluate one inverse-OT solution `theta` with a single tolerance and return
        `InverseOTMetrics`.

    CN: `theta_star`/`wall_time`/`peak_memory_mb`/`iterations` 可选，便于调用方附加诊断。
    EN: `theta_star`/`wall_time`/`peak_memory_mb`/`iterations` are optional so callers can
        attach diagnostics.
    """
    value = np.asarray(theta, dtype=np.float64).reshape(-1)
    check = is_optimal_plan(problem, value, tol=float(optimality_tol))
    distance = (
        float("nan")
        if theta_star is None
        else float(np.linalg.norm(value - np.asarray(theta_star, dtype=np.float64).reshape(-1)))
    )
    return InverseOTMetrics(
        is_optimal=bool(check.is_optimal),
        l0_gap=float(check.absolute_gap),
        moment_error=moment_error(problem, value),
        dist_to_theta_star=distance,
        wall_time=float(wall_time),
        peak_memory_mb=float(peak_memory_mb),
        iterations=iterations,
    )


__all__ = [
    "DEFAULT_OPTIMALITY_TOL",
    "InverseOTMetrics",
    "evaluate_inverse_ot",
    "feature_moment",
    "l0_gap",
    "moment_error",
]
