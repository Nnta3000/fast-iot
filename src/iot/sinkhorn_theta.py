"""
CN: θ-空间的 Sinkhorn 基线：在 Θ 上最小化熵 OT 间隙
    `L_ε(θ) = <C_θ, π_hat> - <C_θ, P_ε(θ)>`。
EN: theta-space Sinkhorn baseline: minimize the entropic OT gap
    `L_ε(θ) = <C_θ, π_hat> - <C_θ, P_ε(θ)>` over Θ.

CN: 这是与约束生成 loop 信息对齐的基线（两者都看同一个 π_hat）。由包络定理，
    `∇L_ε(θ)_k = <Φ_k, π_hat - P_ε(θ)>`，即特征矩之差；投影梯度（可选 Adam）步进，
    ε 几何退火。复用 `FeatureMap`/`ThetaSet`/`sinkhorn_plan`，不重写 Sinkhorn。
EN: This is the information-matched baseline to the constraint-generation loop (both see
    the same π_hat). By the envelope theorem `∇L_ε(θ)_k = <Φ_k, π_hat - P_ε(θ)>`, the
    feature-moment difference; projected gradient (optionally Adam) with geometric epsilon
    annealing. Reuses `FeatureMap`/`ThetaSet`/`sinkhorn_plan`; Sinkhorn is not reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .entropic import project_psd, sinkhorn_plan
from .problem import InverseOTProblem
from .theta import BilinearPSDTheta, SimplexTheta


@dataclass(frozen=True)
class SinkhornThetaIteration:
    """
    CN: 一步外层迭代：当前 ε、熵间隙与梯度范数。
    EN: One outer iteration: current epsilon, entropic gap, and gradient norm.
    """

    index: int
    epsilon: float
    entropic_gap: float
    gradient_norm: float


@dataclass(frozen=True)
class SinkhornThetaResult:
    """
    CN: θ-空间 Sinkhorn 基线的结果；`theta` 是最后一次迭代（已投影回 Θ）。
    EN: Result of the theta-space Sinkhorn baseline; `theta` is the last iterate
        (projected back to Theta).
    """

    theta: np.ndarray
    iterations: tuple[SinkhornThetaIteration, ...]
    final_epsilon: float
    initial_gradient_norm: float
    final_gradient_norm: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", np.ascontiguousarray(self.theta, dtype=np.float64))


def project_simplex(values: np.ndarray) -> np.ndarray:
    """
    CN: 欧氏投影到概率单纯形（排序算法）。
    EN: Euclidean projection onto the probability simplex (sorting algorithm).
    """
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 1:
        return np.ones(1, dtype=np.float64)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    index = np.arange(1, vector.size + 1, dtype=np.float64)
    positive = ordered - cumulative / index > 0.0
    rho = int(np.nonzero(positive)[0][-1])
    threshold = cumulative[rho] / float(rho + 1)
    return np.ascontiguousarray(np.maximum(vector - threshold, 0.0), dtype=np.float64)


def _project_psd_theta(values: np.ndarray, point_dimension: int) -> np.ndarray:
    """CN: 投影到 {A ⪰ 0, Tr A = 1}。EN: Project onto {A ⪰ 0, Tr A = 1}."""
    dim = int(point_dimension)
    matrix = project_psd(np.asarray(values, dtype=np.float64).reshape(dim, dim))
    trace = float(np.trace(matrix))
    if not np.isfinite(trace) or trace <= 0.0:
        matrix = np.eye(dim, dtype=np.float64)
        trace = float(dim)
    return np.ascontiguousarray(matrix / trace, dtype=np.float64).reshape(-1)


def project_theta(theta: np.ndarray, theta_set: SimplexTheta | BilinearPSDTheta) -> np.ndarray:
    """
    CN: 按 Θ 的类型投影；目前支持单纯形与 PSD+trace1。
    EN: Project onto Theta by type; currently the simplex and PSD+trace1 are supported.
    """
    if isinstance(theta_set, SimplexTheta):
        return project_simplex(theta)
    if isinstance(theta_set, BilinearPSDTheta):
        return _project_psd_theta(theta, int(theta_set.point_dimension))
    raise TypeError(
        "fit_theta_sinkhorn supports SimplexTheta and BilinearPSDTheta; "
        f"got {type(theta_set).__name__}"
    )


def _default_initial_theta(theta_set: SimplexTheta | BilinearPSDTheta) -> np.ndarray:
    if isinstance(theta_set, SimplexTheta):
        size = int(theta_set.dimension)
        return np.full(size, 1.0 / float(size), dtype=np.float64)
    if isinstance(theta_set, BilinearPSDTheta):
        dim = int(theta_set.point_dimension)
        return (np.eye(dim, dtype=np.float64) / float(dim)).reshape(-1)
    raise TypeError(f"unsupported theta set {type(theta_set).__name__}")


def _all_edges(n_source: int, n_target: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.repeat(np.arange(int(n_source), dtype=np.int64), int(n_target))
    cols = np.tile(np.arange(int(n_target), dtype=np.int64), int(n_source))
    return rows, cols


def fit_theta_sinkhorn(
    problem: InverseOTProblem,
    *,
    initial_theta: np.ndarray | None = None,
    num_outer: int = 200,
    epsilon_start: float | None = None,
    epsilon_end: float = 1e-3,
    epsilon_decay: float = 0.95,
    learning_rate: float = 0.05,
    step_rule: Literal["adam", "plain"] = "adam",
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    sinkhorn_iterations: int = 500,
    sinkhorn_tolerance: float = 1e-9,
) -> SinkhornThetaResult:
    """
    CN: 用投影梯度/Adam 在 Θ 上最小化熵 OT 间隙，返回 θ 与 ε 退火轨迹。
    EN: Minimize the entropic OT gap over Theta with projected gradient/Adam and return
        theta with the epsilon-annealing trace.

    CN: ε 从 `epsilon_start`（默认 0.1 * std(C_θ0)）按 `epsilon_decay` 几何退火到 `epsilon_end`。
    EN: epsilon anneals geometrically from `epsilon_start` (default `0.1 * std(C_theta0)`)
        to `epsilon_end` at rate `epsilon_decay`.
    """
    theta_set = problem.theta_set
    if not isinstance(theta_set, (SimplexTheta, BilinearPSDTheta)):
        raise TypeError(
            "fit_theta_sinkhorn supports SimplexTheta and BilinearPSDTheta; "
            f"got {type(theta_set).__name__}"
        )
    if int(num_outer) < 1:
        raise ValueError("num_outer must be >= 1")
    if float(epsilon_decay) <= 0.0 or float(epsilon_decay) > 1.0:
        raise ValueError("epsilon_decay must lie in (0, 1]")

    theta = (
        _default_initial_theta(theta_set)
        if initial_theta is None
        else np.asarray(initial_theta, dtype=np.float64).reshape(-1)
    )
    if theta.size != int(problem.n_features):
        raise ValueError("initial_theta length must equal problem.n_features")
    theta = project_theta(theta, theta_set)

    source_mass = problem.resolved_source_mass
    target_mass = problem.resolved_target_mass
    rows, cols = _all_edges(int(problem.n_source), int(problem.n_target))
    feature_matrix = problem.features.features(rows, cols)  # (mn, K)
    observed_moment = feature_matrix.T @ problem.plan.toarray().reshape(-1)

    def cost_of(values: np.ndarray) -> np.ndarray:
        flat = feature_matrix @ values
        return flat.reshape(int(problem.n_source), int(problem.n_target))

    if epsilon_start is None:
        epsilon = 0.1 * float(np.std(cost_of(theta)))
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            epsilon = 1e-1
    else:
        epsilon = float(epsilon_start)

    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    records: list[SinkhornThetaIteration] = []
    initial_gradient_norm = float("nan")
    for step in range(int(num_outer)):
        cost = cost_of(theta)
        plan = sinkhorn_plan(
            cost,
            source_mass,
            target_mass,
            epsilon=float(epsilon),
            max_iterations=int(sinkhorn_iterations),
            tolerance=float(sinkhorn_tolerance),
        )
        fitted_moment = feature_matrix.T @ plan.reshape(-1)
        gradient = observed_moment - fitted_moment
        entropic_gap = float(theta @ gradient)
        gradient_norm = float(np.linalg.norm(gradient))
        if step == 0:
            initial_gradient_norm = gradient_norm

        if step_rule == "adam":
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
            corrected_first = first_moment / (1.0 - beta1 ** (step + 1))
            corrected_second = second_moment / (1.0 - beta2 ** (step + 1))
            direction = corrected_first / (np.sqrt(corrected_second) + float(adam_epsilon))
        else:
            direction = gradient
        theta = project_theta(theta - float(learning_rate) * direction, theta_set)
        records.append(
            SinkhornThetaIteration(
                index=step,
                epsilon=float(epsilon),
                entropic_gap=entropic_gap,
                gradient_norm=gradient_norm,
            )
        )
        epsilon = max(float(epsilon) * float(epsilon_decay), float(epsilon_end))

    final_plan = sinkhorn_plan(
        cost_of(theta),
        source_mass,
        target_mass,
        epsilon=max(float(epsilon), float(epsilon_end)),
        max_iterations=int(sinkhorn_iterations),
        tolerance=float(sinkhorn_tolerance),
    )
    final_gradient = observed_moment - feature_matrix.T @ final_plan.reshape(-1)
    return SinkhornThetaResult(
        theta=theta,
        iterations=tuple(records),
        final_epsilon=float(max(epsilon, float(epsilon_end))),
        initial_gradient_norm=float(initial_gradient_norm),
        final_gradient_norm=float(np.linalg.norm(final_gradient)),
    )


__all__ = [
    "SinkhornThetaIteration",
    "SinkhornThetaResult",
    "fit_theta_sinkhorn",
    "project_simplex",
    "project_theta",
]
