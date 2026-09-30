"""
CN: 熵正则化 OT 上的 PSD 度量拟合：用 Sinkhorn 的平滑梯度替代精确 OT 的次梯度。
EN: PSD metric fitting over entropic OT: a Sinkhorn-smoothed gradient instead of the
    exact-OT subgradient.

CN: 精确 OT 的 plan 是 A 的分段常值函数，moment 的差只是次梯度，普通次梯度下降会震荡。
    熵正则把 plan 变光滑，使 `fitted_moment - observed_moment` 成为真梯度；再配合 Adam、
    Polyak 平均与 epsilon 退火，可稳定收敛。
EN: The exact-OT plan is piecewise constant in A, so `fitted_moment - observed_moment` is
    only a subgradient and plain subgradient descent zig-zags. Entropic regularization
    makes the plan smooth, turning that difference into a true gradient; Adam, Polyak
    averaging, and epsilon annealing then give stable convergence.

CN: 复用 `BilinearPSDFeatures`/`dense_cost_matrix` 计算代价，不重复实现代价语义。
EN: Reuse `BilinearPSDFeatures`/`dense_cost_matrix` for the cost; cost semantics are not
    duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .features import BilinearPSDFeatures, dense_cost_matrix


def project_psd(matrix: np.ndarray) -> np.ndarray:
    """
    CN: 把对称矩阵投影到半正定锥（按特征值截断的 Frobenius 投影）。
    EN: Project a symmetric matrix onto the PSD cone (Frobenius projection by clipping
        negative eigenvalues).
    """
    value = np.asarray(matrix, dtype=np.float64)
    symmetric = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, 0.0, None)
    return np.ascontiguousarray((eigenvectors * clipped) @ eigenvectors.T, dtype=np.float64)


@dataclass(frozen=True)
class SinkhornResult:
    """
    CN: 一次 log-domain Sinkhorn 的完整结果：计划、末次缩放对数势、迭代数、收敛标志与边际误差。
    EN: Full result of one log-domain Sinkhorn run: plan, final scaled log-potentials, iteration
        count, convergence flag, and marginal error.

    CN: `log_source_potential`/`log_target_potential` 是算法内部以 epsilon 缩放的势，可直接作为
        下一次（更小 epsilon）调用的热启动初值。
    EN: `log_source_potential`/`log_target_potential` are the epsilon-scaled potentials used
        internally and can be passed straight back as the warm start of the next (smaller
        epsilon) call.
    """

    plan: np.ndarray
    log_source_potential: np.ndarray
    log_target_potential: np.ndarray
    iterations: int
    converged: bool
    marginal_error: float


def _sinkhorn_marginal_error(
    plan: np.ndarray, source: np.ndarray, target: np.ndarray
) -> float:
    """CN: 行列和相对目标质量的无穷范数偏差。EN: Max-abs row/column mass deviation."""
    row = float(np.max(np.abs(plan.sum(axis=1) - source)))
    column = float(np.max(np.abs(plan.sum(axis=0) - target)))
    return max(row, column)


def sinkhorn_plan_with_potentials(
    cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    *,
    epsilon: float,
    max_iterations: int = 1000,
    tolerance: float = 1e-9,
    initial_log_source_potential: np.ndarray | None = None,
    initial_log_target_potential: np.ndarray | None = None,
) -> SinkhornResult:
    """
    CN: log-domain 稳定 Sinkhorn，返回计划、缩放对数势与边际误差；支持势热启动。
    EN: Log-domain stabilized Sinkhorn returning the plan, scaled log-potentials, and the
        marginal error, with optional warm-started potentials.

    CN: epsilon 越小条件数越差；调用方负责不要把 epsilon 压到 float64 无法处理的程度。
        `initial_*` 允许把上一次（更大 epsilon）的解作为初值，减少迭代数。
    EN: Smaller epsilon is worse conditioned; callers must not push epsilon below what
        float64 can handle. `initial_*` warm-starts from the previous (larger epsilon)
        solution to cut iterations.
    """
    cost_matrix = np.asarray(cost, dtype=np.float64)
    source = np.asarray(source_mass, dtype=np.float64).reshape(-1)
    target = np.asarray(target_mass, dtype=np.float64).reshape(-1)
    if cost_matrix.shape != (source.size, target.size):
        raise ValueError("cost shape must be (len(source_mass), len(target_mass))")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be > 0")
    if np.any(source <= 0.0) or np.any(target <= 0.0):
        raise ValueError("source_mass and target_mass must be strictly positive")

    log_source = np.log(source)
    log_target = np.log(target)
    if initial_log_source_potential is None:
        potential_source = np.zeros(source.size, dtype=np.float64)
    else:
        potential_source = np.ascontiguousarray(
            np.asarray(initial_log_source_potential, dtype=np.float64).reshape(-1)
        )
        if potential_source.size != source.size:
            raise ValueError("initial_log_source_potential has the wrong length")
    if initial_log_target_potential is None:
        potential_target = np.zeros(target.size, dtype=np.float64)
    else:
        potential_target = np.ascontiguousarray(
            np.asarray(initial_log_target_potential, dtype=np.float64).reshape(-1)
        )
        if potential_target.size != target.size:
            raise ValueError("initial_log_target_potential has the wrong length")

    inverse_epsilon = 1.0 / float(epsilon)
    performed = 0
    converged = False
    for _ in range(int(max_iterations)):
        previous = potential_source
        potential_source = float(epsilon) * (
            log_source - logsumexp((potential_target[None, :] - cost_matrix) * inverse_epsilon, axis=1)
        )
        potential_target = float(epsilon) * (
            log_target - logsumexp((potential_source[:, None] - cost_matrix) * inverse_epsilon, axis=0)
        )
        performed += 1
        if float(np.max(np.abs(potential_source - previous))) <= float(tolerance):
            converged = True
            break
    log_plan = (potential_source[:, None] + potential_target[None, :] - cost_matrix) * inverse_epsilon
    plan = np.ascontiguousarray(np.exp(log_plan), dtype=np.float64)
    return SinkhornResult(
        plan=plan,
        log_source_potential=np.ascontiguousarray(potential_source),
        log_target_potential=np.ascontiguousarray(potential_target),
        iterations=int(performed),
        converged=bool(converged),
        marginal_error=_sinkhorn_marginal_error(plan, source, target),
    )


def sinkhorn_plan(
    cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    *,
    epsilon: float,
    max_iterations: int = 1000,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """
    CN: log-domain 稳定 Sinkhorn，返回熵正则 OT 方案 P（`sinkhorn_plan_with_potentials` 的便捷封装）。
    EN: Log-domain stabilized Sinkhorn returning the entropic OT plan P (a thin wrapper over
        `sinkhorn_plan_with_potentials`).
    """
    return sinkhorn_plan_with_potentials(
        cost,
        source_mass,
        target_mass,
        epsilon=float(epsilon),
        max_iterations=int(max_iterations),
        tolerance=float(tolerance),
    ).plan


def moment_from_plan(
    plan: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """
    CN: 交叉矩 `X^T P Y`（向量化，等价于逐边上 outer 的加权和）。
    EN: Cross moment `X^T P Y` (vectorized; equals the weighted sum of edge outer products).
    """
    plan_matrix = np.asarray(plan, dtype=np.float64)
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if plan_matrix.shape != (source.shape[0], target.shape[0]):
        raise ValueError("plan shape must match the point-cloud sizes")
    return np.ascontiguousarray(source.T @ plan_matrix @ target, dtype=np.float64)


def scale_invariant_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    CN: 归一化后的 Frobenius 距离，消除 A 的整体尺度。
    EN: Frobenius distance after normalizing each matrix, removing the overall scale.
    """
    reference_norm = float(np.linalg.norm(reference, "fro"))
    estimate_norm = float(np.linalg.norm(estimate, "fro"))
    if reference_norm == 0.0 or estimate_norm == 0.0:
        return float("inf")
    return float(
        np.linalg.norm(
            reference / reference_norm - estimate / estimate_norm,
            "fro",
        )
    )


@dataclass(frozen=True)
class EntropicIteration:
    """
    CN: 一步外层迭代的记录；objective 是熵 OT 值 <C, P_eps>。
    EN: One outer-iteration record; objective is the entropic OT value <C, P_eps>.
    """

    index: int
    epsilon: float
    objective: float
    moment_error: float


@dataclass(frozen=True)
class EntropicFitResult:
    """
    CN: 拟合结果；`matrix` 是 Polyak 平均（推荐用于评估），`last_matrix` 是最后一次迭代。
    EN: Fit result; `matrix` is the Polyak average (recommended for evaluation) and
        `last_matrix` is the last iterate.
    """

    matrix: np.ndarray
    last_matrix: np.ndarray
    iterations: tuple[EntropicIteration, ...]
    initial_moment_error: float
    final_moment_error: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", np.ascontiguousarray(self.matrix, dtype=np.float64))
        object.__setattr__(self, "last_matrix", np.ascontiguousarray(self.last_matrix, dtype=np.float64))


def _normalized_trace_one(matrix: np.ndarray) -> np.ndarray:
    """CN: 归一化到 Tr = 1，退化时回退到 I/d。EN: Normalize to Tr = 1, falling back to I/d."""
    trace = float(np.trace(matrix))
    if not np.isfinite(trace) or trace <= 0.0:
        dim = int(matrix.shape[0])
        return np.ascontiguousarray(np.eye(dim, dtype=np.float64) / dim)
    return np.ascontiguousarray(matrix / trace, dtype=np.float64)


def fit_psd_moment_entropic(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    observed_moment: np.ndarray,
    initial_matrix: np.ndarray,
    num_outer: int = 100,
    epsilon_start: float | None = None,
    epsilon_end: float = 1e-3,
    epsilon_decay: float = 0.95,
    learning_rate: float = 0.05,
    ridge: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    sinkhorn_iterations: int = 500,
    sinkhorn_tolerance: float = 1e-9,
    keep_trace_one: bool = True,
) -> EntropicFitResult:
    """
    CN: 用 Adam + Polyak 平均 + epsilon 退火，从观测交叉矩拟合 PSD 度量 A。
    EN: Fit a PSD metric A from an observed cross moment using Adam + Polyak averaging +
        epsilon annealing.

    CN: 熵 OT 使 plan 对 A 可微，`fitted_moment - observed_moment` 成为真梯度。
    EN: Entropic OT makes the plan differentiable in A, so
        `fitted_moment - observed_moment` is a true gradient.
    """
    source = np.ascontiguousarray(np.asarray(source_points, dtype=np.float64))
    target = np.ascontiguousarray(np.asarray(target_points, dtype=np.float64))
    source_mass = np.ascontiguousarray(np.asarray(source_mass, dtype=np.float64).reshape(-1))
    target_mass = np.ascontiguousarray(np.asarray(target_mass, dtype=np.float64).reshape(-1))
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source_points and target_points must be 2D with a shared dimension")
    if source_mass.size != source.shape[0] or target_mass.size != target.shape[0]:
        raise ValueError("mass vectors must match the point-cloud sizes")
    if abs(float(source_mass.sum()) - float(target_mass.sum())) > 1e-8:
        raise ValueError("source and target masses must have equal total mass")
    if int(num_outer) < 1:
        raise ValueError("num_outer must be >= 1")

    dim = int(source.shape[1])
    observed = np.ascontiguousarray(np.asarray(observed_moment, dtype=np.float64).reshape(dim, dim))
    features = BilinearPSDFeatures(source, target)
    matrix = project_psd(np.asarray(initial_matrix, dtype=np.float64).reshape(dim, dim))
    if bool(keep_trace_one):
        matrix = _normalized_trace_one(matrix)

    if epsilon_start is None:
        initial_cost = dense_cost_matrix(features, matrix.reshape(-1))
        epsilon = 0.1 * float(np.std(initial_cost))
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            epsilon = 1e-1
    else:
        epsilon = float(epsilon_start)

    averaged = matrix.copy()
    first_moment = np.zeros_like(matrix)
    second_moment = np.zeros_like(matrix)
    records: list[EntropicIteration] = []
    initial_moment_error = float("nan")
    for step in range(int(num_outer)):
        cost = dense_cost_matrix(features, matrix.reshape(-1))
        plan = sinkhorn_plan(
            cost,
            source_mass,
            target_mass,
            epsilon=float(epsilon),
            max_iterations=int(sinkhorn_iterations),
            tolerance=float(sinkhorn_tolerance),
        )
        fitted_moment = moment_from_plan(plan, source, target)
        residual = fitted_moment - observed
        gradient = residual + (2.0 * float(ridge)) * matrix if float(ridge) != 0.0 else residual
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
        corrected_first = first_moment / (1.0 - beta1 ** (step + 1))
        corrected_second = second_moment / (1.0 - beta2 ** (step + 1))
        matrix = matrix - float(learning_rate) * corrected_first / (np.sqrt(corrected_second) + float(adam_epsilon))
        matrix = project_psd(matrix)
        if bool(keep_trace_one):
            matrix = _normalized_trace_one(matrix)
        averaged = (step * averaged + matrix) / (step + 1)
        if step == 0:
            initial_moment_error = float(np.linalg.norm(residual))
        records.append(
            EntropicIteration(
                index=step,
                epsilon=float(epsilon),
                objective=float(np.sum(plan * cost, dtype=np.float64)),
                moment_error=float(np.linalg.norm(residual)),
            )
        )
        epsilon = max(float(epsilon) * float(epsilon_decay), float(epsilon_end))

    final_moment_error = records[-1].moment_error if records else float("nan")
    return EntropicFitResult(
        matrix=averaged,
        last_matrix=matrix,
        iterations=tuple(records),
        initial_moment_error=initial_moment_error,
        final_moment_error=final_moment_error,
    )


__all__ = [
    "EntropicFitResult",
    "EntropicIteration",
    "SinkhornResult",
    "fit_psd_moment_entropic",
    "moment_from_plan",
    "project_psd",
    "scale_invariant_error",
    "sinkhorn_plan",
    "sinkhorn_plan_with_potentials",
]
