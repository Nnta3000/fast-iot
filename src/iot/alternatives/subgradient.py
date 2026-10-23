"""
CN: 精确 OT 次梯度法：在 Θ 上直接最小化 L0 间隙
    `L0(theta) = <C_theta, pi_hat> - min_P <C_theta, P>`。
EN: Exact-OT subgradient method: directly minimise the L0 gap
    `L0(theta) = <C_theta, pi_hat> - min_P <C_theta, P>` over Theta.

CN: 精确 OT 的方案对 theta 分段常值，因此 `fitted - observed` 只是次梯度而非梯度；本模块
    提供 constant / inv_sqrt / polyak 三种投影次梯度步长，用来与熵平滑梯度法对照。内层
    OT 求解器通过 `ExactPlanSolver` 注入，因此 POT 与 HELLO-OT 都可复用同一段循环。
EN: The exact-OT plan is piecewise constant in theta, so `fitted - observed` is only a
    subgradient rather than a gradient. This module offers constant / inv_sqrt / polyak
    projected step rules to compare against the entropic smoothed-gradient method. The inner
    OT solver is injected as an `ExactPlanSolver`, so POT and HELLO-OT share the same loop.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy import sparse

from ..core.features import FeatureMap, dense_cost_matrix
from ..core.problem import InverseOTProblem
from .sinkhorn_theta import project_theta
from ..core.theta import BilinearPSDTheta, SimplexTheta

StepRule = Literal["constant", "inv_sqrt", "polyak"]
ExactPlanSolver = Callable[[np.ndarray], sparse.spmatrix]


@dataclass(frozen=True)
class SubgradientConfig:
    """
    CN: 次梯度法的超参数。`optimal_value` 仅在 polyak 步长中用到（干净实例下 L0* = 0）。
    EN: Subgradient hyperparameters. `optimal_value` is used by the polyak step only
        (L0* = 0 on clean instances).
    """

    step_rule: StepRule = "inv_sqrt"
    learning_rate: float = 0.1
    max_iterations: int = 2000
    tol: float = 1e-7
    optimal_value: float = 0.0

    def __post_init__(self) -> None:
        if self.step_rule not in ("constant", "inv_sqrt", "polyak"):
            raise ValueError(f"unknown step_rule {self.step_rule!r}")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be >= 1")


@dataclass(frozen=True)
class SubgradientIteration:
    """
    CN: 一步次梯度迭代的诊断：步长、当前 L0 间隙、次梯度范数、theta 与累计墙钟。
    EN: Diagnostics of one subgradient step: step size, current L0 gap, subgradient norm,
        theta, and cumulative wall time.
    """

    index: int
    step_size: float
    gap: float
    gradient_norm: float
    theta: tuple[float, ...]
    wall_time: float


@dataclass(frozen=True)
class SubgradientResult:
    """
    CN: 次梯度法的结果：末态 theta、最优（最小间隙）theta、迭代轨迹、是否达到 tol、间隙与墙钟。
    EN: Subgradient result: terminal theta, best (smallest-gap) theta, iteration trace, whether
        tol was reached, gaps, and wall time.
    """

    theta: np.ndarray
    best_theta: np.ndarray
    iterations: tuple[SubgradientIteration, ...]
    converged: bool
    final_gap: float
    best_gap: float
    wall_time: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", np.ascontiguousarray(self.theta, dtype=np.float64))
        object.__setattr__(
            self, "best_theta", np.ascontiguousarray(self.best_theta, dtype=np.float64)
        )


def indexed_feature_moment(
    features: FeatureMap,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """
    CN: 在给定支撑上的特征矩 `sum_e values_e * Phi_e`，复杂度 O(E * K)，从不 materialize m*n。
    EN: Feature moment on a given support, `sum_e values_e * Phi_e`, in O(E * K) and never
        materialising m*n.
    """
    weight = np.asarray(values, dtype=np.float64).reshape(-1)
    if weight.size == 0:
        return np.zeros(int(features.n_features), dtype=np.float64)
    row_index = np.asarray(rows, dtype=np.int64).reshape(-1)
    col_index = np.asarray(cols, dtype=np.int64).reshape(-1)
    phi = np.asarray(features.features(row_index, col_index), dtype=np.float64)
    return np.ascontiguousarray(phi.T @ weight, dtype=np.float64)


def support_moment(features: FeatureMap, plan: sparse.spmatrix | np.ndarray) -> np.ndarray:
    """
    CN: 任意（稀疏或稠密）方案的特征矩，只遍历非零元。
    EN: Feature moment of an arbitrary (sparse or dense) plan, touching nonzero entries only.
    """
    coo = sparse.coo_matrix(plan)
    mask = coo.data > 0.0
    return indexed_feature_moment(features, coo.row[mask], coo.col[mask], coo.data[mask])


def observed_feature_moment(problem: InverseOTProblem) -> np.ndarray:
    """
    CN: 观测方案 pi_hat 的特征矩 `Phi^T vec(pi_hat)`，只遍历 supp(pi_hat)。
    EN: Feature moment of the observed plan, `Phi^T vec(pi_hat)`, over supp(pi_hat) only.
    """
    rows, cols, values = problem.support()
    return indexed_feature_moment(problem.features, rows, cols, values)


def subgradient_from_plan(
    problem: InverseOTProblem,
    theta: np.ndarray,
    plan: sparse.spmatrix | np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    CN: 由精确 OT 方案给出次梯度与 L0 间隙：`g = Phi^T vec(pi_hat - P*)`，`L0 = <theta, g>`。
    EN: Subgradient and L0 gap from an exact-OT plan: `g = Phi^T vec(pi_hat - P*)`,
        `L0 = <theta, g>`.
    """
    observed = observed_feature_moment(problem)
    fitted = support_moment(problem.features, plan)
    gradient = np.ascontiguousarray(observed - fitted, dtype=np.float64)
    value = float(np.asarray(theta, dtype=np.float64).reshape(-1) @ gradient)
    return gradient, value


def make_pot_solver(
    problem: InverseOTProblem, *, num_itermax: int = 2_000_000
) -> ExactPlanSolver:
    """
    CN: 默认内层求解器：POT 精确 OT（稠密代价，CPU）。POT 惰性 import。
    EN: Default inner solver: POT exact OT (dense cost, CPU). POT is imported lazily.
    """

    def solve(theta: np.ndarray) -> sparse.spmatrix:
        import ot

        value = np.asarray(theta, dtype=np.float64).reshape(-1)
        cost = dense_cost_matrix(problem.features, value)
        plan = ot.emd(
            problem.resolved_source_mass,
            problem.resolved_target_mass,
            cost,
            numItermax=int(num_itermax),
        )
        return sparse.csr_matrix(plan)

    return solve


def _step_size(
    rule: StepRule,
    *,
    iteration: int,
    learning_rate: float,
    gap: float,
    gradient_norm: float,
    optimal_value: float,
) -> float:
    """
    CN: 三种步长规则；polyak 在已知最优值时自适应，`denom=0` 时退化为 0。
    EN: The three step rules; polyak is adaptive when the optimal value is known and
        degenerates to 0 when the denominator vanishes.
    """
    if rule == "constant":
        return float(learning_rate)
    if rule == "inv_sqrt":
        return float(learning_rate) / math.sqrt(float(iteration))
    denominator = float(gradient_norm) ** 2
    if denominator <= 0.0:
        return 0.0
    return max(float(gap) - float(optimal_value), 0.0) / denominator


def fit_theta_subgradient(
    problem: InverseOTProblem,
    *,
    config: SubgradientConfig | None = None,
    initial_theta: np.ndarray | None = None,
    exact_plan_solver: ExactPlanSolver | None = None,
    on_iteration: Callable[[SubgradientIteration], None] | None = None,
) -> SubgradientResult:
    """
    CN: 投影次梯度法最小化 L0；每步一次精确 OT 求解，返回轨迹。
    EN: Projected subgradient descent on L0; one exact-OT solve per step, returning the trace.

    CN: theta 直接在 Θ 上参数化并在每步后投影回 Θ；内层求解器可注入（POT 或 HELLO-OT）。
    EN: theta is parametrised directly on Theta and projected back after every step; the inner
        solver is injectable (POT or HELLO-OT).
    """
    settings = SubgradientConfig() if config is None else config
    solver = make_pot_solver(problem) if exact_plan_solver is None else exact_plan_solver
    theta_set = problem.theta_set
    # CN: project_theta 只支持单纯形与 PSD+trace1；在此显式收窄，既给出清晰报错，也让类型检查通过。
    # EN: project_theta only supports the simplex and PSD+trace1; narrow explicitly here, which
    #     both gives a clear error and lets the type checker verify the call.
    if not isinstance(theta_set, (SimplexTheta, BilinearPSDTheta)):
        raise TypeError(
            "fit_theta_subgradient supports SimplexTheta and BilinearPSDTheta; "
            f"got {type(theta_set).__name__}"
        )
    dimension = int(problem.n_features)
    theta = project_theta(
        np.full(dimension, 1.0 / float(dimension), dtype=np.float64)
        if initial_theta is None
        else np.asarray(initial_theta, dtype=np.float64).reshape(-1),
        theta_set,
    )
    observed = observed_feature_moment(problem)

    started = time.perf_counter()
    records: list[SubgradientIteration] = []
    converged = False
    final_gap = float("inf")
    best_gap = float("inf")
    best_theta = theta.copy()
    for step in range(1, int(settings.max_iterations) + 1):
        plan = solver(theta)
        fitted = support_moment(problem.features, plan)
        gradient = np.ascontiguousarray(observed - fitted, dtype=np.float64)
        gap = float(theta @ gradient)
        gradient_norm = float(np.linalg.norm(gradient))
        final_gap = gap
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()
        step_size = _step_size(
            settings.step_rule,
            iteration=step,
            learning_rate=float(settings.learning_rate),
            gap=gap,
            gradient_norm=gradient_norm,
            optimal_value=float(settings.optimal_value),
        )
        record = SubgradientIteration(
            index=step,
            step_size=float(step_size),
            gap=float(gap),
            gradient_norm=float(gradient_norm),
            theta=tuple(float(value) for value in theta),
            wall_time=float(time.perf_counter() - started),
        )
        records.append(record)
        if on_iteration is not None:
            on_iteration(record)
        if gap <= float(settings.tol):
            converged = True
            break
        theta = project_theta(theta - float(step_size) * gradient, theta_set)

    if final_gap < best_gap:
        best_gap = final_gap
        best_theta = theta.copy()
    return SubgradientResult(
        theta=theta,
        best_theta=best_theta,
        iterations=tuple(records),
        converged=bool(converged),
        final_gap=float(final_gap),
        best_gap=float(best_gap),
        wall_time=float(time.perf_counter() - started),
    )


__all__ = [
    "ExactPlanSolver",
    "StepRule",
    "SubgradientConfig",
    "SubgradientIteration",
    "SubgradientResult",
    "fit_theta_subgradient",
    "indexed_feature_moment",
    "make_pot_solver",
    "observed_feature_moment",
    "subgradient_from_plan",
    "support_moment",
]
