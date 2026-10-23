"""
CN: 经典 Kelley 割平面（K-维 cutting plane；不做 bundle 稳定化、不做二次规划）。以结构化 OT
    求解器为分离 oracle，在 theta 空间维护一个割银行，master 是 LP（K+1 个变量、t 行），与 n 无关。
EN: Classic Kelley cutting planes (K-dimensional cutting plane; no bundle stabilisation, no
    quadratic program). A structured OT solver is the separation oracle, a bank of theta-space
    cuts is maintained, and the master is an LP (K+1 variables, t rows) independent of n.

CN: 数学 / Mathematics
---------------------
代 价 对 theta 线性：`C_theta = sum_k theta_k Phi_k`。给定观测方案 pi_hat 与任一顶点
`P`（OT 最优方案），令 `g_P = Phi^T (pi_hat - P)`，则

    L0(theta) = <C_theta, pi_hat> - min_P <C_theta, P> = max_P <g_P, theta> >= 0,

且每个 `g_P` 都是凸函数 `L0` 的一个次梯度。theta 可容许（使 pi_hat 最优）当且仅当
对所有顶点 P 有 `<g_P, theta> <= 0`。因此可容许集合是 R^K 中的一个多面体锥，每次 OT 求解
返回它的一个被违反的面。这与结构化 SVM 的割平面情形一致：内层是 loss-augmented inference。

CN: 本模块是经典 Kelley 割平面，不做加速、不做 bundle 稳定化、不做二次规划。
EN: This module is classic Kelley cutting planes: no acceleration, no bundle stabilisation,
    no quadratic program.

CN: 只用 `FeatureMap` 与 `LinearThetaSet` 两个既有的协议缝隙；从不物化 m x n，也不
    自己计算代价。内层 OT 求解器可注入（默认 POT，经 `subgradient.make_pot_solver`）。
EN: It uses only the existing `FeatureMap` and `LinearThetaSet` seams; it never materialises
    m x n and never computes a cost itself. The inner OT solver is injectable (default POT
    via `subgradient.make_pot_solver`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.optimize import linprog

from ..core.problem import InverseOTError, InverseOTProblem
from .subgradient import (
    ExactPlanSolver,
    make_pot_solver,
    observed_feature_moment,
    support_moment,
)
from ..core.theta import LinearThetaSet

MasterMode = Literal["max_margin", "kelley"]


@dataclass(frozen=True)
class CuttingPlaneCut:
    """
    CN: 一条有效割 `<g, theta> <= 0`，其中 `g = Phi^T (pi_hat - P)` 由一次 OT 方案给出。
    EN: One valid cut `<g, theta> <= 0`, where `g = Phi^T (pi_hat - P)` comes from one OT
        plan.
    """

    index: int
    gradient: np.ndarray
    gap: float
    theta: np.ndarray

    def __post_init__(self) -> None:
        gradient = np.ascontiguousarray(np.asarray(self.gradient, dtype=np.float64).reshape(-1))
        theta = np.ascontiguousarray(np.asarray(self.theta, dtype=np.float64).reshape(-1))
        if gradient.size < 1:
            raise ValueError("cut gradient must be non-empty")
        if theta.size != gradient.size:
            raise ValueError("cut theta and gradient must have the same length")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("cut gradient must be finite")
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "theta", theta)


@dataclass(frozen=True)
class CuttingPlaneIteration:
    """
    CN: 一轮割平面迭代：查询点的真实间隙、当前割银行的 margin（上一轮 master 的值）、割数与时耗。
    EN: One cutting-plane iteration: the true gap at the query point, the current cut bank's
        margin (from the previous master), the cut count, and the timings.
    """

    index: int
    gap: float
    margin: float
    n_cuts: int
    theta: tuple[float, ...]
    wall_time: float
    master_time: float
    oracle_time: float


@dataclass(frozen=True)
class CuttingPlaneConfig:
    """
    CN: 经典割平面的超参数。`master_mode` 选择 master 的查询点：`"max_margin"` 是
        `max m s.t. G theta + m <= 0, m >= 0`（spec 描述的「最大 margin」），
        `"kelley"` 是教科书 `min m s.t. G theta <= m`（min-max 下界）。两者都是 K+1 变量 LP。
    EN: Classic cutting-plane hyperparameters. `master_mode` picks the master's query point:
        `"max_margin"` is `max m s.t. G theta + m <= 0, m >= 0` (the "maximum margin" of the
        spec), `"kelley"` is the textbook `min m s.t. G theta <= m` (the min-max lower bound).
        Both are LPs in K+1 variables.
    """

    master_mode: MasterMode = "max_margin"
    max_iterations: int = 100
    tol: float = 1e-9
    # CN: 归一化割只改变 master 的几何，不改变割所定义的半空间；默认开启，因为它把割数
    #     大约减半（见 CUTTING_PLANE_REPORT.md）。设为 False 即 spec 的未归一化 master。
    # EN: Normalising the cuts only changes the master's geometry, not the halfspaces the cuts
    #     define; it is on by default because it roughly halves the cut count (see
    #     CUTTING_PLANE_REPORT.md). Set False for the spec's unnormalised master.
    normalize_cuts: bool = True

    def __post_init__(self) -> None:
        if self.master_mode not in ("max_margin", "kelley"):
            raise ValueError("master_mode must be 'max_margin' or 'kelley'")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be >= 1")
        # CN: tol 允许为负：与 subgradient 模块一致，负 tol 表示「跑到预算上限」（永不提前停止）。
        # EN: tol may be negative: matching the subgradient module, a negative tol means
        #     "run to the budget" (never stop early).
        if not np.isfinite(float(self.tol)):
            raise ValueError("tol must be finite")


@dataclass(frozen=True)
class CutMasterResult:
    """
    CN: 割平面 master 的解：下一点 theta 与当前割银行的 margin/模型值。
    EN: The cut-plane master solution: the next theta and the margin/model value of the
        current cut bank.
    """

    theta: np.ndarray
    margin: float
    model_value: float
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "theta", np.ascontiguousarray(np.asarray(self.theta, dtype=np.float64).reshape(-1))
        )


@dataclass(frozen=True)
class CuttingPlaneResult:
    """
    CN: 割平面解：末态与最优 theta、割银行、迭代轨迹、是否在 tol 内、间隙与墙钟。
    EN: Cutting-plane solution: terminal and best theta, the cut bank, the iteration trace,
        whether it reached tol, the gaps, and wall time.
    """

    theta: np.ndarray
    best_theta: np.ndarray
    cuts: tuple[CuttingPlaneCut, ...]
    iterations: tuple[CuttingPlaneIteration, ...]
    converged: bool
    final_gap: float
    best_gap: float
    wall_time: float
    n_oracle_calls: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", np.ascontiguousarray(np.asarray(self.theta, dtype=np.float64)))
        object.__setattr__(
            self, "best_theta", np.ascontiguousarray(np.asarray(self.best_theta, dtype=np.float64))
        )


def solve_cut_master(
    gradients: np.ndarray,
    *,
    theta_set: LinearThetaSet,
    mode: MasterMode = "max_margin",
    normalize_cuts: bool = False,
    norm_floor: float = 1e-12,
) -> CutMasterResult:
    """
    CN: 求解割平面 master：给定割矩阵 G（t x K），
        `"max_margin"` -> `max m s.t. G theta + m <= 0, m >= 0, theta in Theta`；
        `"kelley"`     -> `min m s.t. G theta <= m, theta in Theta`。
        只使用 `LinearThetaSet.linear_constraints()`，因此对任何线性 Theta 都成立。
    EN: Solve the cutting-plane master. Given the cut matrix G (t x K),
        `"max_margin"` -> `max m s.t. G theta + m <= 0, m >= 0, theta in Theta`;
        `"kelley"`     -> `min m s.t. G theta <= m, theta in Theta`.
        It only uses `LinearThetaSet.linear_constraints()`, so it works for any linear Theta.
    """
    if mode not in ("max_margin", "kelley"):
        raise ValueError("mode must be 'max_margin' or 'kelley'")
    matrix = np.asarray(gradients, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError(f"gradients must be a non-empty 2D matrix; got shape={matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("gradients must be finite")
    # CN: g = 0 的割是恒真半空间（0·theta <= 0），不携带信息；丢弃它，否则会错误地把 margin 压成 0。
    # EN: A cut with g = 0 is the always-true halfspace (0 theta <= 0) and carries no
    #     information; drop it, otherwise it wrongly caps the margin at 0.
    norms = np.linalg.norm(matrix, axis=1)
    informative = norms > float(norm_floor)
    if not bool(np.any(informative)):
        raise InverseOTError("cut bank has no informative (nonzero) cuts")
    matrix = matrix[informative]
    if bool(normalize_cuts):
        matrix = matrix / norms[informative, None]
    n_cuts = int(matrix.shape[0])
    dimension = int(matrix.shape[1])
    constraints = theta_set.linear_constraints()
    if int(constraints.dimension) != dimension:
        raise ValueError(
            f"theta_set dimension {constraints.dimension} does not match cut dimension {dimension}"
        )

    objective = np.zeros(dimension + 1, dtype=np.float64)
    if mode == "max_margin":
        # CN: G theta + m <= 0，最大化 m，m >= 0。
        # EN: G theta + m <= 0, maximise m, m >= 0.
        a_ub = np.hstack([matrix, np.ones((n_cuts, 1), dtype=np.float64)])
        objective[dimension] = -1.0
        margin_bounds: tuple[float, float] = (0.0, np.inf)
    else:
        # CN: G theta - m <= 0，最小化 m（即 min_theta max_s <g_s, theta>）。
        # EN: G theta - m <= 0, minimise m (i.e. min_theta max_s <g_s, theta>).
        a_ub = np.hstack([matrix, -np.ones((n_cuts, 1), dtype=np.float64)])
        objective[dimension] = 1.0
        margin_bounds = (-np.inf, np.inf)
    b_ub = np.zeros(n_cuts, dtype=np.float64)

    n_eq = int(constraints.a_eq.shape[0])
    a_eq = np.hstack([np.asarray(constraints.a_eq, dtype=np.float64), np.zeros((n_eq, 1))])
    bounds: list[tuple[float, float]] = [
        *constraints.bounds(),
        (float(margin_bounds[0]), float(margin_bounds[1])),
    ]
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.asarray(constraints.b_eq, dtype=np.float64),
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        raise InverseOTError(
            f"cutting-plane master failed ({mode}): {result.message}"
        )
    raw = float(result.x[dimension])
    if mode == "max_margin":
        margin = raw
        model_value = -raw
    else:
        margin = -raw
        model_value = raw
    return CutMasterResult(
        theta=np.ascontiguousarray(result.x[:dimension], dtype=np.float64),
        margin=float(margin),
        model_value=float(model_value),
        status=str(result.message),
    )


def _initial_theta(problem: InverseOTProblem, initial_theta: np.ndarray | None) -> np.ndarray:
    """CN: 初始查询点：默认单纯形均匀点，否则使用调用方提供的 theta。EN: Initial query point."""
    dimension = int(problem.n_features)
    if initial_theta is None:
        value = np.full(dimension, 1.0 / float(dimension), dtype=np.float64)
    else:
        value = np.asarray(initial_theta, dtype=np.float64).reshape(-1)
        if value.size != dimension:
            raise ValueError("initial_theta length must equal problem.n_features")
    if not problem.theta_set.contains(value, tol=1e-9):
        raise ValueError("initial_theta must lie inside Theta")
    return np.ascontiguousarray(value, dtype=np.float64)


def solve_cutting_plane(
    problem: InverseOTProblem,
    *,
    config: CuttingPlaneConfig | None = None,
    initial_theta: np.ndarray | None = None,
    exact_plan_solver: ExactPlanSolver | None = None,
    on_iteration: Callable[[CuttingPlaneIteration], None] | None = None,
) -> CuttingPlaneResult:
    """
    CN: 求解逆 OT 的经典割平面法。每轮查询一个 theta，用注入的 OT 求解器返回方案 P，得到
        次梯度 `g = Phi^T (pi_hat - P)` 与真实间隙 `L0 = <theta, g>`；若 `L0 <= tol` 则停止
        （OT 方案本身给出证书），否则把 g 加入银行并解 master 得到下一个 theta。
    EN: Solve inverse OT by classic cutting planes. Each round queries one theta, the injected
        OT solver returns a plan P, giving the subgradient `g = Phi^T (pi_hat - P)` and the
        true gap `L0 = <theta, g>`; stop when `L0 <= tol` (the OT plan itself certifies
        optimality), otherwise add g to the bank and solve the master for the next theta.

    CN: 只用线性 Theta；SDP 谱面体没有 LP 视图，必须用专用 master，故此处明确拒绝。
    EN: Linear Theta only; the SDP spectrahedron has no LP view and needs a dedicated master,
        so it is rejected explicitly here.
    """
    settings = CuttingPlaneConfig() if config is None else config
    if not isinstance(problem.theta_set, LinearThetaSet):
        raise InverseOTError(
            "the cutting-plane LP master requires a LinearThetaSet; "
            f"got {type(problem.theta_set).__name__}. Pair a non-linear Theta with a "
            "dedicated master."
        )
    solver: ExactPlanSolver = make_pot_solver(problem) if exact_plan_solver is None else exact_plan_solver
    theta = _initial_theta(problem, initial_theta)
    observed = observed_feature_moment(problem)

    cuts: list[CuttingPlaneCut] = []
    records: list[CuttingPlaneIteration] = []
    converged = False
    final_gap = float("inf")
    best_gap = float("inf")
    best_theta = theta.copy()
    previous_margin = float("nan")
    started = time.perf_counter()
    n_oracle_calls = 0

    for iteration in range(1, int(settings.max_iterations) + 1):
        iteration_started = time.perf_counter()
        oracle_started = time.perf_counter()
        plan = solver(theta)
        fitted = support_moment(problem.features, plan)
        gradient = np.ascontiguousarray(observed - fitted, dtype=np.float64)
        gap = float(theta @ gradient)
        oracle_time = time.perf_counter() - oracle_started
        n_oracle_calls += 1
        final_gap = gap
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()

        if gap <= float(settings.tol):
            converged = True
            records.append(
                CuttingPlaneIteration(
                    index=iteration,
                    gap=float(gap),
                    margin=float(previous_margin),
                    n_cuts=len(cuts),
                    theta=tuple(float(value) for value in theta),
                    wall_time=float(time.perf_counter() - iteration_started),
                    master_time=0.0,
                    oracle_time=float(oracle_time),
                )
            )
            if on_iteration is not None:
                on_iteration(records[-1])
            break

        cuts.append(
            CuttingPlaneCut(
                index=iteration,
                gradient=gradient,
                gap=float(gap),
                theta=theta.copy(),
            )
        )
        master_started = time.perf_counter()
        master = solve_cut_master(
            np.vstack([cut.gradient for cut in cuts]),
            theta_set=problem.theta_set,
            mode=settings.master_mode,
            normalize_cuts=bool(settings.normalize_cuts),
        )
        master_time = time.perf_counter() - master_started
        previous_margin = float(master.margin)
        theta = master.theta
        record = CuttingPlaneIteration(
            index=iteration,
            gap=float(gap),
            margin=float(master.margin),
            n_cuts=len(cuts),
            theta=tuple(float(value) for value in cuts[-1].theta),
            wall_time=float(time.perf_counter() - iteration_started),
            master_time=float(master_time),
            oracle_time=float(oracle_time),
        )
        records.append(record)
        if on_iteration is not None:
            on_iteration(record)

    if best_gap > final_gap:
        best_gap = final_gap
        best_theta = theta.copy()
    return CuttingPlaneResult(
        theta=theta,
        best_theta=best_theta,
        cuts=tuple(cuts),
        iterations=tuple(records),
        converged=bool(converged),
        final_gap=float(final_gap),
        best_gap=float(best_gap),
        wall_time=float(time.perf_counter() - started),
        n_oracle_calls=int(n_oracle_calls),
    )


__all__ = [
    "CutMasterResult",
    "CuttingPlaneConfig",
    "CuttingPlaneCut",
    "CuttingPlaneIteration",
    "CuttingPlaneResult",
    "MasterMode",
    "solve_cut_master",
    "solve_cutting_plane",
]
