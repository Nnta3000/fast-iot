"""
CN: 逆 OT 的验证工具：固定对偶势下的 theta 范围，以及 pi_hat 的最优性判定。
EN: Validation utilities for inverse OT: theta ranges under a fixed dual pair, and the
    optimality check for pi_hat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot
from scipy.optimize import linprog

from .edges import EdgeSet
from .features import dense_cost_matrix
from .master import coordinate_extrema
from .problem import InverseOTError, InverseOTProblem
from .theta import LinearThetaSet

# CN: POT 网络单纯形的返回码 1 表示最优；3 表示达到 numItermax（见 emd_wrap.check_result）。
# EN: POT network-simplex result code 1 means optimal; 3 means numItermax was reached
#     (see emd_wrap.check_result).
_POT_OPTIMAL_RESULT_CODE = 1
# CN: 验证用的默认迭代上限；足够覆盖 n≈3200（需 ~2e6），远大于 POT 的 1e5 默认。
# EN: Default iteration cap for validation; covers n≈3200 (needs ~2e6) and is far above
#     POT's 1e5 default.
_DEFAULT_VALIDATION_NUM_ITERMAX = 10_000_000


def _dual_vector(value: np.ndarray, *, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != int(size):
        raise ValueError(f"{name} must have length {size}; got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(array, dtype=np.float64)


def _all_edges(n_source: int, n_target: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.repeat(np.arange(int(n_source), dtype=np.int64), int(n_target))
    cols = np.tile(np.arange(int(n_target), dtype=np.int64), int(n_source))
    return rows, cols


@dataclass(frozen=True)
class ThetaRange:
    """
    CN: 可行 theta 集合逐坐标的 [lower, upper] 区间。
    EN: Per-coordinate [lower, upper] intervals of the feasible theta set.
    """

    lower: np.ndarray
    upper: np.ndarray
    statuses: tuple[str, ...]

    def __post_init__(self) -> None:
        lower = np.ascontiguousarray(np.asarray(self.lower, dtype=np.float64).reshape(-1))
        upper = np.ascontiguousarray(np.asarray(self.upper, dtype=np.float64).reshape(-1))
        if lower.size != upper.size:
            raise ValueError("lower and upper must have the same length")
        if len(self.statuses) != lower.size:
            raise ValueError("statuses must have one entry per coordinate")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dimension(self) -> int:
        return int(self.lower.size)

    def width(self) -> np.ndarray:
        return np.ascontiguousarray(self.upper - self.lower, dtype=np.float64)

    def contains(self, theta: np.ndarray, *, tol: float = 1e-7) -> bool:
        value = np.asarray(theta, dtype=np.float64).reshape(-1)
        if value.size != self.dimension:
            return False
        lower_ok = bool(np.all(value >= self.lower - float(tol)))
        upper_ok = bool(np.all(value <= self.upper + float(tol)))
        return lower_ok and upper_ok


@dataclass(frozen=True)
class OptimalityCheck:
    """
    CN: pi_hat 相对代价 c(theta) 的最优性检查结果。
    EN: Result of checking whether pi_hat is optimal for the cost c(theta).
    """

    is_optimal: bool
    plan_cost: float
    optimal_cost: float
    absolute_gap: float
    relative_gap: float


@dataclass(frozen=True)
class _Polytope:
    a_ub: np.ndarray | None
    b_ub: np.ndarray | None
    a_eq: np.ndarray
    b_eq: np.ndarray
    bounds: list[tuple[float, float]]


def _fixed_dual_polytope(
    problem: InverseOTProblem,
    source_dual: np.ndarray,
    target_dual: np.ndarray,
    *,
    equality_tol: float,
    feasibility_tol: float,
) -> _Polytope:
    """
    CN: 构造固定 (f, g) 时 theta 的可行多面体。
    EN: Build the feasible polytope of theta for a fixed (f, g) pair.

    CN: S 上的等式与 S 外的不等式都放入容差带，以吸收 OT 求解器的数值残差。
    EN: Equalities on S and inequalities off S sit inside tolerance bands to absorb the
        numerical residual of the OT solver.
    """
    m = int(problem.n_source)
    n = int(problem.n_target)
    if not isinstance(problem.theta_set, LinearThetaSet):
        raise InverseOTError("fixed_dual_theta_ranges requires a LinearThetaSet")
    support_rows, support_cols, _values = problem.support()
    all_rows, all_cols = _all_edges(m, n)
    source = _dual_vector(source_dual, size=m, name="source_dual")
    target = _dual_vector(target_dual, size=n, name="target_dual")

    support_features = problem.features.features(support_rows, support_cols)
    support_rhs = source[support_rows] + target[support_cols]
    constraints = problem.theta_set.linear_constraints()

    all_keys = all_rows * n + all_cols
    off_support = ~np.isin(all_keys, problem.support_keys())
    off_rows = all_rows[off_support]
    off_cols = all_cols[off_support]

    blocks: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    if support_features.size:
        blocks.append(support_features)
        rhs.append(support_rhs + float(equality_tol))
        blocks.append(-support_features)
        rhs.append(-support_rhs + float(equality_tol))
    if off_rows.size:
        off_features = problem.features.features(off_rows, off_cols)
        off_rhs = source[off_rows] + target[off_cols]
        blocks.append(-off_features)
        rhs.append(-off_rhs + float(feasibility_tol))
    if blocks:
        a_ub = np.ascontiguousarray(np.vstack(blocks), dtype=np.float64)
        b_ub = np.ascontiguousarray(np.concatenate(rhs), dtype=np.float64)
    else:
        a_ub = None
        b_ub = None
    return _Polytope(
        a_ub=a_ub,
        b_ub=b_ub,
        a_eq=np.ascontiguousarray(constraints.a_eq, dtype=np.float64),
        b_eq=np.ascontiguousarray(constraints.b_eq, dtype=np.float64),
        bounds=constraints.bounds(),
    )


def _solve_theta_lp(polytope: _Polytope, objective: np.ndarray) -> tuple[float, str]:
    result = linprog(
        np.asarray(objective, dtype=np.float64),
        A_ub=polytope.a_ub,
        b_ub=polytope.b_ub,
        A_eq=polytope.a_eq,
        b_eq=polytope.b_eq,
        bounds=polytope.bounds,
        method="highs",
    )
    if not result.success or result.fun is None:
        return float("nan"), str(result.message)
    return float(result.fun), str(result.message)


def fixed_dual_theta_ranges(
    problem: InverseOTProblem,
    source_dual: np.ndarray,
    target_dual: np.ndarray,
    *,
    equality_tol: float = 1e-6,
    feasibility_tol: float = 1e-6,
) -> ThetaRange:
    """
    CN: 固定 OT 对偶势 (f, g)，对每个坐标 k 求可行 theta 的 min/max（e0 实验）。
    EN: With the OT dual pair (f, g) fixed, minimize/maximize each theta coordinate over
        the feasible region (the e0 experiment).
    """
    polytope = _fixed_dual_polytope(
        problem,
        source_dual,
        target_dual,
        equality_tol=float(equality_tol),
        feasibility_tol=float(feasibility_tol),
    )
    dimension = int(problem.n_features)
    lower = np.empty(dimension, dtype=np.float64)
    upper = np.empty(dimension, dtype=np.float64)
    statuses: list[str] = []
    for index in range(dimension):
        direction = np.zeros(dimension, dtype=np.float64)
        direction[index] = 1.0
        low, low_status = _solve_theta_lp(polytope, direction)
        high, high_status = _solve_theta_lp(polytope, -direction)
        if not np.isfinite(low) or not np.isfinite(high):
            raise InverseOTError(
                "fixed-dual theta region is empty or unbounded at coordinate "
                f"{index}: min status={low_status!r}, max status={high_status!r}"
            )
        lower[index] = low
        upper[index] = -high
        statuses.append(f"min={low_status}; max={high_status}")
    return ThetaRange(lower=lower, upper=upper, statuses=tuple(statuses))


def optimal_theta_ranges(
    problem: InverseOTProblem,
    *,
    max_elements: int = 50_000_000,
) -> ThetaRange:
    """
    CN: 在 (f, g) 自由时，对每个 theta 坐标求最优 theta 集合的 min/max（e1）。
    EN: With (f, g) free, minimize/maximize each theta coordinate over the set of
        optimal theta (the e1 experiment).
    """
    m = int(problem.n_source)
    n = int(problem.n_target)
    if m * n > int(max_elements):
        raise ValueError(
            f"optimal_theta_ranges would build {m * n} edges, above max_elements={max_elements}"
        )
    rows, cols = _all_edges(m, n)
    edges = EdgeSet.from_arrays(rows, cols, n_target=n)
    lower, upper, statuses = coordinate_extrema(problem, edges, delta=0.0)
    return ThetaRange(lower=lower, upper=upper, statuses=statuses)


def plan_cost(problem: InverseOTProblem, theta: np.ndarray) -> float:
    """
    CN: 计算 <c(theta), pi_hat>。
    EN: Compute <c(theta), pi_hat>.
    """
    cost = dense_cost_matrix(problem.features, theta)
    plan = np.asarray(problem.plan.toarray(), dtype=np.float64)
    return float(np.sum(plan * cost, dtype=np.float64))


def is_optimal_plan(
    problem: InverseOTProblem,
    theta: np.ndarray,
    *,
    tol: float = 1e-9,
    max_elements: int = 50_000_000,
    num_itermax: int = _DEFAULT_VALIDATION_NUM_ITERMAX,
) -> OptimalityCheck:
    """
    CN: 用 POT 网络单纯形精确求解 c(theta) 的 OT，比较 pi_hat 的目标值（真正的验证）。
    EN: Solve the exact OT for c(theta) with POT's network simplex and compare the
        pi_hat objective (the real validation).

    CN: 这是判定 theta 是否使 pi_hat 最优的权威检查；绝不断言 theta == theta_star。
        `num_itermax` 透传给 POT；若 POT 未到达最优（返回码 != 1 或有警告），本函数
        直接抛出 `InverseOTError`，避免用被迭代上限截断的「最优代价」静默误判。
    EN: This is the authoritative check that a theta makes pi_hat optimal; never assert
        theta == theta_star. `num_itermax` is forwarded to POT; if POT does not reach
        optimality (result code != 1 or any warning), this raises `InverseOTError` instead
        of silently comparing against an iteration-capped "optimal" cost.
    """
    value = np.asarray(theta, dtype=np.float64).reshape(-1)
    if value.size != int(problem.n_features):
        raise ValueError("theta length does not match features.n_features")
    if not problem.theta_set.contains(value, tol=1e-6):
        raise InverseOTError("theta is not (within tolerance) inside the parameter set Theta")
    cost = dense_cost_matrix(problem.features, value, max_elements=int(max_elements))
    source_mass = problem.resolved_source_mass
    target_mass = problem.resolved_target_mass
    if abs(float(source_mass.sum()) - float(target_mass.sum())) > 1e-8:
        raise InverseOTError("source and target masses must have equal total mass")
    transport, log = ot.emd(
        source_mass, target_mass, cost, numItermax=int(num_itermax), log=True
    )
    result_code = int(log["result_code"])
    warning = log["warning"]
    if result_code != _POT_OPTIMAL_RESULT_CODE or warning is not None:
        raise InverseOTError(
            "POT network simplex did not reach optimality while validating pi_hat "
            f"(result_code={result_code}, warning={warning!r}); the reference optimal cost "
            f"would be untrustworthy. Increase num_itermax (currently {int(num_itermax)})."
        )
    plan = np.asarray(problem.plan.toarray(), dtype=np.float64)
    optimal_cost = float(np.sum(transport * cost, dtype=np.float64))
    current_cost = float(np.sum(plan * cost, dtype=np.float64))
    absolute_gap = current_cost - optimal_cost
    scale = max(1.0, abs(optimal_cost))
    relative_gap = absolute_gap / scale
    return OptimalityCheck(
        is_optimal=bool(absolute_gap <= float(tol) * scale),
        plan_cost=current_cost,
        optimal_cost=optimal_cost,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
    )


def _max_dual_violation(
    source_dual: np.ndarray,
    target_dual: np.ndarray,
    cost: np.ndarray,
    *,
    block_rows: int = 256,
) -> float:
    """
    CN: max_ij (u_i + v_j - C_ij)，分块计算以免额外物化第二个 m x n 矩阵。
    EN: max_ij (u_i + v_j - C_ij), computed blockwise to avoid materialising a second
        m x n matrix.
    """
    worst = -np.inf
    rows = int(cost.shape[0])
    step = max(1, int(block_rows))
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        block = source_dual[start:stop, None] + target_dual[None, :] - cost[start:stop]
        worst = max(worst, float(np.max(block)))
    return float(worst)


@dataclass(frozen=True)
class DualCertificate:
    """
    CN: 观测方案 pi_hat 的独立对偶证书：POT 返回的 (u, v) 是否对偶可行，且在 supp(pi_hat)
        上取等。不依赖 POT 的收敛标志，因此不被迭代上限蒙混。
    EN: Independent dual certificate for the observed plan pi_hat: whether POT's (u, v) are
        dual feasible and tight on supp(pi_hat). It does not rely on POT's convergence flag,
        so an iteration cap cannot make it pass silently.
    """

    certified: bool
    max_dual_violation: float
    support_residual: float
    result_code: int
    warning: str | None


def dual_optimality_certificate(
    problem: InverseOTProblem,
    theta: np.ndarray,
    *,
    num_itermax: int = _DEFAULT_VALIDATION_NUM_ITERMAX,
    feasibility_tol: float = 1e-7,
    tightness_tol: float = 1e-7,
    max_elements: int = 50_000_000,
) -> DualCertificate:
    """
    CN: 用 c(theta) 的对偶势验证 pi_hat 最优：要求 u_i + v_j <= C_ij（容差内）对所有边，
        且 u_i + v_j == C_ij 在 supp(pi_hat) 上。这两条成立即由弱对偶证明 pi_hat 最优，
        无需信任求解器的收敛声明。
    EN: Certify pi_hat's optimality from dual potentials of c(theta): require
        u_i + v_j <= C_ij (within tolerance) on every edge, and u_i + v_j == C_ij on
        supp(pi_hat). These two conditions prove pi_hat optimal by weak duality, without
        trusting the solver's convergence claim.
    """
    value = np.asarray(theta, dtype=np.float64).reshape(-1)
    if value.size != int(problem.n_features):
        raise ValueError("theta length does not match features.n_features")
    cost = dense_cost_matrix(problem.features, value, max_elements=int(max_elements))
    source_mass = problem.resolved_source_mass
    target_mass = problem.resolved_target_mass
    if abs(float(source_mass.sum()) - float(target_mass.sum())) > 1e-8:
        raise InverseOTError("source and target masses must have equal total mass")
    _transport, log = ot.emd(
        source_mass, target_mass, cost, numItermax=int(num_itermax), log=True
    )
    source_dual = np.asarray(log["u"], dtype=np.float64).reshape(-1)
    target_dual = np.asarray(log["v"], dtype=np.float64).reshape(-1)
    if source_dual.size != int(problem.n_source) or target_dual.size != int(problem.n_target):
        raise InverseOTError("POT returned dual potentials of unexpected shape")
    max_dual_violation = _max_dual_violation(source_dual, target_dual, cost)
    rows, cols, _values = problem.support()
    if rows.size:
        support_residual = float(
            np.max(np.abs(source_dual[rows] + target_dual[cols] - cost[rows, cols]))
        )
    else:
        support_residual = 0.0
    certified = bool(
        max_dual_violation <= float(feasibility_tol)
        and support_residual <= float(tightness_tol)
    )
    return DualCertificate(
        certified=certified,
        max_dual_violation=max_dual_violation,
        support_residual=support_residual,
        result_code=int(log["result_code"]),
        warning=log["warning"],
    )


__all__ = [
    "DualCertificate",
    "OptimalityCheck",
    "ThetaRange",
    "dual_optimality_certificate",
    "fixed_dual_theta_ranges",
    "is_optimal_plan",
    "optimal_theta_ranges",
    "plan_cost",
]