"""
CN: 边集上的 master 问题（spec 第 2 节）及其 LP 求解器。
EN: The master problem on an edge set (spec section 2) and its LP solver.

CN: master 通过 `MasterSolver` 协议访问，便于将来替换为 SDP / 非线性代价后端。
EN: The master is accessed through the `MasterSolver` protocol so that a future
    SDP / non-linear cost backend can replace the LP without touching the loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix, vstack

from .edges import EdgeSet
from .problem import InverseOTError, InverseOTProblem
from .theta import LinearThetaSet

# CN: "lex_generic" 是 spec/证明第 6 节的两步 tie-break：先最大化 delta，再在
#     delta = delta* 的面上最大化一个整轮固定、运行开始时抽取一次的通用方向 r。
# EN: "lex_generic" is the two-solve tie-break of proof section 6: maximise delta, then
#     maximise one generic direction r on the face delta = delta*, with r drawn once for
#     the whole run and held fixed.
TieBreak = Literal["lex", "random", "none", "lex_generic"]
_DEFAULT_RANDOM_SEED = 0


@dataclass(frozen=True)
class MasterSolution:
    """
    CN: master 的解 (delta, theta, f, g)；sigma_ij = f_i + g_j - c_ij(theta)。
    EN: Master solution (delta, theta, f, g); sigma_ij = f_i + g_j - c_ij(theta).
    """

    delta: float
    theta: np.ndarray
    source_dual: np.ndarray
    target_dual: np.ndarray
    status: str

    @property
    def objective(self) -> float:
        return float(self.delta)


@runtime_checkable
class MasterSolver(Protocol):
    """
    CN: 在给定边集上求解 master 的协议。
    EN: Protocol for solving the master on a given edge set.
    """

    def solve(
        self,
        problem: InverseOTProblem,
        edges: EdgeSet,
        *,
        tie_break: TieBreak = "none",
    ) -> MasterSolution:
        """CN: 求解 master。EN: Solve the master."""
        ...


@dataclass(frozen=True)
class _VariableLayout:
    delta: int
    theta_start: int
    source_start: int
    target_start: int
    size: int


@dataclass(frozen=True)
class _MasterSystem:
    a_ub: csr_matrix
    b_ub: np.ndarray
    a_eq: csr_matrix
    b_eq: np.ndarray
    bounds: list[tuple[float | None, float | None]]
    layout: _VariableLayout


def _variable_layout(n_source: int, n_target: int, n_features: int) -> _VariableLayout:
    theta_start = 1
    source_start = theta_start + int(n_features)
    target_start = source_start + int(n_source)
    size = target_start + int(n_target)
    return _VariableLayout(
        delta=0,
        theta_start=theta_start,
        source_start=source_start,
        target_start=target_start,
        size=size,
    )


def _finite_bound(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def build_master_system(problem: InverseOTProblem, edges: EdgeSet) -> _MasterSystem:
    """
    CN: 由 F 上的线性特征构造 master 的稀疏 LP 系统。
    EN: Build the sparse LP system of the master from the linear features on F.
    """
    if edges.size < 1:
        raise InverseOTError("master requires a non-empty edge set")
    if not isinstance(problem.theta_set, LinearThetaSet):
        raise InverseOTError(
            "the LP master requires a LinearThetaSet; "
            f"got {type(problem.theta_set).__name__}. Pair a non-linear Theta with a "
            "dedicated master solver."
        )
    layout = _variable_layout(int(problem.n_source), int(problem.n_target), int(problem.n_features))
    support_keys = problem.support_keys()
    in_support = np.isin(edges.keys, support_keys)
    feature_matrix = problem.features.features(edges.rows, edges.cols)
    if feature_matrix.shape != (edges.size, int(problem.n_features)):
        raise InverseOTError("features() returned an unexpected number of feature columns")

    n_source = int(problem.n_source)
    equality_rows = int(np.count_nonzero(in_support))
    inequality_rows = int(edges.size - equality_rows)
    theta_constraints = problem.theta_set.linear_constraints()
    n_eq = equality_rows + theta_constraints.a_eq.shape[0] + 1  # +1 for g[0] = 0
    n_ub = inequality_rows

    a_eq = lil_matrix((n_eq, layout.size), dtype=np.float64)
    b_eq = np.zeros(n_eq, dtype=np.float64)
    a_ub = lil_matrix((n_ub, layout.size), dtype=np.float64)
    b_ub = np.zeros(n_ub, dtype=np.float64)

    eq_cursor = 0
    ub_cursor = 0
    for edge_index in range(edges.size):
        row = int(edges.rows[edge_index])
        col = int(edges.cols[edge_index])
        feature_row = feature_matrix[edge_index]
        if bool(in_support[edge_index]):
            a_eq[eq_cursor, layout.source_start + row] = 1.0
            a_eq[eq_cursor, layout.target_start + col] = 1.0
            a_eq[eq_cursor, layout.theta_start : layout.theta_start + int(problem.n_features)] = -feature_row
            b_eq[eq_cursor] = 0.0
            eq_cursor += 1
        else:
            a_ub[ub_cursor, layout.delta] = 1.0
            a_ub[ub_cursor, layout.source_start + row] = 1.0
            a_ub[ub_cursor, layout.target_start + col] = 1.0
            a_ub[ub_cursor, layout.theta_start : layout.theta_start + int(problem.n_features)] = -feature_row
            b_ub[ub_cursor] = 0.0
            ub_cursor += 1

    # CN: 嵌入 Theta 的线性等式约束。
    # EN: Embed the linear equality constraints of Theta.
    theta_block = theta_constraints.a_eq
    for local_row in range(theta_block.shape[0]):
        a_eq[eq_cursor, layout.theta_start : layout.theta_start + int(problem.n_features)] = theta_block[local_row]
        b_eq[eq_cursor] = float(theta_constraints.b_eq[local_row])
        eq_cursor += 1
    # CN: 规范固定 g[0] = 0。
    # EN: Gauge fixing g[0] = 0.
    a_eq[eq_cursor, layout.target_start] = 1.0
    b_eq[eq_cursor] = 0.0
    eq_cursor += 1

    bounds: list[tuple[float | None, float | None]] = [(0.0, None)]
    for index in range(int(problem.n_features)):
        bounds.append(
            (
                _finite_bound(float(theta_constraints.lower[index])),
                _finite_bound(float(theta_constraints.upper[index])),
            )
        )
    bounds.extend([(None, None)] * (n_source + int(problem.n_target)))
    if len(bounds) != layout.size:
        raise InverseOTError("internal error: variable layout and bounds disagree")

    return _MasterSystem(
        a_ub=csr_matrix(a_ub),
        b_ub=b_ub,
        a_eq=csr_matrix(a_eq),
        b_eq=b_eq,
        bounds=bounds,
        layout=layout,
    )


def _solve_system(
    system: _MasterSystem,
    objective: np.ndarray,
    *,
    extra_eq: list[np.ndarray] | None = None,
    extra_eq_rhs: list[float] | None = None,
    extra_ub: list[np.ndarray] | None = None,
    extra_ub_rhs: list[float] | None = None,
    bounds: list[tuple[float | None, float | None]] | None = None,
) -> tuple[np.ndarray | None, float | None, str]:
    a_eq = system.a_eq
    b_eq = system.b_eq
    a_ub = system.a_ub
    b_ub = system.b_ub
    if extra_eq:
        a_eq = vstack([a_eq, csr_matrix(np.vstack(extra_eq))], format="csr")
        b_eq = np.concatenate([b_eq, np.asarray(extra_eq_rhs, dtype=np.float64)])
    if extra_ub:
        a_ub = vstack([a_ub, csr_matrix(np.vstack(extra_ub))], format="csr")
        b_ub = np.concatenate([b_ub, np.asarray(extra_ub_rhs, dtype=np.float64)])
    # CN: scipy 的 `spmatrix.shape` 属性没有类型标注，pyright 会误判为 Optional；此处显式收敛。
    # EN: scipy's `spmatrix.shape` property is untyped; pyright wrongly sees Optional here.
    a_ub_shape = cast("tuple[int, int]", a_ub.shape)
    a_eq_shape = cast("tuple[int, int]", a_eq.shape)
    result = linprog(
        np.asarray(objective, dtype=np.float64),
        A_ub=a_ub if a_ub_shape[0] else None,
        b_ub=b_ub if a_ub_shape[0] else None,
        A_eq=a_eq if a_eq_shape[0] else None,
        b_eq=b_eq if a_eq_shape[0] else None,
        bounds=system.bounds if bounds is None else bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        return None, None, str(result.message)
    return np.asarray(result.x, dtype=np.float64), float(result.fun), str(result.message)


def _unit_row(size: int, index: int, value: float = 1.0) -> np.ndarray:
    row = np.zeros(size, dtype=np.float64)
    row[index] = float(value)
    return row


def _extract(solution: np.ndarray, layout: _VariableLayout, status: str) -> MasterSolution:
    return MasterSolution(
        delta=float(solution[layout.delta]),
        theta=np.ascontiguousarray(solution[layout.theta_start : layout.source_start]),
        source_dual=np.ascontiguousarray(solution[layout.source_start : layout.target_start]),
        target_dual=np.ascontiguousarray(solution[layout.target_start :]),
        status=status,
    )


def draw_generic_direction(
    *,
    random_seed: int,
    n_features: int,
    n_source: int,
    n_target: int,
) -> np.ndarray:
    """
    CN: 为 `lex_generic` 抽取整轮固定的 tie-break 方向 r ~ N(0, I)，覆盖 (theta, f, g)。
    EN: Draw the run-fixed tie-break direction r ~ N(0, I) over (theta, f, g) used by
        `lex_generic`.

    CN: 证明第 6 节只在 Lebesgue 零测集上取坏方向，因此一次高斯抽取以概率 1 有效；
        调用方应在整轮开始前只调用一次，并把 seed/向量记录下来。
    EN: Proof section 6 excludes only a Lebesgue-null set of bad directions, so one
        Gaussian draw is valid with probability 1; callers should draw it exactly once
        before the run and log the seed/vector.
    """
    size = int(n_features) + int(n_source) + int(n_target)
    if size < 1:
        raise ValueError("draw_generic_direction needs at least one coordinate")
    rng = np.random.default_rng(int(random_seed))
    return np.ascontiguousarray(rng.standard_normal(size), dtype=np.float64)


def coordinate_extrema(
    problem: InverseOTProblem,
    edges: EdgeSet,
    *,
    delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """
    CN: 在 delta 固定（默认 0）时，对每个 theta 坐标求 min/max，即可辨识性范围。
    EN: With delta fixed (default 0), minimize/maximize each theta coordinate, i.e. the
        identifiability range.

    CN: delta 只出现在不等式里，把它设为 0 只会放松约束，因此最优 theta 集合不变。
    EN: delta only appears in inequalities, so setting it to 0 only relaxes them and
        leaves the set of optimal theta unchanged.
    """
    system = build_master_system(problem, edges)
    bounds = list(system.bounds)
    bounds[system.layout.delta] = (float(delta), float(delta))
    dimension = int(problem.n_features)
    lower = np.empty(dimension, dtype=np.float64)
    upper = np.empty(dimension, dtype=np.float64)
    statuses: list[str] = []
    for index in range(dimension):
        variable = system.layout.theta_start + index
        low, _low_value, low_status = _solve_system(
            system, _unit_row(system.layout.size, variable, 1.0), bounds=bounds
        )
        high, _high_value, high_status = _solve_system(
            system, _unit_row(system.layout.size, variable, -1.0), bounds=bounds
        )
        if low is None or high is None:
            raise InverseOTError(
                f"theta coordinate {index} range failed: min={low_status!r}, max={high_status!r}"
            )
        lower[index] = float(low[variable])
        upper[index] = float(high[variable])
        statuses.append(f"min={low_status}; max={high_status}")
    return lower, upper, tuple(statuses)


@dataclass(frozen=True)
class LinearMasterSolver:
    """
    CN: 线性代价的 master LP 求解器，支持 lex / random / none / lex_generic 四种 tie-breaking。
    EN: Master LP solver for linear costs, with lex / random / none / lex_generic
        tie-breaking.

    CN: `lex_generic` 实现证明第 6 节的两步构造：先最大 delta，再加等式 delta = delta*
        并在该面上最大化固定的通用方向 r。`generic_direction` 为 None 时按 `random_seed`
        抽取（同一 seed 必得同一 r）；调用方可显式传入，以便整轮只抽一次并记录。
    EN: `lex_generic` implements the two-solve construction of proof section 6: maximise
        delta, add the equality delta = delta*, then maximise the fixed generic direction
        r on that face. With `generic_direction=None` it is drawn from `random_seed` (the
        same seed always yields the same r); callers may pass it explicitly so that it is
        drawn exactly once per run and can be logged.
    """

    random_seed: int = _DEFAULT_RANDOM_SEED
    generic_direction: tuple[float, ...] | None = field(default=None, repr=False)

    def solve(
        self,
        problem: InverseOTProblem,
        edges: EdgeSet,
        *,
        tie_break: TieBreak = "none",
    ) -> MasterSolution:
        if tie_break not in {"lex", "random", "none", "lex_generic"}:
            raise ValueError("tie_break must be one of: lex, random, none, lex_generic")
        system = build_master_system(problem, edges)
        layout = system.layout
        objective = _unit_row(layout.size, layout.delta, -1.0)
        solution, value, status = _solve_system(system, objective)
        if solution is None or value is None:
            raise InverseOTError(f"master problem is infeasible or unbounded: {status}")
        delta_star = -float(value)
        if tie_break == "none":
            return _extract(solution, layout, status)

        extra_eq = [_unit_row(layout.size, layout.delta)]
        extra_eq_rhs = [delta_star]
        if tie_break == "lex":
            order = list(range(layout.theta_start, layout.target_start + int(problem.n_target)))
            for variable in order:
                direction = _unit_row(layout.size, variable, -1.0)
                chosen, chosen_value, chosen_status = _solve_system(
                    system, direction, extra_eq=extra_eq, extra_eq_rhs=extra_eq_rhs
                )
                if chosen is None or chosen_value is None:
                    raise InverseOTError(f"lexicographic master step failed: {chosen_status}")
                extra_eq.append(_unit_row(layout.size, variable))
                extra_eq_rhs.append(float(chosen[variable]))
                solution = chosen
        elif tie_break == "lex_generic":
            direction = np.zeros(layout.size, dtype=np.float64)
            if self.generic_direction is None:
                direction[layout.theta_start :] = draw_generic_direction(
                    random_seed=int(self.random_seed),
                    n_features=int(problem.n_features),
                    n_source=int(problem.n_source),
                    n_target=int(problem.n_target),
                )
            else:
                provided = np.asarray(self.generic_direction, dtype=np.float64).reshape(-1)
                expected = layout.size - layout.theta_start
                if provided.size != expected:
                    raise ValueError(
                        "generic_direction must have n_features + n_source + n_target "
                        f"entries ({expected}); got {provided.size}"
                    )
                direction[layout.theta_start :] = provided
            chosen, chosen_value, chosen_status = _solve_system(
                system,
                -direction,
                extra_eq=extra_eq,
                extra_eq_rhs=extra_eq_rhs,
            )
            if chosen is None or chosen_value is None:
                raise InverseOTError(f"lex_generic master step failed: {chosen_status}")
            solution = chosen
        else:
            rng = np.random.default_rng(int(self.random_seed))
            direction = np.zeros(layout.size, dtype=np.float64)
            direction[layout.theta_start :] = rng.standard_normal(layout.size - layout.theta_start)
            if not np.any(direction):
                direction[layout.delta] = 0.0
                direction[layout.theta_start] = 1.0
            chosen, chosen_value, chosen_status = _solve_system(
                system,
                -direction,
                extra_eq=extra_eq,
                extra_eq_rhs=extra_eq_rhs,
            )
            if chosen is None or chosen_value is None:
                raise InverseOTError(f"randomized master step failed: {chosen_status}")
            solution = chosen
        return _extract(solution, layout, status)


@dataclass(frozen=True)
class RegularizedMasterSolver:
    """
    CN: 固定线性正则化的 master：目标为 `delta + eps * <r, (theta, f, g)>`，r 在一次运行内固定。
    EN: Master with a fixed linear regularizer: objective
        `delta + eps * <r, (theta, f, g)>` with r fixed for the whole run.

    CN: 单次求解即可，接近「用通用泛函选出唯一面」，比每轮更换 r 更稳定。
    EN: One solve; approximates selecting a unique face by a generic functional, more
        stable than re-randomizing r each iteration.
    """

    epsilon: float = 1e-4
    random_seed: int = 0

    def solve(
        self,
        problem: InverseOTProblem,
        edges: EdgeSet,
        *,
        tie_break: TieBreak = "none",
    ) -> MasterSolution:
        del tie_break
        system = build_master_system(problem, edges)
        layout = system.layout
        rng = np.random.default_rng(int(self.random_seed))
        direction = np.zeros(layout.size, dtype=np.float64)
        direction[layout.theta_start :] = rng.standard_normal(layout.size - layout.theta_start)
        scale = float(np.abs(direction).max()) or 1.0
        objective = np.zeros(layout.size, dtype=np.float64)
        objective[layout.delta] = -1.0
        objective += -float(self.epsilon) * (direction / scale)
        solution, value, status = _solve_system(system, objective)
        if solution is None or value is None:
            raise InverseOTError(f"regularized master is infeasible or unbounded: {status}")
        return MasterSolution(
            delta=float(solution[layout.delta]),
            theta=np.ascontiguousarray(solution[layout.theta_start : layout.source_start]),
            source_dual=np.ascontiguousarray(solution[layout.source_start : layout.target_start]),
            target_dual=np.ascontiguousarray(solution[layout.target_start :]),
            status=status,
        )


class WarmStartMasterSolver:
    """
    CN: 增量式 HiGHS master（**新增选项，非默认**）。变量 (delta, theta, f, g)、S 上的等式、
        Theta 约束与规范固定只建一次；每次迭代只把 F\\S 不等式行在「激活 (upper=0) / 停用
        (upper=+inf)」之间切换，因此每次重解都从上一轮的 LP 基热启动，而不是像
        `LinearMasterSolver`/`RegularizedMasterSolver` 那样整表重建。
    EN: Incremental HiGHS master (**new option, not the default**). The variables
        (delta, theta, f, g), the S equalities, the Theta constraints and the gauge are built
        once; each iteration only toggles the F\\S inequality rows between active (upper=0) and
        inactive (upper=+inf), so every re-solve warm-starts from the previous LP basis instead
        of rebuilding the whole tableau like `LinearMasterSolver`/`RegularizedMasterSolver`.

    CN: 与 `RegularizedMasterSolver` 一样只做单次求解，目标默认 `delta + eps*<rtilde,(theta,f,g)>`
        （`epsilon=0` 时即纯 `max delta`）。对任何 `LinearThetaSet` 都成立，不限于 diagonal weights。
    EN: Like `RegularizedMasterSolver`, it performs a single solve per call, with objective
        `delta + eps*<rtilde,(theta,f,g)>` by default (pure `max delta` when `epsilon=0`). It
        works for any `LinearThetaSet`, not only diagonal weights.

    CN: 凸性/可行集/最优值与原 LP 完全一致，因此不改变证明；只在退化时可能返回不同的最优顶点，
        所以仅保证 delta 与证书一致，不保证 (f,g) 逐位相同。
    EN: The feasible set and optimal value are identical to the full LP, so the proof is
        unchanged; only when degenerate may it return a different optimal vertex, so we
        guarantee delta and the certificate, not bitwise-identical (f, g).
    """

    def __init__(
        self,
        *,
        epsilon: float = 1e-4,
        random_seed: int = 0,
        presolve: str = "choose",
        purge: bool = True,
    ) -> None:
        self.epsilon = float(epsilon)
        self.random_seed = int(random_seed)
        self.presolve = str(presolve)
        # CN: purge=True 时直接删除离开 F 的不等式行（保持模型行数 = |F\S|）；
        #     purge=False 时仅把行设为非激活（保留历史行，便于对照）。
        # EN: purge=True deletes inequality rows that leave F (model rows stay = |F\S|);
        #     purge=False only deactivates them (keeps history, for comparison).
        self.purge = bool(purge)
        # CN: highspy 无类型存根，模型句柄是窄互操作边界，类型为 Any 是刻意的。
        # EN: highspy has no type stubs; the model handle is a narrow interop boundary, so Any
        #     is deliberate here.
        self._highs: Any = None
        self._layout: _VariableLayout | None = None
        self._eq_count: int = 0
        self._key_row: dict[int, int] = {}
        self._active: set[int] = set()
        self._signature: tuple[int, int, int] | None = None
        self._last_status: str = ""
        # CN: 纯 `Highs.run()` 累计耗时与次数，便于与冷启的 `_solve_system` 计时对比。
        # EN: Accumulated pure `Highs.run()` time and count, comparable to the cold path's
        #     `_solve_system` timer.
        self.lp_seconds: float = 0.0
        self.n_solves: int = 0

    def solve(
        self,
        problem: InverseOTProblem,
        edges: EdgeSet,
        *,
        tie_break: TieBreak = "none",
    ) -> MasterSolution:
        del tie_break  # single-solve regime; the tie-break is not part of this option
        import highspy  # CN: 惰性导入，避免让 `import iot` 依赖 highspy。EN: lazy import.

        support_keys = problem.support_keys()
        in_support = np.isin(edges.keys, support_keys)
        nonsupport_keys = edges.keys[~in_support]
        signature = (int(problem.n_source), int(problem.n_target), int(problem.n_features))
        if self._highs is None or self._signature != signature:
            self._build(highspy, problem, edges, in_support, nonsupport_keys, signature)
        else:
            self._update(highspy, problem, nonsupport_keys)
        assert self._highs is not None and self._layout is not None
        solve_started = time.perf_counter()
        self._highs.run()
        self.lp_seconds += time.perf_counter() - solve_started
        self.n_solves += 1
        model_status = self._highs.getModelStatus()
        optimal = highspy.HighsModelStatus.kOptimal
        if model_status != optimal:
            name = self._highs.modelStatusToString(model_status)
            if model_status in (
                highspy.HighsModelStatus.kInfeasible,
                highspy.HighsModelStatus.kUnbounded,
                highspy.HighsModelStatus.kUnboundedOrInfeasible,
            ):
                raise InverseOTError(f"master problem is infeasible or unbounded: {name}")
            raise InverseOTError(f"warm-start master failed: {name}")
        solution = np.asarray(self._highs.getSolution().col_value, dtype=np.float64)
        return _extract(solution, self._layout, "optimal")

    @property
    def active_inequality_rows(self) -> int:
        """
        CN: 当前模型中处于激活状态的 F\\S 不等式行数（purge=True 时应等于 |F\\S|）。
        EN: Number of active F\\S inequality rows currently in the model (equals |F\\S| when
            purge=True).
        """
        return len(self._key_row)

    def _objective(self, layout: _VariableLayout, problem: InverseOTProblem) -> np.ndarray:
        objective = np.zeros(layout.size, dtype=np.float64)
        objective[layout.delta] = -1.0
        if self.epsilon:
            direction = draw_generic_direction(
                random_seed=self.random_seed,
                n_features=int(problem.n_features),
                n_source=int(problem.n_source),
                n_target=int(problem.n_target),
            )
            scale = float(np.abs(direction).max()) or 1.0
            objective[layout.theta_start :] -= self.epsilon * (direction / scale)
        return objective

    def _build(self, highspy, problem, edges, in_support, nonsupport_keys, signature) -> None:
        system = build_master_system(problem, edges)
        layout = system.layout
        self._layout = layout
        self._signature = signature
        model = highspy.Highs()
        model.setOptionValue("output_flag", False)
        if self.presolve != "choose":
            model.setOptionValue("presolve", self.presolve)
        bounds = system.bounds
        lower = np.array(
            [-np.inf if lo is None else float(lo) for lo, _ in bounds], dtype=np.float64
        )
        upper = np.array(
            [np.inf if hi is None else float(hi) for _, hi in bounds], dtype=np.float64
        )
        model.addCols(
            layout.size, self._objective(layout, problem), lower, upper, 0,
            np.zeros(layout.size + 1, dtype=np.int32),
            np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64),
        )
        a_eq = system.a_eq.tocsr()
        a_eq_shape = cast("tuple[int, int]", a_eq.shape)
        if a_eq_shape[0]:
            model.addRows(
                a_eq_shape[0], system.b_eq, system.b_eq, int(a_eq.nnz),
                a_eq.indptr.astype(np.int32), a_eq.indices.astype(np.int32),
                a_eq.data.astype(np.float64),
            )
        self._eq_count = int(a_eq_shape[0])
        a_ub = system.a_ub.tocsr()
        rows = int(cast("tuple[int, int]", a_ub.shape)[0])
        if rows:
            model.addRows(
                rows, np.full(rows, -np.inf), np.zeros(rows), int(a_ub.nnz),
                a_ub.indptr.astype(np.int32), a_ub.indices.astype(np.int32),
                a_ub.data.astype(np.float64),
            )
        self._key_row = {
            int(key): self._eq_count + position
            for position, key in enumerate(nonsupport_keys.tolist())
        }
        self._active = set(self._key_row)
        self._highs = model

    def _update(self, highspy, problem, nonsupport_keys) -> None:
        assert self._highs is not None and self._layout is not None
        target = set(int(key) for key in nonsupport_keys.tolist())
        if self.purge:
            removed = [key for key in self._key_row if key not in target]
            if removed:
                old_rows = np.array(sorted(self._key_row[key] for key in removed), dtype=np.int32)
                self._highs.deleteRows(int(old_rows.size), old_rows)
                retained: dict[int, int] = {}
                for key, row in self._key_row.items():
                    if key in target:
                        retained[key] = row - int(np.searchsorted(old_rows, row, side="left"))
                self._key_row = retained
            for key in sorted(target - set(self._key_row)):
                self._add_row(problem, key)
        else:
            for key in self._active - target:
                self._highs.changeRowBounds(self._key_row[key], -np.inf, np.inf)
            for key in sorted(target - self._active):
                if key in self._key_row:
                    self._highs.changeRowBounds(self._key_row[key], -np.inf, 0.0)
                else:
                    self._add_row(problem, key)
        self._active = target

    def _add_row(self, problem: InverseOTProblem, key: int) -> None:
        """CN: 为边 key 追加 `delta + f_i + g_j − c_ij(theta) <= 0` 一行。EN: Add one row."""
        assert self._highs is not None and self._layout is not None
        layout = self._layout
        n_target = int(problem.n_target)
        row = int(key) // n_target
        col = int(key) % n_target
        feature = problem.features.features(
            np.array([row], dtype=np.int64), np.array([col], dtype=np.int64)
        )[0]
        indices = np.concatenate(
            [
                np.array([layout.delta, layout.source_start + row, layout.target_start + col]),
                np.arange(layout.theta_start, layout.source_start, dtype=np.int64),
            ]
        ).astype(np.int32)
        values = np.concatenate([np.ones(3), -feature]).astype(np.float64)
        self._highs.addRow(-np.inf, 0.0, int(indices.size), indices, values)
        self._key_row[int(key)] = int(self._highs.getNumRow()) - 1


__all__ = [
    "LinearMasterSolver",
    "MasterSolution",
    "MasterSolver",
    "RegularizedMasterSolver",
    "TieBreak",
    "WarmStartMasterSolver",
    "build_master_system",
    "coordinate_extrema",
    "draw_generic_direction",
]
