"""
CN: 谱面体 Theta 的半正定规划 master（cvxpy 为可选依赖）。
EN: Semidefinite-programming master for the spectrahedron Theta (cvxpy is an optional
    dependency).

CN: 该 master 与 `LinearMasterSolver` 实现同一 `MasterSolver` 协议，循环无需改动。
EN: This master implements the same `MasterSolver` protocol as `LinearMasterSolver`, so
    the loop is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .edges import EdgeSet
from .features import BilinearPSDFeatures
from .master import MasterSolution, TieBreak
from .problem import InverseOTError, InverseOTProblem
from .theta import BilinearPSDTheta


def _require_cvxpy() -> Any:
    """
    CN: 惰性加载 cvxpy，缺失时给出安装提示。
    EN: Load cvxpy lazily and give an install hint when it is missing.
    """
    try:
        import cvxpy as cp
    except ImportError as error:
        raise InverseOTError(
            "SdpMasterSolver requires cvxpy; install it with `pip install cvxpy` "
            "(the optional 'psd' extra)."
        ) from error
    return cp


@dataclass(frozen=True)
class SdpMasterSolver:
    """
    CN: bilinear-PSD Theta 的 master：A 半正定、Tr A = 1，支持 lex / none。
    EN: Master for the bilinear-PSD Theta: A positive semidefinite, Tr A = 1, with
        lex / none tie-breaking.
    """

    solver: str | None = None

    def solve(
        self,
        problem: InverseOTProblem,
        edges: EdgeSet,
        *,
        tie_break: TieBreak = "lex",
    ) -> MasterSolution:
        if tie_break not in {"lex", "none"}:
            raise ValueError("SdpMasterSolver supports tie_break 'lex' or 'none'")
        if not isinstance(problem.features, BilinearPSDFeatures):
            raise InverseOTError("SdpMasterSolver requires BilinearPSDFeatures")
        if not isinstance(problem.theta_set, BilinearPSDTheta):
            raise InverseOTError("SdpMasterSolver requires BilinearPSDTheta")
        dim = int(problem.features.point_dimension)
        if int(problem.theta_set.point_dimension) != dim:
            raise InverseOTError("feature and theta point dimensions disagree")
        cp = _require_cvxpy()
        source = problem.features.source_points
        target = problem.features.target_points
        m, n = int(problem.n_source), int(problem.n_target)

        matrix = cp.Variable((dim, dim), symmetric=True)
        source_dual = cp.Variable(m)
        target_dual = cp.Variable(n)
        delta = cp.Variable()
        constraints = [matrix >> 0, cp.trace(matrix) == 1.0, target_dual[0] == 0.0, delta >= 0.0]
        support_keys = problem.support_keys()
        in_support = np.isin(edges.keys, support_keys)
        for index in range(int(edges.size)):
            row = int(edges.rows[index])
            col = int(edges.cols[index])
            outer = np.outer(source[row], target[col])
            cost = -cp.sum(cp.multiply(matrix, outer))
            if bool(in_support[index]):
                constraints.append(source_dual[row] + target_dual[col] == cost)
            else:
                constraints.append(source_dual[row] + target_dual[col] + delta <= cost)

        master = cp.Problem(cp.Maximize(delta), constraints)
        self._solve_or_raise(cp, master, "SDP master")
        delta_star = float(delta.value)
        if tie_break == "none":
            return self._extract(matrix, source_dual, target_dual, delta)

        # CN: 先固定 delta，再按 (A, f, g) 的顺序做字典序最大化。
        # CN: 用下界固定而非等式，避免不精确 SDP 解导致的不可行（ε-lex 代理）。
        # EN: Fix delta, then lexicographically maximize in the order (A, f, g).
        # EN: Fix with a lower bound rather than an equality to avoid infeasibility from
        #     inaccurate SDP solves (an epsilon-lex surrogate).
        constraints.append(delta >= delta_star - 1e-6 * (1.0 + abs(delta_star)))
        lex_variables: list[Any] = []
        for row in range(dim):
            for col in range(row, dim):
                lex_variables.append(matrix[row, col])
        for row in range(m):
            lex_variables.append(source_dual[row])
        for col in range(1, n):
            lex_variables.append(target_dual[col])
        for variable in lex_variables:
            step = cp.Problem(cp.Maximize(variable), constraints)
            self._solve_or_raise(cp, step, "SDP lex step")
            value = float(np.asarray(variable.value).reshape(-1)[0])
            constraints.append(variable >= value - 1e-6 * (1.0 + abs(value)))
        return self._extract(matrix, source_dual, target_dual, delta)

    @staticmethod
    def _preferred_solver(cp: Any, requested: str | None) -> str | None:
        if requested is not None:
            return requested
        return "CLARABEL" if "CLARABEL" in cp.installed_solvers() else None

    def _solve_or_raise(self, cp: Any, master: Any, label: str) -> None:
        solver = self._preferred_solver(cp, self.solver)
        try:
            master.solve(solver=solver)
        except Exception as error:  # pragma: no cover - solver-specific failures
            if solver is not None:
                try:
                    master.solve()
                except Exception as retry_error:
                    raise InverseOTError(f"{label} failed to solve: {retry_error}") from retry_error
            else:
                raise InverseOTError(f"{label} failed to solve: {error}") from error
        if master.status not in {"optimal", "optimal_inaccurate"}:
            raise InverseOTError(f"{label} is infeasible or failed: status={master.status}")

    @staticmethod
    def _extract(matrix: Any, source_dual: Any, target_dual: Any, delta: Any) -> MasterSolution:
        return MasterSolution(
            delta=float(delta.value),
            theta=np.ascontiguousarray(np.asarray(matrix.value, dtype=np.float64).reshape(-1)),
            source_dual=np.ascontiguousarray(np.asarray(source_dual.value, dtype=np.float64).reshape(-1)),
            target_dual=np.ascontiguousarray(np.asarray(target_dual.value, dtype=np.float64).reshape(-1)),
            status="optimal",
        )


__all__ = [
    "SdpMasterSolver",
]
