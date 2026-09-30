"""
CN: 支撑森林消元 master（**新增选项，非默认**）。在支撑图 G_S 的每个连通分量上取一棵生成树，
    把 f, g 表示为 `(theta, tau)` 的仿射函数（tau 为每个分量一个自由平移），代回 S 外的不等式，
    得到只含 `(delta, theta, tau)` 的约化 LP：变量数 `1 + K + p`（p 为 G_S 的分量数），
    而非原来的 `1 + K + m + n`。
EN: Spanning-forest-eliminated master (**new option, not the default**). On each connected
    component of the support graph G_S we take a spanning tree and write f, g as affine
    functions of `(theta, tau)` (one free shift tau per component); substituting into the
    off-S inequalities yields a reduced LP in `(delta, theta, tau)` with `1 + K + p` variables
    (p = number of components of G_S), instead of `1 + K + m + n`.

CN: 这是可逆的仿射重参数化，可行集在 (delta, theta) 上的投影、最优值 delta* 与证书都与原
    master 一致，因此不改变证明。退化时可能返回不同的最优顶点；`lex` 的顺序在约化坐标下不同。
    **对任意 `LinearThetaSet` 成立，不限于 diagonal weights。**
EN: This is an invertible affine reparameterisation: the (delta, theta) projection of the
    feasible set, the optimal value delta* and the certificate are identical to the full master,
    so the proof is unchanged. When degenerate it may return a different optimal vertex, and the
    `lex` coordinate order differs in reduced coordinates. It works for any `LinearThetaSet`,
    not only diagonal weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix

from .edges import EdgeSet
from .master import MasterSolution, draw_generic_direction
from .problem import InverseOTError, InverseOTProblem
from .theta import LinearThetaSet


@dataclass(frozen=True)
class ForestRepresentation:
    """
    CN: 支撑森林的仿射表示：节点 node 的势 = `A[node]·theta + s[node]·tau[comp[node]]`。
    EN: Affine representation from the support forest: node potential
        `A[node]·theta + s[node]·tau[comp[node]]`.
    """

    coefficients: np.ndarray  # (m+n, K)
    signs: np.ndarray  # (m+n,)
    component: np.ndarray  # (m+n,)
    n_components: int


def build_forest_representation(problem: InverseOTProblem) -> ForestRepresentation:
    """
    CN: 由 S 构造生成森林，并给出每个节点的 `(A, s, comp)`。
    EN: Build the spanning forest of S and return `(A, s, comp)` for every node.
    """
    m = int(problem.n_source)
    n = int(problem.n_target)
    k = int(problem.n_features)
    support_rows, support_cols, _values = problem.support()
    support_features = problem.features.features(support_rows, support_cols)
    total = m + n

    parent = np.arange(total, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(total)]
    for edge_index in range(support_rows.size):
        i = int(support_rows[edge_index])
        j = m + int(support_cols[edge_index])
        adjacency[i].append((j, edge_index))
        adjacency[j].append((i, edge_index))
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    coefficients = np.zeros((total, k), dtype=np.float64)
    signs = np.zeros(total, dtype=np.float64)
    component = np.full(total, -1, dtype=np.int64)
    n_components = 0
    for start in range(total):
        if component[start] != -1:
            continue
        component[start] = n_components
        coefficients[start] = 0.0
        signs[start] = 1.0
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor, edge_index in adjacency[node]:
                if component[neighbor] != -1:
                    continue
                # CN: 边等式 node + neighbor = c_e(theta)，故 neighbor = Phi_e·theta − node。
                # EN: node + neighbor = c_e(theta), so neighbor = Phi_e·theta − node.
                coefficients[neighbor] = support_features[edge_index] - coefficients[node]
                signs[neighbor] = -signs[node]
                component[neighbor] = n_components
                stack.append(neighbor)
        n_components += 1
    return ForestRepresentation(
        coefficients=coefficients, signs=signs, component=component, n_components=n_components
    )


class SpanningForestMasterSolver:
    """
    CN: 约化 master 求解器（新增选项）。单次求解；目标默认与 `RegularizedMasterSolver` 相同
        `delta + eps*<rtilde,(theta,f,g)>`（epsilon=0 时即纯 max delta）。对任意
        `LinearThetaSet` 成立。
    EN: Reduced master solver (new option). Single solve; by default the objective matches
        `RegularizedMasterSolver`, `delta + eps*<rtilde,(theta,f,g)>` (pure max delta when
        epsilon=0). Works for any `LinearThetaSet`.
    """

    def __init__(self, *, epsilon: float = 1e-4, random_seed: int = 0) -> None:
        self.epsilon = float(epsilon)
        self.random_seed = int(random_seed)
        # CN: 计时累加器（与 warm-start solver 一致，供基准对比）。
        # EN: Timing accumulators (consistent with the warm-start solver, for benchmarking).
        self.lp_seconds: float = 0.0
        self.n_solves: int = 0

    def solve(
        self,
        problem: InverseOTProblem,
        edges: EdgeSet,
        *,
        tie_break: str = "none",
    ) -> MasterSolution:
        del tie_break
        if not isinstance(problem.theta_set, LinearThetaSet):
            raise InverseOTError(
                "the forest-eliminated master requires a LinearThetaSet; "
                f"got {type(problem.theta_set).__name__}"
            )
        m = int(problem.n_source)
        n = int(problem.n_target)
        k = int(problem.n_features)
        representation = build_forest_representation(problem)
        p = representation.n_components
        delta_col = 0
        theta_start = 1
        tau_start = theta_start + k
        size = tau_start + p

        theta_constraints = problem.theta_set.linear_constraints()
        # CN: 等式 = theta 的线性等式 + 规范 g_0 = 0。
        # EN: Equalities = theta linear constraints + the gauge g_0 = 0.
        n_eq = int(theta_constraints.a_eq.shape[0]) + 1
        a_eq = lil_matrix((n_eq, size), dtype=np.float64)
        b_eq = np.zeros(n_eq, dtype=np.float64)
        cursor = 0
        for row in range(int(theta_constraints.a_eq.shape[0])):
            a_eq[cursor, theta_start : theta_start + k] = theta_constraints.a_eq[row]
            b_eq[cursor] = float(theta_constraints.b_eq[row])
            cursor += 1
        gauge_node = m  # target index 0
        a_eq[cursor, theta_start : theta_start + k] = representation.coefficients[gauge_node]
        a_eq[cursor, tau_start + int(representation.component[gauge_node])] = float(
            representation.signs[gauge_node]
        )
        b_eq[cursor] = 0.0

        support_keys = problem.support_keys()
        in_support = np.isin(edges.keys, support_keys)
        off_rows = edges.rows[~in_support]
        off_cols = edges.cols[~in_support]
        off_count = int(off_rows.size)
        a_ub = lil_matrix((off_count, size), dtype=np.float64)
        b_ub = np.zeros(off_count, dtype=np.float64)
        off_features = (
            problem.features.features(off_rows, off_cols)
            if off_count
            else np.zeros((0, k), dtype=np.float64)
        )
        coeff = representation.coefficients
        signs = representation.signs
        components = representation.component
        for position in range(off_count):
            i = int(off_rows[position])
            j = m + int(off_cols[position])
            a_ub[position, delta_col] = 1.0
            a_ub[position, theta_start : theta_start + k] = coeff[i] + coeff[j] - off_features[position]
            a_ub[position, tau_start + int(components[i])] += float(signs[i])
            a_ub[position, tau_start + int(components[j])] += float(signs[j])

        objective = np.zeros(size, dtype=np.float64)
        objective[delta_col] = -1.0
        if self.epsilon:
            direction = draw_generic_direction(
                random_seed=self.random_seed, n_features=k, n_source=m, n_target=n
            )
            scale = float(np.abs(direction).max()) or 1.0
            rtheta = direction[:k] / scale
            rsource = direction[k : k + m] / scale
            rtarget = direction[k + m :] / scale
            pull_theta = rtheta + rsource @ coeff[:m] + rtarget @ coeff[m:]
            pull_tau = np.zeros(p, dtype=np.float64)
            np.add.at(pull_tau, components[:m], rsource * signs[:m])
            np.add.at(pull_tau, components[m:], rtarget * signs[m:])
            objective[theta_start : theta_start + k] -= self.epsilon * pull_theta
            objective[tau_start:] -= self.epsilon * pull_tau

        lower = np.empty(size, dtype=np.float64)
        upper = np.empty(size, dtype=np.float64)
        lower[delta_col], upper[delta_col] = 0.0, np.inf
        lower[theta_start : theta_start + k] = theta_constraints.lower
        upper[theta_start : theta_start + k] = theta_constraints.upper
        lower[tau_start:] = -np.inf
        upper[tau_start:] = np.inf
        bounds = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]

        started = time.perf_counter()
        result = linprog(
            objective,
            A_ub=csr_matrix(a_ub) if off_count else None,
            b_ub=b_ub if off_count else None,
            A_eq=csr_matrix(a_eq) if n_eq else None,
            b_eq=b_eq if n_eq else None,
            bounds=bounds,
            method="highs",
        )
        self.lp_seconds += time.perf_counter() - started
        self.n_solves += 1
        if not result.success or result.x is None:
            raise InverseOTError(f"forest master is infeasible or unbounded: {result.message}")
        x = np.asarray(result.x, dtype=np.float64)
        theta = np.ascontiguousarray(x[theta_start : theta_start + k])
        tau = x[tau_start:]
        source_dual = np.ascontiguousarray(coeff[:m] @ theta + signs[:m] * tau[components[:m]])
        target_dual = np.ascontiguousarray(coeff[m:] @ theta + signs[m:] * tau[components[m:]])
        return MasterSolution(
            delta=float(x[delta_col]),
            theta=theta,
            source_dual=source_dual,
            target_dual=target_dual,
            status=str(result.message),
        )


__all__ = [
    "ForestRepresentation",
    "SpanningForestMasterSolver",
    "build_forest_representation",
]
