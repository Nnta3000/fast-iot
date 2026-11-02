"""
CN: **松弛循环的两段式选择** —— 可选的第二阶段约束生成（`"central"` 模式）。

    CN: 默认 `objective.selection == "none"`，所以这个模块在默认路径上**不会执行**；它存在是为了
        "在 `R <= R* + eta` 上取中心点"这一变体。抽出来的原因纯粹是可读性：这两块占了
        原 `loop_regret.py` 的 200 行，而默认路径从不调用。

    CN: 两者都**不替换** stage-1 的解：调用方只把返回的 `max sigma` 当诊断量。

EN: **The relaxed loop's two-stage selection** -- the optional second constraint-generation phase
    (`"central"` mode).

    The default is `objective.selection == "none"`, so this module is **never executed on the default path**;
    it exists for the "central point on `R <= R* + eta`" variant. It was extracted purely for readability:
    these two blocks were 200 lines of `loop_regret.py` that the default path never calls.

    Neither **replaces** the stage-1 solution: callers use the returned `max sigma` as a diagnostic only.
"""

from __future__ import annotations

import numpy as np

from ..core.edges import EdgeSet
from ..core.problem import InverseOTProblem
from ..relaxed.master_regret import RegretMasterSolver

# CN: 内部名字 `_select_central_cg` 在模块内仍然使用；导出时给出公开名字。
# EN: the internal name `_select_central_cg` is kept inside the module; the public one is exported.
__all__ = ["select_central_cg", "select_nearest_uniform"]


def select_nearest_uniform(
    problem: InverseOTProblem,
    finite: EdgeSet,
    *,
    eta: float,
    tol: float,
    max_iterations: int = 30,
) -> np.ndarray:
    """
    CN: **中心化选择**（约束生成）：先解一次得 `R*`，再限制 `R <= R* + eta` 并最小化
        `||theta - 1/K||_1`；在**全图**上扫描、插入违例边，重复直到 `max sigma <= tol`。
        L1 用 `p, n >= 0` 线性化：`theta + n - p = 1/K`，目标 `sum(p) + sum(n)`。
    EN: **Central selection** (constraint generation): one solve gives `R*`; cap `R <= R* + eta` and
        minimise `||theta - 1/K||_1`; **full-graph** scan, insert violators, repeat until
        `max sigma <= tol`. The L1 is linearised with `p, n >= 0`, `theta + n - p = 1/K`, objective
        `sum(p) + sum(n)`.
    """
    from scipy import sparse as _sparse
    from scipy.optimize import linprog

    from ..backends.lp_common import LpLayout, repair_point
    from .master_regret import build_regret_lp

    m, n = int(problem.n_source), int(problem.n_target)
    k = int(problem.n_features)
    target = 1.0 / float(k)
    layout = LpLayout(has_delta=False, n_features=k, n_source=m, n_target=n)
    r_idx, c_idx = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    all_rows, all_cols = r_idx.reshape(-1), c_idx.reshape(-1)
    sup_rows, sup_cols, sup_vals = problem.support()
    current = finite
    objective = regret_objective(problem)
    theta = np.full(k, target)
    for _round in range(int(max_iterations)):
        system = build_regret_lp(problem, current)
        base = linprog(
            objective,
            A_ub=system.a_ub,
            b_ub=system.b_ub,
            A_eq=system.a_eq,
            b_eq=system.b_eq,
            bounds=system.bounds,
            method="highs",
        )
        if not base.success or base.x is None:
            raise RuntimeError(f"selection: base solve failed: {base.message!r}")
        bound = float(base.fun) + float(eta)
        size = system.size

        # CN: 右侧零块的行数必须与**对应矩阵的行数**一致（a_ub 是边数，不是变量数）。
        # EN: the right-hand zero blocks need as many rows as the matrix they extend (edges, not variables).
        zeros_ub = _sparse.csr_matrix((system.a_ub.shape[0], 2 * k))
        a_ub = _sparse.vstack(
            [
                _sparse.hstack([system.a_ub, zeros_ub], format="csr"),
                _sparse.csr_matrix(np.append(objective, np.zeros(2 * k))[None, :]),
            ],
            format="csr",
        )
        b_ub = np.concatenate([system.b_ub, [bound]])

        # CN: regret LP 无 delta 列，theta 从 0 开始（见 LpLayout(has_delta=False)）。
        # EN: the regret LP has no delta column, so theta starts at 0 (see LpLayout(has_delta=False)).
        rows_pn = np.zeros((k, size + 2 * k))
        for i in range(k):
            rows_pn[i, i] = 1.0  # theta_i
            rows_pn[i, size + i] = -1.0  # -p_i
            rows_pn[i, size + k + i] = 1.0  # +n_i
        a_eq = _sparse.vstack(
            [
                _sparse.hstack(
                    [system.a_eq, _sparse.csr_matrix((system.a_eq.shape[0], 2 * k))],
                    format="csr",
                ),
                _sparse.csr_matrix(rows_pn),
            ],
            format="csr",
        )
        b_eq = np.concatenate([system.b_eq, np.full(k, target)])
        # CN: theta + n - p = target  =>  p 的系数应为 -1（见上），n 为 +1。
        # EN: theta + n - p = target, hence coefficient -1 on p and +1 on n (set above).
        c_obj = np.concatenate([np.zeros(size), np.ones(2 * k)])
        res = linprog(
            c_obj,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=list(system.bounds) + [(0.0, None)] * (2 * k),
            method="highs",
        )
        if not res.success or res.x is None:
            raise RuntimeError(f"selection: central solve failed: {res.message!r}")
        repaired = repair_point(
            problem,
            np.asarray(res.x[:size], dtype=np.float64),
            layout,
            all_rows=all_rows,
            all_cols=all_cols,
            support_rows=sup_rows,
            support_cols=sup_cols,
            support_values=sup_vals,
            objective=objective,
        )
        theta = repaired.theta
        sigma_all = _sigma(
            problem,
            theta,
            repaired.source_dual,
            repaired.target_dual,
            all_rows,
            all_cols,
        )
        if float(sigma_all.max()) <= float(tol):
            return theta
        viol = sigma_all > float(tol)
        current = current.union(
            EdgeSet.from_arrays(all_rows[viol], all_cols[viol], n_target=n)
        )
    return theta


def select_central_cg(
    problem: InverseOTProblem,
    finite: EdgeSet,
    *,
    eta: float,
    tol: float,
    tau_prune: float,
    gamma: int,
    max_iterations: int,
    solver: RegretMasterSolver,
) -> float:
    """
    CN: **可选的第二阶段约束生成**：选择 LP 限制 `R <= R* + eta` 并最大化最小余量，然后在**全图**上扫描，
        插入违例边，重复直到 `max sigma <= tol` 或预算耗尽。返回最终 `max sigma`。
        **不替换** stage-1 的解；调用方只用它作诊断。
    EN: **Optional second constraint-generation phase**: the selection LP caps `R <= R* + eta` and maximises the
        minimum slack, then a **full-graph** scan inserts violators, repeating until `max sigma <= tol` or the
        budget is spent. Returns the final `max sigma`. It does **not** replace the stage-1 solution; callers
        use it as a diagnostic only.
    """
    from scipy import sparse as _sparse
    from scipy.optimize import linprog

    from .master_regret import build_regret_lp

    m, n = int(problem.n_source), int(problem.n_target)
    r_idx, c_idx = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    all_rows, all_cols = r_idx.reshape(-1), c_idx.reshape(-1)
    current = finite
    objective = regret_objective(problem)
    for _round in range(int(max_iterations)):
        system = build_regret_lp(problem, current)
        base = linprog(
            objective,
            A_ub=system.a_ub,
            b_ub=system.b_ub,
            A_eq=system.a_eq,
            b_eq=system.b_eq,
            bounds=system.bounds,
            method="highs",
        )
        if not base.success or base.x is None:
            raise RuntimeError(f"selection: base solve failed: {base.message!r}")
        bound = float(base.fun) + float(eta)
        size = system.size
        a_ub = _sparse.hstack(
            [system.a_ub, _sparse.csr_matrix(np.ones((system.a_ub.shape[0], 1)))],
            format="csr",
        )
        a_ub = _sparse.vstack(
            [a_ub, _sparse.csr_matrix(np.append(objective, 0.0)[None, :])], format="csr"
        )
        b_ub = np.concatenate([system.b_ub, [bound]])
        a_eq = _sparse.hstack(
            [system.a_eq, _sparse.csr_matrix((system.a_eq.shape[0], 1))], format="csr"
        )
        res = linprog(
            np.concatenate([np.zeros(size), [-1.0]]),
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=system.b_eq,
            bounds=list(system.bounds) + [(None, 0.0)],
            method="highs",
        )
        if not res.success or res.x is None:
            raise RuntimeError(f"selection: max-margin solve failed: {res.message!r}")
        primal = np.asarray(res.x[:size], dtype=np.float64)
        theta = primal[: system.n_features]
        f = primal[system.n_features : system.n_features + system.n_source]
        g = primal[system.n_features + system.n_source :]
        sigma_all = _sigma(problem, theta, f, g, all_rows, all_cols)
        worst = float(sigma_all.max())
        if worst <= float(tol):
            return worst
        viol = sigma_all > float(tol)
        current = current.union(
            EdgeSet.from_arrays(all_rows[viol], all_cols[viol], n_target=n)
        )
    sigma_all = _sigma(problem, theta, f, g, all_rows, all_cols)
    return float(sigma_all.max())
