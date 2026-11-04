"""
EN: **Independent reference for Gate B -- the joint formulation solved WITHOUT the custom Newton-CG.**

    Purpose. `iot.entropic.joint_master` is a hand-written Newton-CG with a primal active set, and Gate B
    (free theta) fails on it. Before patching it again, this module answers the prior question: does the
    JOINT FORMULATION itself have the expected optimum? Nothing here imports `joint_master`, and nothing
    here reuses its gradient, Hessian-vector product, reduction, active set or line search.

    It is written straight from the reduced objective, which is derived (not copied) here:

    Starting from
        Psi(theta,f,g) = <phi_bar,theta> - a.f - b.g + eps * sum_ij exp((f_i + g_j - c_ij)/eps),
    the source marginal constraint `sum_j pi_ij = a_i` is used to eliminate `f`. With
        z_ij = (g_j - c_ij)/eps,   L_i = log sum_j exp(z_ij),   f_i = eps*(log a_i - L_i),
    the plan is  pi_ij = exp((f_i + g_j - c_ij)/eps) = a_i * exp(z_ij - L_i), so rows sum to `a_i`
    EXACTLY by construction. Substituting and using `sum_ij pi_ij = sum_i a_i`:

        Psi_red(theta, g) = <phi_bar,theta> - b.g + eps * sum_i a_i * (1 - log a_i + L_i).

    Gradients follow from `dL_i/dg_j = pi_ij/a_i` and `dL_i/dtheta = -(1/(eps*a_i)) sum_j pi_ij grad c_ij`,
    giving the two expressions the review asked to derive independently:

        d/dtheta Psi_red = phi_bar - sum_ij pi_ij grad_theta c_ij,
        d/dg_j  Psi_red  = sum_i pi_ij - b_j.

    Constraints are handled SEPARATELY, as required:
      * theta:  equality `sum theta = 1` plus nonnegativity, given to SLSQP as a real equality constraint
        with explicit bounds -- no coordinate is eliminated, no pivot is chosen;
      * g: `g_0 = 0` is held by parameterising ONLY `g_1..g_{n-1}`. The gradient of that block is
        `d/dg_j` for `j >= 1`; `d/dg_0` is NOT subtracted, because coordinate 0 is fixed, not a variable
        and not a reference.

    Marginal feasibility is checked on BOTH sides. The source side is exact by construction; the target
    side is the actual stationarity condition for `g` and is the honest test of an optimum.

CN: **Gate B 的独立参考实现 —— 不使用自定义 Newton-CG 求解联合形式。**

    目的：先回答"联合形式本身的最优点对不对"，再决定是否修 `joint_master`。本模块不导入它，也不复用
    它的梯度、HVP、约化、主动集或线搜索。

    由约化目标直接实现：
        `Psi_red(theta,g) = <phi_bar,theta> - b.g + eps * sum_i a_i * (1 - log a_i + L_i)`
    梯度独立推导为 `phi_bar - sum pi grad c` 与 `sum_i pi_ij - b_j`。

    约束**分开处理**：θ 用真实等式约束 + 非负界交给 SLSQP（不消元、不选枢轴）；g 只优化 `g_1..g_{n-1}`，
    其梯度就是 `d/dg_j`（j>=1），**不减去** `d/dg_0`。
"""

from __future__ import annotations

import dataclasses

import numpy as np
from jaxtyping import Float, Int
from scipy.special import logsumexp


@dataclasses.dataclass(frozen=True)
class ReducedProblem:
    """
    EN: everything the reduced objective needs, assembled once. Every field is recorded in the JSON so a
        run can be reproduced without assuming two scripts mean the same instance.
    CN: 约化目标所需的数据；全部字段写入 JSON，确保可复现。
    """

    cost_features: Float[np.ndarray, "edges k"]
    rows: Int[np.ndarray, "edges"]
    cols: Int[np.ndarray, "edges"]
    source_mass: Float[np.ndarray, "m"]
    target_mass: Float[np.ndarray, "n"]
    phi_bar: Float[np.ndarray, "k"]
    epsilon: float

    def __post_init__(self) -> None:
        """EN: fail early on shapes that cannot represent a transport problem. CN: 形状不对就尽早失败。"""
        edges = int(self.rows.size)
        if self.cols.size != edges or self.cost_features.shape[0] != edges:
            raise ValueError("rows, cols and cost_features must agree on the edge count")
        if float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive")
        if abs(float(self.source_mass.sum()) - float(self.target_mass.sum())) > 1e-8:
            raise ValueError("source and target mass must agree")
        # EN: a zero marginal entry cannot be logged; refuse rather than silently drop it.
        # CN: 零边缘的质量取对数无定义，直接拒绝而不是悄悄丢掉。
        if np.any(self.source_mass <= 0.0):
            raise ValueError("zero source mass is not supported by this reference")


def evaluate(problem: ReducedProblem, theta: Float[np.ndarray, "k"], g: Float[np.ndarray, "n"]
             ) -> tuple[float, Float[np.ndarray, "k"], Float[np.ndarray, "n"], Float[np.ndarray, "edges"]]:
    """
    EN: return `(Psi_red, grad_theta, grad_g, plan)` computed from the reduced objective.

        The log-sum-exp is done per source row with `scipy.special.logsumexp`, which is stable. `grad_g`
        is the FULL vector `colsum - b`; the caller decides which entries are variables.
    CN: 返回 `(Psi_red, grad_theta, grad_g, plan)`。逐行 log-sum-exp，`grad_g` 返回完整向量。
    """
    eps = float(problem.epsilon)
    cost = np.asarray(problem.cost_features @ np.asarray(theta, dtype=np.float64), dtype=np.float64)
    z = (np.asarray(g, dtype=np.float64)[problem.cols] - cost) / eps
    # EN: per-row log-sum-exp via a sort-free scipy routine; `-inf` never appears because every row has
    #     at least one edge on the full grid.
    # CN: 逐行 log-sum-exp；全网格下每行至少一条边，不会出现 -inf。
    row_lse = np.asarray(logsumexp(z.reshape(problem.source_mass.size, -1), axis=1), dtype=np.float64)
    row_lse_edges = row_lse[problem.rows]
    plan = np.asarray(problem.source_mass, dtype=np.float64)[problem.rows] * np.exp(z - row_lse_edges)
    colsum = np.bincount(problem.cols, weights=plan, minlength=problem.target_mass.size)
    psi = (float(np.asarray(theta) @ problem.phi_bar)
           - float(problem.target_mass @ np.asarray(g, dtype=np.float64))
           + eps * float(np.sum(problem.source_mass * (1.0 - np.log(problem.source_mass) + row_lse))))
    grad_theta = np.asarray(problem.phi_bar - problem.cost_features.T @ plan, dtype=np.float64)
    grad_g = np.asarray(colsum - problem.target_mass, dtype=np.float64)
    return psi, grad_theta, grad_g, plan


def marginals(problem: ReducedProblem, plan: Float[np.ndarray, "edges"]
              ) -> tuple[float, float]:
    """
    EN: worst relative violation of each marginal. CN: 两个边缘的最大相对违背。
    """
    rowsum = np.bincount(problem.rows, weights=np.asarray(plan), minlength=problem.source_mass.size)
    colsum = np.bincount(problem.cols, weights=np.asarray(plan), minlength=problem.target_mass.size)
    row_err = float(np.max(np.abs(rowsum - problem.source_mass) / problem.source_mass))
    col_err = float(np.max(np.abs(colsum - problem.target_mass) / problem.target_mass))
    return row_err, col_err


@dataclasses.dataclass(frozen=True)
class ReferenceRun:
    """EN: one independent run from one starting point. CN: 从一个起点出发的一次独立运行。"""

    start_label: str
    theta: Float[np.ndarray, "k"]
    g: Float[np.ndarray, "n"]
    psi: float
    grad_theta: Float[np.ndarray, "k"]
    grad_g: Float[np.ndarray, "n"]
    plan: Float[np.ndarray, "edges"]
    simplex_multiplier: float
    simplex_kkt_violation: float
    row_marginal_error: float
    col_marginal_error: float
    status: int
    message: str
    iterations: int
    objective_calls: int


def solve(problem: ReducedProblem, theta_start: Float[np.ndarray, "k"],
          g_start: Float[np.ndarray, "n"], *, label: str = "start",
          max_iterations: int = 4000, tolerance: float = 1e-12) -> ReferenceRun:
    """
    EN: minimise `Psi_red` by SLSQP over `(theta, g_1..g_{n-1})`.

        theta enters only through the equality `sum theta = 1` and the bounds `theta >= 0`; `g_0` is
        excluded from the variables entirely. Objective and gradient come from `evaluate`, which is a
        direct transcription of the reduced objective, not an adaptation of the custom solver.
    CN: 用 SLSQP 在 `(theta, g_1..g_{n-1})` 上最小化 `Psi_red`。θ 只通过等式与下界进入；`g_0` 不在变量中。
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    k = int(np.asarray(theta_start).size)
    n = int(np.asarray(g_start).size)
    nu = n - 1

    def unpack(x: Float[np.ndarray, "variables"]) -> tuple[Float[np.ndarray, "k"],
                                                          Float[np.ndarray, "n"]]:
        """EN: `(theta, g_1..g_{n-1})` -> `(theta, g)` with `g_0 = 0`. CN: 展开变量，`g_0 = 0`。"""
        theta = np.asarray(x[:k], dtype=np.float64)
        g = np.zeros(n, dtype=np.float64)
        g[1:] = x[k:]
        return theta, g

    def objective(x: Float[np.ndarray, "variables"]) -> tuple[float, Float[np.ndarray, "variables"]]:
        """EN: reduced objective and its gradient w.r.t. the optimisation variables.

            The `g` block of the gradient is `d/dg_j` for `j >= 1` ONLY: coordinate 0 is fixed, so it is
            neither a variable nor the reference for this block.
        CN: 约化目标及其对优化变量的梯度。`g` 块只取 `j>=1` 的 `d/dg_j`；坐标 0 固定，既不是变量也不是参照。
        """
        theta, g = unpack(x)
        psi, grad_theta, grad_g, _plan = evaluate(problem, theta, g)
        return psi, np.concatenate([grad_theta, grad_g[1:]])

    x0 = np.concatenate([np.asarray(theta_start, dtype=np.float64),
                         np.asarray(g_start, dtype=np.float64)[1:]])
    result = minimize(
        objective, x0, jac=True, method="SLSQP",
        bounds=[(0.0, None)] * k + [(None, None)] * nu,
        constraints=[{"type": "eq", "fun": lambda x: np.sum(x[:k]) - 1.0,
                      "jac": lambda x: np.concatenate([np.ones(k), np.zeros(nu)])}],
        options={"maxiter": int(max_iterations), "ftol": float(tolerance)})
    theta, g = unpack(np.asarray(result.x, dtype=np.float64))
    psi, grad_theta, grad_g, plan = evaluate(problem, theta, g)
    row_err, col_err = marginals(problem, plan)
    multiplier, violation = simplex_kkt(grad_theta, theta)
    return ReferenceRun(label, theta, g, float(psi), grad_theta, grad_g, plan, float(multiplier),
                        float(violation), row_err, col_err, int(result.status),
                        str(result.message), int(getattr(result, "nit", -1)),
                        int(getattr(result, "nfev", -1)))


def simplex_kkt(grad_theta: Float[np.ndarray, "k"], theta: Float[np.ndarray, "k"],
                *, tolerance: float = 1e-7) -> tuple[float, float]:
    """
    EN: simplex KKT residuals, with the multiplier taken FROM THE THETA EQUATIONS.

        Lagrangian `Psi - lam*(sum theta - 1) - mu_k*theta_k` with `mu_k >= 0` gives
        `grad_k = lam + mu_k`. So on the positive set `grad_k` is common (= `lam`, taken as the mean over
        positive coordinates), `mu_k = grad_k - lam = 0` there, and on the zero set
        `mu_k = grad_k - lam >= 0` is required. Returned: `(lam, worst violation)`, where the violation is
        the largest |grad_k - lam| over positive coordinates and the largest negative
        `(grad_k - lam)` over zero coordinates -- both are zero at a KKT point.
    CN: 单纯形 KKT；乘子取自 θ 方程本身。正坐标上 `grad_k` 应相等（`lam`），零坐标上 `grad_k - lam >= 0`。
    """
    theta = np.asarray(theta, dtype=np.float64)
    grad = np.asarray(grad_theta, dtype=np.float64)
    positive = theta > tolerance
    if not np.any(positive):
        return float("nan"), float("inf")
    multiplier = float(np.mean(grad[positive]))
    spread = float(np.max(np.abs(grad[positive] - multiplier)))
    zero = ~positive
    boundary = float(max(0.0, -np.min(grad[zero] - multiplier))) if np.any(zero) else 0.0
    return multiplier, float(max(spread, boundary))


def solve_on_face(problem: ReducedProblem, indices: Int[np.ndarray, "face"],
                  theta_seed: Float[np.ndarray, "k"], *, label: str = "face",
                  max_iterations: int = 4000, tolerance: float = 1e-14) -> ReferenceRun:
    """
    EN: minimise `Psi_red` with every coordinate OUTSIDE `indices` held at exactly 0.

        SLSQP's inequality handling is not exact for a coordinate driven to its bound (it leaves ~1e-8 and
        then the equality is only approximately satisfied), so a face cannot be explored by constraining
        the full problem. Here the free coordinates are the real variables and the equality is the
        renormalised sum; the excluded coordinates never exist as variables, so they cannot drift.
    CN: 在**被排除坐标恒为 0** 的面上最小化。SLSQP 对触界坐标不精确，因此这里把自由坐标作为真正的变量。
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    indices = np.asarray(indices, dtype=np.int64)
    k = int(problem.phi_bar.size)
    n = int(problem.target_mass.size)
    size = int(indices.size)
    if size < 1:
        raise ValueError("a face needs at least one coordinate")

    def expand(z: Float[np.ndarray, "face"] ) -> Float[np.ndarray, "k"]:
        """EN: face coordinates -> full theta, excluded coordinates exactly zero. CN: 展开到完整 θ。"""
        theta = np.zeros(k, dtype=np.float64)
        theta[indices] = np.asarray(z, dtype=np.float64)
        return theta

    def objective(z: Float[np.ndarray, "variables"]) -> tuple[float, Float[np.ndarray, "variables"]]:
        """EN: `Psi_red` on the face, and its gradient in the face + g variables. CN: 面上的目标与梯度。"""
        theta = expand(z[:size])
        psi, grad_theta, grad_g, _plan = evaluate(problem, theta,
                                                  np.concatenate([[0.0], np.asarray(z[size:])]))
        return psi, np.concatenate([grad_theta[indices], grad_g[1:]])

    seed = np.zeros(k, dtype=np.float64)
    mass = float(np.asarray(theta_seed, dtype=np.float64)[indices].sum())
    seed[indices] = (np.asarray(theta_seed, dtype=np.float64)[indices] / mass) if mass > 0 else \
        np.full(size, 1.0 / size)
    seed[indices] = np.maximum(seed[indices], 1e-9)
    seed[indices] /= seed[indices].sum()
    x0 = np.concatenate([seed[indices], np.zeros(n - 1)])
    result = minimize(objective, x0, jac=True, method="SLSQP",
                      bounds=[(0.0, None)] * size + [(None, None)] * (n - 1),
                      constraints=[{"type": "eq", "fun": lambda z: np.sum(z[:size]) - 1.0,
                                    "jac": lambda z: np.concatenate([np.ones(size),
                                                                     np.zeros(n - 1)])}],
                      options={"maxiter": int(max_iterations), "ftol": float(tolerance)})
    theta = expand(np.asarray(result.x[:size], dtype=np.float64))
    g = np.concatenate([[0.0], np.asarray(result.x[size:], dtype=np.float64)])
    psi, grad_theta, grad_g, plan = evaluate(problem, theta, g)
    positive = indices[theta[indices] > 1e-9]
    multiplier = float(np.mean(grad_theta[positive])) if positive.size else float("nan")
    excluded = np.setdiff1d(np.arange(k), indices)
    boundary = float(max(0.0, -np.min(grad_theta[excluded] - multiplier))) if excluded.size else 0.0
    spread = float(np.max(np.abs(grad_theta[positive] - multiplier))) if positive.size else float("inf")
    row_err, col_err = marginals(problem, plan)
    return ReferenceRun(label, theta, g, float(psi), grad_theta, grad_g, plan, multiplier,
                        float(max(spread, boundary)), row_err, col_err, int(result.status),
                        str(result.message), int(getattr(result, "nit", -1)),
                        int(getattr(result, "nfev", -1)))


def face_enumeration(problem: ReducedProblem, *, theta_start: Float[np.ndarray, "k"],
                     max_iterations: int = 4000) -> list[dict[str, object]]:
    """
    EN: independently enumerate all non-empty faces of the simplex; on each face minimise over the face.

        On a face, the free coordinates are optimised by SLSQP subject to positivity and the renormalised
        sum being 1; the excluded coordinates are held at exactly 0. Each face is started from a few
        feasible points, and the best result is kept, so a face cannot be dismissed because of a bad start.
        The excluded-coordinate multipliers `grad_k - lam` are reported per face.
    CN: 独立枚举单纯形的全部非空面；每个面上在面内最小化，并报告被排除坐标的乘子。
    """
    import itertools  # noqa: PLC0415

    k = int(problem.phi_bar.size)
    records: list[dict[str, object]] = []
    starts: list[np.ndarray] = [np.asarray(theta_start, dtype=np.float64)]
    for index in range(k):
        vertex = np.zeros(k, dtype=np.float64)
        vertex[index] = 1.0
        starts.append(vertex)
    for size in range(k, 0, -1):
        for face in itertools.combinations(range(k), size):
            indices = np.asarray(face, dtype=np.int64)
            best: ReferenceRun | None = None
            for start in starts:
                seed = np.zeros(k, dtype=np.float64)
                mass = float(start[indices].sum())
                seed[indices] = start[indices] / mass if mass > 0 else 1.0 / indices.size
                seed[indices] = np.maximum(seed[indices], 1.0 / (10.0 * indices.size))
                seed[indices] /= seed[indices].sum()
                # EN: optimise strictly WITHIN the face, so the excluded coordinates cannot drift.
                # CN: 严格在面内优化，被排除的坐标不会漂移。
                cand = solve_on_face(problem, indices, seed, label=f"face{face}",
                                     max_iterations=max_iterations)
                if best is None or cand.psi < best.psi:
                    best = cand
            if best is None:
                continue
            excluded = np.setdiff1d(np.arange(k), indices)
            multiplier = float(np.mean(best.grad_theta[indices]))
            excluded_multipliers = (best.grad_theta[excluded] - multiplier) if excluded.size else \
                np.zeros(0, dtype=np.float64)
            records.append({
                "face": [int(v) for v in indices],
                "psi": float(best.psi),
                "theta": [round(float(v), 10) for v in best.theta],
                "excluded_multipliers": [round(float(v), 6) for v in excluded_multipliers],
                "excluded_kkt_ok": bool(excluded.size == 0
                                        or np.all(excluded_multipliers >= -1e-6)),
                "row_marginal_error": float(best.row_marginal_error),
                "col_marginal_error": float(best.col_marginal_error),
                "simplex_kkt_violation": float(best.simplex_kkt_violation),
                "status": int(best.status),
            })
    records.sort(key=lambda record: float(record["psi"]))
    return records


__all__ = ["ReducedProblem", "ReferenceRun", "evaluate", "face_enumeration", "marginals",
           "simplex_kkt", "solve", "solve_on_face"]
