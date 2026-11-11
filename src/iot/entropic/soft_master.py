"""
EN: **EOT's soft master** -- minimise, on a given edge set `F`,

        `Psi_F(theta,f,g) = <phi_bar, theta> - a.f - b.g + eps * sum_{e in F} exp(sigma_e/eps)`

    with `sigma_e = f_i + g_j - theta.phi_e`.

    **NOTE, and it matters:** this is the NESTED formulation, not EOT proper. It runs two levels --
    `theta` by SLSQP, and an ordinary entropic OT solve for `(f, g)` at every `theta`. EOT is a
    SINGLE joint solve over `(theta, f, g)` (see `joint_hvp`), where the plan falls out of the
    converged potentials instead of being recomputed per `theta`. Keep this for the P1 reference
    comparison; do not cite its cost as EOT's cost.

    Two levels:
      - **inner** (theta fixed): `f` is eliminated in closed form by a per-row segment log-sum-exp,
        and `g` is solved by `inner_f.solve_inner_on_f`;
      - **outer**: `theta` over Theta, gradient `dPsi/dtheta = phi_bar - sum_e pi_e phi_e`, by SLSQP
        with the exact gradient, stopping on the **Frank-Wolfe gap <= 1e-4 * eps** (scale-free,
        bounds the optimality gap directly by convexity).

    `Psi_F` is **convex** in `(theta, f, g)`, so any converged point is a global minimiser of the
    nested objective.

CN: **EOT 的软 master** —— 在给定边集 `F` 上最小化

        `Ψ_F(θ,f,g) = <φ̄, θ> − a·f − b·g + ε Σ_{e∈F} exp(σ_e/ε)`

    其中 `σ_e = f_i + g_j − θ·φ_e`。

    注意：这是**嵌套**形式，不是 EOT 本身。两级：内层（给定 θ）用 `inner_f.solve_inner_on_f`
    解 `(f,g)`；外层 `θ` 用 SLSQP，停止判据 `Frank–Wolfe gap ≤ 1e-4·ε`。
    EOT 是 `(θ,f,g)` 的**单次联合**求解（见 `joint_hvp`）。`Ψ_F` 对三者凸，收敛点即全局最优。
"""
from __future__ import annotations

import dataclasses

import numpy as np
from jaxtyping import Float

from ..core.edges import EdgeSet
from ..core.problem import InverseOTProblem
from ..core.theta import frank_wolfe_gap
from ..relaxed.master_regret import regret_objective
from .inner_f import SparseInnerResult, sinkhorn_sweeps_on_f

FRANK_WOLFE_RTOL: float = 1e-4
"""EN: stop when the Frank-Wolfe gap `<= rtol * eps`. CN: 停止判据 `gap <= rtol * eps`."""


@dataclasses.dataclass(frozen=True)
class SoftMasterResult:
    """EN: one soft-master solution. CN: 一次软 master 的解。"""

    theta: Float[np.ndarray, "k"]
    source_dual: Float[np.ndarray, "m"]
    target_dual: Float[np.ndarray, "n"]
    psi: float
    plan_on_f: Float[np.ndarray, "edges"]
    frank_wolfe_gap: float
    converged: bool
    outer_iterations: int
    inner_sweeps: int
    inner_marginal_error: float


def _edge_features(problem: InverseOTProblem, edges: EdgeSet) -> Float[np.ndarray, "edges k"]:
    """EN: the feature rows on `F`. CN: `F` 上的特征行。"""
    return np.asarray(problem.features.features(edges.rows, edges.cols), dtype=np.float64)


def soft_master(problem: InverseOTProblem, edges: EdgeSet, *, epsilon: float,
                warm: SoftMasterResult | None = None, max_outer: int = 200,
                frank_wolfe_rtol: float = FRANK_WOLFE_RTOL,
                inner_tolerance: float = 1e-8, inner_sweeps: int = 20000,
                max_newton: int = 60) -> SoftMasterResult:
    """
    EN: solve the soft master on `F`. `epsilon` is **absolute** (frozen by the caller).
    CN: 在 `F` 上解软 master。`epsilon` 是**绝对值**（由调用方冻结）。
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    k = int(problem.n_features)
    m = int(problem.n_source)
    n = int(problem.n_target)
    a: Float[np.ndarray, "m"] = np.asarray(problem.resolved_source_mass, dtype=np.float64).reshape(-1)
    b: Float[np.ndarray, "n"] = np.asarray(problem.resolved_target_mass, dtype=np.float64).reshape(-1)
    coef = regret_objective(problem)
    phi_bar: Float[np.ndarray, "k"] = coef[:k]
    features_f = _edge_features(problem, edges)

    constraints = problem.theta_set.linear_constraints()
    lower = np.asarray(constraints.lower, dtype=np.float64)
    upper = np.asarray(constraints.upper, dtype=np.float64)
    bounds = [(None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi))
              for lo, hi in zip(lower, upper)]
    a_eq = np.asarray(constraints.a_eq, dtype=np.float64)
    b_eq = np.asarray(constraints.b_eq, dtype=np.float64)

    state: dict[str, object] = {"sweeps": 0, "last": None, "g_warm": None}

    def value_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        cf = features_f @ np.asarray(theta, dtype=np.float64)
        # EN: use **alternating Sinkhorn scaling** (measured: Newton gives up on hard instances,
        #     scaling does not).
        # CN: 用交替 Sinkhorn 缩放（实测困难实例上牛顿会放弃，缩放不会）。
        inner: SparseInnerResult = sinkhorn_sweeps_on_f(
            edges, cf, a, b, epsilon=epsilon, g_init=state["g_warm"],  # type: ignore[arg-type]
            max_sweeps=int(inner_sweeps), marginal_tolerance=float(inner_tolerance))
        state["sweeps"] = int(state["sweeps"]) + int(inner.newton_steps)  # type: ignore[arg-type]
        state["g_warm"] = np.asarray(inner.potential_target, dtype=np.float64)
        state["last"] = inner
        pi = np.asarray(inner.plan_on_f, dtype=np.float64)
        penalty = float(pi.sum())
        psi = float(theta @ phi_bar) - float(a @ inner.potential_source) \
            - float(b @ inner.potential_target) + float(epsilon) * penalty
        grad = np.asarray(phi_bar - pi @ features_f, dtype=np.float64)
        return psi, grad

    start = (np.full(k, 1.0 / k, dtype=np.float64) if warm is None
             else np.asarray(warm.theta, dtype=np.float64).reshape(-1))
    result = minimize(value_and_grad, start, jac=True, method="SLSQP", bounds=bounds,
                      constraints=[{"type": "eq", "fun": lambda t: float(a_eq @ t - b_eq),
                                    "jac": lambda t: a_eq}],
                      options={"maxiter": int(max_outer), "ftol": 1e-16})
    theta: Float[np.ndarray, "k"] = np.asarray(result.x, dtype=np.float64)
    psi, grad = value_and_grad(theta)
    gap = frank_wolfe_gap(problem.theta_set, theta, grad)
    last = state["last"]
    assert isinstance(last, SparseInnerResult)
    return SoftMasterResult(
        theta=theta, source_dual=np.asarray(last.potential_source, dtype=np.float64),
        target_dual=np.asarray(last.potential_target, dtype=np.float64), psi=float(psi),
        plan_on_f=np.asarray(last.plan_on_f, dtype=np.float64), frank_wolfe_gap=float(gap),
        converged=bool(gap <= float(frank_wolfe_rtol) * float(epsilon)),
        outer_iterations=int(getattr(result, "nit", -1)), inner_sweeps=int(state["sweeps"]),
        inner_marginal_error=float(last.marginal_error))


__all__ = ["FRANK_WOLFE_RTOL", "SoftMasterResult", "soft_master"]
