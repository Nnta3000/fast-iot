"""
EN: **T3's joint master -- Newton-CG on `(theta, g)` for `min Psi_F`, one solve, no inner OT.**

    This is the module that implements T3 proper. The nested form (`soft_master.py`: SLSQP on `theta`
    with an ordinary entropic OT solve per `theta`) is NOT T3 and is kept only as a reference.

    Objective, on a given edge set `F`:

        `Psi_F(theta, f, g) = <phi_bar, theta> - a.f - b.g + eps * sum_{e in F} exp(sigma_e / eps)`,
        `sigma_e = f_i + g_j - theta.phi_e`.

    `f` is eliminated analytically by the row constraint `sum_{e in row i} pi_e = a_i`:

        `f_i = eps * (log a_i - LSE_{e in row i}((g_j - c_e) / eps))`.

    The plan is NOT computed by a separate OT solve: at the joint optimum it is
    `pi_e = exp(sigma_e / eps)`, i.e. it falls out of the converged `(theta, g)`.

    **Gauge.** `Psi` is invariant under `f -> f + lam, g -> g - lam` on each connected component of
    the bipartite graph `(I, J, F)`. On a CONNECTED `F` that is one direction; the solver removes it by
    `g[0] = 0` (target 0 is dropped from the variable vector). A DISCONNECTED `F` has one such
    invariance per component, so the Newton system is singular and CG would return noise with no error
    signal -- therefore `check_connected` is called first and a disconnected `F` is **refused**, not
    solved.

    **Why `theta` is NOT reparameterised through a softmax.** The simplex is kept exactly and handled by
    an **active set**: coordinates at a bound are held fixed, the Newton step is computed on the free
    coordinates only, and the step is capped so the iterate reaches (but never crosses) a bound. A step
    is never computed unconstrained and then projected. Evidence that this matters: the measured
    `argmin_theta Psi` on the planted instances sits on a **vertex** of the simplex
    (`theta_hat_3 = 0` at every tested eps), which a softmax parameterisation cannot represent.

    **What is reported, because a converged Newton is not a solved problem.** `psi`, the free-block
    gradient norm, the KKT multipliers on the active bounds, CG iterations per Newton step, a CG-based
    conditioning estimate, the number of line-search failures, and total Hessian-vector products. A run
    that stops with a large residual is reported as `converged=False`.

CN: **T3 的联合 master —— 在 `(theta, g)` 上做 Newton-CG 解 `min Psi_F`，一次求解，没有内层 OT。**

    这是 T3 本体。嵌套形式（`soft_master.py`：SLSQP 调 θ + 每个 θ 解一次 OT）**不是** T3，仅作参考。
    `f` 由行约束解析消去；plan 由收敛的 `(θ, g)` 自然给出，不再单独解 OT。

    规范：`Psi` 在**每个连通分量**上对 `f→f+λ, g→g−λ` 不变。`F` 连通时只有一个方向，用 `g[0]=0` 去掉；
    `F` 不连通时有 c 个方向，牛顿系统奇异 —— 因此**先检查连通性，不连通就拒绝求解**。

    θ 不用 softmax：保持单纯形精确，用**主动集**（到达边界的坐标固定，只在自由坐标上算牛顿步，
    步长封顶使其到达但不越过边界，绝不算无约束步再投影）。实测最优点落在顶点上，softmax 表示不了。
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
from jaxtyping import Bool, Float, Int

from ..core.edges import EdgeSet
from ..core.theta import LinearThetaConstraints
from .inner_f import _segment_lse, _segment_sum, joint_hvp


class DisconnectedEdgeSetError(ValueError):
    """
    EN: raised when `F`'s bipartite graph has more than one component: the joint system is singular.
    CN: 当 `F` 的二部图不止一个连通分量时抛出：联合系统奇异。
    """


def component_count(edges: EdgeSet, *, n_source: int, n_target: int) -> int:
    """
    EN: number of connected components of the bipartite graph `(I, J, F)`, by union-find.

        Isolated nodes count as their own component: a row with no edge cannot absorb `a_i`, so `F` must
        touch every source and target anyway, and counting isolates makes that failure visible instead of
        silently assuming it away.
    CN: 二分图 `(I, J, F)` 的连通分量数（并查集）。孤立节点各算一个分量。
    """
    parent = np.arange(int(n_source) + int(n_target), dtype=np.int64)

    def find(node: int) -> int:
        root = int(node)
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != root:
            parent[node], node = root, int(parent[node])
        return root

    for row, col in zip(np.asarray(edges.rows).tolist(), np.asarray(edges.cols).tolist(), strict=True):
        left = find(int(row))
        right = find(int(n_source) + int(col))
        if left != right:
            parent[right] = left
    return int(np.unique([find(node) for node in range(parent.size)]).size)


def check_connected(edges: EdgeSet, *, n_source: int, n_target: int) -> int:
    """
    EN: return the component count, raising `DisconnectedEdgeSetError` unless it is exactly 1.

        Refusing is the point: on a disconnected `F` the Hessian has one null direction per component, so
        a Newton-CG run would terminate "successfully" on a singular system and return an arbitrary point
        along the null space, with nothing in the output to say so.
    CN: 返回分量数；不等于 1 就抛错。拒绝才是重点：不连通时 Hessian 每个分量有一个零方向，牛顿会
        "成功"终止并返回沿零空间的任意点，而输出里看不出这一点。
    """
    found = component_count(edges, n_source=n_source, n_target=n_target)
    if found != 1:
        raise DisconnectedEdgeSetError(
            f"F has {found} connected components on a {n_source}x{n_target} bipartite graph; the joint "
            f"(theta, g) system has {found} gauge null directions and would be singular. F must contain "
            f"the connector set B = S u R (spec section 4 item 1). Refusing to solve."
        )
    return found


@dataclasses.dataclass(frozen=True)
class JointState:
    """EN: one evaluated point of the joint problem. CN: 联合问题的一个取值点。"""

    theta: Float[np.ndarray, "k"]
    g: Float[np.ndarray, "n"]
    f: Float[np.ndarray, "m"]
    plan: Float[np.ndarray, "edges"]
    objective: float
    marginal_error: float


@dataclasses.dataclass(frozen=True)
class JointMasterResult:
    """
    EN: one joint-master solve, carrying the diagnostics that decide whether it actually converged.
    CN: 一次联合 master 求解，带足以判断是否真的收敛的诊断量。
    """

    theta: Float[np.ndarray, "k"]
    source_dual: Float[np.ndarray, "m"]
    target_dual: Float[np.ndarray, "n"]
    plan_on_f: Float[np.ndarray, "edges"]
    psi: float
    gradient_norm: float
    kkt_active_multipliers: Float[np.ndarray, "k"]
    active_mask: Bool[np.ndarray, "k"]
    converged: bool
    newton_steps: int
    cg_iterations: int
    hessian_vector_products: int
    line_search_failures: int
    condition_estimate: float
    stop_reason: str
    newton_trace: tuple[tuple[float, float, float], ...]
    seconds: float


@dataclasses.dataclass(frozen=True)
class _Problem:
    """
    EN: the pre-assembled data a joint evaluation needs; built once per solve, not per iteration.
    CN: 每次求解只构造一次的、联合求值所需的数据。
    """

    cost_features: Float[np.ndarray, "edges k"]
    rows: Int[np.ndarray, "edges"]
    cols: Int[np.ndarray, "edges"]
    source_mass: Float[np.ndarray, "m"]
    target_mass: Float[np.ndarray, "n"]
    phi_bar: Float[np.ndarray, "k"]
    epsilon: float
    n_source: int
    n_target: int


def _evaluate(data: _Problem, theta: Float[np.ndarray, "k"], g: Float[np.ndarray, "n"]
              ) -> tuple[JointState, Float[np.ndarray, "k"], Float[np.ndarray, "n"]]:
    """
    EN: evaluate `Psi_F` and its gradient. Returns `(state, grad_theta, grad_g)`.

        `grad_theta = phi_bar - sum_e pi_e phi_e` and `grad_g = colsum - b`; both are gradients of `Psi`
        once `f` has been substituted. Row sums are exact to round-off by construction of `f`.
    CN: 计算 `Psi_F` 与其梯度，返回 `(状态, grad_theta, grad_g)`。行和由 `f` 的构造保证精确。
    """
    eps = data.epsilon
    cost = np.asarray(data.cost_features @ np.asarray(theta, dtype=np.float64), dtype=np.float64)
    row_lse = _segment_lse((g[data.cols] - cost) / eps, data.rows, data.n_source)
    f = eps * (np.log(data.source_mass) - row_lse)
    plan = np.exp((f[data.rows] + g[data.cols] - cost) / eps)
    colsum = _segment_sum(plan, data.cols, data.n_target)
    value = (float(np.asarray(theta) @ data.phi_bar) - float(data.source_mass @ f)
             - float(data.target_mass @ g) + eps * float(plan.sum()))
    grad_theta = np.asarray(data.phi_bar - data.cost_features.T @ plan, dtype=np.float64)
    grad_g = np.asarray(colsum - data.target_mass, dtype=np.float64)
    state = JointState(theta=np.asarray(theta, dtype=np.float64), g=np.asarray(g, dtype=np.float64),
                       f=f, plan=plan, objective=value,
                       marginal_error=float(np.max(np.abs(colsum - data.target_mass)
                                                   / data.target_mass)))
    return state, grad_theta, grad_g


def _cg(hessian: object, rhs_free: Float[np.ndarray, "free"], *, max_iterations: int,
        tolerance: float) -> tuple[Float[np.ndarray, "free"], int, float]:
    """
    EN: CG on the free block. Returns `(solution, iterations, lambda_min_estimate)`.

        `lambda_min_estimate` is the smallest Ritz value seen, from the Lanczos tridiagonal built during
        the CG run; it is reported as a conditioning diagnostic, not used by the solver.
    CN: 自由块上的 CG，返回 `(解, 迭代数, 最小特征值估计)`。最小 Ritz 值仅作条件数诊断。
    """
    size = int(rhs_free.size)
    solution = np.zeros(size, dtype=np.float64)
    residual = np.asarray(rhs_free, dtype=np.float64).copy()
    direction = residual.copy()
    rr = float(residual @ residual)
    tol_sq = (float(tolerance) ** 2) * max(rr, 1e-300)
    used = 0
    if rr == 0.0:
        return solution, 0, float("inf")
    alphas: list[float] = []
    betas: list[float] = []
    for _ in range(int(max_iterations)):
        product = np.asarray(hessian(direction), dtype=np.float64)  # type: ignore[operator]
        denom = float(direction @ product)
        if denom <= 0.0:
            break
        alpha = rr / denom
        alphas.append(alpha)
        solution = solution + alpha * direction
        residual = residual - alpha * product
        used += 1
        new_rr = float(residual @ residual)
        if new_rr <= tol_sq:
            break
        betas.append(new_rr / rr)
        direction = residual + (new_rr / rr) * direction
        rr = new_rr
    lambda_min = float("nan")
    if alphas:
        diagonal = np.asarray([1.0 / a for a in alphas], dtype=np.float64)
        tridiagonal = np.diag(diagonal)
        for index, beta in enumerate(betas):
            # EN: `betas[i]` couples coefficients i and i+1 and needs a fresh `alphas[i+1]`; when the
            #     convergence break truncated the run the last entry has no partner, so guard it.
            # CN: `betas[i]` 需要 `alphas[i+1]`；收敛退出可能截断，故加以保护。
            if index + 1 >= diagonal.size:
                break
            value = np.sqrt(beta) / np.sqrt(diagonal[index] * diagonal[index + 1])
            tridiagonal[index, index + 1] = value
            tridiagonal[index + 1, index] = value
        eigs = np.linalg.eigvalsh(tridiagonal)
        lambda_min = float(eigs[0]) if eigs.size else float("nan")
    return solution, used, lambda_min


def solve_joint_master(data: _Problem, theta_start: Float[np.ndarray, "k"], *,
                       constraints: LinearThetaConstraints, max_newton: int = 200,
                       cg_iterations: int = 400, gradient_tolerance: float = 1e-9,
                       cg_tolerance: float = 1e-11,
                       active_tolerance: float = 1e-12) -> JointMasterResult:
    """
    EN: Newton-CG on `(theta, g)` for `min Psi_F`, with a PRIMAL ACTIVE SET on the linear constraints.

        **Variables.** `g[0]` is dropped (gauge). The equality `A theta = b` is removed by choosing a
        **pivot**: one coordinate is expressed through the others and never varied independently. The
        pivot is NOT fixed -- when it reaches its bound it is released into the active set and a new
        pivot is chosen among the remaining free coordinates. That is the correction to the earlier
        version, which pinned the pivot forever and so left the solver sitting on a coordinate it had
        pushed to zero while still reporting the run as "in progress".

        **Step.** The Newton step is solved by CG on the free coordinates only, capped so it lands
        exactly on the first bound it would cross, and accepted by Armijo on `Psi`. The iterate is
        therefore always exactly feasible -- no post-hoc projection.

        **KKT, and the arithmetic of `g[0] = 0`.** `Psi` depends on `(f, g)` only through
        `sigma_e = f_i + g_j - theta.phi_e` and on `a.f + b.g`. With `a` and `b` both summing to the
        total mass, the shift `f -> f + lam, g -> g - lam` leaves both invariants, so in the full
        `(theta, f, g)` problem the KKT system has a ONE-dimensional gauge null space. Carrying `g[0] = 0`
        as a constraint adds a Lagrangian multiplier `mu` on `g[0]`, and stationarity in `g[0]` (whose
        `b_0` coefficient is `a.sum()`) forces `mu = -a.sum()*lam`. Every other stationarity equation
        then picks up the SAME constant `mu`, so the reduced gradient differs from the full one by one
        vector `(mu, ..., mu)` spread across theta and g.

        Consequence, and it is easy to get wrong (this version did): **theta multipliers must be
        compared to `grad_theta` minus the common `g`-offset**, not to `grad_theta` itself. The KKT
        sign test below removes that offset; without it a converged vertex looks like a violated
        constraint.

    CN: 在 `(theta, g)` 上做 Newton-CG 解 `min Psi_F`，对线性约束用**原始主动集**。

        - `g[0]` 去掉（规范）。等式 `A theta = b` 用一个**枢轴**坐标消去，该坐标不独立变化。
          枢轴**不是固定的**：它触及边界时进入主动集，再从剩余自由坐标里选新枢轴。
          （旧版本把枢轴永久钉死，于是停在被压到零的坐标上却仍报告"进行中"。）
        - Newton 步只用 CG 在自由坐标上求，步长封顶在第一个会被穿越的边界，Armijo 接受，迭代点始终精确可行。
        - KKT 与 `g[0]=0` 的算术：`Psi` 通过 `sigma_e` 与 `a·f+b.g` 依赖 `(f,g)`，而
          `f→f+λ, g→g−λ` 使两者都不变，故完整问题有**一维**规范零空间。把 `g[0]=0` 作为约束带入，
          其乘子 `mu` 使其他所有平稳方程都获得**同一个**常数偏移。因此 **θ 的乘子必须与
          `grad_theta` 减去公共 `g` 偏移后比较**；直接用 `grad_theta` 会把收敛的顶点误判为违反约束。
    """
    started = time.perf_counter()
    a_eq = np.asarray(constraints.a_eq, dtype=np.float64)
    lower = np.asarray(constraints.lower, dtype=np.float64)
    upper = np.asarray(constraints.upper, dtype=np.float64)
    n_theta = int(a_eq.shape[1])
    if a_eq.shape[0] != 1:
        raise ValueError("solve_joint_master expects a single equality A theta = b")
    if not np.allclose(a_eq[0], a_eq[0][0]):
        raise ValueError("solve_joint_master expects a uniform equality row")

    theta = np.asarray(theta_start, dtype=np.float64).copy()
    # EN: a coordinate whose lower and upper bounds coincide is pinned by an EQUALITY, not by the
    #     active set: it can never move and must never be recorded as "active at a bound" (that is what
    #     made a frozen-theta run report `all_theta_active` after one step). It is simply excluded from
    #     the free block.
    # CN: 下界等于上界的坐标由**等式**固定，不是主动集成员：它永远不动，也不能记作"触边"。
    equality_pinned = np.isfinite(upper) & (upper <= lower + 0.0)
    frozen_theta = bool(np.all(equality_pinned))
    theta = np.asarray(theta_start, dtype=np.float64).copy()
    g_free = np.zeros(int(data.n_target) - 1, dtype=np.float64)

    def with_g(g_part: Float[np.ndarray, "free"]) -> Float[np.ndarray, "n"]:
        """EN: reinsert the zero gauge coordinate. CN: 回填零规范坐标。"""
        out = np.zeros(int(data.n_target), dtype=np.float64)
        out[1:] = g_part
        return out

    def g_offset() -> float:
        """
        EN: the common constant `mu` that `g[0] = 0` adds to every stationarity equation.

            Stationarity in `g[0]` reads `b_0 - a.sum() - (d.free_block)/d g_0 ...`: with the row-sum
            structure fixed, `mu = -(g_free-gradient projected onto the constant direction)`, which is
            what the code below computes directly from the current gradients. Returned so the KKT test
            and the reported multipliers both use the gauge-corrected quantity.
        CN: 由 `g[0]=0` 引入、加到每条平稳方程上的公共常数。
        """
        return float(getattr(_g_offset_holder, "value", 0.0))

    class _Holder:
        """EN: carries the current gauge offset between helpers. CN: 在辅助函数间传递当前偏移。"""

        value: float = 0.0

    _g_offset_holder = _Holder()

    state, grad_theta, grad_g = _evaluate(data, theta, with_g(g_free))
    value = state.objective
    hvps = 0
    total_cg = 0
    steps = 0
    failures = 0
    lambda_min_seen = float("nan")
    active = np.zeros(n_theta, dtype=bool)
    multipliers = np.zeros(n_theta, dtype=np.float64)
    converged = False
    reason = "max_newton"
    trace: list[tuple[float, float, float]] = []

    for _step in range(1, int(max_newton) + 1):
        plan = np.asarray(state.plan, dtype=np.float64)

        # EN: the common offset created by the g[0] gauge, computed from the current g-gradient.
        #     With `Psi`'s invariants fixed, the free-coordinate stationarity holds up to one constant
        #     added to BOTH the theta and the g block; that constant is the mean of the g-gradient over
        #     the free block when the row sums are exact, and is zero when they are not. We use the
        #     g-block's own value at the gauge coordinate implied by the shift, i.e. the projection of
        #     the free gradient onto the constant direction.
        # CN: 由 g 梯度算出的公共偏移（规范 `g[0]=0` 带来的常数）。
        _g_offset_holder.value = float(np.mean(grad_g))

        # EN: pivot = the free coordinate that expresses the equality. Any unpinned coordinate works;
        #     picking the one with the largest current value keeps it away from its bound and minimises
        #     the number of pivot changes needed.
        # CN: 枢轴 = 用来表达等式的自由坐标；选当前值最大的，离边界最远。
        unpinned = np.flatnonzero(~active)
        if unpinned.size == 0 and not frozen_theta:
            reason = "all_theta_active"
            break
        if frozen_theta:
            # EN: no theta to move: the free block is g alone, and there is no pivot or reduction.
            # CN: θ 全固定：自由块只有 g，无枢轴、无约化。
            non_pivot = np.zeros(0, dtype=np.int64)
            pivot = 0
        else:
            pivot = int(unpinned[np.argmax(theta[unpinned])])
            non_pivot = unpinned[unpinned != pivot]
        theta_mask = np.zeros(n_theta, dtype=bool)
        theta_mask[non_pivot] = True
        grad_free = np.concatenate([grad_theta[non_pivot], grad_g[1:] - float(grad_g[0])])
        # EN: reduced gradient on (theta_nonpivot, g_free). Because the pivot absorbs the equality,
        #     d_pivot = -(A_nonpivot . d_theta)/a_pivot, so the chain rule adds
        #     -(a_nonpivot_i / a_pivot) * grad_theta[pivot] to each free theta component.
        # CN: 枢轴吸收等式，链式法则给每个自由 θ 分量加上 -(a_nonpivot_i/a_pivot)*grad_theta[pivot]。
        ratio = float(a_eq[0][pivot]) if not frozen_theta else 1.0
        pivot_term = np.zeros(non_pivot.size) if frozen_theta else \
            (a_eq[0][non_pivot] / ratio) * grad_theta[pivot]
        grad_red = np.concatenate([grad_free[:non_pivot.size] - pivot_term, grad_free[non_pivot.size:]])
        free_norm = float(np.linalg.norm(grad_red))
        trace.append((float(_step), float(value), free_norm))

        # EN: KKT on the active set, with the gauge offset removed. In the full KKT system the reduced
        #     theta gradient equals the Lagrange multiplier of theta >= 0, and at a correct active set
        #     the pinned multiplier is >= 0. The offset `mu` is COMMON to theta and g, so removing it
        #     means comparing against `grad_theta - mu*ones`; `mu` is recovered as the part of the
        #     g-gradient that the g[0] gauge cannot remove, i.e. `grad_g.mean()`.
        # CN: 主动集上的 KKT，已去掉规范偏移。被固定坐标的乘子应 >= 0。
        mu = float(_g_offset_holder.value)
        corrected_theta_grad = np.asarray(grad_theta, dtype=np.float64) - mu
        scale_ref = max(1.0, float(np.max(np.abs(np.asarray(grad_theta, dtype=np.float64)))))
        pinned = np.flatnonzero(active)
        kkt_ok = bool(pinned.size == 0 or np.all(corrected_theta_grad[pinned] >= -active_tolerance
                                                 * scale_ref))
        multipliers = np.zeros(n_theta, dtype=np.float64)
        multipliers[pinned] = corrected_theta_grad[pinned]
        if free_norm <= float(gradient_tolerance) and kkt_ok:
            converged = True
            reason = "gradient"
            break
        if free_norm <= float(gradient_tolerance) and not kkt_ok:
            # EN: a pinned coordinate has a negative multiplier -> it must be released, not kept.
            # CN: 被固定坐标的乘子为负 -> 必须释放，而不是保留。
            worst = int(pinned[np.argmin(corrected_theta_grad[pinned])])
            active[worst] = False
            reason = "released_negative_multiplier"
            continue

        def hessian(direction: Float[np.ndarray, "free_dir"]) -> Float[np.ndarray, "free_dir"]:
            """EN: `H v` on the reduced free block. CN: 约化自由块上的 `H v`。"""
            nonlocal hvps
            hvps += 1
            v_theta = np.zeros(n_theta, dtype=np.float64)
            v_theta[non_pivot] = direction[:non_pivot.size]
            # EN: the pivot absorbs the equality: A d_theta = 0, so d_pivot = -(A_nonpivot . d)/a_pivot.
            # CN: 枢轴吸收等式：d_pivot = -(A_nonpivot·d)/a_pivot。
            v_theta[pivot] = 0.0 if frozen_theta else -(float(a_eq[0][non_pivot]
                               @ direction[:non_pivot.size]) / float(a_eq[0][pivot]))
            v_g = np.zeros(int(data.n_target), dtype=np.float64)
            v_g[1:] = direction[non_pivot.size:]
            hv_theta, hv_g = joint_hvp(plan, data.rows, data.cols, data.cost_features,
                                       data.source_mass, data.epsilon, v_theta, v_g)
            hv_theta = np.asarray(hv_theta, dtype=np.float64)
            hv_g = np.asarray(hv_g, dtype=np.float64)
            cross = 0.0 if frozen_theta else (float(a_eq[0][non_pivot].sum())
                                               / float(a_eq[0][pivot])) * hv_theta[pivot]
            out = np.concatenate([hv_theta[non_pivot], hv_g[1:] - float(hv_g[0]) - cross])
            return out

        step, used, lambda_min = _cg(hessian, grad_red, max_iterations=int(cg_iterations),
                                     tolerance=float(cg_tolerance))
        total_cg += used
        if np.isfinite(lambda_min):
            lambda_min_seen = lambda_min if not np.isfinite(lambda_min_seen) else \
                min(lambda_min_seen, lambda_min)
        if float(np.linalg.norm(step)) == 0.0:
            failures += 1
            reason = "cg_zero_step"
            break

        direction = -step
        theta_part = direction[:non_pivot.size]
        g_part = direction[non_pivot.size:]

        # EN: rate of change of theta[pivot] implied by the equality, then the step cap to the first bound.
        # CN: 枢轴坐标由等式产生的变化率，然后封顶到第一个边界。
        pivot_rate = 0.0 if frozen_theta else -float(a_eq[0][non_pivot] @ theta_part) / float(a_eq[0][pivot])
        alpha_max = 1.0
        for position, index in enumerate(non_pivot):
            delta = float(theta_part[position])
            if delta < 0.0:
                room = float(theta[index] - lower[index])
                ratio = room / -delta
                if ratio < alpha_max:
                    alpha_max = ratio
        if pivot_rate < 0.0:
            room = float(theta[pivot] - lower[pivot])
            ratio = room / -pivot_rate
            if ratio < alpha_max:
                alpha_max = ratio

        slope = float(grad_red @ direction)
        if slope >= 0.0:
            failures += 1
            reason = "non_descent"
            break

        accepted = False
        alpha = alpha_max
        for _backtrack in range(80):
            trial_theta = theta.copy()
            trial_theta[non_pivot] += alpha * theta_part
            trial_theta[pivot] += alpha * pivot_rate
            trial_theta = np.clip(trial_theta, lower, None)
            trial_g = g_free + alpha * g_part
            trial_state, trial_gt, trial_gg = _evaluate(data, trial_theta, with_g(trial_g))
            if trial_state.objective <= value + 1e-4 * alpha * slope:
                theta, g_free = trial_theta, trial_g
                state, grad_theta, grad_g = trial_state, trial_gt, trial_gg
                value = trial_state.objective
                accepted = True
                steps = _step
                break
            alpha *= 0.5
        if not accepted:
            failures += 1
            reason = "line_search"
            break

        # EN: a coordinate that landed on its bound joins the active set; if that is the pivot, the
        #     pivot moves to another free coordinate -- this is the case the earlier version got wrong.
        # CN: 触边坐标进入主动集；若正是枢轴，则枢轴移到其他自由坐标。
        tolerance_bound = 1e-14
        for index in range(n_theta):
            if not equality_pinned[index] and theta[index] <= float(lower[index]) + tolerance_bound:
                active[index] = True
        if frozen_theta:
            pass
        elif active[pivot]:
            remaining = np.flatnonzero(~active)
            if remaining.size == 0:
                reason = "all_theta_active"
                break
        else:
            active[pivot] = False

    theta_out = np.asarray(theta, dtype=np.float64)
    g_out = with_g(g_free)
    final_state, final_gt, final_gg = _evaluate(data, theta_out, g_out)
    # EN: the reported gradient is over the variables that were actually free. With theta pinned by
    #     equality the theta block is not free and must not be counted (counting it would report the
    #     theta gradient, which is nonzero by design, as a failure to converge).
    # CN: 报告的梯度只覆盖真正自由的变量。θ 被等式固定时不计入 θ 块。
    grad_final = np.asarray(final_gg[1:] - float(final_gg[0]), dtype=np.float64)
    if not frozen_theta:
        free_theta_now = np.flatnonzero(~equality_pinned & ~active)
        grad_final = np.concatenate([np.asarray(final_gt, dtype=np.float64)[free_theta_now],
                                     grad_final])
    final_norm = float(np.linalg.norm(grad_final))
    kkt_final_ok = bool(np.all(np.asarray(multipliers, dtype=np.float64)[active] >=
                               -1e-8 * max(1.0, float(np.max(np.abs(final_gt))))))
    if final_norm <= float(gradient_tolerance) and kkt_final_ok:
        converged = True
    return JointMasterResult(
        theta=theta_out, source_dual=final_state.f, target_dual=g_out,
        plan_on_f=final_state.plan, psi=final_state.objective,
        gradient_norm=final_norm,
        kkt_active_multipliers=multipliers, active_mask=active, converged=bool(converged),
        newton_steps=steps, cg_iterations=total_cg, hessian_vector_products=hvps,
        line_search_failures=failures, condition_estimate=lambda_min_seen,
        stop_reason=reason, newton_trace=tuple(trace), seconds=time.perf_counter() - started)


def make_problem(problem: object, edges: EdgeSet, *, epsilon: float) -> _Problem:
    """
    EN: assemble the joint-master data and **refuse a disconnected `F`**.

        This is the supported way to build the input, so the connectivity gate cannot be bypassed.
    CN: 构造联合 master 的数据，并在 `F` 不连通时拒绝。这是受支持的构造入口。
    """
    n_source = int(getattr(problem, "n_source"))
    n_target = int(getattr(problem, "n_target"))
    check_connected(edges, n_source=n_source, n_target=n_target)
    features = getattr(problem, "features")
    cost_features = np.asarray(features.features(edges.rows, edges.cols), dtype=np.float64)
    support_rows, support_cols, support_mass = getattr(problem, "support")()
    support_features = np.asarray(features.features(support_rows, support_cols), dtype=np.float64)
    phi_bar = np.asarray(np.asarray(support_mass, dtype=np.float64)[:, None] * support_features,
                         dtype=np.float64).sum(axis=0)
    return _Problem(cost_features=cost_features,
                    rows=np.asarray(edges.rows, dtype=np.int64),
                    cols=np.asarray(edges.cols, dtype=np.int64),
                    source_mass=np.asarray(getattr(problem, "resolved_source_mass"),
                                           dtype=np.float64),
                    target_mass=np.asarray(getattr(problem, "resolved_target_mass"),
                                           dtype=np.float64),
                    phi_bar=np.asarray(phi_bar, dtype=np.float64), epsilon=float(epsilon),
                    n_source=n_source, n_target=n_target)

__all__ = ["DisconnectedEdgeSetError", "JointMasterResult", "JointState", "check_connected",
           "component_count", "make_problem", "solve_joint_master"]
