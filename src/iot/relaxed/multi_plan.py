"""
EN: **One parameter, several observed plans** — the shared-`theta` arm for a multi-observation instance, with the
    relaxed (regret) formulation and an **exact** OT forward model.

    Given `L` observed plans `pi_hat_l`, each with its own marginals, minimise the summed regret

        F(theta) = sum_l R_l(theta),    R_l(theta) = <c(theta), pi_hat_l> - OT(c(theta); a_l, b_l),

    over `theta` on the simplex. Every `R_l >= 0`, so `F` is convex (linear in `theta` plus `-OT`, and `OT` is
    concave in the cost), and `F(theta) = 0` exactly when every observed plan is optimal at the common `c(theta)`.
    Its subgradient is

        g = sum_l Phi^T (pi_hat_l - P_l(theta)),     P_l(theta) an optimal plan of the forward problem,

    one exact OT solve per window. The solver is Frank–Wolfe on the simplex with the standard `2/(t+2)` step.

    **Two hypothesis classes.** `distance_power` uses only inter-zone distance. A cost built from distance alone
    cannot express zone-level structure, and measured on NYC taxi it collapsed onto one distance column
    (`theta = (0,1,0,0)`), i.e. "trips are short" and nothing else. `distance_lowrank` adds `R` non-negative
    rank-one terms `u_r(i) v_r(j)` obtained by a non-negative factorisation of the **training** flows, so the cost
    can be cheap between a specific origin and a specific destination for reasons other than distance. The factors
    are computed from the flows passed in, so nothing from a held-out window enters them.

    Note what the low-rank terms are *not*: a term of the form `a_i + b_j` (a per-zone attractiveness) would be
    absorbed by the OT duals and would leave every plan unchanged — only genuine interactions `u_i v_j` move the
    optimum. That is also why "add per-zone potentials" is not a bigger hypothesis class here.

    The forward model is exact OT, so a predicted plan is a basic feasible solution with at most `m + n - 1`
    positive entries. On a flow instance that is a real restriction and it is reported, not worked around: the
    question this arm answers is whether the relaxed formulation's own quantity — the excess cost of the observed
    plan under the fitted cost — transfers across windows, and secondarily what a plan decoded from that cost
    scores on a flow metric.

CN: **一个参数、多个观测计划** —— 多观测实例的共享 `theta` 臂，使用**松弛（regret）**形式与**精确** OT 前向模型。
    给定 `L` 个观测计划（各带 marginal），在单纯形上最小化求和 regret `F(theta) = sum_l R_l(theta)`。每个 `R_l >= 0`，
    故 `F` 凸（`theta` 线性项加 `-OT`，而 `OT` 对代价是凹的），且 `F(theta)=0` 恰当全部观测计划都在共同 `c(theta)` 处最优。
    次梯度为 `g = sum_l Phi^T(pi_hat_l - P_l(theta))`，每窗一次精确 OT 求解。求解器为单纯形上的 Frank–Wolfe
    （标准 `2/(t+2)` 步长）。前向模型是精确 OT，故预测计划是基本可行解，最多 `m+n-1` 个正元素；在流量实例上这是真实限制，
    如实报告而不绕过：该臂要回答的是"松弛形式自身的量（观测计划在拟合代价下的超出代价）能否跨窗迁移"，其次才是解码出的
    计划在流量指标上的得分。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import ot
from jaxtyping import Float

__all__ = ["SharedThetaFit", "distance_features", "fit_shared_theta", "mixed_features",
           "nonnegative_factors", "regret_of"]


def nonnegative_factors(flow: Float[np.ndarray, "n n"], *, rank: int, iterations: int = 200,
                        seed: int = 0) -> tuple[Float[np.ndarray, "n rank"], Float[np.ndarray, "n rank"]]:
    """
    EN: non-negative factorisation `flow ~ U V^T` by Lee–Seung multiplicative updates (Frobenius objective).

        Used to turn the training flows into interaction features. Non-negative is not cosmetic: `theta` is on the
        simplex, so every feature column must be non-negative for the cost to stay non-negative. A signed SVD would
        produce columns that cannot be combined this way.

    CN: 用 Lee–Seung 乘性更新（Frobenius 目标）做非负分解 `flow ~ U Vᵀ`，把训练流量变成交互特征。非负不是装饰：`theta`
        在单纯形上，特征列必须非负才能保证代价非负；带符号的 SVD 无法这样组合。
    """
    matrix = np.asarray(flow, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"flow must be square; got {matrix.shape}")
    if np.any(matrix < 0.0):
        raise ValueError("flow must be non-negative")
    if int(rank) < 1:
        raise ValueError("rank must be at least 1")
    rng = np.random.default_rng(int(seed))
    n = matrix.shape[0]
    scale = float(matrix.max()) if matrix.max() > 0 else 1.0
    u = rng.random((n, int(rank))) * 0.1 * scale + 1e-12
    v = rng.random((n, int(rank))) * 0.1 * scale + 1e-12
    for _ in range(int(iterations)):
        v *= (matrix.T @ u) / np.maximum(v @ (u.T @ u), 1e-300)
        u *= (matrix @ v) / np.maximum(u @ (v.T @ v), 1e-300)
    return u, v


def distance_features(distance: Float[np.ndarray, "n n"], *, family: str = "distance_power",
                      include_constant: bool = False) -> Float[np.ndarray, "n n k"]:
    """
    EN: the feature family for a distance-structured cost, as `(n, n, K)` non-negative columns.

        `distance_power` uses `{1, s, s^2, log(1 + s)}` with `s = d / max(d)`, plus a same-zone indicator. The family
        is deliberately **not** a set of bandwidths of one kernel: `exp(-d/sigma)` at different `sigma` are monotone
        transforms of one another, so they rank every candidate pair identically and `theta` cannot express a
        preference between them (measured on another instance in `notes/KERNEL_DICTIONARY_AXIS.md`). A weighted
        *sum* of `1, s, s^2, log(1+s)` changes the ranking.

        **A constant column is excluded by default, and that is a measured decision, not a tidiness one.** Adding a
        constant to a cost leaves every OT plan unchanged, so a constant column looks like a harmless gauge
        direction. It is not, under `min_theta R`: a *constant cost* makes every plan matching the marginals
        optimal, so `R = 0` for **any** observation. On the NYC taxi instance the arm with the constant column
        returned `theta = e_constant` and `R_test / R_uniform = 5e-5` on all six held-out windows — a perfect
        "generalisation" that says nothing about the data. The column is therefore off unless explicitly requested,
        and the degeneracy is documented where the fit is reported.

    CN: 距离结构代价的特征族，`(n, n, K)` 的非负列。`distance_power` 用 `{s, s², log(1+s)}`（`s=d/max(d)`）加同
        zone 指示列。刻意不是单一核的多带宽：不同 `sigma` 的 `exp(-d/sigma)` 互为单调变换，排序相同，`theta` 无法表达
        偏好（在另一实例上实测）。**常数列默认排除，这是实测决定**：代价加常数不改变任何 OT 计划，看似无害的规范方向；
        但在 `min_theta R` 下，**常数代价使任何满足 marginal 的计划都最优**，故对任何观测都有 `R=0`。在纽约出租车实例上，
        带常数列的该臂返回 `theta = e_常数`，六个留出窗的 `R_test/R_uniform` 均为 `5e-5` —— 一个不说明任何数据的"完美
        泛化"。因此该列默认关闭，并在报告处写明这一退化。
    """
    spacing = np.asarray(distance, dtype=np.float64)
    if spacing.ndim != 2 or spacing.shape[0] != spacing.shape[1]:
        raise ValueError(f"distance must be square; got {spacing.shape}")
    if np.any(spacing < 0.0):
        raise ValueError("distance must be non-negative")
    scale = float(spacing.max())
    if scale <= 0.0:
        raise ValueError("distance is identically zero")
    normalised = spacing / scale
    if str(family) == "distance_power":
        columns = [
            normalised,
            normalised ** 2,
            np.log1p(normalised),
            np.eye(spacing.shape[0], dtype=np.float64),
        ]
        if bool(include_constant):
            columns.insert(0, np.ones_like(normalised))
        return np.ascontiguousarray(np.stack(columns, axis=2), dtype=np.float64)
    raise ValueError(f"unknown feature family {family!r}")


def mixed_features(distance: Float[np.ndarray, "n n"], flows: Float[np.ndarray, "windows n n"], *,
                   family: str = "distance_power", rank: int = 4,
                   include_constant: bool = False) -> Float[np.ndarray, "n n k"]:
    """
    EN: the feature tensor for a family, built from the given (training) flows.

        `distance_power` is the distance block only. `distance_lowrank` appends `rank` non-negative interaction
        columns `u_r(i) v_r(j)` from `nonnegative_factors` applied to the **sum of the flows passed in**. Each column
        is scaled to unit Frobenius norm, so `theta`'s split between the distance block and the interaction block is
        a shape choice rather than a magnitude choice (the same gauge as elsewhere in this study).

        Passing the training flows is what keeps the factors leakage-free: a held-out window is never seen here.

    CN: 某个族的特征张量，由给定（训练）流量构造。`distance_power` 只有距离块；`distance_lowrank` 追加 `rank` 个非负交互
        列 `u_r(i)v_r(j)`，由对**传入流量之和**做非负分解得到。每列缩放到单位 Frobenius 范数，使 `theta` 在距离块与交互块
        之间的分配是形状选择而不是幅度选择（与本研究其它地方的规范一致）。传入训练流量正是因子不泄漏的原因：留出窗从不参与。
    """
    base = distance_features(distance, family="distance_power", include_constant=include_constant)
    if str(family) == "distance_power":
        return base
    if str(family) != "distance_lowrank":
        raise ValueError(f"unknown family {family!r}")
    observed = np.asarray(flows, dtype=np.float64)
    if observed.ndim != 3:
        raise ValueError(f"flows must be (windows, n, n); got {observed.shape}")
    u, v = nonnegative_factors(observed.sum(axis=0), rank=int(rank))
    columns = [base]
    for r in range(int(rank)):
        column = u[:, r][:, None] * v[:, r][None, :]
        norm = float(np.linalg.norm(column))
        if norm <= 0.0:
            raise ValueError(f"low-rank column {r} is identically zero")
        columns.append((column / norm)[:, :, None])
    return np.ascontiguousarray(np.concatenate(columns, axis=2), dtype=np.float64)


@dataclass(frozen=True)
class SharedThetaFit:
    """
    EN: the fitted shared parameter, the features it refers to, and the optimisation trace.
    CN: 拟合出的共享参数、对应的特征，以及优化轨迹。
    """

    theta: Float[np.ndarray, "k"]
    features: Float[np.ndarray, "n n k"]
    objective: float
    history: tuple[float, ...] = field(default=())
    iterations: int = 0

    @property
    def n_features(self) -> int:
        """EN: `K`. CN: `K`。"""
        return int(self.theta.size)

    def cost(self) -> Float[np.ndarray, "n n"]:
        """EN: the fitted cost matrix. CN: 拟合出的代价矩阵。"""
        return np.tensordot(self.features, self.theta, axes=([2], [0]))

    def predict(self, source_mass: Float[np.ndarray, "n"], target_mass: Float[np.ndarray, "n"]
                ) -> Float[np.ndarray, "n n"]:
        """EN: exact OT plan at the fitted cost and the given marginals. CN: 拟合代价与给定 marginal 下的精确 OT 计划。"""
        rows = np.asarray(source_mass, dtype=np.float64).reshape(-1)
        cols = np.asarray(target_mass, dtype=np.float64).reshape(-1)
        return np.asarray(ot.emd(rows, cols, self.cost(), numItermax=2_000_000), dtype=np.float64)


def regret_of(theta: Float[np.ndarray, "k"], features: Float[np.ndarray, "n n k"],
              plan: Float[np.ndarray, "n n"]) -> float:
    """
    EN: `R(theta) = <c(theta), pi_hat> - OT(c(theta))` for one plan, at the plan's own marginals.

        `R >= 0` and `R = 0` exactly when the plan is optimal at `c(theta)`; it is the formulation's own measure of
        how far the observation is from being explained. Reported on held-out windows, normalised by the value at
        the uniform `theta` so the number is comparable across instances.

    CN: 单个计划在自身 marginal 下的 `R(theta) = <c(theta),pi_hat> - OT(c(theta))`。`R>=0`，且 `R=0` 恰当计划
        在 `c(theta)` 处最优；它是该形式自身的"观测被解释到什么程度"的度量。在留出窗上报告，并用均匀 `theta` 处的值归一，
        使跨实例可比。
    """
    cost = np.tensordot(features, np.asarray(theta, dtype=np.float64), axes=([2], [0]))
    flow = np.asarray(plan, dtype=np.float64)
    rows = flow.sum(axis=1)
    cols = flow.sum(axis=0)
    return float(np.sum(cost * flow) - ot.emd2(rows, cols, cost, numItermax=2_000_000))


def fit_shared_theta(
    flows: Float[np.ndarray, "windows n n"],
    distance: Float[np.ndarray, "n n"],
    *,
    family: str = "distance_power",
    rank: int = 4,
    iterations: int = 200,
) -> SharedThetaFit:
    """
    EN: Frank–Wolfe on the simplex for the summed regret.

        Each iteration needs one exact OT solve per window, so the iteration count is kept modest and the objective
        trace is returned: a reader can see whether the run stopped at a plateau or hit the cap.

    CN: 单纯形上的 Frank–Wolfe，最小化求和 regret。每次迭代每窗一次精确 OT 求解，故迭代数取得适中，并返回目标轨迹：
        读者能看出是到平台还是被迭代上限截断。`family` 选择假设类，`rank` 只对 `distance_lowrank` 有意义。
    """
    observed = np.asarray(flows, dtype=np.float64)
    if observed.ndim != 3:
        raise ValueError(f"flows must be (windows, n, n); got {observed.shape}")
    if np.any(observed < 0.0):
        raise ValueError("flows must be non-negative")
    if int(iterations) < 1:
        raise ValueError("iterations must be at least 1")
    features = mixed_features(distance, observed, family=str(family), rank=int(rank))
    n_windows = int(observed.shape[0])
    k = int(features.shape[2])
    theta = np.full(k, 1.0 / k, dtype=np.float64)
    history: list[float] = []
    objective = _summed_regret(theta, features, observed, n_windows)
    history.append(objective)
    for step in range(1, int(iterations) + 1):
        gradient = np.zeros(k, dtype=np.float64)
        for window in range(n_windows):
            plan = observed[window]
            rows = plan.sum(axis=1)
            cols = plan.sum(axis=0)
            cost = np.tensordot(features, theta, axes=([2], [0]))
            forward = np.asarray(ot.emd(rows, cols, cost, numItermax=2_000_000), dtype=np.float64)
            gradient += np.tensordot(features, plan - forward, axes=([0, 1], [0, 1]))
        vertex = np.zeros(k, dtype=np.float64)
        vertex[int(np.argmin(gradient))] = 1.0
        step_size = 2.0 / (step + 2.0)
        candidate = (1.0 - step_size) * theta + step_size * vertex
        candidate_value = _summed_regret(candidate, features, observed, n_windows)
        # EN: FW with a 2/(t+2) step is monotone for a convex objective up to rounding; keep the better point.
        # CN: 凸目标下 2/(t+2) 步长的 FW 在舍入误差内单调；保留更好的点。
        if candidate_value <= objective:
            theta, objective = candidate, candidate_value
        history.append(objective)
    return SharedThetaFit(theta=theta, features=features, objective=float(objective),
                          history=tuple(history), iterations=int(iterations))


def _summed_regret(theta: Float[np.ndarray, "k"], features: Float[np.ndarray, "n n k"],
                   observed: Float[np.ndarray, "windows n n"], n_windows: int) -> float:
    """
    EN: `F(theta) = sum_l [<c, pi_hat_l> - OT_l(c)]`, one forward solve per window.

        Both terms are linear-or-concave in `theta`, so `F` is convex and `F >= 0`; `F = 0` exactly when every
        observed plan is optimal at `c(theta)`.

    CN: `F(theta) = sum_l [<c,pi_hat_l> - OT_l(c)]`，每窗一次前向求解。两项对 `theta` 分别线性与凹，故 `F` 凸且
        `F>=0`；`F=0` 恰当全部观测计划都在 `c(theta)` 处最优。
    """
    cost = np.tensordot(features, theta, axes=([2], [0]))
    total = 0.0
    for window in range(n_windows):
        plan = observed[window]
        rows = plan.sum(axis=1)
        cols = plan.sum(axis=0)
        total += float(np.sum(cost * plan) - ot.emd2(rows, cols, cost, numItermax=2_000_000))
    return float(total)
