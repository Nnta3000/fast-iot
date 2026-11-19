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

__all__ = ["DISTANCE_PROFILES", "SharedThetaFit", "distance_features", "fit_shared_theta", "mixed_features",
           "nonnegative_factors", "regret_of", "regret_share"]


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


DISTANCE_PROFILES: tuple[str, ...] = ("s", "s2", "log1p", "same_zone", "s3", "sqrt", "exp", "s4")


def distance_features(distance: Float[np.ndarray, "n n"], *, family: str = "distance_power",
                      include_constant: bool = False, unsupported: str = "median",
                      fill_value: float | None = None, normalise: bool = True, width: int = 4
                      ) -> Float[np.ndarray, "n n k"]:
    """
    EN: the feature family for a distance-structured cost, as `(n, n, K)` non-negative columns.

        `distance_power` uses `{1, s, s^2, log(1 + s)}` with `s = d / max(d)`, plus a same-zone indicator. The family
        is deliberately **not** a set of bandwidths of one kernel: `exp(-d/sigma)` at different `sigma` are monotone
        transforms of one another, so they rank every candidate pair identically and `theta` cannot express a
        preference between them (measured on another instance in `notes/KERNEL_DICTIONARY_AXIS.md`). A weighted
        *sum* of `1, s, s^2, log(1+s)` changes the ranking.

        **Pairs with no training trip carry `NaN`, and are never silently zero.** Two declared treatments:
        `unsupported="median"` replaces them by `fill_value` (the median observed distance when that is `None`), a
        single stated assumption; `unsupported="mask"` keeps the distance columns at zero there and appends one
        indicator column equal to 1 on exactly those pairs. The mask can only *add* cost where theta >= 0, so it can
        never make an unexplored pair cheaper than an observed one -- it is reported as a sensitivity, not as the
        model.

        **A constant column is excluded by default, and that is a measured decision, not a tidiness one.** Adding a
        constant to a cost leaves every OT plan unchanged, so a constant column looks like a harmless gauge
        direction. It is not, under `min_theta R`: a *constant cost* makes every plan matching the marginals
        optimal, so `R = 0` for **any** observation. On the NYC taxi instance the arm with the constant column
        returned `theta = e_constant` and `R_test / R_uniform = 5e-5` on all six held-out windows — a perfect
        "generalisation" that says nothing about the data. The column is therefore off unless explicitly requested,
        and the degeneracy is documented where the fit is reported.

    CN: **无训练行程的对存 `NaN`，绝不静默置零。** 两种声明式处理：`unsupported="median"` 用 `fill_value`（未给出时取
        观测距离的中位数）替换，是一个明确陈述的假设；`unsupported="mask"` 让距离列在这些对上取 0，并追加一列指示（恰在这些
        对上为 1）。由于 `theta>=0`，掩码只能**增加**代价，永远不能让未探索的对比已观测的对更便宜 —— 故作为敏感性报告，而不是
        主模型。

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
    missing = np.isnan(spacing)
    observed = ~missing
    if np.any(spacing[observed] < 0.0):
        raise ValueError("distance must be non-negative where observed")
    if not np.any(observed):
        raise ValueError("distance has no observed entry")
    treatment = str(unsupported)
    if treatment not in {"median", "mask"}:
        raise ValueError(f"unsupported must be 'median' or 'mask'; got {unsupported!r}")
    fill = float(fill_value) if fill_value is not None else float(np.median(spacing[observed]))
    if treatment == "median":
        spacing = np.where(missing, fill, spacing)
    else:
        spacing = np.where(missing, 0.0, spacing)
    scale = float(spacing.max())
    if scale <= 0.0:
        raise ValueError("distance is identically zero")
    normalised = spacing / scale
    if str(family) == "distance_power":
        if int(width) not in {4, 8}:
            raise ValueError(f"width must be 4 or 8; got {width!r}")
        identity = np.eye(spacing.shape[0], dtype=np.float64)
        profiles = {
            "s": normalised,
            "s2": normalised ** 2,
            "log1p": np.log1p(normalised),
            "same_zone": identity,
            "s3": normalised ** 3,
            "sqrt": np.sqrt(normalised),
            "exp": np.exp(-normalised),
            "s4": normalised ** 4,
        }
        # EN: the list is ordered, so `width` is a prefix: 4 reproduces the original family, 8 adds four profiles.
        # CN: 列表有序，故 `width` 取前缀：4 即原族，8 再加四种剖面。
        columns = [profiles[name] for name in DISTANCE_PROFILES[: int(width)]]
        if treatment == "mask":
            columns.append(missing.astype(np.float64))
        if bool(include_constant):
            columns.insert(0, np.ones_like(normalised))
        tensor = np.stack(columns, axis=2)
        if bool(normalise):
            # EN: **scale gauge.** `R(alpha c) = alpha R(c)`, so over unnormalised columns the simplex can lower the
            #     regret by selecting the column with the smallest entries, with no better explanation. Scaling every
            #     column to unit Frobenius norm removes exactly that direction; the recovered scale is not reported
            #     because the cost is only identified up to it.
            # CN: **尺度规范。** `R(alpha c) = alpha R(c)`，故在未归一化的列上，单纯形可以通过选择元素最小的列来降低
            #     regret，而解释并未变好。把每列缩放到单位 Frobenius 范数恰好去掉这个方向；恢复出的尺度不报告，因为代价只在
            #     该意义下可辨识。
            flat = tensor.reshape(-1, tensor.shape[2])
            norms = np.linalg.norm(flat, axis=0)
            if np.any(norms <= 0.0):
                raise ValueError("a feature column is identically zero")
            tensor = tensor / norms[None, None, :]
        return np.ascontiguousarray(tensor, dtype=np.float64)
    raise ValueError(f"unknown feature family {family!r}")


def mixed_features(distance: Float[np.ndarray, "n n"], flows: Float[np.ndarray, "windows n n"], *,
                   family: str = "distance_power", rank: int = 4,
                   include_constant: bool = False, unsupported: str = "median",
                   fill_value: float | None = None, width: int = 4) -> Float[np.ndarray, "n n k"]:
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
    base = distance_features(distance, family="distance_power", include_constant=include_constant,
                             unsupported=str(unsupported), fill_value=fill_value, width=int(width))
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
    feature_maps: tuple[Float[np.ndarray, "n n k"], ...]
    objective: float
    history: tuple[float, ...] = field(default=())
    iterations: int = 0
    forbidden_mass: float = 0.0

    @property
    def n_features(self) -> int:
        """EN: `K`. CN: `K`。"""
        return int(self.theta.size)

    @property
    def features(self) -> Float[np.ndarray, "n n k"]:
        """EN: the feature tensor of the first window (all of them when the geometry is shared). CN: 第一个窗的特征张量。"""
        return self.feature_maps[0]

    def cost(self, window: int = 0) -> Float[np.ndarray, "n n"]:
        """
        EN: the fitted cost on window `window`'s geometry.

            With a shared geometry every window gives the same matrix; with fresh points per window they differ,
            and the cost *function* is the same while its values are not.

        CN: 在窗 `window` 的几何上的拟合代价。几何共享时各窗相同；每窗新取点时不同 —— 代价**函数**相同而取值不同。
        """
        return np.tensordot(self.feature_maps[int(window)], self.theta, axes=([2], [0]))

    def predict(self, source_mass: Float[np.ndarray, "n"], target_mass: Float[np.ndarray, "n"], *,
                window: int = 0, features: Float[np.ndarray, "n n k"] | None = None
                ) -> Float[np.ndarray, "n n"]:
        """
        EN: exact OT plan at the fitted cost and the given marginals.

            `features` is the geometry to score on when it is not one the fit saw — a held-out window whose points
            were never given to `fit_shared_theta` has no entry in `feature_maps`, so the caller passes its feature
            tensor here. Without it, `window` selects one of the training geometries.

        CN: 拟合代价与给定 marginal 下的精确 OT 计划。当被预测的几何不在拟合所见之内（留出窗的点从未交给
            `fit_shared_theta`，故 `feature_maps` 里没有它），由调用方在 `features` 传入其特征张量；不给时用 `window`
            选择某个训练几何。
        """
        rows = np.asarray(source_mass, dtype=np.float64).reshape(-1)
        cols = np.asarray(target_mass, dtype=np.float64).reshape(-1)
        cost = (np.tensordot(np.asarray(features, dtype=np.float64), self.theta, axes=([2], [0]))
                if features is not None else self.cost(window))
        return np.asarray(ot.emd(rows, cols, cost, numItermax=2_000_000), dtype=np.float64)


def regret_of(theta: Float[np.ndarray, "k"], features: Float[np.ndarray, "n n k"],
              plan: Float[np.ndarray, "n n"], *, allowed: np.ndarray | None = None) -> float:
    """
    EN: `R(theta) = <c(theta), pi_hat> - OT_E(c(theta))` for one plan, at the plan's own marginals.

        `R >= 0` and `R = 0` exactly when the plan is optimal at `c(theta)` (over `E` when `allowed` is given). It is
        the formulation's own measure of how far the observation is from being explained. Reported on held-out
        windows, normalised by the value at the uniform `theta`.

    CN: 单个计划在自身 marginal 下的 `R(theta) = <c(theta),pi_hat> - OT_E(c(theta))`（给出 `allowed` 时在 `E` 上取
        OT）。`R>=0`，且 `R=0` 恰当计划在 `c(theta)` 处最优。它是该形式自身的"观测被解释到什么程度"的度量；在留出窗上报告，
        并用均匀 `theta` 处的值归一。
    """
    cost = np.tensordot(features, np.asarray(theta, dtype=np.float64), axes=([2], [0]))
    flow = np.asarray(plan, dtype=np.float64)
    rows = flow.sum(axis=1)
    cols = flow.sum(axis=0)
    forward = _forward_plan(cost, rows, cols, allowed)
    return float(np.sum(cost * flow) - np.sum(cost * forward))


def regret_share(theta: Float[np.ndarray, "k"], features: Float[np.ndarray, "n n k"],
                 plan: Float[np.ndarray, "n n"], *, allowed: np.ndarray | None = None) -> tuple[float, float]:
    """
    EN: `(share, fraction)` — two scale-free readings of the same regret.

        `share = R(plan) / R(independent plan)`: the independent plan is the product of the marginals, i.e. "pickup and
        drop-off are unrelated". 0 means the observed flow is as cheap as the forward optimum, 1 means it is as
        wasteful as the unrelated plan. A share above 1 means the observation is *more* wasteful than independence.

        `fraction = R(plan) / <c, plan>`: the regret expressed as a share of the flow's own transport cost, which is
        the "how much of the total cost is waste" reading.

        Both are invariant to `c -> alpha c` for `alpha > 0`, so neither can be lowered by shrinking the cost — the
        bias a bare `R` has on a simplex over features of different magnitudes.

    CN: `(share, fraction)` —— 同一个 regret 的两种无量纲读法。`share = R(计划)/R(独立计划)`：独立计划是 marginal 的乘积，
        即"上车与下车无关"。0 表示观测流量与前向最优一样便宜，1 表示它与无关计划一样浪费；大于 1 表示比独立还浪费。
        `fraction = R(计划)/<c,计划>`：把 regret 表示为该流量自身运输代价的一部分，即"总代价里有多少是浪费"。两者对
        `c -> alpha c`（`alpha>0`）不变，故不能靠缩小代价来降低 —— 这正是裸 `R` 在不同幅度特征上的偏差。
    """
    rows = np.asarray(plan, dtype=np.float64).sum(axis=1)
    cols = np.asarray(plan, dtype=np.float64).sum(axis=0)
    independent = np.outer(rows, cols) / float(rows.sum())
    own = regret_of(theta, features, plan, allowed=allowed)
    unrelated = regret_of(theta, features, independent, allowed=allowed)
    cost = np.tensordot(features, np.asarray(theta, dtype=np.float64), axes=([2], [0]))
    transport = float(np.sum(cost * np.asarray(plan, dtype=np.float64)))
    share = own / unrelated if unrelated > 0.0 else float("nan")
    fraction = own / transport if transport > 0.0 else float("nan")
    return float(share), float(fraction)


def fit_shared_theta(
    flows: Float[np.ndarray, "windows n n"],
    distance: Float[np.ndarray, "n n"] | tuple[Float[np.ndarray, "n n"], ...],
    *,
    family: str = "distance_power",
    rank: int = 4,
    iterations: int = 200,
    unsupported: str = "median",
    restrict_edges: bool = False,
    width: int = 4,
) -> SharedThetaFit:
    """
    EN: Frank–Wolfe on the simplex for the summed regret.

        Each iteration needs one exact OT solve per window, so the iteration count is kept modest and the objective
        trace is returned: a reader can see whether the run stopped at a plateau or hit the cap.

    EN: `restrict_edges=True` forbids every pair that had no training trip, which is the edge-set formulation (E
        strictly inside the full grid) and the analogue of the graph restriction in Stuart & Wolfram. Forbidden
        pairs are given a large finite cost rather than an infinite one, so the LP stays finite; the mass the forward
        plan places on forbidden pairs is reported by `SharedThetaFit.forbidden_mass` and must be zero for the
        restriction to be doing what it says.

    CN: `restrict_edges=True` 禁止所有没有训练行程的对，这正是边集形式（E 严格包含于全网格），也是 Stuart & Wolfram
        图限制的类比。被禁的对给一个**大而有限**的代价而不是无穷，使 LP 保持有限；前向计划落在被禁对上的质量由
        `SharedThetaFit.forbidden_mass` 报告，必须为 0 才算限制真正生效。

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
    maps = _feature_maps(distance, observed, family=str(family), rank=int(rank),
                         unsupported=str(unsupported), width=int(width))
    if len(maps) not in {1, int(observed.shape[0])}:
        raise ValueError(f"got {len(maps)} distance matrices for {observed.shape[0]} windows; "
                         "pass one matrix or one per window")
    if len(maps) == 1:
        maps = maps * int(observed.shape[0])
    alloweds: list[np.ndarray | None] = [None] * len(maps)
    if bool(restrict_edges):
        alloweds = [~np.isnan(np.asarray(d, dtype=np.float64))
                    for d in ([distance] if np.ndim(distance) == 2 else list(distance))]
    features = maps[0]
    n_windows = int(observed.shape[0])
    k = int(features.shape[2])
    theta = np.full(k, 1.0 / k, dtype=np.float64)
    history: list[float] = []
    objective = _summed_regret(theta, maps, observed, alloweds)
    history.append(objective)
    for step in range(1, int(iterations) + 1):
        gradient = np.zeros(k, dtype=np.float64)
        for window in range(n_windows):
            plan = observed[window]
            rows = plan.sum(axis=1)
            cols = plan.sum(axis=0)
            window_features = maps[window]
            cost = np.tensordot(window_features, theta, axes=([2], [0]))
            forward = _forward_plan(cost, rows, cols, alloweds[window])
            gradient += np.tensordot(window_features, plan - forward, axes=([0, 1], [0, 1]))
        vertex = np.zeros(k, dtype=np.float64)
        vertex[int(np.argmin(gradient))] = 1.0
        step_size = 2.0 / (step + 2.0)
        candidate = (1.0 - step_size) * theta + step_size * vertex
        candidate_value = _summed_regret(candidate, maps, observed, alloweds)
        # EN: FW with a 2/(t+2) step is monotone for a convex objective up to rounding; keep the better point.
        # CN: 凸目标下 2/(t+2) 步长的 FW 在舍入误差内单调；保留更好的点。
        if candidate_value <= objective:
            theta, objective = candidate, candidate_value
        history.append(objective)
    forbidden_mass = 0.0
    if any(allowed is not None for allowed in alloweds):
        # EN: report the mass the forward plan places on forbidden pairs; a nonzero value means the restriction is
        #     not binding as stated (the LP needed those edges), which must not be silently accepted.
        # CN: 报告前向计划落在被禁对上的质量；非零说明限制并不像声明的那样生效（LP 需要那些边），不能静默接受。
        masses = []
        for w in range(int(observed.shape[0])):
            if alloweds[w] is None:
                continue
            cost = np.tensordot(maps[w], theta, axes=([2], [0]))
            plan = _forward_plan(cost, observed[w].sum(axis=1), observed[w].sum(axis=0), alloweds[w])
            masses.append(float(plan[~alloweds[w]].sum()))
        forbidden_mass = float(sum(masses))
    return SharedThetaFit(theta=theta, feature_maps=tuple(maps), objective=float(objective),
                          history=tuple(history), iterations=int(iterations),
                          forbidden_mass=forbidden_mass)


FORBIDDEN_COST: float = 1e3


def _feature_maps(distance: Float[np.ndarray, "n n"] | tuple[Float[np.ndarray, "n n"], ...],
                  flows: Float[np.ndarray, "windows n n"], *, family: str, rank: int, unsupported: str,
                  width: int = 4) -> list[Float[np.ndarray, "n n k"]]:
    """
    EN: one feature tensor per window, from one shared distance matrix or one matrix per window.

        The low-rank family needs a single geometry (its factors are built from the pooled flows), so a per-window
        distance list is rejected for it rather than silently averaged.

    CN: 每窗一个特征张量；输入可以是共享的一个距离矩阵，也可以是每窗一个。低秩族需要单一几何（其因子由汇总流量构造），
        故对逐窗距离列表直接报错，而不是悄悄平均。
    """
    if np.ndim(distance) == 2:
        return [mixed_features(np.asarray(distance, dtype=np.float64), flows, family=str(family),
                               rank=int(rank), unsupported=str(unsupported), width=int(width))]
    matrices = [np.asarray(d, dtype=np.float64) for d in distance]           # type: ignore[union-attr]
    if str(family) != "distance_power":
        raise ValueError(f"family {family!r} needs one shared geometry; got {len(matrices)} distance matrices")
    return [mixed_features(d, flows, family=str(family), rank=int(rank), unsupported=str(unsupported),
                           width=int(width))
            for d in matrices]


def _forward_plan(cost: Float[np.ndarray, "n n"], rows: Float[np.ndarray, "n"], cols: Float[np.ndarray, "n"],
                  allowed: np.ndarray | None) -> Float[np.ndarray, "n n"]:
    """
    EN: exact OT plan, restricted to `allowed` pairs when given.

        Forbidden pairs get `FORBIDDEN_COST` (large but finite) instead of infinity: POT's network simplex works with
        finite costs, and a large one keeps the problem feasible. `SharedThetaFit.forbidden_mass` reports whether any
        mass still landed there.

    CN: 精确 OT 计划；给出 `allowed` 时限制在这些对上。被禁对用 `FORBIDDEN_COST`（大而有限）而不是无穷：POT 的网络单纯形
        使用有限代价，大的有限值使问题保持可行。是否有质量仍落在那里由 `SharedThetaFit.forbidden_mass` 报告。
    """
    cost = np.asarray(cost, dtype=np.float64)
    if allowed is not None:
        cost = np.where(np.asarray(allowed, dtype=bool), cost, FORBIDDEN_COST)
    return np.asarray(ot.emd(np.asarray(rows, dtype=np.float64), np.asarray(cols, dtype=np.float64), cost,
                             numItermax=2_000_000), dtype=np.float64)


def _summed_regret(theta: Float[np.ndarray, "k"], feature_maps: list[Float[np.ndarray, "n n k"]],
                   observed: Float[np.ndarray, "windows n n"],
                   alloweds: list[np.ndarray | None]) -> float:
    """
    EN: `F(theta) = sum_l [<c, pi_hat_l> - OT_l(c)]`, one forward solve per window.

        Both terms are linear-or-concave in `theta`, so `F` is convex and `F >= 0`; `F = 0` exactly when every
        observed plan is optimal at `c(theta)`.

    CN: `F(theta) = sum_l [<c,pi_hat_l> - OT_l(c)]`，每窗一次前向求解。两项对 `theta` 分别线性与凹，故 `F` 凸且
        `F>=0`；`F=0` 恰当全部观测计划都在 `c(theta)` 处最优。
    """
    total = 0.0
    for window in range(int(observed.shape[0])):
        cost = np.tensordot(feature_maps[window], theta, axes=([2], [0]))
        plan = observed[window]
        rows = plan.sum(axis=1)
        cols = plan.sum(axis=0)
        forward = _forward_plan(cost, rows, cols, alloweds[window])
        total += float(np.sum(cost * plan) - np.sum(cost * forward))
    return float(total)
