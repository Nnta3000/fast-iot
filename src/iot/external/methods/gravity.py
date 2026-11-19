"""
EN: **Gravity model** — the parametric transport baseline reported in M-IOT's Table 1 (Bao, Li & Sun 2026, §6.3).

    The model in the paper: `T_ij ∝ O_i^α D_j^β d_ij^{-γ}`, with `α, β, γ` fitted by regression. This module fits the
    log-linear form by ordinary least squares and then puts the prediction on the target marginals by iterative
    proportional fitting, so the prediction is in the same absolute units as the observed flow.

    Every choice the paper leaves open is stated here:
      * the regression uses entries with **observed** flow and a **positive** distance (up to a common constant);
      * a pair with no observed trip still gets a prediction, because the exponent form is defined for any `d > 0`;
      * `d` is the mean reported trip distance of the **training** trips between that pair (see
        `studies/related/nyc_taxi/instance.py`); a pair with no train trip uses the global mean distance;
      * the same-zone pair (`d = 0`) is floored at `distance_floor`, since `d^{-γ}` is undefined there; the gravity
        model has no separate same-zone term;
      * one fit is shared by all training windows (the paper fits a single parametric model, and `α, β, γ` are not
        window-specific).

CN: **引力模型** —— M-IOT 表 1 中的参数化传输基线。模型为 `T_ij ∝ O_i^α D_j^β d_ij^{-γ}`，`α, β, γ` 由回归拟合。
    本模块用普通最小二乘在 log 线性形式上拟合，再用 IPF 把预测放到目标 marginal 上，使其与观测流量同量纲。论文未规定的
    选择都在此写明：回归只用**有观测流量**且**距离为正**的元素；无观测行程的对仍有预测（指数形式对任意 `d>0` 有定义）；
    `d` 是该对**训练**行程的平均报告距离，无训练行程的对用全局平均距离；同 zone 对（`d=0`）用 `distance_floor` 兜底，
    引力模型没有单独的同 zone 项；一次拟合适用于全部训练窗（论文只给一组参数）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from ..metrics import align_to_marginals

__all__ = ["GravityFit", "fit_gravity"]


@dataclass(frozen=True)
class GravityFit:
    """
    EN: the four fitted coefficients and the distance floor. `alpha` and `beta` are the demand elasticities,
        `gamma` the distance-decay exponent (positive: a larger `gamma` means flow decays faster with distance).
    CN: 四个拟合系数与距离下限。`alpha`、`beta` 是需求弹性，`gamma` 是距离衰减指数（越大衰减越快）。
    """

    intercept: float
    alpha: float
    beta: float
    gamma: float
    distance_floor: float
    distance_fill: float = 0.0

    def predict(
        self,
        source_mass: Float[np.ndarray, "n"],
        target_mass: Float[np.ndarray, "n"],
        distance: Float[np.ndarray, "n n"],
        *,
        iterations: int = 500,
    ) -> Float[np.ndarray, "n n"]:
        """
        EN: predict the flow matrix at the given marginals.

            `T = exp(intercept + alpha*log O + beta*log D - gamma*log d)`, then rescaled onto `(source, target)` by
            IPF. The rescaling is what makes the prediction comparable in absolute units; without it the fitted
            constant sets an arbitrary total.

        CN: 在给定 marginal 下预测流量矩阵：先按指数形式算 `T`，再用 IPF 缩放到 `(source, target)`。重标定使预测与观测
            同量纲；不做的话拟合常数决定了一个任意总量。
        """
        rows = np.asarray(source_mass, dtype=np.float64).reshape(-1)
        cols = np.asarray(target_mass, dtype=np.float64).reshape(-1)
        spacing = np.asarray(distance, dtype=np.float64)
        if spacing.shape != (rows.size, cols.size):
            raise ValueError(f"distance shape {spacing.shape} does not match ({rows.size}, {cols.size})")
        # EN: a pair with no training trip has no distance (`NaN`); it is filled with the declared `distance_fill`
        #     (the median observed distance) before the floor is applied. NaN must not reach `log`.
        # CN: 无训练行程的对没有距离（`NaN`），先用声明的 `distance_fill`（观测距离中位数）补齐，再取下限；NaN 绝不能进入
        #     `log`。
        spacing = np.where(np.isnan(spacing), float(self.distance_fill), spacing)
        safe_distance = np.maximum(spacing, float(self.distance_floor))
        log_rows = np.log(np.maximum(rows, 1e-300))[:, None]
        log_cols = np.log(np.maximum(cols, 1e-300))[None, :]
        log_flow = (float(self.intercept) + float(self.alpha) * log_rows + float(self.beta) * log_cols
                    - float(self.gamma) * np.log(safe_distance))
        # EN: subtract the maximum before exponentiating, then rescale; this keeps the pattern and avoids overflow.
        # CN: 指数化前先减去最大值，再重标定；保留模式且避免溢出。
        flow = np.exp(log_flow - float(log_flow.max()))
        # EN: the floor above keeps every entry positive, which is what IPF needs.
        # CN: 上面的下限保证每个元素为正，这正是一步 IPF 所需要的。
        return align_to_marginals(flow, rows, cols, iterations=int(iterations))


def fit_gravity(
    train_flows: Float[np.ndarray, "windows n n"],
    distance: Float[np.ndarray, "n n"],
    *,
    distance_floor: float | None = None,
) -> GravityFit:
    """
    EN: least squares on `log T = c + α log O + β log D − γ log d`.

        Only entries with `T > 0` and `d > 0` enter the regression (a log of zero is not an observation). The
        design matrix is `[1, log O, log D, log d]` and `γ` is returned as `−coefficient(log d)` to keep the
        paper's sign convention, so a positive `gamma` means the flow decays with distance.

        With fewer than five usable entries the least-squares problem is under-determined; that is raised rather
        than answered with an arbitrary solution.

    CN: 对 `log T = c + α log O + β log D − γ log d` 做最小二乘。只有 `T>0` 且 `d>0` 的元素进入回归（零的对数不是观测）。
        无训练行程的对（距离为 `NaN`）不进入回归，预测时用观测距离中位数补齐（`distance_fill`）。
        设计矩阵为 `[1, log O, log D, log d]`，返回的 `γ = −coef(log d)` 以保持论文的符号约定（`gamma` 为正表示随距离衰减）。
        可用元素少于五个时问题欠定，直接报错而不是给出任意解。
    """
    flows = np.asarray(train_flows, dtype=np.float64)
    spacing = np.asarray(distance, dtype=np.float64)
    if flows.ndim != 3:
        raise ValueError(f"train_flows must be (windows, n, n); got {flows.shape}")
    if spacing.shape != flows.shape[1:]:
        raise ValueError(f"distance shape {spacing.shape} does not match the flows {flows.shape[1:]}")
    if np.any(flows < 0.0):
        raise ValueError("train_flows must be non-negative")
    floor = float(distance_floor) if distance_floor is not None else _default_floor(spacing)
    # EN: the imputation for pairs with no training trip: the median observed distance, one stated assumption.
    # CN: 无训练行程对的插补值：观测距离的中位数，一个明确陈述的假设。
    observed_distances = spacing[~np.isnan(spacing)]
    fill = float(np.median(observed_distances)) if observed_distances.size else 0.0

    rows_source = flows.sum(axis=2)          # (windows, n)
    rows_target = flows.sum(axis=1)          # (windows, n)
    design: list[np.ndarray] = []
    response: list[np.ndarray] = []
    for window in range(flows.shape[0]):
        observed = (flows[window] > 0.0) & (spacing > 0.0)
        if not np.any(observed):
            continue
        index = np.argwhere(observed)
        i, j = index[:, 0], index[:, 1]
        design.append(np.column_stack([
            np.ones(i.size),
            np.log(rows_source[window][i]),
            np.log(rows_target[window][j]),
            np.log(spacing[i, j]),
        ]))
        response.append(np.log(flows[window][i, j]))
    if not design:
        raise ValueError("no entry has both observed flow and a positive distance")
    a = np.vstack(design)
    y = np.concatenate(response)
    if a.shape[0] < 5:
        raise ValueError(f"only {a.shape[0]} usable entries; the four-parameter fit is under-determined")
    coefficients, *_ = np.linalg.lstsq(a, y, rcond=None)
    intercept, alpha, beta, log_distance = (float(value) for value in coefficients)
    return GravityFit(intercept=intercept, alpha=alpha, beta=beta, gamma=-log_distance,
                      distance_floor=floor, distance_fill=fill)


def _default_floor(distance: Float[np.ndarray, "n n"]) -> float:
    """EN: half the smallest positive pair distance (a same-zone pair has zero distance). CN: 最小正距离的一半。"""
    positive = distance[distance > 0.0]
    if positive.size == 0:
        raise ValueError("distance has no positive entry")
    return 0.5 * float(positive.min())
