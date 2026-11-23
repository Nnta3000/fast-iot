"""
EN: **Metrics the comparison papers report**, plus the one preprocessing step a flow comparison needs.

    All four functions take arrays and return a float; none of them knows what generated the plan. That keeps them
    usable from a baseline, from our own arm, and from a test, without importing anything from `experiments`.

    Definitions follow the papers quoted in `notes/RELATED_WORK_AND_COMPARABILITY.md`:

      * `rmse` and `pearson` are the usual ones, computed over **every** entry of the flow matrix (including zeros),
        which is what M-IOT's Table 1 reports;
      * `cpc` is the Common Part of Commuters, `2 * sum(min(X, X_hat)) / (sum(X) + sum(X_hat))`, in `[0, 1]`;
      * `relative_frobenius` is Ma et al.'s cost-recovery error, `||a - b||_F / ||b||_F`.

    `align_to_marginals` exists because a flow comparison is in **absolute** units: two plans can order the same
    pairs but differ in total mass, and RMSE between them is then dominated by the mass. Iterative proportional
    fitting puts a predicted plan on the observed row and column sums without touching its relative structure, which
    is the standard way to compare a transport model's *pattern* at the observed demand.

CN: **比较论文所报告的指标**，以及流量比较所需的那一步预处理。四个函数只接收数组、返回浮点数，不关心计划由谁生成，
    因此基线、我们自己的臂和测试都能用，且不导入 `experiments`。定义取自 `notes/RELATED_WORK_AND_COMPARABILITY.md`
    所引论文：`rmse`/`pearson` 在流量矩阵**全部**元素上计算（含零，M-IOT 表 1 即如此）；`cpc` 为
    `2*sum(min(X, X_hat))/(sum(X)+sum(X_hat))`，取值 `[0,1]`；`relative_frobenius` 为 Ma 等的代价恢复误差。
    `align_to_marginals` 存在的原因：流量比较是**绝对**量纲，两个计划可以对相同的配对排序而总质量不同，此时 RMSE 被质量
    主导；IPF 把预测计划放到观测的行列和上而不改变其相对结构，是比较传输模型"模式"的标准做法。
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float

__all__ = [
    "FlowScores",
    "align_to_marginals",
    "cpc",
    "flow_scores",
    "pearson",
    "relative_frobenius",
    "rmse",
]


def _as_flow(value: Float[np.ndarray, "m n"]) -> Float[np.ndarray, "m n"]:
    """EN: validate a flow matrix. CN: 校验流量矩阵。"""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"a flow matrix must be 2D; got shape={array.shape}")
    if array.size < 1:
        raise ValueError("a flow matrix must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("a flow matrix must be finite")
    if np.any(array < 0.0):
        raise ValueError("a flow matrix must be non-negative")
    return array


def rmse(predicted: Float[np.ndarray, "m n"], observed: Float[np.ndarray, "m n"]) -> float:
    """
    EN: root mean squared error over every entry. CN: 逐元素的均方根误差。
    """
    prediction = _as_flow(predicted)
    target = _as_flow(observed)
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {target.shape}")
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def pearson(predicted: Float[np.ndarray, "m n"], observed: Float[np.ndarray, "m n"]) -> float:
    """
    EN: Pearson correlation over every entry; `nan` when either side is constant (undefined, not zero).
    CN: 逐元素的 Pearson 相关；任一侧为常数时返回 `nan`（未定义，而非 0）。
    """
    prediction = _as_flow(predicted).reshape(-1)
    target = _as_flow(observed).reshape(-1)
    if prediction.size != target.size:
        raise ValueError(f"size mismatch: {prediction.size} vs {target.size}")
    centred_prediction = prediction - prediction.mean()
    centred_target = target - target.mean()
    denominator = float(np.linalg.norm(centred_prediction) * np.linalg.norm(centred_target))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(centred_prediction, centred_target) / denominator)


def cpc(predicted: Float[np.ndarray, "m n"], observed: Float[np.ndarray, "m n"]) -> float:
    """
    EN: Common Part of Commuters, `2*sum(min(X, X_hat)) / (sum(X) + sum(X_hat))`; `nan` when both are empty.
    CN: 通勤公共部分 `2*sum(min(X, X_hat))/(sum(X)+sum(X_hat))`；两者皆空时返回 `nan`。
    """
    prediction = _as_flow(predicted)
    target = _as_flow(observed)
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {target.shape}")
    total = float(prediction.sum() + target.sum())
    if total <= 0.0:
        return float("nan")
    return float(2.0 * np.minimum(prediction, target).sum() / total)


def relative_frobenius(estimate: Float[np.ndarray, "m n"], truth: Float[np.ndarray, "m n"]) -> float:
    """
    EN: `||estimate - truth||_F / ||truth||_F` (Ma et al.'s cost-recovery error).
    CN: `||估计 - 真值||_F / ||真值||_F`（Ma 等的代价恢复误差）。
    """
    value = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(truth, dtype=np.float64)
    if value.shape != reference.shape:
        raise ValueError(f"shape mismatch: {value.shape} vs {reference.shape}")
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(value - reference) / denominator)


def align_to_marginals(
    plan: Float[np.ndarray, "m n"],
    source_mass: Float[np.ndarray, "m"],
    target_mass: Float[np.ndarray, "n"],
    *,
    iterations: int = 500,
    tol: float = 1e-12,
) -> Float[np.ndarray, "m n"]:
    """
    EN: iterative proportional fitting: rescale `plan` onto the given row/column sums.

        The result is `diag(r) @ plan @ diag(c)`, so what is preserved is the multiplicative structure (the
        cross-ratios `x_ij x_kl / (x_il x_kj)`), not each row's internal ratios: the column factors differ per
        column, so within-row proportions do move. That is the standard property of IPF and the reason it is the
        right comparison for a *pattern* at a different total demand.

        `plan` must be strictly positive wherever a row or column has mass to place (a zero row with positive
        `source_mass` cannot be rescaled). The result satisfies `rows = source_mass` and `cols = target_mass` to
        `tol`, or a `RuntimeError` is raised: a silently unconverged rescaling would move the metric.

    CN: 迭代比例拟合：把 `plan` 缩放到给定的行/列和上。结果是 `diag(r) @ plan @ diag(c)`，因此保留的是乘性结构
        （交叉比 `x_ij x_kl/(x_il x_kj)`），而不是每行内部的比例：列因子逐列不同，行内比例确实会变。这是 IPF 的标准性质，
        也正是在不同总需求下比较"模式"用它的原因。凡有质量要分配的行列，`plan` 必须严格正（正
        `source_mass` 的整行零无法缩放）。结果满足 `rows=source_mass`、`cols=target_mass`（误差 `tol`），否则抛
        `RuntimeError`：静默未收敛的重标定会改变指标。
    """
    value = np.asarray(plan, dtype=np.float64)
    rows = np.asarray(source_mass, dtype=np.float64).reshape(-1)
    cols = np.asarray(target_mass, dtype=np.float64).reshape(-1)
    if value.ndim != 2:
        raise ValueError(f"plan must be 2D; got shape={value.shape}")
    if value.shape[0] != rows.size or value.shape[1] != cols.size:
        raise ValueError(f"plan shape {value.shape} does not match masses ({rows.size}, {cols.size})")
    if np.any(value < 0.0) or np.any(rows < 0.0) or np.any(cols < 0.0):
        raise ValueError("plan and masses must be non-negative")
    if abs(float(rows.sum()) - float(cols.sum())) > 1e-8 * max(1.0, float(rows.sum())):
        raise ValueError(f"masses are incompatible: sum(source)={rows.sum()!r} sum(target)={cols.sum()!r}")
    # EN: the stopping rule is **relative** to the total mass: trips count in the hundreds of thousands, so an
    #     absolute 1e-12 is below the arithmetic's resolution and would never be reached.
    # CN: 收敛判据相对总质量：行程计数在十万量级，绝对 1e-12 低于浮点分辨率，永远达不到。
    scale = max(1.0, float(rows.sum()))
    out = np.array(value, dtype=np.float64, copy=True)
    # EN: a row/column with mass but no support cannot be rescaled; fail here rather than produce NaN.
    # CN: 有质量却无支撑的行列无法缩放，先在此报错，避免产生 NaN。
    if np.any((rows > 0.0) & (out.sum(axis=1) <= 0.0)):
        raise ValueError("a row with positive source_mass has no support in the plan")
    if np.any((cols > 0.0) & (out.sum(axis=0) <= 0.0)):
        raise ValueError("a column with positive target_mass has no support in the plan")
    for _ in range(int(iterations)):
        row_sums = out.sum(axis=1)
        out *= np.divide(rows, row_sums, out=np.zeros_like(rows), where=row_sums > 0.0)[:, None]
        col_sums = out.sum(axis=0)
        out *= np.divide(cols, col_sums, out=np.zeros_like(cols), where=col_sums > 0.0)[None, :]
        residual = max(float(np.abs(out.sum(axis=1) - rows).max()), float(np.abs(out.sum(axis=0) - cols).max()))
        if residual <= float(tol) * scale:
            return out
    raise RuntimeError(f"align_to_marginals did not converge in {iterations} iterations "
                       f"(residual={residual:.3e}, scale={scale:.3e})")


def flow_scores(predicted: Float[np.ndarray, "m n"], observed: Float[np.ndarray, "m n"]) -> "FlowScores":
    """
    EN: the three metrics M-IOT's Table 1 reports, computed together. CN: M-IOT 表 1 的三个指标，一次算完。
    """
    return FlowScores(rmse=rmse(predicted, observed), correlation=pearson(predicted, observed),
                      cpc=cpc(predicted, observed))


class FlowScores:
    """
    EN: the triple reported for a flow prediction, with the direction of each metric in the docstring so a table
        cannot invert one by accident (`rmse` lower is better; `correlation` and `cpc` higher is better).
    CN: 流量预测所报告的三个指标；方向写在文档里，避免表格把某个指标的方向搞反（`rmse` 越低越好；`correlation`、`cpc`
        越高越好）。
    """

    __slots__ = ("rmse", "correlation", "cpc")

    def __init__(self, *, rmse: float, correlation: float, cpc: float) -> None:
        self.rmse = float(rmse)
        self.correlation = float(correlation)
        self.cpc = float(cpc)

    def to_dict(self) -> dict[str, float]:
        """EN: JSON-ready. CN: 可直接写 JSON。"""
        return {"rmse": self.rmse, "correlation": self.correlation, "cpc": self.cpc}

    def __repr__(self) -> str:  # pragma: no cover - representation only
        return (f"FlowScores(rmse={self.rmse:.4g}, correlation={self.correlation:.4g}, "
                f"cpc={self.cpc:.4g})")
