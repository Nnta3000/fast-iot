"""
CN: **两个循环共用的边集推进** —— `F_{t+1} = mandatory ∪ pruned ∪ V`。

    CN: 精确循环与松弛循环执行的是**同一个代数**：必需边（支撑 + 连接边）永远保留，可选边按
        slack 剪枝，加上本轮扫描到的违反边。它们的差别只在**如何算出 `pruned`**：
          - 精确：`slack = c - f - g - delta <= tau_prune`，可按 hysteresis 细化；
          - 松弛：`sigma = f + g - c >= -tau`（无 delta 列），或整段剪枝关闭。
        因此这里把**代数**共用，把**策略**留给调用方：`pruned` 由调用方传入。

    CN: 语义与两处原实现逐字一致——只是把一行 `mandatory.union(pruned).union(candidates)` 提出来，
        并顺带共用两个正负号不同的 slack 计算。

EN: **Edge-set advance shared by both loops** -- `F_{t+1} = mandatory ∪ pruned ∪ V`.

    The exact loop and the relaxed loop run the **same algebra**: mandatory edges (support + connectors) are
    always kept, optional edges are pruned by slack, and this round's violating edges are added. They differ
    only in **how `pruned` is computed**:
      - exact: `slack = c - f - g - delta <= tau_prune`, optionally refined by hysteresis;
      - relaxed: `sigma = f + g - c >= -tau` (no delta column), or pruning switched off entirely.
    So the **algebra** is shared here and the **policy** stays with the caller: `pruned` is passed in.

    The semantics are verbatim those of the two original sites: the line
    `mandatory.union(pruned).union(candidates)` is lifted out, and the two opposite-sign slack helpers are
    shared alongside it.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int

from .edges import EdgeSet
from .problem import InverseOTProblem


def slack_toward_cost(
    problem: InverseOTProblem,
    theta: Float[np.ndarray, "k"],
    source_dual: Float[np.ndarray, "m"],
    target_dual: Float[np.ndarray, "n"],
    rows: Int[np.ndarray, "edges"],
    cols: Int[np.ndarray, "edges"],
    *,
    delta: float = 0.0,
) -> Float[np.ndarray, "edges"]:
    """
    CN: `slack = c(theta) - f - g - delta`，**越大越松**（精确循环用它判断能否剪掉）。
        `delta=0` 时就是 `-sigma`。
    EN: `slack = c(theta) - f - g - delta`, **larger means looser** (the exact loop prunes on it). With
        `delta = 0` it is `-sigma`.
    """
    if int(rows.size) == 0:
        return np.zeros(0, dtype=np.float64)
    cost = np.asarray(problem.features.costs(theta, rows, cols), dtype=np.float64)
    return cost - np.asarray(source_dual, dtype=np.float64)[rows] - \
        np.asarray(target_dual, dtype=np.float64)[cols] - float(delta)


def slack_toward_duals(
    problem: InverseOTProblem,
    theta: Float[np.ndarray, "k"],
    source_dual: Float[np.ndarray, "m"],
    target_dual: Float[np.ndarray, "n"],
    rows: Int[np.ndarray, "edges"],
    cols: Int[np.ndarray, "edges"],
) -> Float[np.ndarray, "edges"]:
    """
    CN: `sigma = f + g - c(theta)`，**越大越紧**（松弛循环用它找违反边）。等于 `-slack_toward_cost`。
    EN: `sigma = f + g - c(theta)`, **larger means tighter** (the relaxed loop finds violators with it).
        Equals `-slack_toward_cost`.
    """
    return -slack_toward_cost(problem, theta, source_dual, target_dual, rows, cols)


def keep_by_tightness(
    candidates: EdgeSet,
    slack: Float[np.ndarray, "edges"],
    *,
    tau_prune: float,
) -> EdgeSet:
    """
    CN: 只保留 `slack <= tau_prune` 的边——即**松到不能剪**的那些。两个循环共用同一条规则，只是
        slack 的正负号约定不同（见上面两个函数）。
    EN: keep only edges with `slack <= tau_prune`, i.e. those loose enough that they cannot be pruned.
        Both loops use this rule; only the slack sign convention differs (see the two functions above).
    """
    return candidates.select_mask(np.asarray(slack, dtype=np.float64) <= float(tau_prune))


def advance_edges(
    mandatory: EdgeSet,
    pruned: EdgeSet,
    candidates: EdgeSet,
) -> tuple[EdgeSet, int]:
    """
    CN: `F_{t+1} = mandatory ∪ pruned ∪ V`，并返回净增边数。
        `mandatory` 必须始终保留——它含支撑与连接边，剪掉会让 master 无界。
    EN: `F_{t+1} = mandatory ∪ pruned ∪ V`, returning the net number of added edges.
        `mandatory` must always survive: it holds the support and the connectors, and pruning it would make
        the master unbounded.

    CN: 返回的 `added` 是 `|F_{t+1}| - |F_t|`，由调用方传入当前 `F_t` 时更精确；这里给的是**相对
        `mandatory` 的绝对规模**，与两处原实现一致（原实现也是用 `edges` 减出来的）。
    EN: `added` here is the absolute size of the new set; that matches both original sites, which computed
        the delta against the previous `edges` themselves.
    """
    merged = mandatory.union(pruned).union(candidates)
    return merged, int(merged.size)


__all__ = ["advance_edges", "keep_by_tightness", "slack_toward_cost", "slack_toward_duals"]
