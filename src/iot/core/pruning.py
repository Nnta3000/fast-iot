"""
CN: **迟滞剪枝** —— 只有一条边**连续两轮**都松（`slack > tau`）才剪掉。

    CN: 为什么需要迟滞：一阶后端（CuPDLPx）给出的对偶带 ~1e-4 量级的噪声，单轮规则会把噪声误判成
        "这条边松了"，于是永久剪掉一条本该保留的边。要求连续两轮才剪，正好压掉这种误判。

    CN: 状态是**上一轮松掉的边的键**（`int64`，按身份匹配）。规模上界是 `|F|`（不是 `m·n`），
        并且每轮**整体替换**而不是累积，因此不会随轮数增长。实测 m=40 时 `|F|` 从 80 长到 325，
        松边集合是它的子集，只有几 KB。

    CN: 对象**不可变**：`step()` 返回 `(保留的边, 新的剪枝器)`。这样两个循环都能用同一份逻辑，
        且不共享可变状态。

EN: **Hysteresis pruning** -- drop an edge only when it was loose (`slack > tau`) in **two consecutive
    rounds**.

    Why hysteresis is needed: a first-order backend (CuPDLPx) returns duals with ~1e-4 noise, so a
    single-round rule can read noise as "this edge is loose" and permanently drop an edge that should have
    been kept. Requiring two consecutive rounds suppresses exactly that.

    The state is **the keys of the edges that were loose last round** (`int64`, matched by identity). It is
    bounded by `|F|` (not `m·n`) and is **replaced** every round rather than accumulated, so it does not grow
    with the number of rounds. Measured at m=40, `|F|` grew 80 -> 325 and the loose set was a subset of it:
    a few KB.

    The object is **immutable**: `step()` returns `(kept edges, next pruner)`, so both loops can use the same
    logic without sharing mutable state.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from jaxtyping import Bool, Float, Int

from .edges import EdgeSet


@dataclasses.dataclass(frozen=True)
class HysteresisPruner:
    """
    CN: 单轮 vs 迟滞的开关就在 `previous_positive_keys` 是否为空：
        第一轮为空 ⇒ 没有任何"连续两轮"证据 ⇒ 什么都不剪，与精确循环原有行为一致。
    EN: The switch between one-shot and hysteresis is whether `previous_positive_keys` is empty: on the
        first round it is empty, so there is no "two consecutive rounds" evidence and nothing is pruned --
        identical to the exact loop's existing behaviour.
    """

    tau_prune: float
    previous_positive_keys: Int[np.ndarray, "previous"]

    @classmethod
    def fresh(cls, *, tau_prune: float) -> "HysteresisPruner":
        """CN: 初始状态：没有任何历史证据。EN: initial state: no history."""
        return cls(tau_prune=float(tau_prune),
                   previous_positive_keys=np.zeros(0, dtype=np.int64))

    def step(
        self,
        candidates: EdgeSet,
        slack: Float[np.ndarray, "edges"],
    ) -> tuple[EdgeSet, "HysteresisPruner"]:
        """
        CN: 返回 `(保留的边, 新的剪枝器)`。`slack = c - f - g - delta`，越大越松。
            `candidates` 与 `slack` 必须同序（调用方用同一个 `EdgeSet` 的 rows/cols 算出来）。
        EN: returns `(kept edges, next pruner)`. `slack = c - f - g - delta`, larger is looser.
            `candidates` and `slack` must be in the same order (callers compute both from the same
            `EdgeSet` rows/cols).
        """
        value = np.asarray(slack, dtype=np.float64).reshape(-1)
        if value.size != int(candidates.size):
            raise ValueError(f"slack has {value.size} entries, candidates has {int(candidates.size)}")
        positive: Bool[np.ndarray, "edges"] = value > float(self.tau_prune)
        if int(self.previous_positive_keys.size) == 0:
            # CN: 第一轮没有历史 ⇒ 不剪（保守）。
            # EN: no history on the first round, so prune nothing (conservative).
            keep: Bool[np.ndarray, "edges"] = np.ones(int(candidates.size), dtype=bool)
        else:
            # CN: 身份匹配：键是稳定的整数，只有连续两轮同一条边都松才剪。
            # EN: identity matching: keys are stable integers, so only an edge loose in both rounds is
            #     dropped.
            keep = ~(positive & np.isin(candidates.keys, self.previous_positive_keys))
        next_state = HysteresisPruner(tau_prune=self.tau_prune,
                                      previous_positive_keys=np.ascontiguousarray(
                                          candidates.keys[positive], dtype=np.int64))
        return candidates.select_mask(keep), next_state


__all__ = ["HysteresisPruner"]
