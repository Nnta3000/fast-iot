"""
CN: **层构造** —— `BuildLevels`（`HierarchicalIOT.typ` 的 Lemma 1）。

    CN: 关键：从**支撑边**抽样，而不是从点抽样。按点抽样会让某些保留下来的点没有支撑边（违反 H1），
        并让支撑图碎裂。按边抽样保证每个保留点至少是一条支撑边的端点。
        另外拒绝"诱导支撑等于整个粗网格"的抽样——那种情况下 master 没有不等式约束，margin 无界。

EN: **Level construction** -- `BuildLevels` (Lemma 1 of `HierarchicalIOT.typ`).

    Sample **support edges**, not points: point sampling can leave a retained point with no support edge
    (violating H1) and fragments the support graph. Edge sampling guarantees every retained point is an
    endpoint of at least one support edge.
    A draw whose induced support fills the coarse grid is rejected: there the master has no margin
    constraint and the margin is unbounded.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from jaxtyping import Int

from ..core.edges import EdgeSet
from ..core.problem import InverseOTProblem


@dataclasses.dataclass(frozen=True)
class Level:
    """
    CN: 一层：源/目标索引集与**诱导支撑** `S_ℓ = S ∩ (I_ℓ × J_ℓ)`。
        索引是**全局**索引；`rows[pos]` 是第 `pos` 个保留源点。
    EN: One level: source/target index sets and the **induced support** `S_ℓ = S ∩ (I_ℓ × J_ℓ)`.
        Indices are **global**; `rows[pos]` is the `pos`-th retained source.
    """

    rows: Int[np.ndarray, "sources"]
    cols: Int[np.ndarray, "targets"]
    support: EdgeSet

    @property
    def n_source(self) -> int:
        """CN: 保留源点数。EN: number of retained sources."""
        return int(self.rows.size)

    @property
    def n_target(self) -> int:
        """CN: 保留目标点数。EN: number of retained targets."""
        return int(self.cols.size)

    @property
    def grid_size(self) -> int:
        """CN: 粗网格大小 `|I|·|J|`。EN: coarse grid size `|I|·|J|`."""
        return int(self.rows.size) * int(self.cols.size)


def full_level(problem: InverseOTProblem) -> Level:
    """CN: 第 0 层（完整问题）。EN: level 0 (the full problem)."""
    s_rows, s_cols, _ = problem.support()
    return Level(rows=np.arange(int(problem.n_source), dtype=np.int64),
                 cols=np.arange(int(problem.n_target), dtype=np.int64),
                 support=EdgeSet.from_arrays(s_rows, s_cols, n_target=int(problem.n_target)))


def coarsen(problem: InverseOTProblem, parent: Level, *, rho: float, seed: int,
            n_target: int) -> Level:
    """
    CN: 由 `parent` 构造下一层：抽 `⌈rho|S_parent|⌉` 条诱导支撑边，取端点。
        若点集没有变小或诱导支撑等于整网格，则 rho 减半重抽（Lemma 1 的接受条件）。
    EN: build the next level from `parent`: draw `ceil(rho|S_parent|)` induced support edges and take their
        endpoints. If the point set does not shrink or the induced support fills the grid, halve rho and
        redraw (Lemma 1's acceptance condition).
    """
    s_rows = parent.support.rows
    s_cols = parent.support.cols
    rng = np.random.default_rng(int(seed))
    ratio = float(rho)
    for _attempt in range(60):
        count = max(1, int(np.ceil(ratio * int(s_rows.size))))
        pick = rng.choice(int(s_rows.size), size=min(count, int(s_rows.size)), replace=False)
        c_rows = np.unique(s_rows[pick])
        c_cols = np.unique(s_cols[pick])
        keep = np.isin(s_rows, c_rows) & np.isin(s_cols, c_cols)
        induced = EdgeSet.from_arrays(np.ascontiguousarray(s_rows[keep]),
                                      np.ascontiguousarray(s_cols[keep]), n_target=int(n_target))
        shrunk = c_rows.size + c_cols.size < parent.n_source + parent.n_target
        # CN: 拒绝"诱导支撑 == 粗网格"：没有不等式，margin 无界。
        # EN: reject "induced support == coarse grid": no inequality, so the margin is unbounded.
        informative = int(induced.size) < int(c_rows.size * c_cols.size)
        if shrunk and informative:
            return Level(rows=np.ascontiguousarray(c_rows), cols=np.ascontiguousarray(c_cols),
                         support=induced)
        ratio *= 0.5
    raise RuntimeError("coarsening could not produce an admissible level")


def build_levels(problem: InverseOTProblem, *, rho: float, max_levels: int, grid_budget: int,
                 seed: int) -> list[Level]:
    """
    CN: 返回 `[level_0, level_1, ..., level_L]`，从细到粗；`level_0` 是完整问题。
        当粗网格大小 `<= grid_budget` 或层数达到 `max_levels` 时停止。
    EN: returns `[level_0, ..., level_L]` fine to coarse; `level_0` is the full problem. Stops when the
        coarse grid is `<= grid_budget` or `max_levels` coarse levels have been built.
    """
    levels = [full_level(problem)]
    for index in range(int(max_levels)):
        parent = levels[-1]
        if parent.grid_size <= int(grid_budget):
            break
        child = coarsen(problem, parent, rho=float(rho), seed=int(seed) + index,
                        n_target=int(problem.n_target))
        levels.append(child)
    return levels


__all__ = ["Level", "build_levels", "coarsen", "full_level"]
