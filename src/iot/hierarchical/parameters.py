"""
CN: **层级参数** —— 一个不可变对象，同时给精确循环与松弛循环使用。
    `None`（默认）表示不使用层级，行为与之前**逐位相同**。

EN: **Hierarchy parameters** -- one immutable object handed to both the exact and the relaxed solver.
    `None` (the default) means no hierarchy, and behaviour is **bit-identical** to before.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class HierarchyParameters:
    """
    CN: 两级（或多级）粗到细的旋钮。语义见 `theory/HierarchicalIOT.typ`。

        用法：
            from iot.hierarchical import HierarchyParameters
            solve_inverse_ot(problem, hierarchical=HierarchyParameters(rho=0.25, kappa=3))

    EN: Knobs for coarse-to-fine levels. Semantics in `theory/HierarchicalIOT.typ`.

        Usage:
            from iot.hierarchical import HierarchyParameters
            solve_inverse_ot(problem, hierarchical=HierarchyParameters(rho=0.25, kappa=3))

    CN: 参数含义：
        - `rho`：每层从**支撑**里抽样的边的比例（抽边而非抽点，保证每点至少连一条支撑边）。
        - `max_levels`：最多构造几层（1 表示一个粗层 + 细层，即两级）。
        - `grid_budget`：粗网格 `|I|·|J|` 的预算；达到即停止继续粗化。
        - `kappa`：`Seed` 中每行/每列保留的最大 slack 边数（`A_row`/`A_col`）。
        - `level_seed`：抽样随机种子；同一条边集下可复现。
        - `carry`：是否把粗层最终边集并入 `F₀`（`A_carry`）。

    EN: Parameter meanings:
        - `rho`: fraction of **support** edges sampled per level (sample edges, not points, so every
          retained point keeps at least one support edge).
        - `max_levels`: how many coarse levels to build (1 = one coarse level plus the fine one).
        - `grid_budget`: budget on a coarse grid `|I|·|J|`; stop coarsening once reached.
        - `kappa`: per row/column count kept by `Seed` (`A_row`/`A_col`).
        - `level_seed`: RNG seed for the edge sampling; reproducible for the same edge set.
        - `carry`: whether to fold the coarse final edge set into `F₀` (`A_carry`).
    """

    rho: float = 0.25
    max_levels: int = 1
    grid_budget: int = 1_000_000
    kappa: int = 3
    level_seed: int = 0
    carry: bool = False

    def __post_init__(self) -> None:
        """CN: 早失败。EN: fail early."""
        if not 0.0 < float(self.rho) < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {self.rho!r}")
        if int(self.max_levels) < 1:
            raise ValueError(f"max_levels must be >= 1, got {self.max_levels!r}")
        if int(self.grid_budget) < 1:
            raise ValueError(f"grid_budget must be >= 1, got {self.grid_budget!r}")
        if int(self.kappa) < 1:
            raise ValueError(f"kappa must be >= 1, got {self.kappa!r}")


__all__ = ["HierarchyParameters"]
