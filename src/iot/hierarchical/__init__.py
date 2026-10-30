"""
CN: **层级逆 OT 的驱动** —— 粗到细地求解同一个逆 OT 问题。

    CN: 它**不**替换任何一个求解器：每一层都调用调用方传入的 `solve_level`（也就是精确循环或松弛循环），
        层级只负责**构造粗层**与**计算细层的起点 `F₀`**。因此：
          - 用精确循环 ⇒ 精确层级（Lemma 2 的单调界与 Theorem 6 的"输出与 flat 相同"都适用）；
          - 用松弛循环 ⇒ 松弛层级（只是热启动，**没有**界，见 Typst 文件 §5）。

    CN: 这种"注入求解器"的写法有两个好处：
        1. 两个变体共用一份构造/传递代码，不可能分叉；
        2. 测试可以注入一个假的 `solve_level`，只验证 `BuildLevels`/`Prolong`/`Seed`。

EN: **Hierarchical inverse-OT driver** -- solve the same inverse-OT problem coarse to fine.

    It **does not replace** either solver: each level calls the injected `solve_level` (the exact loop or
    the relaxed loop). The hierarchy only **builds the coarse level** and **computes the fine start set
    `F₀`**. Therefore:
      - with the exact loop you get the *exact* hierarchy (Lemma 2's monotone bounds and Theorem 6's
        "same output as flat" both apply);
      - with the relaxed loop you get the *relaxed* hierarchy (warm start only, **no bounds**; see §5 of the
        Typst note).

    Injecting the solver has two benefits:
      1. both variants share one implementation of construction/transfer, so they cannot diverge;
      2. a test can inject a fake `solve_level` and check `BuildLevels`/`Prolong`/`Seed` alone.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

from beartype.typing import Callable

import numpy as np
import scipy.sparse as sparse
# CN: 本模块只用 Protocols 与 dataclasses；jaxtyping 未使用。
# EN: this module only needs protocols and dataclasses; jaxtyping is unused.

from ..core.edges import EdgeSet
from ..core.problem import InverseOTProblem
from .levels import Level, build_levels
from .parameters import HierarchyParameters
from .transfer import prolong, seed


@runtime_checkable
class LevelOutcome(Protocol):
    """
    CN: 一层求解器必须返回的字段。`runtime_checkable` 是必需的——`beartype` 在运行时检查 dataclass 字段。
    EN: what an injected level solver must return. `runtime_checkable` is required because
        `beartype` checks dataclass fields at runtime.
    """

    theta: np.ndarray
    source_dual: np.ndarray
    target_dual: np.ndarray


@dataclasses.dataclass(frozen=True)
class SeedRecord:
    """CN: 一层的种子统计（用于诊断，不参与正确性）。EN: one level's seed stats (diagnostics only)."""

    level: int
    coarse_grid: str
    n_mandatory: int
    n_row_col: int
    n_carry: int
    f0_size: int


@dataclasses.dataclass(frozen=True)
class HierarchicalRun:
    """CN: 层级运行的结果：细层解 + 每层的种子统计。EN: result: the fine solution plus per-level seed stats."""

    fine: LevelOutcome
    n_levels: int
    seeds: tuple[SeedRecord, ...]


def _coarse_problem(problem: InverseOTProblem, level: Level) -> InverseOTProblem:
    """
    CN: 用**诱导**方案在粗网格上重建问题：`pi_hat` 限制到 `I_ℓ × J_ℓ`。
        诱导支撑是 `S ∩ (I × J)`，与 Lemma 2 的要求一致。
    EN: rebuild the problem on the coarse grid with the **induced** plan: `pi_hat` restricted to
        `I_ℓ × J_ℓ`. The induced support is `S ∩ (I x J)`, which is what Lemma 2 requires.
    """
    features = problem.features
    source = np.asarray(features.source_points, dtype=np.float64)[level.rows]
    target = np.asarray(features.target_points, dtype=np.float64)[level.cols]
    dense = np.asarray(problem.plan.todense(), dtype=np.float64)
    restricted = np.ascontiguousarray(dense[np.ix_(level.rows, level.cols)])
    coarse_features = type(features)(np.ascontiguousarray(source), np.ascontiguousarray(target))
    return InverseOTProblem(features=coarse_features, theta_set=problem.theta_set,
                            plan=sparse.csr_matrix(restricted))


def solve_hierarchical(
    problem: InverseOTProblem,
    solve_level: Callable[[InverseOTProblem, EdgeSet | None], LevelOutcome],
    parameters: HierarchyParameters,
    *,
    mandatory_of: Callable[[InverseOTProblem], EdgeSet],
) -> HierarchicalRun:
    """
    CN: 粗到细求解。`solve_level(problem, initial_extra)` 由调用方提供：精确循环传
        `lambda p, e: solve_inverse_ot(p, ...)`，松弛循环传 `lambda p, e: solve_inverse_ot_regret(p,
        initial_extra=e, ...)`。`mandatory_of` 给出该问题的必需边集 `B = S ∪ R`。

    CN: 流程：
        1. `levels = build_levels(...)`，`levels[0]` 是完整问题；
        2. 从最粗层开始：全边求解（网格小，扫描可省）；
        3. 逐层细化：`prolong` → `seed` → 用 `F₀` 求解；
        4. 返回最细层的解。

    EN: coarse-to-fine. `solve_level(problem, initial_extra)` is supplied by the caller: the exact loop
        passes `lambda p, e: solve_inverse_ot(p, ...)`, the relaxed loop passes
        `lambda p, e: solve_inverse_ot_regret(p, initial_extra=e, ...)`. `mandatory_of` returns that
        problem's mandatory set `B = S ∪ R`.

    Flow: build levels (`levels[0]` is the full problem); solve the coarsest densely; then refine
    `prolong -> seed -> solve from F0` at each finer level; return the finest solution.
    """
    levels = build_levels(problem, rho=float(parameters.rho), max_levels=int(parameters.max_levels),
                          grid_budget=int(parameters.grid_budget), seed=int(parameters.level_seed))
    outcomes: list[LevelOutcome] = []
    seed_records: list[SeedRecord] = []

    # CN: 最粗层：全边（`None` 表示不额外给起点）。网格小，扫描代价可接受。
    # EN: coarsest level: all edges (`None` = no extra start). The grid is small, so scanning is fine.
    coarsest = levels[-1]
    coarse_problem = problem if coarsest is levels[0] else _coarse_problem(problem, coarsest)
    outcomes.append(solve_level(coarse_problem, None))

    for index in range(len(levels) - 2, -1, -1):
        coarse = levels[index + 1]
        fine = levels[index]
        fine_problem = problem if fine is levels[0] else _coarse_problem(problem, fine)
        coarse_out = outcomes[-1]
        f_full, g_full = prolong(fine_problem, coarse, fine,
                                 np.asarray(coarse_out.theta, dtype=np.float64),
                                 np.asarray(coarse_out.source_dual, dtype=np.float64),
                                 np.asarray(coarse_out.target_dual, dtype=np.float64))
        mandatory = mandatory_of(fine_problem)
        f0, stats = seed(fine_problem, fine, mandatory, theta=np.asarray(coarse_out.theta, dtype=np.float64),
                         source_dual=np.asarray(f_full, dtype=np.float64),
                         target_dual=np.asarray(g_full, dtype=np.float64),
                         kappa=int(parameters.kappa), carry_keys=None,
                         n_target=int(fine_problem.n_target))
        seed_records.append(SeedRecord(level=index, coarse_grid=f"{coarse.n_source}x{coarse.n_target}",
                                       n_mandatory=stats["n_mandatory"], n_row_col=stats["n_row_col"],
                                       n_carry=stats["n_carry"], f0_size=int(f0.size)))
        outcomes.append(solve_level(fine_problem, f0))

    return HierarchicalRun(fine=outcomes[-1], n_levels=len(levels), seeds=tuple(seed_records))


__all__ = ["HierarchicalRun", "LevelOutcome", "SeedRecord", "solve_hierarchical"]
