"""
CN: **求解器参数** —— 把两个循环各自散落的 18 / 26 个关键字参数收成几个不可变对象。

    CN: 分三类：
        - **共用**（两个目标函数都一样）：`ScanParameters`、`PruningParameters`、`SolverParameters`；
        - **目标函数专属**：`ExactObjective` / `RelaxedObjective` —— 它们各自带自己的旋钮，
          因此"给错目标的参数"在类型上就不存在；
        - **层级**：`HierarchyParameters`（见 `iot.hierarchical`），是 `SolverParameters` 的字段，
          因为它对两个目标函数同样适用。

    CN: `objective` 是 **union** 而不是 Protocol：结果类型随目标函数变化
        （`ExactObjective → InverseOTResult`，`RelaxedObjective → RegretResult`），
        union 让 `match` 可穷尽、类型检查器能提示遗漏，而 Protocol 会把返回类型退化成 `Any`。

EN: **Solver parameters** -- the 18 / 26 keyword arguments scattered across the two loops, gathered into a
    few immutable objects.

    Three groups:
      - **shared** by both objectives: `ScanParameters`, `PruningParameters`, `SolverParameters`;
      - **objective-specific**: `ExactObjective` / `RelaxedObjective`, each carrying its own knobs, so
        "a parameter for the wrong objective" does not exist as a type;
      - **hierarchy**: `HierarchyParameters` (see `iot.hierarchical`), a field of `SolverParameters`,
        because it applies to both objectives identically.

    `objective` is a **union**, not a Protocol: the result type changes with the objective
    (`ExactObjective -> InverseOTResult`, `RelaxedObjective -> RegretResult`), and a union makes `match`
    exhaustive and lets the type checker flag omissions, where a Protocol would degrade the return type
    to `Any`.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from ..core.edges import EdgeSet
from ..hierarchical.parameters import HierarchyParameters

TieBreak = Literal["lex", "lex_generic", "random", "none"]
"""CN: 精确循环的挑选规则（证明第 4 节）。EN: the exact loop's selection rule (proof section 4)."""

SelectionMode = Literal["central", "uniform", "none"]
"""CN: 松弛循环的两段式第二段。EN: the relaxed loop's second stage."""

TAU_PRUNE_DEFAULT: float = 1e-12
"""
CN: **剪枝阈值，两个循环共用同一个默认值**。

    语义：边 `e` 在 `slack_e = c_e - f_i - g_j (- delta) > tau_prune` 时才是"可以剪"的。
    证明只覆盖**严格正** slack 的剪枝，所以 `1e-12` 是唯一有依据的取值——它等价于
    "只在 slack 严格为正时剪，仅留浮点误差余量"。

    CN: 历史上松弛循环用的是 `1e-7`，那是把相邻的 `DEFAULT_TOL = 1e-7` 抄过来的结果，**没有推导**。
        两个默认值相差 1e5，且 `1e-7` 比一阶后端（CuPDLPx，精度 ~1e-4）的噪声还紧，
        因此它既不被证明覆盖、也不比 `1e-12` 更安全。**已统一为 `1e-12`。**

EN: **The prune threshold; both loops now share one default.**

    Semantics: edge `e` is prunable only when `slack_e = c_e - f_i - g_j (- delta) > tau_prune`.
    The proof covers pruning of **strictly positive** slack only, so `1e-12` is the only defensible value:
    it is "prune only when slack is strictly positive", with a floating-point margin.

    The relaxed loop historically used `1e-7`, copied from the adjacent `DEFAULT_TOL = 1e-7` with **no
    derivation**. The two defaulted 1e5 apart, and `1e-7` is tighter than the first-order backend's noise
    (~1e-4), so it was neither proved nor safer than `1e-12`. **Unified to `1e-12`.**
"""


@dataclasses.dataclass(frozen=True)
class ScanParameters:
    """
    CN: **分离扫描**（每轮在全网格上找违反边）。

    EN: **Separation scan** (finds violated edges on the whole grid every round).

    CN: 字段：
        - `backend`：`"numpy"`（CPU，参考）、`"hello"`（HELLO-OT 的 torch 内核，仅对角族）或
          `"torch"`（通用 torch，任意线性于 θ 的族）。非法值**直接报错**，绝不静默回退。
        - `device`：torch 后端的设备（`"auto"` 取第一个可用设备）。
        - `max_tile_elements`：NumPy 后端每块的元素数；只影响内存，不影响结果。
        - `max_tile_bytes`：GPU 后端每块的字节数；同理只影响内存。

    EN: Fields:
        - `backend`: `"numpy"` (CPU reference), `"hello"` (HELLO-OT's torch kernel, diagonal only) or
          `"torch"` (generic torch, any family linear in theta). An unknown value **raises**; never a silent
          fallback.
        - `device`: device for the torch backends (`"auto"` picks the first available).
        - `max_tile_elements`: elements per tile for the NumPy backend; affects memory only, not results.
        - `max_tile_bytes`: bytes per tile for the GPU backends; likewise memory only.
    """

    backend: str = "numpy"
    device: str = "auto"
    max_tile_elements: int = 1 << 20
    max_tile_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        """CN: 早失败。EN: fail early."""
        if str(self.backend) not in ("numpy", "hello", "torch"):
            raise ValueError(f"scan backend must be 'numpy', 'hello' or 'torch', got {self.backend!r}")
        if int(self.max_tile_elements) < 1:
            raise ValueError(f"max_tile_elements must be >= 1, got {self.max_tile_elements!r}")
        if int(self.max_tile_bytes) < 1:
            raise ValueError(f"max_tile_bytes must be >= 1, got {self.max_tile_bytes!r}")


@dataclasses.dataclass(frozen=True)
class PruningParameters:
    """
    CN: **剪枝**（从 `F` 里去掉已经松掉的边，控制 `|F|`）。

    EN: **Pruning** (drop edges that have gone loose, to control `|F|`).

    CN: 字段：
        - `enabled`：是否剪枝。**默认关闭**：剪枝是启发式，只有"剪掉严格正 slack 的边"被证明端到端安全，
          保留多余边永远安全（只是更大更慢）。要 `enabled=True` 请显式给出。
        - `hysteresis`：只有**连续两轮**都松才剪。一阶后端的对偶噪声约 1e-4，单轮规则会把噪声当成
          "这条边松了"并永久剪掉它。
        - `tau_prune`：多松才算"松"，见 `TAU_PRUNE_DEFAULT`。两个目标函数共用这一个值。

    EN: Fields:
        - `enabled`: whether to prune. **Off by default**: pruning is a heuristic and only dropping
          strictly-positive-slack edges is proved safe end to end; keeping extra edges is always safe (just
          bigger and slower). Pass `enabled=True` explicitly.
        - `hysteresis`: prune only when loose in **two consecutive rounds**. A first-order backend's duals
          carry ~1e-4 noise, so a single-round rule reads noise as "loose" and drops the edge for good.
        - `tau_prune`: how loose counts as loose; see `TAU_PRUNE_DEFAULT`. One value, shared by both
          objectives.
    """

    enabled: bool = False
    hysteresis: bool = True
    tau_prune: float = TAU_PRUNE_DEFAULT

    def __post_init__(self) -> None:
        """CN: 早失败。EN: fail early."""
        if float(self.tau_prune) < 0.0:
            raise ValueError(f"tau_prune must be >= 0, got {self.tau_prune!r}")


@dataclasses.dataclass(frozen=True)
class ExactObjective:
    """
    CN: **精确目标**：`max delta`，支撑集上是**等式** `f_i + g_j = c_ij(θ)`，其余边
        `f_i + g_j + delta <= c_ij(θ)`。

    EN: **Exact objective**: `max delta`, with **equalities** `f_i + g_j = c_ij(theta)` on the support and
        `f_i + g_j + delta <= c_ij(theta)` elsewhere.

    CN: 字段：
        - `tie_break`：挑选规则。`"lex"`（字典序，最慢最严格）、`"lex_generic"`（两次求解 + 固定通用方向，
          证明覆盖）、`"random"`（同 `lex_generic`，历史名称）、`"none"`（默认求解器输出，可能循环）。
        - `random_seed`：`lex_generic` 抽通用方向 `r` 的种子；整轮只抽一次。
        - `validate`：结束时是否跑权威的 `is_optimal_plan`（POT 精确 OT）。**贵**：`m=2000` 时是几十秒。
        - `validate_max_elements`：`m·n` 超过此值就跳过验证（静默跳过，结果为 `is_optimal=None`）。

    EN: Fields:
        - `tie_break`: the selection rule. `"lex"` (lexicographic, slowest and strictest), `"lex_generic"`
          (two solves plus a fixed generic direction; covered by the proof), `"random"` (same, historical
          name), `"none"` (default solver output; may cycle).
        - `random_seed`: seed for the generic direction `r` in `lex_generic`; drawn once per run.
        - `validate`: run the authoritative `is_optimal_plan` (exact POT OT) at the end. **Expensive**:
          tens of seconds at `m=2000`.
        - `validate_max_elements`: skip validation when `m*n` exceeds this (silently; the result is
          `is_optimal=None`).
    """

    tie_break: TieBreak = "lex"
    random_seed: int = 0
    validate: bool = True
    validate_max_elements: int = 50_000_000

    def __post_init__(self) -> None:
        """CN: 早失败。EN: fail early."""
        if str(self.tie_break) not in ("lex", "lex_generic", "random", "none"):
            raise ValueError(f"tie_break must be lex/lex_generic/random/none, got {self.tie_break!r}")
        if int(self.validate_max_elements) < 1:
            raise ValueError(f"validate_max_elements must be >= 1, got {self.validate_max_elements!r}")


@dataclasses.dataclass(frozen=True)
class RelaxedObjective:
    """
    CN: **松弛目标**：`min R`（regret），`F` 上**全部是不等式**（含支撑集），没有 `delta` 列。

    EN: **Relaxed objective**: `min R` (regret), **all inequalities** on `F` (support included), no `delta`
        column.

    CN: 字段：
        - `eps_r_rel`：证书区间 `U - L <= eps_r_rel * |R_uniform|` 即视为收敛。**相对量**，因为 regret 的
          尺度随实例变化。
        - `selection`：两段式的第二段。`"central"`（在 `R <= R* + eta` 上取中心点）、`"uniform"`、
          `"none"`（跳过）。
        - `eta`：`"central"` 允许的 regret 放大比例。
        - `check_pot`：结束时用 POT 重算 regret 并核对标识 `pot_matches`。**贵**，仅诊断用。
        - `initial_extra`：额外的初始边集（热启动/层级用）；必须包含 `B = S ∪ R` 的超集关系由循环检查。
        - `identity_rtol`：regret 恒等式的容差；`None` 时按后端取（一阶 ~1e-8，精确 ~1e-10）。
        - `warm_start`：把**上一轮**的 `(θ, f, g)` 交给一阶求解器作起点（`setInitSol`）。只影响迭代起点，
          不影响最优值；布局在轮间不变。

    EN: Fields:
        - `eps_r_rel`: convergence when the certificate bracket `U - L <= eps_r_rel * |R_uniform|`. A
          **relative** quantity, because the regret scale varies with the instance.
        - `selection`: second stage of the two-solve rule. `"central"` (central point on `R <= R* + eta`),
          `"uniform"`, `"none"` (skip).
        - `eta`: how far `"central"` may raise the regret.
        - `check_pot`: recompute the regret with POT and report the identity flag `pot_matches`. Expensive;
          diagnostics only.
        - `initial_extra`: extra starting edges (warm start / hierarchy). The loop checks that the support
          stays inside `F`.
        - `identity_rtol`: tolerance of the regret identity; `None` picks per backend (first-order ~1e-8,
          exact ~1e-10).
        - `warm_start`: hand the **previous** round's `(theta, f, g)` to the first-order solver
          (`setInitSol`). Affects the iteration start only, never the optimum; the layout is unchanged
          between rounds.
    """

    eps_r_rel: float = 1e-6
    selection: SelectionMode = "central"
    eta: float = 1e-7
    check_pot: bool = True
    initial_extra: EdgeSet | None = None
    identity_rtol: float | None = None
    warm_start: bool = False

    def __post_init__(self) -> None:
        """CN: 早失败。EN: fail early."""
        if float(self.eps_r_rel) <= 0.0:
            raise ValueError(f"eps_r_rel must be > 0, got {self.eps_r_rel!r}")
        if str(self.selection) not in ("central", "uniform", "none"):
            raise ValueError(f"selection must be central/uniform/none, got {self.selection!r}")
        if float(self.eta) < 0.0:
            raise ValueError(f"eta must be >= 0, got {self.eta!r}")
        if self.identity_rtol is not None and float(self.identity_rtol) <= 0.0:
            raise ValueError(f"identity_rtol must be > 0 when given, got {self.identity_rtol!r}")


Objective = ExactObjective | RelaxedObjective
"""CN: 目标函数是 union，`match` 可穷尽。EN: the objective is a union, so `match` is exhaustive."""


@dataclasses.dataclass(frozen=True)
class SolverParameters:
    """
    CN: **两个目标函数共用的旋钮**。

    EN: **Knobs shared by both objectives.**

    CN: 字段：
        - `tol`：违反判据。边 `e` 在 `sigma_e + delta > tol` 时算违反。`None` **按代价尺度自动定**
          （`max(1e-5, 1e-4) * median|c|`），松弛循环一直这么做；精确循环用 `None` 时也走同一规则。
        - `gamma`：每轮每行/每列保留的违反边数。`1` 最保守（最少边、最多轮）；`2`–`4` 常能少几轮。
        - `max_iterations`：硬上限。
        - `scan`：`ScanParameters`，见其文档。
        - `pruning`：`PruningParameters`，见其文档。
        - `hierarchical`：`HierarchyParameters | None`。`None`（默认）是扁平求解，与引入层级前**逐位相同**。

    EN: Fields:
        - `tol`: violation test. Edge `e` violates when `sigma_e + delta > tol`. `None` **derives** it from
          the cost scale (`max(1e-5, 1e-4) * median|c|`), which the relaxed loop always did; the exact loop
          follows the same rule when given `None`.
        - `gamma`: violated edges kept per row and per column each round. `1` is the most conservative (fewest
          edges, most rounds); `2`-`4` often save rounds.
        - `max_iterations`: hard cap.
        - `scan`: `ScanParameters`, see its docstring.
        - `pruning`: `PruningParameters`, see its docstring.
        - `hierarchical`: `HierarchyParameters | None`. `None` (default) is the flat solve, **bit-identical**
          to before the hierarchy existed.
    """

    tol: float | None = None
    gamma: int = 1
    max_iterations: int = 100
    scan: ScanParameters = dataclasses.field(default_factory=ScanParameters)
    pruning: PruningParameters = dataclasses.field(default_factory=PruningParameters)
    hierarchical: HierarchyParameters | None = None

    def __post_init__(self) -> None:
        """CN: 早失败。EN: fail early."""
        if self.tol is not None and float(self.tol) < 0.0:
            raise ValueError(f"tol must be >= 0 when given, got {self.tol!r}")
        if int(self.gamma) < 1:
            raise ValueError(f"gamma must be >= 1, got {self.gamma!r}")
        if int(self.max_iterations) < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations!r}")


__all__ = [
    "Objective", "ExactObjective", "PruningParameters", "RelaxedObjective", "ScanParameters",
    "SelectionMode", "SolverParameters", "TAU_PRUNE_DEFAULT", "TieBreak",
]
