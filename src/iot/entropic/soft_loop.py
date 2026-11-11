"""
CN: **EOT 的循环** —— 软 master + 分离扫描 + 剪枝 + 边集推进，停在**逐点超出质量**上。

    CN: 每一轮：
        1. `soft_master(problem, F)` 解出 `(θ, f, g)` 与 `Ψ_F`；
        2. 扫描 `U∖F` 得到 `σ`、**逐点**超出质量 `M_out[i]`（`core/oracle.py` 的 `mass_epsilon` 选项），
           以及按 `σ` 的 top-γ 候选（**无阈值**：惩罚质量是 `e^{σ/ε}`，所以按 `σ` 排序）；
        3. 区间下界 `Ψ_F`，上界 `Ψ_F + ε·ΣM_out`；
        4. 停止：`max_i M_out[i]/a_i <= eta_mass`；
        5. 剪枝：`F∖B` 中 `π_e < eta_prune` 的边去掉；
        6. 推进：`F ← B ∪ T ∪ V`（复用 `core/edges_advance.py`）。

    CN: `B = S ∪ R` 由 `exact/components.mandatory_edges` 给出，永不被剪——缺了它 `Ψ_F` 会无下界。

EN: **EOT's loop** -- soft master, separation scan, prune, edge advance, stopping on **per-node excess mass**.

    Each round: solve the soft master on `F`; scan `U∖F` for `σ`, the **per-node** excess mass, and the
    top-γ candidates ranked by `σ` with **no threshold** (mass is `e^(σ/eps)`, so `σ` ranks it); form the
    bracket `[Ψ_F, Ψ_F + eps·ΣM_out]`; stop when `max_i M_out[i]/a_i <= eta_mass`; prune `F∖B` edges with
    `π_e < eta_prune`; advance `F <- B ∪ T ∪ V` reusing `core/edges_advance.py`.

    `B = S ∪ R` comes from `exact/components.mandatory_edges` and is never pruned -- without it `Ψ_F` has no
    lower bound.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
from jaxtyping import Float

from ..core.edges import EdgeSet
from ..core.edges_advance import advance_edges
from ..core.problem import InverseOTProblem
from ..core.scan import scan_for_violations
from ..exact.components import mandatory_edges
from ..exact.master import MasterSolution
from .soft_master import SoftMasterResult, soft_master


@dataclasses.dataclass(frozen=True)
class SoftLoopRound:
    """CN: 一轮的读数。EN: one round's reading."""

    index: int
    f_size: int
    psi: float
    lower: float
    upper: float
    max_excess_fraction: float
    frank_wolfe_gap: float
    outer_iterations: int
    inner_sweeps: int
    seconds: float


@dataclasses.dataclass(frozen=True)
class SoftLoopResult:
    """CN: 软循环的结果。EN: the soft loop's result."""

    theta: Float[np.ndarray, "k"]
    final_edges: EdgeSet
    rounds: tuple[SoftLoopRound, ...]
    converged: bool
    stall_status: str
    total_seconds: float


def solve_entropic_soft(problem: InverseOTProblem, *, epsilon_absolute: float,
                        gamma: int = 1, eta_mass: float = 1e-3, eta_prune: float = 1e-12,
                        max_rounds: int = 30, max_outer: int = 200,
                        inner_tolerance: float = 1e-8,
                        scan_backend: str = "numpy") -> SoftLoopResult:
    """
    CN: 跑 EOT 循环。`epsilon_absolute` 由调用方冻结（不做相对量换算）。
    EN: run the EOT loop. `epsilon_absolute` is frozen by the caller (no relative scaling here).
    """
    m = int(problem.n_source)
    n = int(problem.n_target)
    a: Float[np.ndarray, "m"] = np.asarray(problem.resolved_source_mass, dtype=np.float64).reshape(-1)
    b: Float[np.ndarray, "n"] = np.asarray(problem.resolved_target_mass, dtype=np.float64).reshape(-1)
    mandatory, _partition = mandatory_edges(problem)
    finite = mandatory
    warm: SoftMasterResult | None = None
    rounds: list[SoftLoopRound] = []
    stall_status = "max_rounds"
    started = time.perf_counter()

    for index in range(1, int(max_rounds) + 1):
        t0 = time.perf_counter()
        master = soft_master(problem, finite, epsilon=epsilon_absolute, warm=warm,
                             max_outer=int(max_outer), inner_tolerance=float(inner_tolerance))
        warm = master
        # CN: 扫描把 `(θ,f,g)` 当成 master 解；`delta=0`，因为 EOT 没有 delta 列。
        # EN: present `(theta,f,g)` as the master solution; `delta=0`, since EOT has no delta column.
        as_master = MasterSolution(delta=0.0, theta=master.theta, source_dual=master.source_dual,
                                   target_dual=master.target_dual, status="ok")
        scan = scan_for_violations(problem, as_master, excluded=finite, tol=-np.inf,
                                   gamma=int(gamma), scan_backend=str(scan_backend),
                                   mass_epsilon=float(epsilon_absolute))
        # CN: 逐点超出质量 -> 质量分数（`eta_mass` 是分数，不是绝对量）。
        # EN: per-node excess mass -> a mass fraction (`eta_mass` is a fraction, not an absolute number).
        excess_rows = scan.excess_mass_rows
        excess_cols = scan.excess_mass_cols
        if excess_rows is None or excess_cols is None:
            raise RuntimeError("the scan did not return excess masses; pass mass_epsilon")
        # CN: 行质量除以 `a`（长度 m），列质量除以 `b`（长度 n）——两者长度可以不同（planting 会剪列）。
        # EN: row mass divides by `a` (length m), column mass by `b` (length n); the two can differ in length
        #     (planting prunes columns).
        fraction = max(float(np.max(excess_rows / a)), float(np.max(excess_cols / b)))
        total_mass = float(np.sum(excess_rows))
        lower = float(master.psi)
        upper = lower + float(epsilon_absolute) * total_mass
        rounds.append(SoftLoopRound(index=index, f_size=int(finite.size), psi=float(master.psi),
                                    lower=lower, upper=upper, max_excess_fraction=fraction,
                                    frank_wolfe_gap=float(master.frank_wolfe_gap),
                                    outer_iterations=int(master.outer_iterations),
                                    inner_sweeps=int(master.inner_sweeps),
                                    seconds=float(time.perf_counter() - t0)))
        if fraction <= float(eta_mass):
            stall_status = "converged"
            break
        candidates = EdgeSet.from_arrays(scan.rows, scan.cols, n_target=n)
        # CN: 剪枝只看 `F∖B`；`pi_e` 与 `finite` 的边一一对应（master 在同一个 `F` 上解的）。
        # EN: prune only `F∖B`; `pi_e` is aligned with `finite` (the master solved on the same `F`).
        in_b = np.isin(finite.keys, mandatory.keys)
        keep = in_b | (np.asarray(master.plan_on_f, dtype=np.float64) >= float(eta_prune))
        pruned = finite.select_mask(keep)
        finite, _size = advance_edges(mandatory, pruned.difference(mandatory), candidates)

    theta = np.asarray(warm.theta, dtype=np.float64) if warm is not None else np.zeros(
        int(problem.n_features), dtype=np.float64)
    return SoftLoopResult(theta=theta, final_edges=finite, rounds=tuple(rounds),
                          converged=stall_status == "converged", stall_status=stall_status,
                          total_seconds=float(time.perf_counter() - started))


__all__ = ["SoftLoopResult", "SoftLoopRound", "solve_entropic_soft"]
