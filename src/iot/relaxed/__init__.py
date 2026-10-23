"""
CN: **松弛（最小 regret）逆 OT**：不假设 A1，最小化观测方案的 regret `R >= 0`；总可行。
EN: **Relaxed (minimum-regret) inverse OT**: drops assumption A1 and minimises the observed plan's
    regret `R >= 0`; always feasible.
"""

from .loop_regret import (
    RegretIteration,
    RegretResult,
    SelectionMode,
    full_regret,
    solve_inverse_ot_regret,
)
from .master_regret import (
    RegretMasterSolver,
    RegretSolution,
    build_regret_lp,
    regret_objective,
    support_duals,
)

__all__ = [
    "RegretIteration", "RegretMasterSolver", "RegretResult", "RegretSolution", "SelectionMode",
    "build_regret_lp", "full_regret", "regret_objective", "solve_inverse_ot_regret", "support_duals",
]
