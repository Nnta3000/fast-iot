"""
CN: **熵 OT 族**（逆 OT 的第三个目标函数）。见 `theta.py` 的模块文档。

    CN: 入口 `solve_entropic(problem, objective=EntropicObjective(...))`，与精确/松弛循环并列。
        **它不返回证书**——只有 θ 与损失；`stall_reason` 说明为什么停下。

EN: **Entropic OT family** (the third objective of inverse OT). See the `theta.py` module docstring.

    Entry point `solve_entropic(problem, objective=EntropicObjective(...))`, alongside the exact/relaxed
    loops. **It returns no certificate** -- only theta and the loss; `stall_reason` says why it stopped.
"""

from __future__ import annotations

from ..core.parameters import EntropicObjective
from .sinkhorn import SinkhornResult, sinkhorn_plan_torch
from .theta import EntropicResult, EntropicStep, fit_entropic


def solve_entropic(problem, *, objective: EntropicObjective | None = None) -> EntropicResult:
    """
    CN: 与另外两个求解器同形的入口。`objective=None` 用默认的 `EntropicObjective()`。
    EN: same shape as the other two solvers. `objective=None` uses the default `EntropicObjective()`.
    """
    obj = EntropicObjective() if objective is None else objective
    return fit_entropic(
        problem, relative_epsilon=float(obj.relative_epsilon),
        learning_rate=float(obj.learning_rate), max_steps=int(obj.max_steps),
        sinkhorn_iterations=int(obj.sinkhorn_iterations),
        sinkhorn_tolerance=float(obj.sinkhorn_tolerance),
        gradient_tolerance=float(obj.gradient_tolerance), device=str(obj.device))


__all__ = ["EntropicResult", "EntropicStep", "SinkhornResult", "solve_entropic", "sinkhorn_plan_torch"]
