"""
EN: A study-side wrapper around `RegretMasterSolver` that remembers what it was asked to solve.

    Two experiments needed the same thing -- `corner.py` (which reads the per-round `theta` and the final edge
    set) and `multi_pair.py` (which needs the final edge set `F_p` of each pair) -- and had two copies of the
    class that had already drifted apart (one recorded `theta`, the other did not). This is the single
    definition: it records both, and callers use what they need.

    Nothing under `src/iot` is modified; this is an observation wrapper.

    CN: `RegretMasterSolver` 的研究侧包装：记录被求解的边集与每轮 `theta`。`corner.py` 与 `multi_pair.py`
        原本各有一份且已经漂移（一份记 theta、一份没记），这里是唯一实现。
"""

from __future__ import annotations

import numpy as np

from iot.core.edges import EdgeSet
from iot.core.problem import InverseOTProblem
from iot.relaxed.master_regret import RegretMasterSolver, RegretSolution

__all__ = ["RecordingMaster"]


class RecordingMaster(RegretMasterSolver):
    """CN: 记录边集与每轮 `theta` 的主问题求解器。EN: a master solver that records the edge set and each `theta`."""

    def __init__(self, **kwargs: object) -> None:
        """EN: same keyword arguments as the parent. CN: 关键字参数与父类一致。"""
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.last_edges: EdgeSet | None = None
        self.thetas: list[np.ndarray] = []

    def solve(self, problem: InverseOTProblem, edges: EdgeSet) -> RegretSolution:
        """EN: solve, then remember the edge set and the `theta` the round produced.
            CN: 求解后记录边集与本轮产生的 `theta`。"""
        self.last_edges = edges
        out = super().solve(problem, edges)
        self.thetas.append(np.asarray(getattr(out, "theta_pre", out.theta), dtype=np.float64).copy())
        return out
