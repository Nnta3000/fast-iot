"""
CN: 逆 OT 循环的结果类型。
EN: Result types for the inverse-OT loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IterationRecord:
    """
    CN: 约束生成循环中一轮的诊断信息。
    EN: Diagnostics for one iteration of the constraint-generation loop.
    """

    index: int
    delta: float
    support_size: int
    violation_count: int
    added: int
    max_violation: float
    wall_time: float
    master_time: float
    oracle_time: float
    prune_time: float


@dataclass(frozen=True)
class InverseOTResult:
    """
    CN: 逆 OT 的解：最大间隔的 theta、对偶势与诊断。
    EN: Inverse-OT solution: the max-margin theta, dual potentials, and diagnostics.

    CN: `is_optimal` 是权威验证（若规模允许），绝不表示 theta 等于 theta_star。
    EN: `is_optimal` is the authoritative validation (when the size allows); it never
        claims theta equals theta_star.
    """

    theta: np.ndarray
    delta: float
    source_dual: np.ndarray
    target_dual: np.ndarray
    n_iterations: int
    support_size: int
    iterations: tuple[IterationRecord, ...]
    converged: bool
    is_optimal: bool | None
    relative_gap: float | None
    total_wall_time: float = 0.0
    final_max_violation: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", np.ascontiguousarray(self.theta, dtype=np.float64))
        object.__setattr__(self, "source_dual", np.ascontiguousarray(self.source_dual, dtype=np.float64))
        object.__setattr__(self, "target_dual", np.ascontiguousarray(self.target_dual, dtype=np.float64))


__all__ = [
    "InverseOTResult",
    "IterationRecord",
]
