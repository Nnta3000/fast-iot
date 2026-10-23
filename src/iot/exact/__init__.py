"""
CN: **精确（max-margin）逆 OT**：要求存在 theta 使观测方案精确最优（假设 A1）。这是主求解器。
EN: **Exact (max-margin) inverse OT**: requires a theta making the observed plan exactly optimal (assumption
    A1). This is the main solver.
"""

from .components import (
    ComponentPartition,
    connector_edges,
    mandatory_edges,
    support_components,
)
from .loop import solve_inverse_ot
from .master import (
    CanonicalGaugeMasterSolver,
    LinearMasterSolver,
    MasterSolution,
    MasterSolver,
    RegularizedMasterSolver,
    TieBreak,
    WarmStartMasterSolver,
    canonical_potentials,
    coordinate_extrema,
    draw_generic_direction,
)
from .oracle import ViolationScan, scan_violations
from .reduced_master import ForestRepresentation, SpanningForestMasterSolver, build_forest_representation
from .sdp import SdpMasterSolver

__all__ = [
    "CanonicalGaugeMasterSolver", "ComponentPartition", "ForestRepresentation", "LinearMasterSolver",
    "MasterSolution", "MasterSolver", "RegularizedMasterSolver", "SdpMasterSolver",
    "SpanningForestMasterSolver", "TieBreak", "ViolationScan", "WarmStartMasterSolver",
    "build_forest_representation", "canonical_potentials", "connector_edges", "coordinate_extrema",
    "draw_generic_direction", "mandatory_edges", "scan_violations", "solve_inverse_ot",
    "support_components",
]
