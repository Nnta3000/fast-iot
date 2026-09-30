"""
CN: hello-iot 公开 API：逆最优传输（约束生成）。
EN: Public API of hello-iot: inverse optimal transport by constraint generation.
"""

from __future__ import annotations

from .components import (
    ComponentPartition,
    connector_edges,
    mandatory_edges,
    support_components,
)
from .edges import EdgeSet, edge_keys
from .entropic import (
    EntropicFitResult,
    EntropicIteration,
    fit_psd_moment_entropic,
    moment_from_plan,
    project_psd,
    scale_invariant_error,
    sinkhorn_plan,
)
from .features import (
    BilinearPSDFeatures,
    DenseFeatureTensor,
    DiagonalWeightFeatures,
    FeatureMap,
    dense_cost_matrix,
)
from .grounded import (
    GroundedInstance,
    make_bilinear_psd_grounded_instance,
    make_diagonal_grounded_instance,
)
from .loop import solve_inverse_ot
from .master import (
    LinearMasterSolver,
    MasterSolution,
    MasterSolver,
    RegularizedMasterSolver,
    TieBreak,
    draw_generic_direction,
)
from .oracle import ViolationScan, scan_violations
from .problem import InverseOTError, InverseOTProblem
from .result import InverseOTResult, IterationRecord
from .sdp import SdpMasterSolver
from .theta import (
    BilinearPSDTheta,
    LinearThetaConstraints,
    LinearThetaSet,
    SimplexTheta,
    ThetaSet,
)
from .validation import (
    DualCertificate,
    OptimalityCheck,
    ThetaRange,
    dual_optimality_certificate,
    fixed_dual_theta_ranges,
    is_optimal_plan,
    optimal_theta_ranges,
    plan_cost,
)

__version__ = "0.1.0"

__all__ = [
    "ComponentPartition",
    "DenseFeatureTensor",
    "DiagonalWeightFeatures",
    "DualCertificate",
    "EdgeSet",
    "EntropicFitResult",
    "EntropicIteration",
    "FeatureMap",
    "GroundedInstance",
    "BilinearPSDFeatures",
    "BilinearPSDTheta",
    "InverseOTError",
    "InverseOTProblem",
    "InverseOTResult",
    "IterationRecord",
    "LinearMasterSolver",
    "LinearThetaConstraints",
    "LinearThetaSet",
    "MasterSolution",
    "MasterSolver",
    "OptimalityCheck",
    "RegularizedMasterSolver",
    "SdpMasterSolver",
    "SimplexTheta",
    "ThetaRange",
    "ThetaSet",
    "TieBreak",
    "ViolationScan",
    "__version__",
    "connector_edges",
    "dense_cost_matrix",
    "draw_generic_direction",
    "dual_optimality_certificate",
    "edge_keys",
    "fit_psd_moment_entropic",
    "fixed_dual_theta_ranges",
    "is_optimal_plan",
    "make_bilinear_psd_grounded_instance",
    "make_diagonal_grounded_instance",
    "mandatory_edges",
    "moment_from_plan",
    "optimal_theta_ranges",
    "plan_cost",
    "project_psd",
    "scale_invariant_error",
    "scan_violations",
    "sinkhorn_plan",
    "solve_inverse_ot",
    "support_components",
]
