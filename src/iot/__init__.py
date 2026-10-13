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
from .cutting_plane import (
    CutMasterResult,
    CuttingPlaneConfig,
    CuttingPlaneCut,
    CuttingPlaneIteration,
    CuttingPlaneResult,
    MasterMode,
    solve_cut_master,
    solve_cutting_plane,
)
from .cupdlpx_master import (
    CuPdlpxLp,
    CuPdlpxMasterSolver,
    build_cupdlpx_lp,
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
    QuadraticFeatures,
    dense_cost_matrix,
)
from .grounded import (
    GroundedInstance,
    make_bilinear_psd_grounded_instance,
    make_diagonal_grounded_instance,
)
from .hello_scan import fast_scan_supported, scan_violations_fast
from .hello_solver import hello_solver_supported, make_hello_solver
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
    draw_generic_direction,
)
from .oracle import ViolationScan, scan_violations
from .problem import InverseOTError, InverseOTProblem
from .reduced_master import (
    ForestRepresentation,
    SpanningForestMasterSolver,
    build_forest_representation,
)
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
    "CanonicalGaugeMasterSolver",
    "CutMasterResult",
    "CuttingPlaneConfig",
    "CuttingPlaneCut",
    "CuttingPlaneIteration",
    "CuttingPlaneResult",
    "CuPdlpxLp",
    "CuPdlpxMasterSolver",
    "DenseFeatureTensor",
    "DiagonalWeightFeatures",
    "DualCertificate",
    "EdgeSet",
    "EntropicFitResult",
    "EntropicIteration",
    "FeatureMap",
    "ForestRepresentation",
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
    "MasterMode",
    "MasterSolution",
    "MasterSolver",
    "OptimalityCheck",
    "QuadraticFeatures",
    "RegularizedMasterSolver",
    "SdpMasterSolver",
    "SimplexTheta",
    "SpanningForestMasterSolver",
    "ThetaRange",
    "ThetaSet",
    "TieBreak",
    "ViolationScan",
    "WarmStartMasterSolver",
    "__version__",
    "build_forest_representation",
    "build_cupdlpx_lp",
    "canonical_potentials",
    "connector_edges",
    "dense_cost_matrix",
    "draw_generic_direction",
    "dual_optimality_certificate",
    "edge_keys",
    "fit_psd_moment_entropic",
    "fixed_dual_theta_ranges",
    "fast_scan_supported",
    "hello_solver_supported",
    "is_optimal_plan",
    "make_bilinear_psd_grounded_instance",
    "make_diagonal_grounded_instance",
    "make_hello_solver",
    "mandatory_edges",
    "moment_from_plan",
    "optimal_theta_ranges",
    "plan_cost",
    "project_psd",
    "scale_invariant_error",
    "scan_violations",
    "scan_violations_fast",
    "sinkhorn_plan",
    "solve_cut_master",
    "solve_cutting_plane",
    "solve_inverse_ot",
    "support_components",
]
