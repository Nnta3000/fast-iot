"""
CN: hello-iot 公开 API。**按用途分组**，每组一个子包：

      iot.core         问题、边集、参数集、特征族、结果类型（词汇层，无算法）
      iot.exact        精确（max-margin）逆 OT —— 主求解器 `solve_inverse_ot`
      iot.relaxed      松弛（最小 regret）逆 OT —— `solve_inverse_ot_regret`，不假设 A1
      iot.backends     可替换后端：CuPDLPx LP、torch/HELLO 分离扫描、双线性嵌入
      iot.alternatives 其他算法：切割平面、精确次梯度、熵正则
      iot.eval         验证与指标
      iot.generators   真值与精确 OT 内层求解器

EN: Public API of hello-iot, **grouped by purpose**, one subpackage per group:

      iot.core         problem, edge set, parameter set, feature families, result types (no algorithms)
      iot.exact        exact (max-margin) inverse OT -- the main solver `solve_inverse_ot`
      iot.relaxed      relaxed (minimum-regret) inverse OT -- `solve_inverse_ot_regret`, drops A1
      iot.backends     swappable backends: CuPDLPx LP, torch/HELLO separation scans, bilinear embedding
      iot.alternatives other algorithms: cutting plane, exact subgradient, entropic
      iot.eval         validation and metrics
      iot.generators   ground truth and the exact-OT inner solver

CN: 顶层只再导出**最常用**的几个名字，便于 `from iot import solve_inverse_ot`。
EN: The top level re-exports only the most common names, so `from iot import solve_inverse_ot` works.
"""

from __future__ import annotations

from . import alternatives, backends, core, eval, exact, generators, relaxed
from .core import (
    BilinearPSDFeatures,
    BilinearPSDTheta,
    DenseFeatureTensor,
    DiagonalWeightFeatures,
    EdgeSet,
    FeatureMap,
    InverseOTError,
    InverseOTProblem,
    InverseOTResult,
    IterationRecord,
    LinearThetaConstraints,
    LinearThetaSet,
    QuadraticFeatures,
    SimplexTheta,
    ThetaSet,
    dense_cost_matrix,
    edge_keys,
)
from .eval import (
    DualCertificate,
    OptimalityCheck,
    ThetaRange,
    dual_optimality_certificate,
    fixed_dual_theta_ranges,
    is_optimal_plan,
    optimal_theta_ranges,
    plan_cost,
)
from .exact import (
    ComponentPartition,
    MasterSolution,
    MasterSolver,
    TieBreak,
    connector_edges,
    mandatory_edges,
    scan_violations,
    solve_inverse_ot,
    support_components,
)
from .backends import (
    CuPdlpxLp,
    CuPdlpxMasterSolver,
    bilinear_embedding,
    build_cupdlpx_lp,
    cuda_available,
    describe_backend,
    fast_scan_supported,
    features_support_bilinear,
    resolve_torch_device,
    scan_violations_fast,
    scan_violations_torch,
    torch_available,
)
from .alternatives import (
    CutMasterResult,
    CuttingPlaneConfig,
    CuttingPlaneCut,
    CuttingPlaneIteration,
    CuttingPlaneResult,
    EntropicFitResult,
    EntropicIteration,
    MasterMode,
    SubgradientConfig,
    SubgradientIteration,
    SubgradientResult,
    fit_psd_moment_entropic,
    fit_theta_subgradient,
    moment_from_plan,
    project_psd,
    scale_invariant_error,
    sinkhorn_plan,
    solve_cut_master,
    solve_cutting_plane,
)
from .exact.sdp import SdpMasterSolver
from .exact.master import (
    CanonicalGaugeMasterSolver,
    LinearMasterSolver,
    RegularizedMasterSolver,
    WarmStartMasterSolver,
    canonical_potentials,
    draw_generic_direction,
)
from .exact.reduced_master import (
    ForestRepresentation,
    SpanningForestMasterSolver,
    build_forest_representation,
)
from .relaxed import (
    RegretIteration,
    RegretMasterSolver,
    RegretResult,
    RegretSolution,
    build_regret_lp,
    full_regret,
    regret_objective,
    solve_inverse_ot_regret,
    support_duals,
)
from .generators import (
    GroundedInstance,
    hello_solver_supported,
    make_bilinear_psd_grounded_instance,
    make_diagonal_grounded_instance,
    make_hello_solver,
)
from .backends import BilinearEmbeddable, BilinearEmbedding
from .alternatives.entropic import SinkhornResult, sinkhorn_plan_with_potentials
from .alternatives.sinkhorn_theta import (
    SinkhornThetaIteration,
    SinkhornThetaResult,
    fit_theta_sinkhorn,
    project_simplex,
    project_theta,
)
from .alternatives.subgradient import (
    observed_feature_moment,
    subgradient_from_plan,
    support_moment,
)
from .eval.evaluation import (
    InverseOTMetrics,
    evaluate_inverse_ot,
    feature_moment,
    l0_gap,
    moment_error,
)

__version__ = "0.2.0"

__all__ = [
    # subpackages
    "alternatives", "backends", "core", "eval", "exact", "generators", "relaxed",
    # core vocabulary
    "BilinearPSDFeatures", "BilinearPSDTheta", "DenseFeatureTensor", "DiagonalWeightFeatures",
    "EdgeSet", "FeatureMap", "InverseOTError", "InverseOTProblem", "InverseOTResult",
    "IterationRecord", "LinearThetaConstraints", "LinearThetaSet", "QuadraticFeatures",
    "SimplexTheta", "ThetaSet", "dense_cost_matrix", "edge_keys",
    # exact
    "CanonicalGaugeMasterSolver", "ComponentPartition", "ForestRepresentation", "LinearMasterSolver",
    "MasterSolution", "MasterSolver", "RegularizedMasterSolver", "SdpMasterSolver",
    "SpanningForestMasterSolver", "TieBreak", "WarmStartMasterSolver",
    "build_forest_representation", "canonical_potentials", "connector_edges",
    "draw_generic_direction", "mandatory_edges", "scan_violations", "solve_inverse_ot",
    "support_components",
    # relaxed
    "RegretIteration", "RegretMasterSolver", "RegretResult", "RegretSolution",
    "build_regret_lp", "full_regret", "regret_objective", "solve_inverse_ot_regret", "support_duals",
    # backends
    "BilinearEmbeddable", "BilinearEmbedding", "CuPdlpxLp", "CuPdlpxMasterSolver",
    "bilinear_embedding", "build_cupdlpx_lp", "cuda_available", "describe_backend",
    "fast_scan_supported", "features_support_bilinear", "resolve_torch_device",
    "scan_violations_fast", "scan_violations_torch", "torch_available",
    # alternatives
    "CutMasterResult", "CuttingPlaneConfig", "CuttingPlaneCut", "CuttingPlaneIteration",
    "CuttingPlaneResult", "EntropicFitResult", "EntropicIteration", "MasterMode", "SinkhornResult",
    "SinkhornThetaIteration", "SinkhornThetaResult", "SubgradientConfig", "SubgradientIteration",
    "SubgradientResult", "fit_psd_moment_entropic", "fit_theta_sinkhorn", "fit_theta_subgradient",
    "moment_from_plan", "observed_feature_moment", "project_psd", "project_simplex", "project_theta",
    "scale_invariant_error", "sinkhorn_plan", "sinkhorn_plan_with_potentials", "solve_cut_master",
    "solve_cutting_plane", "subgradient_from_plan", "support_moment",
    # eval
    "DualCertificate", "InverseOTMetrics", "OptimalityCheck", "ThetaRange",
    "dual_optimality_certificate", "evaluate_inverse_ot", "feature_moment",
    "fixed_dual_theta_ranges", "is_optimal_plan", "l0_gap", "moment_error",
    "optimal_theta_ranges", "plan_cost",
    # generators
    "GroundedInstance", "hello_solver_supported", "make_bilinear_psd_grounded_instance",
    "make_diagonal_grounded_instance", "make_hello_solver",
]
