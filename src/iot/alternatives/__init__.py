"""
CN: **其他逆 OT 算法**（不是约束生成循环的变体）：Kelley 切割平面、精确次梯度、熵正则梯度。
EN: **Other inverse-OT algorithms** (not variants of the constraint-generation loop): Kelley cutting plane,
    exact subgradient, entropic gradient.
"""

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
from .entropic import (
    EntropicFitResult,
    EntropicIteration,
    SinkhornResult,
    fit_psd_moment_entropic,
    moment_from_plan,
    project_psd,
    scale_invariant_error,
    sinkhorn_plan,
    sinkhorn_plan_with_potentials,
)
from .sinkhorn_theta import (
    SinkhornThetaIteration,
    SinkhornThetaResult,
    fit_theta_sinkhorn,
    project_simplex,
    project_theta,
)
from .subgradient import (
    SubgradientConfig,
    SubgradientIteration,
    SubgradientResult,
    fit_theta_subgradient,
    observed_feature_moment,
    subgradient_from_plan,
    support_moment,
)

__all__ = [
    "CutMasterResult", "CuttingPlaneConfig", "CuttingPlaneCut", "CuttingPlaneIteration",
    "CuttingPlaneResult", "EntropicFitResult", "EntropicIteration", "MasterMode", "SinkhornResult",
    "SinkhornThetaIteration", "SinkhornThetaResult", "SubgradientConfig", "SubgradientIteration",
    "SubgradientResult", "fit_psd_moment_entropic", "fit_theta_sinkhorn", "fit_theta_subgradient",
    "moment_from_plan", "observed_feature_moment", "project_psd", "project_simplex", "project_theta",
    "scale_invariant_error", "sinkhorn_plan", "sinkhorn_plan_with_potentials", "solve_cut_master",
    "solve_cutting_plane", "subgradient_from_plan", "support_moment",
]
