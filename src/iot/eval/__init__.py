"""
CN: **评估与验证**：权威最优性判定、对偶证书、theta 可辨识区间、指标。
EN: **Evaluation and validation**: authoritative optimality check, dual certificate, theta ranges, metrics.
"""

from .evaluation import InverseOTMetrics, evaluate_inverse_ot, feature_moment, l0_gap, moment_error
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

__all__ = [
    "DualCertificate", "InverseOTMetrics", "OptimalityCheck", "ThetaRange",
    "dual_optimality_certificate", "evaluate_inverse_ot", "feature_moment",
    "fixed_dual_theta_ranges", "is_optimal_plan", "l0_gap", "moment_error",
    "optimal_theta_ranges", "plan_cost",
]
