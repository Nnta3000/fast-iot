"""
CN: 逆 OT 的**词汇层**：问题、边集、参数集、特征族、结果类型。任何算法都只依赖这一层。
EN: The **vocabulary layer** of inverse OT: problem, edge set, parameter set, feature families and result
    types. Every algorithm depends only on this layer.
"""

from .edges import EdgeSet, edge_keys
from .features import (
    BilinearPSDFeatures,
    ConcatFeatures,
    DenseFeatureTensor,
    DiagonalWeightFeatures,
    KernelMixtureFeatures,
    FeatureMap,
    QuadraticFeatures,
    RadialProfileFeatures,
    dense_cost_matrix,
)
from .problem import InverseOTProblem
from .result import InverseOTResult, IterationRecord
from .theta import (
    BilinearPSDTheta,
    LinearThetaConstraints,
    LinearThetaSet,
    SimplexTheta,
    ThetaSet,
)

__all__ = [
    "BilinearPSDFeatures",
    "ConcatFeatures",
    "BilinearPSDTheta",
    "DenseFeatureTensor",
    "DiagonalWeightFeatures",
    "KernelMixtureFeatures",
    "EdgeSet",
    "FeatureMap",
    "InverseOTProblem",
    "InverseOTResult",
    "IterationRecord",
    "LinearThetaConstraints",
    "LinearThetaSet",
    "QuadraticFeatures",
    "RadialProfileFeatures",
    "SimplexTheta",
    "ThetaSet",
    "dense_cost_matrix",
    "edge_keys",
]
