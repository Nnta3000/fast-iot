"""
CN: 精确 OT 内层求解器的 HELLO-OT 适配器：把 `c_ij(theta) = sum_k theta_k (x_ik - y_jk)^2`
    改写成 `||sqrt(theta) dot x_i - sqrt(theta) dot y_j||^2`，交给 HELLO-OT 的平方欧氏解算器
    （O(m+n) 内存，CPU/GPU）。它是 `src/iot` 允许 import `hello_ot` 的适配器之一。
EN: HELLO-OT adapter for the exact-OT inner solver: rewrite
    `c_ij(theta) = sum_k theta_k (x_ik - y_jk)^2` as
    `||sqrt(theta) dot x_i - sqrt(theta) dot y_j||^2` and hand it to HELLO-OT's squared-Euclidean
    solver (O(m+n) memory, CPU/GPU). One of the `src/iot` adapters allowed to import `hello_ot`.

CN: 供 `fit_theta_subgradient(..., exact_plan_solver=make_hello_solver(problem))` 与割平面
    oracle 注入使用；POT 仍是 `src/iot` 的默认内层求解器。
EN: Used by `fit_theta_subgradient(..., exact_plan_solver=make_hello_solver(problem))` and the
    cutting-plane oracle; POT remains the `src/iot` default inner solver.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..core.features import DiagonalWeightFeatures, FeatureMap
from ..core.problem import InverseOTError, InverseOTProblem
from ..alternatives.subgradient import ExactPlanSolver


def hello_solver_supported(features: FeatureMap) -> bool:
    """CN: 该特征族是否可交给 HELLO-OT（平方欧氏）。EN: Whether HELLO-OT can solve the family."""
    return isinstance(features, DiagonalWeightFeatures)


def make_hello_solver(
    problem: InverseOTProblem,
    *,
    device: str = "auto",
    max_iterations: int = 100,
) -> ExactPlanSolver:
    """
    CN: 返回一个 `ExactPlanSolver`：对 `DiagonalWeightFeatures`，用 sqrt(theta) 重缩放点云并调用
        HELLO-OT 的平方欧氏 OT。θ 分量被 clip 到非负。
    EN: Return an `ExactPlanSolver`: for `DiagonalWeightFeatures`, rescale the point clouds by
        sqrt(theta) and call HELLO-OT's squared-Euclidean OT. θ coordinates are clipped to >= 0.
    """
    if not hello_solver_supported(problem.features):
        raise InverseOTError(
            "make_hello_solver supports DiagonalWeightFeatures (weighted squared Euclidean) "
            f"only; got {type(problem.features).__name__}. Use make_pot_solver for other families."
        )
    features = problem.features
    assert isinstance(features, DiagonalWeightFeatures)
    source = np.ascontiguousarray(features.source_points, dtype=np.float64)
    target = np.ascontiguousarray(features.target_points, dtype=np.float64)
    source_mass = np.ascontiguousarray(problem.resolved_source_mass, dtype=np.float64)
    target_mass = np.ascontiguousarray(problem.resolved_target_mass, dtype=np.float64)

    def solve(theta: np.ndarray) -> sparse.spmatrix:
        import hello_ot
        from hello_ot.config import SolverOptions

        root = np.sqrt(np.clip(np.asarray(theta, dtype=np.float64).reshape(-1), 0.0, None))
        scaled_source = np.ascontiguousarray(source * root[None, :], dtype=np.float32)
        scaled_target = np.ascontiguousarray(target * root[None, :], dtype=np.float32)
        options = SolverOptions(backend="torch", torch_device=str(device), verbose="off")
        result = hello_ot.solve(
            scaled_source,
            scaled_target,
            source_mass=source_mass,
            target_mass=target_mass,
            max_iterations=int(max_iterations),
            options=options,
        )
        solution = result.solution
        return sparse.coo_matrix(
            (solution.values, (solution.rows, solution.cols)), shape=solution.shape
        ).tocsr()

    return solve


__all__ = [
    "hello_solver_supported",
    "make_hello_solver",
]
