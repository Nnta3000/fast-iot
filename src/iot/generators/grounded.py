"""
CN: 用 HELLO-OT 生成带真值的逆 OT 实例。该模块只读调用 hello_ot，不修改它。
EN: Generate inverse-OT instances with known ground truth using HELLO-OT. This module
    calls hello_ot read-only and never modifies it.

CN: 逐坐标加权平方欧氏代价与 HELLO-OT 的双线性 runtime 精确对应：
    c_ij(theta) = sum_k theta_k (x_ik - y_jk)^2
                = (sum_k theta_k x_ik^2) + (sum_k theta_k y_jk^2)
                  - 2 <x_i * sqrt(theta), y_j * sqrt(theta)>.
EN: Per-coordinate weighted squared Euclidean cost matches the HELLO-OT bilinear
    runtime exactly, so HELLO-OT can generate a plan and dual pair for a known theta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ..core.features import BilinearPSDFeatures, DiagonalWeightFeatures
from ..core.problem import InverseOTProblem
from ..core.theta import BilinearPSDTheta, SimplexTheta


@dataclass(frozen=True)
class GroundedInstance:
    """
    CN: 带已知 theta_star 与 HELLO-OT 对偶势的逆 OT 实例。
    EN: An inverse-OT instance with a known theta_star and HELLO-OT dual potentials.
    """

    problem: InverseOTProblem
    theta_star: np.ndarray
    source_dual: np.ndarray
    target_dual: np.ndarray
    objective: float


def _mixing_matrix(dimension: int, strength: float, rng: np.random.Generator) -> np.ndarray | None:
    """
    CN: 构造下三角混合矩阵，使坐标相关（非乘积分布）；strength=0 时返回 None。
    EN: Build a lower-triangular mixing matrix that correlates coordinates (non-product);
        return None when strength is 0.
    """
    if float(strength) <= 0.0:
        return None
    lower = np.tril(rng.standard_normal((int(dimension), int(dimension))), -1)
    return np.eye(int(dimension), dtype=np.float64) + float(strength) * lower


def make_diagonal_grounded_instance(
    *,
    n_source: int,
    n_target: int,
    n_features: int,
    rng: np.random.Generator,
    theta_star: np.ndarray | None = None,
    mixing_strength: float = 0.0,
    max_iterations: int = 100,
    solver_seed: int = 0,
) -> GroundedInstance:
    """
    CN: 生成 x, y, theta_star，用 HELLO-OT 求解加权平方欧氏 OT，返回 pi_hat, f, g。
    EN: Draw x, y, theta_star, solve the weighted squared-Euclidean OT with HELLO-OT,
        and return pi_hat, f, g.

    CN: 对偶势来自 HELLO-OT，因此存在约 1e-7 的可行性残差；验证端用容差吸收。
    EN: The duals come from HELLO-OT and carry roughly 1e-7 residual feasibility error;
        the validation layer absorbs it with tolerances.

    CN: `mixing_strength > 0` 让坐标相关，避免乘积分布下 theta 完全不可辨识。
    EN: `mixing_strength > 0` correlates coordinates, avoiding the product-measure regime
        where theta is unidentifiable for every value.
    """
    if int(n_source) < 1 or int(n_target) < 1 or int(n_features) < 1:
        raise ValueError("n_source, n_target, and n_features must all be >= 1")
    # CN: 惰性 import，保证 `import iot` 不强制加载 hello_ot。
    # EN: Import lazily so `import iot` does not require hello_ot.
    from hello_ot.api import _solve
    from hello_ot.config import SolverOptions
    from hello_ot.cost import _prepare_bilinear_problem

    theta_set = SimplexTheta(int(n_features))
    source_points = rng.normal(size=(int(n_source), int(n_features)))
    target_points = rng.normal(size=(int(n_target), int(n_features)))
    source_mix = _mixing_matrix(int(n_features), float(mixing_strength), rng)
    target_mix = _mixing_matrix(int(n_features), float(mixing_strength), rng)
    if source_mix is not None:
        source_points = source_points @ source_mix
    if target_mix is not None:
        target_points = target_points @ target_mix
    source_points = np.ascontiguousarray(source_points, dtype=np.float64)
    target_points = np.ascontiguousarray(target_points, dtype=np.float64)
    if theta_star is None:
        star = theta_set.sample(rng)
    else:
        star = np.asarray(theta_star, dtype=np.float64).reshape(-1)
        if star.size != int(n_features):
            raise ValueError("theta_star length must equal n_features")
        if not theta_set.contains(star, tol=1e-9):
            raise ValueError("theta_star must lie in the simplex")

    scale = np.sqrt(star)
    source_scaled = np.ascontiguousarray(source_points * scale[None, :], dtype=np.float32)
    target_scaled = np.ascontiguousarray(target_points * scale[None, :], dtype=np.float32)
    source_offset = np.ascontiguousarray(
        np.sum(source_points * source_points * star[None, :], axis=1), dtype=np.float64
    )
    target_offset = np.ascontiguousarray(
        np.sum(target_points * target_points * star[None, :], axis=1), dtype=np.float64
    )
    source_mass = np.full(int(n_source), 1.0 / float(n_source), dtype=np.float64)
    target_mass = np.full(int(n_target), 1.0 / float(n_target), dtype=np.float64)
    prepared = _prepare_bilinear_problem(
        source_points=source_scaled,
        target_points=target_scaled,
        source_offset=source_offset,
        target_offset=target_offset,
        source_mass=source_mass,
        target_mass=target_mass,
        dot_scale=2.0,
        reported_cost_type="diagonal_weight",
    )
    config = SolverOptions(backend="torch", torch_device="cpu", verbose="off")._to_internal_config(
        max_iterations=int(max_iterations),
        random_seed=int(solver_seed),
    )
    result = _solve(prepared, config)
    solution = result.solution
    plan = sparse.csr_matrix(
        (solution.values, (solution.rows, solution.cols)),
        shape=(int(n_source), int(n_target)),
    )
    features = DiagonalWeightFeatures(source_points, target_points)
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=plan)
    return GroundedInstance(
        problem=problem,
        theta_star=np.ascontiguousarray(star, dtype=np.float64),
        source_dual=np.ascontiguousarray(solution.source_dual, dtype=np.float64),
        target_dual=np.ascontiguousarray(solution.target_dual, dtype=np.float64),
        objective=float(result.objective),
    )


def make_bilinear_psd_grounded_instance(
    *,
    n_source: int,
    n_target: int,
    point_dimension: int,
    rng: np.random.Generator,
    theta_star: np.ndarray | None = None,
    mixing_strength: float = 0.0,
    max_iterations: int = 100,
    solver_seed: int = 0,
) -> GroundedInstance:
    """
    CN: 生成 bilinear-PSD 真值实例：c_ij = -x_i^T A y_j，A 半正定、Tr A = 1。
    EN: Generate a bilinear-PSD grounded instance: c_ij = -x_i^T A y_j with A positive
        semidefinite and Tr A = 1.

    CN: 因为 A = L L^T，代价可写成 -<L^T x_i, L^T y_j>，正好是 HELLO-OT 的双线性 runtime
        （dot_scale = 1，零 offset），所以 HELLO-OT 仍可生成真值方案与对偶势。
    EN: Since A = L L^T the cost is -<L^T x_i, L^T y_j>, exactly HELLO-OT's bilinear
        runtime (dot_scale = 1, zero offsets), so HELLO-OT still supplies the ground-truth
        plan and duals.
    """
    if int(n_source) < 1 or int(n_target) < 1 or int(point_dimension) < 1:
        raise ValueError("n_source, n_target, and point_dimension must all be >= 1")
    from hello_ot.api import _solve
    from hello_ot.config import SolverOptions
    from hello_ot.cost import _prepare_bilinear_problem

    dim = int(point_dimension)
    theta_set = BilinearPSDTheta(dim)
    source_points = rng.normal(size=(int(n_source), dim))
    target_points = rng.normal(size=(int(n_target), dim))
    source_mix = _mixing_matrix(dim, float(mixing_strength), rng)
    target_mix = _mixing_matrix(dim, float(mixing_strength), rng)
    if source_mix is not None:
        source_points = source_points @ source_mix
    if target_mix is not None:
        target_points = target_points @ target_mix
    source_points = np.ascontiguousarray(source_points, dtype=np.float64)
    target_points = np.ascontiguousarray(target_points, dtype=np.float64)

    if theta_star is None:
        star = theta_set.sample(rng)
    else:
        star = np.asarray(theta_star, dtype=np.float64).reshape(-1)
        if star.size != dim * dim:
            raise ValueError("theta_star length must equal point_dimension**2")
        if not theta_set.contains(star, tol=1e-7):
            raise ValueError("theta_star must be symmetric PSD with trace 1")

    matrix = star.reshape(dim, dim)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))[None, :]
    source_scaled = np.ascontiguousarray(source_points @ factor, dtype=np.float32)
    target_scaled = np.ascontiguousarray(target_points @ factor, dtype=np.float32)
    source_offset = np.zeros(int(n_source), dtype=np.float64)
    target_offset = np.zeros(int(n_target), dtype=np.float64)
    source_mass = np.full(int(n_source), 1.0 / float(n_source), dtype=np.float64)
    target_mass = np.full(int(n_target), 1.0 / float(n_target), dtype=np.float64)
    prepared = _prepare_bilinear_problem(
        source_points=source_scaled,
        target_points=target_scaled,
        source_offset=source_offset,
        target_offset=target_offset,
        source_mass=source_mass,
        target_mass=target_mass,
        dot_scale=1.0,
        reported_cost_type="bilinear_psd",
    )
    config = SolverOptions(backend="torch", torch_device="cpu", verbose="off")._to_internal_config(
        max_iterations=int(max_iterations),
        random_seed=int(solver_seed),
    )
    result = _solve(prepared, config)
    solution = result.solution
    plan = sparse.csr_matrix(
        (solution.values, (solution.rows, solution.cols)),
        shape=(int(n_source), int(n_target)),
    )
    features = BilinearPSDFeatures(source_points, target_points)
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=plan)
    return GroundedInstance(
        problem=problem,
        theta_star=np.ascontiguousarray(star, dtype=np.float64),
        source_dual=np.ascontiguousarray(solution.source_dual, dtype=np.float64),
        target_dual=np.ascontiguousarray(solution.target_dual, dtype=np.float64),
        objective=float(result.objective),
    )


__all__ = [
    "GroundedInstance",
    "make_bilinear_psd_grounded_instance",
    "make_diagonal_grounded_instance",
]
