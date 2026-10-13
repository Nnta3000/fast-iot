"""
CN: 「更难的代价族」基准的实例生成器与通用评估器。与 `instances.py` 不同，这里不限于
    `DiagonalWeightFeatures`：支持对角、二次、带符号二次/双线性、低秩随机、以及 bilinear-PSD。
    所有族都遵守 `c_ij(theta) = <Phi_ij, theta>`，只在特征映射与 Theta 上不同。
EN: Instance generator and generic evaluator for the "harder cost families" benchmark. Unlike
    `instances.py`, it is not tied to `DiagonalWeightFeatures`: it covers diagonal, quadratic,
    signed quadratic/bilinear, low-rank random, and bilinear-PSD. Every family obeys
    `c_ij(theta) = <Phi_ij, theta>`; only the feature map and Theta differ.

CN: 这不是共享协议（PROTOCOL.md 的 K=8 干净实例仍是权威）。它服务的是一条独立的轴：
    「当代价族不再是可分对角时，方法如何退化」。
EN: This is not the shared protocol (PROTOCOL.md's clean K=8 instance remains authoritative);
    it serves a separate axis: "how the methods degrade when the cost family is no longer
    separable-diagonal".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import ot
from scipy import sparse

from iot import (
    BilinearPSDFeatures,
    BilinearPSDTheta,
    DenseFeatureTensor,
    DiagonalWeightFeatures,
    InverseOTProblem,
    SimplexTheta,
    dense_cost_matrix,
    is_optimal_plan,
)

from .lowrank import LowRankFeatures

Family = Literal[
    "diagonal",
    "quadratic",
    "quadratic-signed",
    "bilinear-signed",
    "lowrank",
    "bilinear-psd",
]
FAMILIES: tuple[str, ...] = (
    "diagonal",
    "quadratic",
    "quadratic-signed",
    "bilinear-signed",
    "lowrank",
    "bilinear-psd",
)
LP_FAMILIES: tuple[str, ...] = (
    "diagonal",
    "quadratic",
    "quadratic-signed",
    "bilinear-signed",
    "lowrank",
)
SIGNED_FAMILIES: frozenset[str] = frozenset({"quadratic-signed", "bilinear-signed"})

DEFAULT_NUM_ITERMAX: int = 2_000_000


@dataclass(frozen=True)
class HardInstance:
    """
    CN: 一个「难」实例：族、点维度 d、特征数 K、真值 theta_star、真值方案与逆问题对象。
    EN: One "hard" instance: family, point dimension d, feature count K, ground-truth
        theta_star, the true plan, and the inverse problem.
    """

    family: str
    n: int
    point_dimension: int
    k: int
    seed: int
    theta_star: np.ndarray
    features: object
    theta_set: object
    problem: InverseOTProblem
    plan: np.ndarray
    source_mass: np.ndarray
    target_mass: np.ndarray

    @property
    def is_lp(self) -> bool:
        """CN: Theta 是否有 LP 视图（简単形）；PSD 不是。EN: Whether Theta is an LP (simplex)."""
        return self.family in LP_FAMILIES

    @property
    def support_over_m_plus_n(self) -> float:
        """CN: 支撑密度 |S|/(m+n)。EN: Support density |S|/(m+n)."""
        return float(self.problem.support_size) / float(self.n + self.n)


def _quadratic_columns(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """CN: (x_ia-y_ja)(x_ib-y_jb), a<=b；形状 (m,n,K)。EN: Quadratic cross-term columns."""
    d = int(source.shape[1])
    columns: list[np.ndarray] = []
    for a in range(d):
        for b in range(a, d):
            columns.append(
                (source[:, a][:, None] - target[:, a][None, :])
                * (source[:, b][:, None] - target[:, b][None, :])
            )
    return np.stack(columns, axis=2)


def _bilinear_columns(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """CN: x_ia y_jb（全部 a,b）；形状 (m,n,d*d)。EN: Bilinear columns x_ia y_jb."""
    d = int(source.shape[1])
    columns: list[np.ndarray] = []
    for a in range(d):
        for b in range(d):
            columns.append(source[:, a][:, None] * target[:, b][None, :])
    return np.stack(columns, axis=2)


def _signed_theta_star(rng: np.random.Generator, base_k: int) -> np.ndarray:
    """
    CN: 带符号族的真值：A = theta+ - theta-，theta+/- 的 L1 质量为 1。
    EN: Ground truth for signed families: A = theta_+ - theta_-, with L1 mass 1 on each side.
    """
    coefficients = rng.normal(size=int(base_k))
    scale = float(np.sum(np.abs(coefficients)))
    if scale <= 0.0:
        coefficients = np.ones(int(base_k), dtype=np.float64)
        scale = float(base_k)
    coefficients = coefficients / scale
    return np.ascontiguousarray(
        np.concatenate([np.maximum(coefficients, 0.0), np.maximum(-coefficients, 0.0)]),
        dtype=np.float64,
    )


def build_hard_instance(
    n: int,
    *,
    family: str,
    seed: int = 0,
    point_dimension: int = 4,
    lowrank_k: int = 8,
    num_itermax: int = DEFAULT_NUM_ITERMAX,
) -> HardInstance:
    """
    CN: 构造一个已求解真值方案的难实例。`theta_star` 先抽，然后抽点云（与 `instances.py` 的
        顺序一致，便于跨族比较）。
    EN: Build one hard instance with a solved ground-truth plan. `theta_star` is drawn first,
        then the point clouds (same order as `instances.py`, so families are comparable).
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if int(n) < 2:
        raise ValueError("n must be >= 2")
    rng = np.random.default_rng(int(seed))
    d = int(point_dimension)

    if family == "bilinear-psd":
        theta_set = BilinearPSDTheta(d)
        theta_star = theta_set.sample(rng)
        source = rng.normal(size=(int(n), d))
        target = rng.normal(size=(int(n), d))
        features = BilinearPSDFeatures(source, target)
        k = d * d
    elif family == "lowrank":
        k = int(lowrank_k)
        theta_set = SimplexTheta(k)
        theta_star = theta_set.sample(rng)
        features = LowRankFeatures(
            rng.normal(size=(int(n), k)), rng.normal(size=(int(n), k))
        )
    else:
        source = rng.normal(size=(int(n), d))
        target = rng.normal(size=(int(n), d))
        if family == "diagonal":
            features = DiagonalWeightFeatures(source, target)
            k = d
            theta_set = SimplexTheta(k)
            theta_star = theta_set.sample(rng)
        else:
            base = (
                _quadratic_columns(source, target)
                if family.startswith("quadratic")
                else _bilinear_columns(source, target)
            )
            base_k = int(base.shape[2])
            if family in SIGNED_FAMILIES:
                tensor = np.concatenate([base, -base], axis=2)
                theta_star = _signed_theta_star(rng, base_k)
            else:
                tensor = base
                theta_star = SimplexTheta(base_k).sample(rng)
            features = DenseFeatureTensor(tensor)
            k = int(tensor.shape[2])
            theta_set = SimplexTheta(k)

    mass = np.full(int(n), 1.0 / float(n), dtype=np.float64)
    cost = dense_cost_matrix(features, theta_star)
    plan = np.ascontiguousarray(
        ot.emd(mass, mass, cost, numItermax=int(num_itermax)), dtype=np.float64
    )
    if not np.all(np.isfinite(plan)):
        raise RuntimeError("POT returned a non-finite ground-truth plan")
    problem = InverseOTProblem(
        features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan)
    )
    return HardInstance(
        family=family,
        n=int(n),
        point_dimension=d,
        k=int(k),
        seed=int(seed),
        theta_star=np.ascontiguousarray(theta_star, dtype=np.float64),
        features=features,
        theta_set=theta_set,
        problem=problem,
        plan=plan,
        source_mass=mass,
        target_mass=mass,
    )


def score_hard(
    instance: HardInstance,
    theta: np.ndarray,
    *,
    observed_plan: np.ndarray | None = None,
    true_plan: np.ndarray | None = None,
    check_optimality: bool = True,
) -> dict:
    """
    CN: 与 `evaluate.py` 同义的 G/R/L0 指标，但适用于任意族。`observed_plan`/`true_plan` 默认为
        `instance.plan`；噪声臂传入 `pi_hat` 与真值 `pi*` 以把「观测间隙」与「真值后悔」分开。
    EN: The same G/R/L0 metrics as `evaluate.py`, for an arbitrary family.
        `observed_plan`/`true_plan` default to `instance.plan`; the noisy arm passes `pi_hat` and
        the true plan `pi*` so the observed gap and the true-cost regret stay separate.
    """
    value = np.asarray(theta, dtype=np.float64).reshape(-1)
    observed = instance.plan if observed_plan is None else np.asarray(observed_plan, dtype=np.float64)
    true = instance.plan if true_plan is None else np.asarray(true_plan, dtype=np.float64)
    cost_theta = dense_cost_matrix(instance.features, value)
    cost_true = dense_cost_matrix(instance.features, instance.theta_star)
    sigma = float(np.std(cost_true))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("std(C_theta*) is not positive; cannot normalise")
    observed_optimal = ot.emd(
        instance.source_mass, instance.target_mass, cost_theta,
        numItermax=int(DEFAULT_NUM_ITERMAX),
    )
    gap_raw = float(np.sum(cost_theta * observed) - np.sum(cost_theta * observed_optimal))
    induced = ot.emd(
        instance.source_mass, instance.target_mass, cost_theta,
        numItermax=int(DEFAULT_NUM_ITERMAX),
    )
    regret_raw = float(np.sum(cost_true * induced) - np.sum(cost_true * true))
    is_optimal: bool | None = None
    if check_optimality:
        is_optimal = bool(is_optimal_plan(instance.problem, value).is_optimal)
    return {
        "l0": gap_raw,
        "G": gap_raw / sigma,
        "R": regret_raw / sigma,
        "R_raw": regret_raw,
        "cost_std": sigma,
        "is_optimal": is_optimal,
    }


def uniform_theta(instance: HardInstance) -> np.ndarray:
    """CN: Theta 统一的无信息参考点（简単形均匀；PSD 用 I/d）。EN: Uniform reference point."""
    if instance.family == "bilinear-psd":
        d = int(instance.point_dimension)
        return np.ascontiguousarray((np.eye(d) / float(d)).reshape(-1), dtype=np.float64)
    return np.full(int(instance.k), 1.0 / float(instance.k), dtype=np.float64)


__all__ = [
    "DEFAULT_NUM_ITERMAX",
    "FAMILIES",
    "LP_FAMILIES",
    "SIGNED_FAMILIES",
    "Family",
    "HardInstance",
    "build_hard_instance",
    "score_hard",
    "uniform_theta",
]
