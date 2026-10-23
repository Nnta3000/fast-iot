"""
CN: 逆最优传输的参数集合 Theta 及其线性约束表示。
EN: Parameter sets Theta for inverse optimal transport and their linear constraint views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class LinearThetaConstraints:
    """
    CN: Theta 的线性视图：lower <= theta <= upper 且 a_eq @ theta == b_eq。
    EN: Linear view of Theta: lower <= theta <= upper together with a_eq @ theta == b_eq.

    CN: 只描述可用 LP 表达的约束；非线性集合（如 bilinear-PSD）必须用专门求解器。
    EN: Only LP-representable constraints live here; non-linear sets (e.g. bilinear-PSD)
        require a dedicated solver.
    """

    lower: np.ndarray
    upper: np.ndarray
    a_eq: np.ndarray
    b_eq: np.ndarray

    def __post_init__(self) -> None:
        lower = np.ascontiguousarray(np.asarray(self.lower, dtype=np.float64).reshape(-1))
        upper = np.ascontiguousarray(np.asarray(self.upper, dtype=np.float64).reshape(-1))
        a_eq = np.ascontiguousarray(np.atleast_2d(np.asarray(self.a_eq, dtype=np.float64)))
        b_eq = np.ascontiguousarray(np.asarray(self.b_eq, dtype=np.float64).reshape(-1))
        if lower.size != upper.size:
            raise ValueError("lower and upper must have the same length")
        if np.any(lower > upper):
            raise ValueError("lower bounds must not exceed upper bounds")
        if a_eq.shape[1] != lower.size:
            raise ValueError("a_eq must have one column per theta coordinate")
        if a_eq.shape[0] != b_eq.size:
            raise ValueError("b_eq must have one entry per row of a_eq")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "a_eq", a_eq)
        object.__setattr__(self, "b_eq", b_eq)

    @property
    def dimension(self) -> int:
        return int(self.lower.size)

    def bounds(self) -> list[tuple[float, float]]:
        """
        CN: scipy.linprog 形式的逐坐标边界。
        EN: Per-coordinate bounds in the form consumed by scipy.linprog.
        """
        return [
            (float(low), float(high))
            for low, high in zip(self.lower.tolist(), self.upper.tolist(), strict=True)
        ]


@runtime_checkable
class ThetaSet(Protocol):
    """
    CN: 紧凑凸参数集合 Theta 的协议（只包含所有集合都具备的能力）。
    EN: Protocol for a compact convex parameter set Theta (only the capabilities shared
        by every set).
    """

    @property
    def dimension(self) -> int:
        """CN: theta 的坐标数。EN: Number of theta coordinates."""
        ...

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """CN: 在 Theta 内均匀-ish 采样一个 theta。EN: Draw a theta from Theta."""
        ...

    def contains(self, theta: np.ndarray, *, tol: float = 1e-9) -> bool:
        """CN: theta 是否（在容差内）属于 Theta。EN: Whether theta lies in Theta within tol."""
        ...


@runtime_checkable
class LinearThetaSet(ThetaSet, Protocol):
    """
    CN: 可用线性约束表达的 Theta；只有这类集合才能交给 LP master。
    EN: A Theta representable by linear constraints; only these can be handed to the LP
        master.

    CN: 非线性的 Theta（如 bilinear-PSD 谱面体）必须搭配专用 master，而不是 LP。
    EN: A non-linear Theta (e.g. the bilinear-PSD spectrahedron) must pair with a
        dedicated master, not the LP.
    """

    def linear_constraints(self) -> LinearThetaConstraints:
        """CN: 返回等价线性约束。EN: Return the equivalent linear constraints."""
        ...


@dataclass(frozen=True)
class SimplexTheta:
    """
    CN: 概率单纯形 `theta >= 0, sum(theta) = 1`；v1 的主场景。
    EN: The probability simplex `theta >= 0, sum(theta) = 1`; the v1 workhorse.
    """

    dimension: int

    def __post_init__(self) -> None:
        if int(self.dimension) < 1:
            raise ValueError("SimplexTheta dimension must be >= 1")

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.ascontiguousarray(
            rng.dirichlet(np.ones(int(self.dimension), dtype=np.float64)),
            dtype=np.float64,
        )

    def contains(self, theta: np.ndarray, *, tol: float = 1e-9) -> bool:
        value = np.asarray(theta, dtype=np.float64).reshape(-1)
        if value.size != int(self.dimension):
            return False
        nonnegative = bool(np.all(value >= -float(tol)))
        normalized = abs(float(value.sum(dtype=np.float64)) - 1.0) <= float(tol)
        return nonnegative and normalized

    def linear_constraints(self) -> LinearThetaConstraints:
        size = int(self.dimension)
        return LinearThetaConstraints(
            lower=np.zeros(size, dtype=np.float64),
            upper=np.full(size, np.inf, dtype=np.float64),
            a_eq=np.ones((1, size), dtype=np.float64),
            b_eq=np.ones(1, dtype=np.float64),
        )


@dataclass(frozen=True)
class BilinearPSDTheta:
    """
    CN: 谱面体 {A 对称半正定, Tr A = 1}；theta 是 A 按行展平的 d*d 维向量。
    EN: The spectrahedron {A symmetric positive semidefinite, Tr A = 1}; theta is the
        row-major flattening of A, of dimension d * d.

    CN: 该集合不是线性可表达的，因此必须搭配 SDP master 而不是 LP master。
    EN: This set is not linearly representable, so it requires an SDP master rather than
        the LP master.
    """

    point_dimension: int

    def __post_init__(self) -> None:
        if int(self.point_dimension) < 1:
            raise ValueError("BilinearPSDTheta point_dimension must be >= 1")

    @property
    def dimension(self) -> int:
        return int(self.point_dimension) * int(self.point_dimension)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        dim = int(self.point_dimension)
        factor = rng.standard_normal((dim, dim))
        matrix = factor @ factor.T
        trace = float(np.trace(matrix))
        if trace <= 0.0:
            matrix = np.eye(dim, dtype=np.float64)
            trace = float(dim)
        matrix = matrix / trace
        return np.ascontiguousarray(matrix, dtype=np.float64).reshape(-1)

    def contains(self, theta: np.ndarray, *, tol: float = 1e-9) -> bool:
        value = np.asarray(theta, dtype=np.float64).reshape(-1)
        dim = int(self.point_dimension)
        if value.size != dim * dim:
            return False
        matrix = value.reshape(dim, dim)
        if not np.allclose(matrix, matrix.T, atol=float(tol)):
            return False
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        if float(eigenvalues.min()) < -float(tol):
            return False
        return abs(float(np.trace(matrix)) - 1.0) <= float(tol)


__all__ = [
    "BilinearPSDTheta",
    "LinearThetaConstraints",
    "LinearThetaSet",
    "SimplexTheta",
    "ThetaSet",
]
