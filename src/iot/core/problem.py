"""
CN: 逆 OT 的输入问题对象：特征、参数集合与观测方案 pi_hat。
EN: The inverse-OT input object: features, parameter set, and observed plan pi_hat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .features import FeatureMap
from .theta import ThetaSet


class InverseOTError(RuntimeError):
    """
    CN: 逆 OT 求解或验证失败。
    EN: Inverse-OT solve or validation failure.
    """


def _vector(value: np.ndarray, *, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != int(size):
        raise ValueError(f"{name} must have length {size}; got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return np.ascontiguousarray(array, dtype=np.float64)


@dataclass(frozen=True)
class InverseOTProblem:
    """
    CN: 已知 pi_hat 在何种线性代价下最优的逆问题输入。
    EN: Input to the problem of finding a linear cost for which pi_hat is optimal.

    CN: `plan` 的行和/列和就是质量向量；显式提供的质量必须与之一致。
    EN: The row/column sums of `plan` are the mass vectors; explicitly supplied masses
        must agree with those marginals.
    """

    features: FeatureMap
    theta_set: ThetaSet
    plan: sparse.csr_matrix
    source_mass: np.ndarray | None = None
    target_mass: np.ndarray | None = None
    support_rows: np.ndarray = field(init=False, repr=False, compare=False)
    support_cols: np.ndarray = field(init=False, repr=False, compare=False)
    support_values: np.ndarray = field(init=False, repr=False, compare=False)
    resolved_source_mass: np.ndarray = field(init=False, repr=False, compare=False)
    resolved_target_mass: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if int(self.theta_set.dimension) != int(self.features.n_features):
            raise ValueError(
                "theta_set.dimension must equal features.n_features; "
                f"got {int(self.theta_set.dimension)} and {int(self.features.n_features)}"
            )
        plan = sparse.csr_matrix(self.plan, dtype=np.float64)
        expected = (int(self.features.n_source), int(self.features.n_target))
        if plan.shape != expected:
            raise ValueError(f"plan shape {plan.shape} does not match features shape {expected}")
        if plan.nnz < 1:
            raise ValueError("plan must contain at least one positive entry")
        if plan.data.size and np.any(plan.data < 0.0):
            raise ValueError("plan must be nonnegative")

        row_sums = np.asarray(plan.sum(axis=1), dtype=np.float64).reshape(-1)
        col_sums = np.asarray(plan.sum(axis=0), dtype=np.float64).reshape(-1)
        if np.any(row_sums <= 0.0) or np.any(col_sums <= 0.0):
            raise ValueError("every row and column of plan must have positive mass")
        if abs(float(row_sums.sum()) - float(col_sums.sum())) > 1e-8 * max(1.0, float(row_sums.sum())):
            raise ValueError("plan source and target total masses differ")

        provided_source = (
            row_sums if self.source_mass is None
            else _vector(self.source_mass, size=expected[0], name="source_mass")
        )
        provided_target = (
            col_sums if self.target_mass is None
            else _vector(self.target_mass, size=expected[1], name="target_mass")
        )
        if self.source_mass is not None and not np.allclose(provided_source, row_sums, atol=1e-8, rtol=1e-8):
            raise ValueError("source_mass must equal the row marginals of plan")
        if self.target_mass is not None and not np.allclose(provided_target, col_sums, atol=1e-8, rtol=1e-8):
            raise ValueError("target_mass must equal the column marginals of plan")

        coo = plan.tocoo(copy=False)
        positive = coo.data > 0.0
        rows = np.ascontiguousarray(coo.row[positive].astype(np.int64, copy=True))
        cols = np.ascontiguousarray(coo.col[positive].astype(np.int64, copy=True))
        values = np.ascontiguousarray(coo.data[positive].astype(np.float64, copy=True))
        order = np.lexsort((cols, rows))
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "support_rows", np.ascontiguousarray(rows[order]))
        object.__setattr__(self, "support_cols", np.ascontiguousarray(cols[order]))
        object.__setattr__(self, "support_values", np.ascontiguousarray(values[order]))
        object.__setattr__(self, "resolved_source_mass", np.ascontiguousarray(provided_source))
        object.__setattr__(self, "resolved_target_mass", np.ascontiguousarray(provided_target))

    @property
    def n_source(self) -> int:
        return int(self.features.n_source)

    @property
    def n_target(self) -> int:
        return int(self.features.n_target)

    @property
    def n_features(self) -> int:
        return int(self.features.n_features)

    @property
    def support_size(self) -> int:
        return int(self.support_rows.size)

    def support(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CN: 支撑 S = {(i, j) : pi_hat[i, j] > 0}，行主序。
        EN: Support S = {(i, j) : pi_hat[i, j] > 0}, in row-major order.
        """
        return self.support_rows, self.support_cols, self.support_values

    def support_keys(self) -> np.ndarray:
        """
        CN: 将支撑边编码为唯一键 `i * n_target + j`，用于集合运算。
        EN: Encode support edges as unique keys `i * n_target + j` for set operations.
        """
        return self.support_rows * int(self.n_target) + self.support_cols


__all__ = [
    "InverseOTError",
    "InverseOTProblem",
]
