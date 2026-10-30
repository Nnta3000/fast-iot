"""
CN: **求解器与指标** —— 在给定几何上做一次完整 `m·n` LP（`box ∩ simplex`），并计算检索指标。

    CN: 这里**不**做预处理（在 `registry`），也不做划分（在 `driver`）；只接收已切好的数组。

EN: **Solver and metrics** -- one full `m*n` LP (`box ∩ simplex`) on a prepared geometry, plus the
    retrieval metrics. No preprocessing here (that is `registry`) and no splitting (that is `driver`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import ot
import scipy.sparse as sparse
from jaxtyping import Bool, Float, Int

from iot.core.edges import EdgeSet
from iot.core.features import DiagonalWeightFeatures, FeatureMap
from iot.core.problem import InverseOTProblem
from iot.core.theta import LinearThetaConstraints
from iot.relaxed.master_regret import RegretMasterSolver, regret_objective

POOL_ORACLE_SIZE: int = 50


class MetricsError(RuntimeError):
    """CN: 指标无法可信计算（例如 oracle 未通过）。EN: metrics cannot be trusted (e.g. oracle failed)."""


@dataclass(frozen=True)
class BoxSimplex:
    """
    CN: `simplex ∩ box`：`Sum(theta) = 1`，`lo <= theta_k <= hi`。`lo/hi` 由 `box_relative` 决定：
        `lo = box_relative / k`，`hi = 1 / (box_relative * k)`（默认 `0.25` 给出 `[0.25/k, 4/k]`）。
    EN: `simplex ∩ box`: `Sum(theta) = 1`, `lo <= theta_k <= hi`, with `lo = box_relative / k` and
        `hi = 1 / (box_relative * k)` (default `0.25` gives `[0.25/k, 4/k]`).
    """

    _dimension: int
    lo: float
    hi: float

    @property
    def dimension(self) -> int:
        """CN: 特征维数 k。EN: feature dimension k."""
        return int(self._dimension)

    def sample(self, rng: np.random.Generator) -> Float[np.ndarray, "k"]:
        """CN: 盒内采样（Dirichlet 后裁剪再归一）。EN: sample inside the box (dirichlet, clip, renorm)."""
        k = int(self._dimension)
        v: Float[np.ndarray, "k"] = np.asarray(rng.dirichlet(np.ones(k)), dtype=np.float64)
        v = np.clip(v, self.lo, self.hi)
        return v / v.sum()

    def contains(self, theta: Float[np.ndarray, "k"], *, tol: float = 1e-9) -> bool:
        """CN: 是否在集合内。EN: membership test."""
        v = np.asarray(theta, dtype=np.float64).reshape(-1)
        return bool(v.size == int(self._dimension) and np.all(v >= self.lo - tol)
                    and np.all(v <= self.hi + tol) and abs(float(v.sum()) - 1.0) <= tol)

    def linear_constraints(self) -> LinearThetaConstraints:
        """CN: 线性约束视图（给 LP master 用）。EN: linear-constraint view for the LP master."""
        k = int(self._dimension)
        return LinearThetaConstraints(lower=np.full(k, float(self.lo)),
                                      upper=np.full(k, float(self.hi)),
                                      a_eq=np.ones((1, k)), b_eq=np.ones(1))


def box_simplex(k: int, *, box_relative: float = 0.25) -> BoxSimplex:
    """CN: 构造 `simplex ∩ box`。EN: build `simplex ∩ box`."""
    if not 0.0 < float(box_relative) <= 1.0:
        raise MetricsError(f"box_relative must be in (0, 1], got {box_relative!r}")
    lower = float(box_relative) / float(k)
    return BoxSimplex(_dimension=int(k), lo=lower, hi=1.0 / (float(box_relative) * float(k)))


@dataclass(frozen=True)
class FitResult:
    """
    CN: 一次学习的读数。`R_ratio = R' / R_uniform`（>1 表示"没有 regret 改进"）。
    EN: one learned fit's read-out. `R_ratio = R' / R_uniform` (`>1` means "no regret improvement").
    """

    theta: Float[np.ndarray, "k"]
    R_prime: float
    R_uniform: float
    R_ratio: float
    nnz: int
    build_s: float
    solve_s: float
    status: str


def diagonal_cost(x: Float[np.ndarray, "m d"], y: Float[np.ndarray, "n d"],
                  theta: Float[np.ndarray, "k"]) -> Float[np.ndarray, "m n"]:
    """
    CN: 对角代价 `sum_k theta_k (x_k - y_k)^2`，用分解式避免物化 `(m, n, d)` 张量。
    EN: diagonal cost via the factorised form, avoiding an `(m, n, d)` tensor.
    """
    xm: Float[np.ndarray, "m k"] = np.asarray(x, dtype=np.float64)
    ym: Float[np.ndarray, "n k"] = np.asarray(y, dtype=np.float64)
    th: Float[np.ndarray, "k"] = np.asarray(theta, dtype=np.float64)
    return ((xm ** 2) @ th)[:, None] - 2.0 * ((xm * th[None, :]) @ ym.T) + ((ym ** 2) @ th)[None, :]


def uniform_cosine(x: Float[np.ndarray, "m d"], y: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "m n"]:
    """CN: `1 - <x, y>`（输入已 L2 归一化）。EN: `1 - <x, y>` (inputs assumed L2-normalised)."""
    return 1.0 - np.asarray(x, dtype=np.float64) @ np.asarray(y, dtype=np.float64).T


def matching_hit(cost: Float[np.ndarray, "n n"], *, seed: int = 0) -> float:
    """
    CN: **方阵** EMD 命中率：打乱候选列、用 `argsort(perm)` 跟踪真伙伴。
        入口先跑一次 oracle（对角 0、其余 1），返回值必须为 1，否则拒绝计算。
    EN: **Square** EMD hit rate: shuffle candidate columns and track the true partner via `argsort(perm)`.
        An oracle (diagonal 0, off-diagonal 1) is checked first and must return 1, else we refuse.
    """
    oracle_cost: Float[np.ndarray, "cycles cycles"] = np.ones(
        (POOL_ORACLE_SIZE, POOL_ORACLE_SIZE), dtype=np.float64)
    np.fill_diagonal(oracle_cost, 0.0)
    if abs(_emd_hit(oracle_cost, seed=0) - 1.0) > 1e-12:
        raise MetricsError("square-EMD oracle failed; hit rates would be meaningless")
    return _emd_hit(np.asarray(cost, dtype=np.float64), seed=seed)


def _emd_hit(cost: Float[np.ndarray, "n n"], *, seed: int) -> float:
    """CN: 实际计算（已假定方阵）。EN: the actual computation (square assumed)."""
    h = int(cost.shape[0])
    if cost.shape[1] != h:
        raise MetricsError(f"square EMD requires a square matrix, got {cost.shape}")
    perm: Int[np.ndarray, "n"] = np.random.default_rng(int(seed)).permutation(h)
    plan: Float[np.ndarray, "n n"] = ot.emd(np.ones(h) / h, np.ones(h) / h,
                                            cost[:, perm], numItermax=10_000_000)
    return float(np.mean(np.argmax(plan, axis=1) == np.argsort(perm)))


def rank_metrics(cost: Float[np.ndarray, "n n"]) -> dict[str, float]:
    """
    CN: 方阵检索指标（目标是恒等置换）：`R@1`、`R@5`、中位排名。
    EN: square retrieval metrics (identity target): `R@1`, `R@5`, median rank.
    """
    values: Float[np.ndarray, "n n"] = np.asarray(cost, dtype=np.float64)
    n = int(values.shape[0])
    order: Int[np.ndarray, "n n"] = np.argsort(values, axis=1, kind="stable")
    match: Bool[np.ndarray, "n n"] = order == np.arange(n)[:, None]
    ranks: Int[np.ndarray, "n"] = np.argmax(match, axis=1) + 1
    return {"R@1": float(np.mean(ranks <= 1)), "R@5": float(np.mean(ranks <= 5)),
            "median": float(np.median(ranks))}


def fit_full_lp(xl: Float[np.ndarray, "m k"], yl: Float[np.ndarray, "m k"], *,
                solver: str, solve_limit_s: float, box_relative: float = 0.25,
                features: FeatureMap | None = None) -> FitResult:
    """
    CN: 在全部 `m·m` 边上解一次 LP（Θ = box ∩ simplex）。失败时 `status` 以 `FAILED` 开头，
        **绝不**回退成 uniform θ。

        CN: `features` 允许使用**任意** `FeatureMap`（对角、二次、低秩……），默认 `None` 表示
            `DiagonalWeightFeatures`，因此所有既有调用与已发布结果**逐位不变**。Θ 的维度取自
            `features.n_features`，而不是 `xl.shape[1]`：对二次族二者不同（`d(d+1)/2` vs `d`）。
        CN: 注意 `m` 是**方案**的维度（对角选择），`xl`/`yl` 可以与它不同长；只有 `features` 的
            行列数必须与之相符。
    EN: one LP solve over all `m*m` edges (Θ = box ∩ simplex). On failure `status` starts with `FAILED`;
        it **never** falls back to uniform theta.

        `features` accepts **any** `FeatureMap` (diagonal, quadratic, low-rank, ...); the default `None`
        means `DiagonalWeightFeatures`, so every existing caller and published result is **bit-identical**.
        The theta dimension comes from `features.n_features`, not `xl.shape[1]`: for the quadratic family
        those differ (`d(d+1)/2` vs `d`).
    """
    m = int(xl.shape[0])
    family: FeatureMap = DiagonalWeightFeatures(xl, yl) if features is None else features
    # CN: 方案是对角 m x m，所以 `features` 必须是 m x m；`xl`/`yl` 自己的行数不参与检查
    #     （允许调用方传入更长的点集再取子集）。
    # EN: the plan is the m x m diagonal, so `features` must be m x m; `xl`/`yl` row counts are not
    #     checked (callers may pass a longer point set and subset it).
    if int(family.n_source) != m or int(family.n_target) != m:
        raise MetricsError(
            f"features shape ({family.n_source}, {family.n_target}) does not match the "
            f"{m} x {m} diagonal plan")
    k = int(family.n_features)
    plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
    problem = InverseOTProblem(features=family,
                               theta_set=box_simplex(k, box_relative=box_relative), plan=plan)
    r: Int[np.ndarray, "m m"]
    c: Int[np.ndarray, "m m"]
    r, c = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")

    t0 = time.perf_counter()
    edges = EdgeSet.from_arrays(r.reshape(-1), c.reshape(-1), n_target=m)
    master = RegretMasterSolver(backend=str(solver), tie_break=False,
                                params={"time_sec_limit": float(solve_limit_s)})
    build_s = time.perf_counter() - t0

    # CN: 用 uniform theta 的可行对偶给出 `R_uniform`，用于 `R'/R_uniform`。
    # EN: feasible duals at uniform theta give `R_uniform`, used for `R'/R_uniform`.
    uni: Float[np.ndarray, "k"] = np.full(k, 1.0 / k, dtype=np.float64)
    cost_l: Float[np.ndarray, "m m"] = np.asarray(problem.features.costs(uni, r.reshape(-1), c.reshape(-1)),
                                                  dtype=np.float64).reshape(m, m)
    f_uni: Float[np.ndarray, "m"] = cost_l.min(axis=1)
    g_uni: Float[np.ndarray, "m"] = np.zeros(m, dtype=np.float64)
    R_uniform = float(regret_objective(problem) @ np.concatenate([uni, f_uni, g_uni]))

    t0 = time.perf_counter()
    try:
        sol = master.solve(problem, edges)
        theta: Float[np.ndarray, "k"] = np.asarray(sol.theta, dtype=np.float64)
        R_prime = float(sol.regret)
        status = "ok"
    except Exception as error:  # noqa: BLE001 - surfaced as FAILED, never substituted
        theta = np.full(k, np.nan, dtype=np.float64)
        R_prime = float("nan")
        status = f"FAILED: {type(error).__name__}: {str(error)[:120]}"
    solve_s = time.perf_counter() - t0
    # CN: `R_uniform ~ 0`（浮点零）意味着 uniform theta 本身已使方案最优，比值无定义——返回 nan 而非伪造。
    # EN: `R_uniform ~ 0` (floating-point zero) means uniform theta already certifies the plan, so the ratio
    #     is undefined; return nan rather than inventing one.
    ratio = R_prime / R_uniform if R_uniform > 1e-12 else float("nan")
    return FitResult(theta=theta, R_prime=R_prime, R_uniform=R_uniform, R_ratio=ratio,
                     nnz=int(np.count_nonzero(np.isfinite(theta) & (theta > 1e-12))),
                     build_s=round(build_s, 2), solve_s=round(solve_s, 2), status=status)


__all__ = [
    "BoxSimplex",
    "FitResult",
    "MetricsError",
    "box_simplex",
    "diagonal_cost",
    "fit_full_lp",
    "matching_hit",
    "rank_metrics",
    "uniform_cosine",
]
