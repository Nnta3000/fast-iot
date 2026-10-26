"""
CN: **分数（Kantorovich）方案构造** —— 把源点滑到**胞边界**并拆分质量，得到 |S| ∈ [n, 2n] 的方案
    （循环针对的 `|S| ≈ 2n-1`、`p=1` 区间）。

CN: 构造：
     对源 i，设其最优目标 j1。沿单位方向 d 滑动 `x_i(t) = x_i + t d`，定义
         Delta_k(t) = [c_{i,k}(theta*) - g_k] - [c_{i,j1}(theta*) - g_{j1}] ,   k != j1
     初始 Delta_k(0) > 0。令
         t* = min{ t in (0, t_cap] : 存在 k, Delta_k(t) = 0 },   j2 = 在该 t* 归零的目标。
     在 `x_i(t*)` 处 j1 与 j2 并列、其它目标仍更差，故对任意 lambda ∈ (0,1)
         pi_{i,j1} = (1-lambda) a_i ,  pi_{i,j2} = lambda a_i
     两条活动边都满足 `f_i + g_j = c_{i,j}`，方案按互补松弛**精确最优**；只有第 i 行改变。

CN: **精确斜率（无有限差分）**。对 `diagonal` 与 `quadratic` 族，`c` 中 x 的二次项在 Delta_k 中相消，
    故 `Delta_k(t) = alpha_k + beta_k t` **精确**成立，其中（令 M = diag(theta) 或由 theta 构造的对称矩阵）
         alpha_k = Delta_k(0) ,   beta_k = -2 * d^T M (y_k - y_{j1}) .
    于是 `t* = min_{k: beta_k<0, alpha_k>0} (-alpha_k / beta_k)`，只保留 `t* <= t_cap`。
    若没有这样的 k（即 t_cap 内无交点），**该源不拆分**，计入 skipped。

CN: low-rank 族没有 `target_points`（因子模型），**不支持**分数路径。

EN: **Fractional (Kantorovich) plan construction** -- slide a source to a **cell boundary** and split its
    mass, giving |S| in [n, 2n] (the |S| ~ 2n-1, p=1 regime).

    Construction: for source i with best target j1, slide `x_i(t) = x_i + t d` and define
        Delta_k(t) = [c_{i,k}(theta*) - g_k] - [c_{i,j1}(theta*) - g_{j1}] ,  k != j1
    with Delta_k(0) > 0. Let `t* = min{ t in (0, t_cap] : exists k, Delta_k(t) = 0 }` and `j2` the target
    that vanishes there. At `x_i(t*)`, j1 and j2 tie and every other target is still worse, so for any
    lambda in (0,1) `pi_{i,j1}=(1-lambda)a_i`, `pi_{i,j2}=lambda a_i` keeps `f_i+g_j=c_{i,j}` on both
    edges -- exactly optimal by complementary slackness; only row i changes.

    **Exact slopes (no finite differences).** For the `diagonal` and `quadratic` families the quadratic
    term in x cancels in Delta_k, so `Delta_k(t) = alpha_k + beta_k t` holds **exactly**, with
    (M = diag(theta) or the symmetric matrix built from theta)
        alpha_k = Delta_k(0) ,  beta_k = -2 * d^T M (y_k - y_{j1}) .
    Hence `t* = min_{k: beta_k<0, alpha_k>0} (-alpha_k/beta_k)` restricted to `t* <= t_cap`. If no such k
    exists (no crossing within t_cap) the source is **left unsplit** and counted in `skipped`.

    The `lowrank` family has no `target_points` (factor model) and is **not** supported on the fractional
    path.

用法 / EN: usage
    from iot_experiments.harness.lib.fractional_planting import plant_fractional
    problem, meta = plant_fractional(family="quadratic", n=2000, point_dimension=8, seed=0,
                                     split_fraction=0.5, direction="target_diff")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from iot import (
    DiagonalWeightFeatures,
    FeatureMap,
    InverseOTProblem,
    QuadraticFeatures,
)

from .costs import dual_feasibility
from .families import FamilySpec, build_family
from .marginals import marginal_spec
from .planting import _best_two, _feature_scale, _subselect_targets

PROBLEM_TYPE = "kantorovich-fractional"
CERT_TOL = 1e-9
# CN: 滑动长度上限的默认值（相对于源-目标距离）。EN: default cap on the slide length (relative to |x_i - y_j1|).
DEFAULT_SLIDE_CAP: float = 4.0


@dataclass(frozen=True)
class FractionalInstance:
    """CN: 分数 planting 的产物与诊断。EN: The fractional planted instance plus diagnostics."""

    problem: InverseOTProblem
    theta_star: np.ndarray
    n_eff: int
    m_eff: int
    target_index: np.ndarray
    support_size: int
    support_over_n: float
    max_dual_violation: float
    max_support_residual: float
    n_split: int
    n_skipped: int
    split_ratio: float
    direction: str
    slide_cap: float
    gap_quantiles: dict[str, float]
    problem_type: str = PROBLEM_TYPE
    mass_splitting: bool = True


def _metric_matrix(features: FeatureMap, theta: np.ndarray) -> np.ndarray:
    """
    CN: 取 `c(x,y) = (x-y)^T M (x-y)` 中的对称矩阵 M：
         - `DiagonalWeightFeatures` → M = diag(theta)；
         - `QuadraticFeatures` → 由 theta 在 (a<=b) 上构造。
        其它族抛错（分数路径仅支持这两族）。
    EN: The symmetric matrix M in `c(x,y) = (x-y)^T M (x-y)`:
        - `DiagonalWeightFeatures` -> M = diag(theta);
        - `QuadraticFeatures` -> built from theta over (a<=b).
        Other families raise (the fractional path supports only these two).
    """
    value = np.asarray(theta, dtype=np.float64).reshape(-1)
    if isinstance(features, DiagonalWeightFeatures):
        if value.size != features.n_features:
            raise ValueError(f"theta has length {value.size}, expected {features.n_features}")
        return np.diag(value)
    if isinstance(features, QuadraticFeatures):
        if value.size != features.n_features:
            raise ValueError(f"theta has length {value.size}, expected {features.n_features}")
        d = int(features.point_dimension)
        matrix = np.zeros((d, d), dtype=np.float64)
        for index, (a, b) in enumerate(features.pairs()):
            if a == b:
                matrix[a, a] = value[index]
            else:
                matrix[a, b] = 0.5 * value[index]
                matrix[b, a] = 0.5 * value[index]
        return matrix
    raise TypeError(
        "fractional planting supports DiagonalWeightFeatures and QuadraticFeatures only "
        f"(got {type(features).__name__}); low-rank has no target_points."
    )


def _first_crossing_exact(features: FeatureMap, theta: np.ndarray, i: int, j1: int,
                          direction: np.ndarray, g: np.ndarray, matrix: np.ndarray,
                          slide_cap: float) -> tuple[float, int] | None:
    """
    CN: **精确**求 `t* = min{t in (0, t_cap] : 存在 k, Delta_k(t)=0}` 与最先归零的 j2。
        用闭式斜率 `beta_k = -2 d^T M (y_k - y_{j1})`；`alpha_k = Delta_k(0)` 直接算一次。
        无有限差分。返回 (t*, j2)，若 t_cap 内无交点则 None。
    EN: **Exact** `t* = min{t in (0, t_cap] : exists k, Delta_k(t)=0}` and the first vanishing target j2.
        Uses the closed-form slope `beta_k = -2 d^T M (y_k - y_{j1})`; `alpha_k = Delta_k(0)` computed
        once. No finite differences. Returns (t*, j2), or None when no crossing lies within t_cap.
    """
    n = int(features.n_target)
    y = np.asarray(features.target_points, dtype=np.float64)
    x = np.asarray(features.source_points[i], dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    y1 = y[j1]
    diff = x[None, :] - y
    quad = np.einsum("ka,ab,kb->k", diff, matrix, diff)
    alpha = (quad - g) - (quad[j1] - g[j1])            # alpha_k = Delta_k(0)
    beta = -2.0 * ((y - y1) @ (matrix @ d))            # beta_k = -2 d^T M (y_k - y1)
    best_t = np.inf
    best_k = -1
    for k in range(n):
        if k == j1 or beta[k] >= 0.0 or alpha[k] <= 0.0:
            continue
        t = -alpha[k] / beta[k]
        if 1e-12 < t <= float(slide_cap) and t < best_t:
            best_t = t
            best_k = k
    if best_k < 0 or not np.isfinite(best_t):
        return None
    return float(best_t), int(best_k)


def plant_fractional(
    *, family: str, n: int, point_dimension: int = 8, seed: int = 0,
    theta_star: np.ndarray | None = None, g: np.ndarray | None = None,
    source_mass: np.ndarray | str | None = None, lowrank_k: int = 8,
    split_fraction: float = 0.5, split_ratio: float = 0.5,
    direction: str = "target_diff", slide_cap: float = DEFAULT_SLIDE_CAP,
    chunk: int = 4096, cert_tol: float = CERT_TOL,
) -> FractionalInstance:
    """
    CN: 构造**分数**逆 OT 实例：对 `split_fraction` 比例的源滑到**胞边界**并拆分质量，得到
        |S| ∈ [n, 2n]。方案按互补松弛精确最优。方向 d、比例 lambda、滑动上限 slide_cap 都是可扫参数。
    EN: Build a **fractional** inverse-OT instance: slide a `split_fraction` share of sources to a **cell
        boundary** and split their mass, giving |S| in [n, 2n]. Exactly optimal by complementary
        slackness. Direction d, ratio lambda and the slide cap are sweep parameters.
    """
    if not 0.0 <= float(split_fraction) <= 1.0:
        raise ValueError("split_fraction must be in [0, 1]")
    if not 0.0 < float(split_ratio) < 1.0:
        raise ValueError("split_ratio must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    family_spec = FamilySpec(family, point_dimension=int(point_dimension), lowrank_k=int(lowrank_k))
    features, theta_set, theta_draw = build_family(family_spec, rng, int(n), int(n))
    # CN: 分数路径要求有点云（diagonal/quadratic）。EN: the fractional path needs a point cloud.
    _metric_matrix(features, np.zeros(features.n_features))  # fail fast for unsupported families
    m = int(features.n_source)

    star = np.asarray(theta_draw if theta_star is None else theta_star, dtype=np.float64).reshape(-1)
    if not theta_set.contains(star, tol=1e-9):
        base = np.asarray(theta_set.sample(rng), dtype=np.float64).reshape(-1)
        star = 0.9 * base + 0.1 / base.size
        star = star / star.sum()
    matrix = _metric_matrix(features, star)

    n_t = int(features.n_target)
    gvec = np.zeros(n_t, dtype=np.float64) if g is None else np.asarray(g, dtype=np.float64).reshape(-1)
    if gvec.size != n_t:
        raise ValueError(f"g must have length {n_t}")

    if source_mass is None:
        a = np.ones(m, dtype=np.float64)
    elif isinstance(source_mass, str):
        a = np.asarray(marginal_spec(source_mass).weights(m), dtype=np.float64)
    else:
        a = np.asarray(source_mass, dtype=np.float64).reshape(-1)
    if a.size != m or np.any(a <= 0.0):
        raise ValueError("source_mass must have length m with strictly positive entries")

    best_j, best, second_j, second = _best_two(features, star, gvec, chunk=int(chunk))
    scale = _feature_scale(features, star)
    gap = (second - best) / scale
    gap_quantiles = {f"{q:.2f}": float(np.quantile(gap, float(q))) for q in (0.0, 0.01, 0.1, 0.5, 0.9, 1.0)}

    n_split_target = int(round(float(split_fraction) * m))
    if n_split_target > 0:
        rng_split = np.random.default_rng(int(seed) + 7)
        selected = rng_split.choice(m, size=min(n_split_target, m), replace=False)
    else:
        selected = np.zeros(0, dtype=np.int64)

    partnered: dict[int, int] = {}
    n_skipped = 0
    for i in selected.tolist():
        j1 = int(best_j[i])
        if j1 < 0:
            n_skipped += 1
            continue
        if direction == "target_diff":
            v = np.asarray(features.target_points[int(second_j[i])] - features.target_points[j1],
                           dtype=np.float64)
        else:
            raise ValueError(f"unknown direction {direction!r}")
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            n_skipped += 1
            continue
        v = v / norm
        found = _first_crossing_exact(features, star, i, j1, v, gvec, matrix, float(slide_cap))
        if found is None:
            n_skipped += 1
            continue
        t_star, j2 = found
        # CN: 只改第 i 行：把源点永久移到胞边界。EN: only row i changes: move the source to the boundary.
        features.source_points[i] = features.source_points[i] + t_star * v  # type: ignore[index]
        partnered[int(i)] = int(j2)

    # CN: 组装方案：拆分源按 lambda 给 (j1, j2)；其余给其（滑动后的）最优目标。
    # EN: assemble: split sources give lambda to (j1, j2); others their (post-slide) best target.
    best_j2, best2, _second_j2, _second2 = _best_two(features, star, gvec, chunk=int(chunk))
    plan_rows: list[int] = []
    plan_cols: list[int] = []
    plan_vals: list[float] = []
    for i in range(m):
        if i in partnered:
            j1 = int(best_j2[i])
            j2 = int(partnered[i])
            plan_rows.extend([i, i])
            plan_cols.extend([j1, j2])
            plan_vals.extend([(1.0 - float(split_ratio)) * float(a[i]), float(split_ratio) * float(a[i])])
        else:
            plan_rows.append(i)
            plan_cols.append(int(best_j2[i]))
            plan_vals.append(float(a[i]))

    rows = np.array(plan_rows, dtype=np.int64)
    cols = np.array(plan_cols, dtype=np.int64)
    plan = sparse.csr_matrix((np.array(plan_vals, dtype=np.float64), (rows, cols)), shape=(m, n_t))
    col_mass = np.asarray(plan.sum(axis=0)).reshape(-1)
    kept = np.nonzero(col_mass > 0.0)[0]
    remap = -np.ones(n_t, dtype=np.int64)
    remap[kept] = np.arange(kept.size, dtype=np.int64)
    coo = plan.tocoo()
    plan_pruned = sparse.csr_matrix((coo.data, (coo.row, remap[coo.col])), shape=(m, kept.size))
    features_pruned = _subselect_targets(features, kept)

    problem = InverseOTProblem(features=features_pruned, theta_set=theta_set, plan=plan_pruned)
    f = np.asarray(best2, dtype=np.float64)
    g_kept = gvec[kept]
    scan = dual_feasibility(features_pruned, star, f, g_kept, problem.support_rows, problem.support_cols,
                            feasibility_tol=float(cert_tol), tightness_tol=float(cert_tol))
    support_cost = np.asarray(features_pruned.costs(star, problem.support_rows, problem.support_cols),
                              dtype=np.float64)
    residual = float(np.max(np.abs(f[problem.support_rows] + g_kept[problem.support_cols] - support_cost)))
    return FractionalInstance(
        problem=problem, theta_star=star, n_eff=int(problem.n_source), m_eff=int(problem.n_target),
        target_index=np.asarray(kept, dtype=np.int64), support_size=int(problem.support_size),
        support_over_n=float(problem.support_size) / float(problem.n_source),
        max_dual_violation=float(scan.max_dual_violation), max_support_residual=residual,
        n_split=int(len(partnered)), n_skipped=int(n_skipped), split_ratio=float(split_ratio),
        direction=str(direction), slide_cap=float(slide_cap), gap_quantiles=gap_quantiles,
        problem_type=PROBLEM_TYPE, mass_splitting=True,
    )


__all__ = ["FractionalInstance", "plant_fractional", "PROBLEM_TYPE", "DEFAULT_SLIDE_CAP"]
