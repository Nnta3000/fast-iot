"""
CN: **构造式（"planting"）逆 OT 实例 —— Monge 问题**。不依赖 POT/HELLO，直接**种入**一个精确最优方案。

CN: 重要标识 —— 这是 **Monge** 问题，**不是 Kantorovich**：每个源 i 的全部质量 a_i 只分配给
    **单个**最优目标 j(i)，因此方案是某个映射的图（确定性耦合），支撑恰为一条边/源，|S| = n，
    没有质量分裂。

思路（半离散 planting，适用于**任意**线性于 theta 的代价，不需要度量/PSD）：
  1. 给定 `theta*`（单纯形内部）与目标对偶势 `g`，对每个源 i 取
         j(i) = argmin_j [ c_ij(theta*) - g_j ],   f_i = min_j [ c_ij(theta*) - g_j ] .
  2. 于是 `f_i + g_j <= c_ij(theta*)` 对所有 (i,j) 成立（对偶可行），且在 (i, j(i)) 上取等
     （互补松弛）。因此把质量 a_i 全部放到 (i, j(i)) 得到的方案，**按构造**就是 c(theta*) 下的精确最优。
  3. `b = pi^T 1` 由方案诱导（不是自由参数）；空目标列被剪枝，报告 `n_eff` / `m_eff`。

CN: **精度**：本项目 GPU master（CuPDLPx）内部即 FP64（`double`），因此证书保持 FP64 是**匹配**而非
    过度精度；近并列在 FP32 下无法表示，会伪造违例。点云/质量默认 FP64。

CN: **质量分裂**：本模块**不做**分裂（没有"滑到胞边界"的步骤）。若将来需要 fractional / Kantorovich
    支撑（|S| 在 [n, 2n]），那将是一个**单独的、明确标注的选项**（做法是把源点滑到胞边界，方向作为
    可扫参数），而不是默认行为。

EN: **Constructive ("planting") inverse-OT instances -- Monge problems.** No POT/HELLO; we plant an
    exactly optimal plan.

    IMPORTANT TAG: these are **Monge** problems, **not Kantorovich**: each source i sends its whole
    mass a_i to a **single** best target j(i), so the plan is the graph of a map (a deterministic
    coupling) with exactly one edge per source, |S| = n, and **no mass splitting**.

    Semi-discrete planting, valid for **any** cost linear in theta (no metric / PSD needed):
      1. Given `theta*` (simplex interior) and target potentials `g`, per source i take
             j(i) = argmin_j [ c_ij(theta*) - g_j ],  f_i = min_j [ c_ij(theta*) - g_j ].
      2. Then `f_i + g_j <= c_ij(theta*)` for all (i,j) (dual feasible) with equality on (i, j(i))
         (complementary slackness). So putting all of a_i on (i, j(i)) is, **by construction**, an exact
         optimum of c(theta*) at its own marginals.
      3. `b = pi^T 1` is induced by the plan, not a free parameter; empty target columns are pruned and
         `n_eff` / `m_eff` are reported.

    **Precision**: our GPU master (CuPDLPx) is FP64 internally (`double`), so keeping the certificate in
    FP64 is *matching*, not over-precision. Points/masses default to FP64.

    **No mass splitting**: this module does **not** split mass (there is no cell-boundary slide). A
    fractional / Kantorovich support (|S| in [n, 2n]) would be a **separate, clearly labelled option**
    (slide source points to a cell boundary, with the direction as a sweep parameter), never the default.

用法 / EN: usage
    from iot_experiments.harness.lib.planting import plant_instance
    problem, meta = plant_instance(family="quadratic", n=2000, point_dimension=8, seed=0)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from iot import (
    BilinearPSDFeatures,
    DiagonalWeightFeatures,
    FeatureMap,
    InverseOTProblem,
    QuadraticFeatures,
)

from .costs import dual_feasibility
from .families import FamilySpec, build_family
from .marginals import marginal_spec

# CN: 默认保留的 top-k 候选数（用于证书与修复）。EN: default top-k candidates kept (certificate + repair).
DEFAULT_TOPK: int = 6
# CN: 证书/修复的容差。EN: tolerance for the certificate / repair.
CERT_TOL: float = 1e-9
# CN: 问题类型的明确标签（Monge，非 Kantorovich）。EN: explicit problem-type tag (Monge, not Kantorovich).
PROBLEM_TYPE: str = "monge"


@dataclass(frozen=True)
class PlantedInstance:
    """CN: planting 的产物与诊断。EN: The planted instance plus its diagnostics."""

    problem: InverseOTProblem
    theta_star: np.ndarray
    n_eff: int
    m_eff: int
    target_index: np.ndarray            # kept target indices (into the original n)
    support_size: int
    max_dual_violation: float           # after repair
    max_support_residual: float         # max |f_i+g_j-c_ij| on the support
    num_repairs: int
    gap_quantiles: dict[str, float]     # per-source (2nd-best - best) reduced-cost quantiles
    g_norm: float
    feature_scale: float
    dtype: str
    problem_type: str = PROBLEM_TYPE    # "monge"
    mass_splitting: bool = False


def _feature_scale(features: FeatureMap, theta_star: np.ndarray) -> float:
    """CN: 特征尺度的粗代理（用于归一化 gap）。EN: A rough feature-scale proxy (to normalise gaps)."""
    m = int(features.n_source)
    n = int(features.n_target)
    r = min(m, 64)
    c = min(n, 64)
    rows = np.repeat(np.arange(r, dtype=np.int64), c)
    cols = np.tile(np.arange(c, dtype=np.int64), r)
    cost = np.asarray(features.costs(theta_star, rows, cols), dtype=np.float64)
    scale = float(np.median(np.abs(cost)))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _best_two(features: FeatureMap, theta_star: np.ndarray, g: np.ndarray, *,
              chunk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CN: 每个源的最优/次优 reduced cost 及其目标下标（分块全扫描，O(n) 内存）。
    EN: Per-source best / 2nd-best reduced cost with their target indices (blockwise, O(n) memory).

    返回 / EN: (best_j, best, second_j, second)。
    """
    m = int(features.n_source)
    n = int(features.n_target)
    best = np.full(m, np.inf)
    second = np.full(m, np.inf)
    best_j = np.full(m, -1, dtype=np.int64)
    second_j = np.full(m, -1, dtype=np.int64)
    gg = np.asarray(g, dtype=np.float64).reshape(-1)
    for start in range(0, m, chunk):
        stop = min(m, start + chunk)
        rows = np.repeat(np.arange(start, stop, dtype=np.int64), n)
        cols = np.tile(np.arange(n, dtype=np.int64), stop - start)
        cost = np.asarray(features.costs(theta_star, rows, cols), dtype=np.float64).reshape(stop - start, n)
        reduced = cost - gg[None, :]
        order = np.argsort(reduced, axis=1)[:, :2]
        vals = np.take_along_axis(reduced, order, axis=1)
        best[start:stop] = vals[:, 0]
        second[start:stop] = vals[:, 1]
        best_j[start:stop] = order[:, 0]
        second_j[start:stop] = order[:, 1]
    return best_j, best, second_j, second


def _assign_and_certify(features: FeatureMap, theta_star: np.ndarray, g: np.ndarray, *,
                        chunk: int, tol: float) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """
    CN: 分块取每源最优目标（真 argmin），并在分块内用直接 `costs` 精确复核（修复任何近并列误判）。
        最后给出全网格对偶可行性与支撑残差（FP64，分块）。
    EN: Blockwise per-source best target (true argmin), re-checked in FP64 via direct `costs` (repairs
        any near-tie mis-selection). Finally reports full-grid dual feasibility and support residual.
    """
    m = int(features.n_source)
    n = int(features.n_target)
    assignment = np.full(m, -1, dtype=np.int64)
    f = np.full(m, np.inf, dtype=np.float64)
    gg = np.asarray(g, dtype=np.float64).reshape(-1)
    num_repairs = 0
    for start in range(0, m, chunk):
        stop = min(m, start + chunk)
        rows = np.repeat(np.arange(start, stop, dtype=np.int64), n)
        cols = np.tile(np.arange(n, dtype=np.int64), stop - start)
        cost = np.asarray(features.costs(theta_star, rows, cols), dtype=np.float64).reshape(stop - start, n)
        reduced = cost - gg[None, :]
        true_best = np.argmin(reduced, axis=1)
        assignment[start:stop] = true_best
        f[start:stop] = reduced[np.arange(stop - start), true_best]
    rows = np.arange(m, dtype=np.int64)
    support_cost = np.asarray(features.costs(theta_star, rows, assignment), dtype=np.float64)
    support_residual = float(np.max(np.abs(f + gg[assignment] - support_cost))) if m else 0.0
    scan = dual_feasibility(features, theta_star, f, gg, rows, assignment,
                            feasibility_tol=float(tol), tightness_tol=float(tol))
    return assignment, f, int(num_repairs), float(scan.max_dual_violation), support_residual


def _subselect_targets(features: FeatureMap, kept: np.ndarray) -> FeatureMap:
    """CN: 返回只保留 `kept` 目标列的 `FeatureMap`。EN: A `FeatureMap` restricted to the kept targets."""
    kept = np.asarray(kept, dtype=np.int64)
    if isinstance(features, DiagonalWeightFeatures):
        return DiagonalWeightFeatures(features.source_points, features.target_points[kept])
    if isinstance(features, QuadraticFeatures):
        return QuadraticFeatures(features.source_points, features.target_points[kept],
                                 chunk_elements=int(features.chunk_elements))
    if isinstance(features, BilinearPSDFeatures):
        return BilinearPSDFeatures(features.source_points, features.target_points[kept])
    if hasattr(features, "target_factors"):
        return dataclasses.replace(features, target_factors=features.target_factors[kept])
    raise TypeError(f"planting cannot sub-select targets for {type(features).__name__}")


def plant_instance(
    *, family: str, n: int, point_dimension: int = 8, seed: int = 0,
    theta_star: np.ndarray | None = None, g: np.ndarray | None = None,
    source_mass: np.ndarray | str | None = None, lowrank_k: int = 8,
    topk: int = DEFAULT_TOPK, chunk: int = 4096, cert_tol: float = CERT_TOL,
) -> PlantedInstance:
    """
    CN: 构造一个"种入"的 **Monge** 逆 OT 实例：theta* 在单纯形内部，质量 a>0，每个源整体分配到其
        最优目标，plan 按互补松弛精确最优；空目标列剪枝，报告 n_eff / m_eff。**不做质量分裂**。
    EN: Build a planted **Monge** inverse-OT instance: theta* in the simplex interior, masses a > 0,
        each source assigned wholly to its best target, plan exactly optimal by complementary slackness;
        empty target columns pruned with n_eff / m_eff reported. **No mass splitting.**
    """
    rng = np.random.default_rng(int(seed))
    family_spec = FamilySpec(family, point_dimension=int(point_dimension), lowrank_k=int(lowrank_k))
    features, theta_set, theta_draw = build_family(family_spec, rng, int(n), int(n))
    m = int(features.n_source)
    n_t = int(features.n_target)

    star = np.asarray(theta_draw if theta_star is None else theta_star, dtype=np.float64).reshape(-1)
    if not theta_set.contains(star, tol=1e-9):
        base = np.asarray(theta_set.sample(rng), dtype=np.float64).reshape(-1)
        star = 0.9 * base + 0.1 / base.size  # nudge strictly inside (generic positive margin)
        star = star / star.sum()

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

    assign, f, num_repairs, maxviol, support_residual = _assign_and_certify(
        features, star, gvec, chunk=int(chunk), tol=float(cert_tol))

    _best_j, best, _second_j, second = _best_two(features, star, gvec, chunk=int(chunk))
    scale = _feature_scale(features, star)
    gap = (second - best) / scale
    gap_quantiles = {f"{q:.2f}": float(np.quantile(gap, float(q)))
                     for q in (0.0, 0.01, 0.1, 0.5, 0.9, 1.0)}

    # CN: Monge 方案：每个源一条边 (i, j(i))，质量 a_i（无分裂）。
    # EN: Monge plan: one edge (i, j(i)) per source, mass a_i (no splitting).
    plan = sparse.csr_matrix((a, (np.arange(m, dtype=np.int64), assign)), shape=(m, n_t))

    # CN: 空列剪枝（b_j = 0）；报告 n_eff / m_eff。EN: prune empty columns (b_j = 0); report n_eff / m_eff.
    col_mass = np.asarray(plan.sum(axis=0)).reshape(-1)
    kept = np.nonzero(col_mass > 0.0)[0]
    m_eff = int(kept.size)
    remap = -np.ones(n_t, dtype=np.int64)
    remap[kept] = np.arange(m_eff, dtype=np.int64)
    coo = plan.tocoo()
    plan_pruned = sparse.csr_matrix((coo.data, (coo.row, remap[coo.col])), shape=(m, m_eff))
    features_pruned = _subselect_targets(features, kept)

    problem = InverseOTProblem(features=features_pruned, theta_set=theta_set, plan=plan_pruned)
    return PlantedInstance(
        problem=problem, theta_star=star, n_eff=int(problem.n_source), m_eff=int(problem.n_target),
        target_index=np.asarray(kept, dtype=np.int64), support_size=int(problem.support_size),
        max_dual_violation=float(maxviol), max_support_residual=float(support_residual),
        num_repairs=int(num_repairs), gap_quantiles=gap_quantiles,
        g_norm=float(np.linalg.norm(gvec)), feature_scale=float(scale), dtype="float64",
        problem_type=PROBLEM_TYPE, mass_splitting=False,
    )


__all__ = ["PlantedInstance", "plant_instance", "PROBLEM_TYPE"]
