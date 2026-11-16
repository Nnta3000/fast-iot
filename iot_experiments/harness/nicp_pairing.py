"""
EN: **NICP pairing: independent per-row argmin vs a globally constrained transport plan.**

    NICP's correspondence step is, per target point `k`, an independent nearest-template-vertex decision:

        i*_k = argmin_i ||F_theta(y_k)_i||^2

    Define `C[k, i] = ||F_theta(y_k)_i||^2`. That is a **rectangular** pairwise cost matrix (`K` target
    points x `M` template vertices), and the argmin is the `K`-independent minimiser of `<P, C>` subject to
    the row-sum constraint `P 1 = a` **only**. Nothing constrains the column sums, so many target points may
    select the same template vertex and others may be selected by none -- the result is not a map.

    The transport alternative adds the second marginal:

        P* = argmin_{P >= 0} <P, C>   s.t.  P 1_M = a,  P^T 1_K = b

    This module decodes either rule and reports the diagnostics that distinguish them. It is the gate for
    "does global assignment help this prior at all": if the transport plan does not reduce matching error on
    held-out shapes, the rest of the inverse-OT programme rests on a false premise.

    **The cost matrix is an input, never built here.** Both arms must see the same `C` from the same network
    and the same sampling, otherwise the comparison measures features rather than the pairing rule.

    **Exactness.** Decoding uses POT's network simplex (`ot.emd`), not Sinkhorn. The question is what the
    *exact* transport plan does, so an entropic approximation would confound the answer with a temperature.

    **Not implemented here, deliberately.** FAUST scans are non-watertight and only ~80 % of vertices are
    reliably aligned, so a *partial* (dustbin) formulation is the principled variant. It is left out until
    there is a cost matrix to calibrate the dustbin cost against: a decoder that silently drops mass is
    worse than one that does not, and an uncalibrated one cannot be validated. Balanced transport and argmin
    are the two arms the first experiment needs.

CN: **NICP 配对：逐行独立 argmin 与全局约束运输方案的对比。** NICP 对每个目标点独立选最近的模板顶点，只满足
    行和、不约束列和，因此多个点可能选中同一模板顶点。运输方案加上第二个边缘约束。本模块只**解码**并给出
    诊断量；代价矩阵由调用方提供，保证两臂特征一致。解码用精确网络单纯形而非 Sinkhorn，以免把熵温度混入
    比较。**部分运输（垃圾桶）故意未实现**：在拿到真实代价矩阵、能标定丢弃代价之前，写出一个无法验证的
    解码器不如不写。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import ot
from jaxtyping import Float, Int

__all__ = [
    "DecodedPairing",
    "PairingDiagnostics",
    "cost_from_offsets",
    "decode_argmin",
    "decode_transport",
    "diagnose",
]


@dataclasses.dataclass(frozen=True)
class DecodedPairing:
    """
    EN: one decoded pairing rule: the chosen template index per target point, plus how it was obtained.
    CN: 一种解码结果：每个目标点选中的模板索引，以及它是怎么得到的。
    """

    assignment: Int[np.ndarray, "k"]
    rule: str
    transport_cost: float


@dataclasses.dataclass(frozen=True)
class PairingDiagnostics:
    """
    EN: what distinguishes a genuine map from a pile of independent decisions.

        On this problem the target is far larger than the template (`K` scan points, `M = 690` template
        vertices), so **repeats are unavoidable**: `K` points cannot occupy `M < K` vertices without reuse.
        A raw "collision count" therefore says nothing. The two measures that do are normalised by what a
        balanced plan would achieve:

        `load_imbalance` -- `max_load / mean_load` with `mean_load = K / M`. A balanced plan is **1.0**;
            an argmin that sends everything to one vertex is `M`. This is the "is it a map" test.

        `excess_repeat_fraction` -- the share of target points whose repeat assignment was **not** forced by
            `K > M`. Zero for any completely balanced plan; large when one vertex hogs the mass.

        `template_coverage` -- fraction of template vertices used at least once. An argmin can leave
            template vertices unused; a balanced plan uses all of them.

        `repeat_rate` -- `(K - unique) / K`, kept only for transparency: it counts ALL repeats, including
            the unavoidable ones, so it is **not** a defect measure on this problem.

    CN: 本问题中目标点远多于模板顶点（K 个扫描点、M=690 个模板顶点），**重复不可避免**，所以原始"碰撞数"
        没有意义。真正有意义的是按平衡方案归一化的两个量：`load_imbalance`（最大负载/平均负载，平衡为 1，
        全部挤在一个顶点为 M）与 `excess_repeat_fraction`（超出 K>M 强制下限的重复比例，平衡方案为 0）。
    """

    rule: str
    template_coverage: float
    max_load: int
    mean_load: float
    load_imbalance: float
    excess_repeat_fraction: float
    repeat_rate: float
    unique_templates: int
    transport_cost: float


def cost_from_offsets(offsets: Float[np.ndarray, "k m 3"]) -> Float[np.ndarray, "k m"]:
    """
    EN: NICP's cost matrix from the predicted offsets: `C[k, i] = ||offset_k_i||^2`.

        NSR's field returns, for each queried target point, `M` ordered offsets toward the template vertices.
        Squaring their norms is exactly the quantity its own `argmin` compares, so this is the cost the
        network already implies rather than a new one.

    CN: 由预测偏移得到 NICP 的代价矩阵 `C[k,i] = ||offset_ki||^2`，即网络本身隐含使用的量。
    """
    arr = np.asarray(offsets, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"offsets must have shape (k, m, 3), got {arr.shape}")
    return np.ascontiguousarray((arr * arr).sum(axis=2), dtype=np.float64)


def _validate_cost(cost: Float[np.ndarray, "k m"]) -> Float[np.ndarray, "k m"]:
    """EN: reject a cost matrix that cannot be transported. CN: 拒绝无法运输的代价矩阵。"""
    arr = np.asarray(cost, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"cost must be 2-D (k target points x m template vertices), got {arr.shape}")
    if arr.size == 0:
        raise ValueError("cost matrix is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("cost matrix contains non-finite entries")
    return np.ascontiguousarray(arr)


def decode_argmin(cost: Float[np.ndarray, "k m"]) -> DecodedPairing:
    """
    EN: the baseline -- every target point independently selects its cheapest template vertex.

        Ties break to the lowest index (`np.argmin`), which is deterministic and matches a per-point loop in
        the original pipeline.

    CN: 基线 —— 每个目标点独立选最便宜的模板顶点；平局取最小索引。
    """
    arr = _validate_cost(cost)
    assignment = np.asarray(np.argmin(arr, axis=1), dtype=np.int64)
    # EN: **same units as `decode_transport`.** Both report `<P, C>` for a plan of UNIT total mass, so the
    #     argmin's row minima must be averaged (its plan puts `1/K` on one entry per row), not summed. An
    #     earlier version summed here and used the weighted form there, which made the argmin look 500x
    #     *cheaper* than the transport plan -- a units artefact that inverts the obvious reading.
    # CN: 与 `decode_transport` 同单位：两者都是总量为 1 的方案的 `<P,C>`，所以 argmin 的行最小要取平均而非求和。
    total = float(arr[np.arange(arr.shape[0]), assignment].mean())
    return DecodedPairing(assignment=assignment, rule="argmin", transport_cost=total)


def decode_transport(cost: Float[np.ndarray, "k m"], *,
                     source_mass: Float[np.ndarray, "k"] | None = None,
                     target_mass: Float[np.ndarray, "m"] | None = None,
                     num_iterations: int = 20_000_000) -> DecodedPairing:
    """
    EN: the alternative -- a globally constrained optimal coupling, decoded by per-row argmax.

        Marginals default to uniform on both sides. The plan is the exact LP optimum (network simplex), so the
        decoded correspondence is a property of the exact transport solution, not of a temperature.

        **Decoding choice.** Row `argmax` discards the softness of the plan; it is used because the downstream
        FAUST comparison needs a discrete correspondence. The barycentric alternative (the plan's average
        template position) is a different object and is not what is measured here.

    CN: 另一种方案 —— 全局约束的最优耦合，按行 argmax 解码。边缘默认均匀；方案是精确 LP 最优解。
    """
    arr = _validate_cost(cost)
    k, m = arr.shape
    a = np.full(k, 1.0 / k, dtype=np.float64) if source_mass is None else \
        np.asarray(source_mass, dtype=np.float64).reshape(-1)
    b = np.full(m, 1.0 / m, dtype=np.float64) if target_mass is None else \
        np.asarray(target_mass, dtype=np.float64).reshape(-1)
    if a.size != k or b.size != m:
        raise ValueError(f"marginal sizes {(a.size, b.size)} do not match cost shape {(k, m)}")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("marginals must be non-negative")
    if abs(float(a.sum()) - float(b.sum())) > 1e-9:
        raise ValueError(f"marginals must have equal total mass, got {a.sum()} vs {b.sum()}")
    plan = np.asarray(ot.emd(a, b, arr, numItermax=int(num_iterations)), dtype=np.float64)
    assignment = np.asarray(np.argmax(plan, axis=1), dtype=np.int64)
    return DecodedPairing(assignment=assignment, rule="transport",
                          transport_cost=float((plan * arr).sum()))


def diagnose(pairing: DecodedPairing, *, n_template: int) -> PairingDiagnostics:
    """
    EN: the diagnostics that separate a map from independent decisions, computed from the assignment alone so
        the numbers are comparable across rules.

        `mean_load = K / M` is what a balanced plan gives every template vertex; `excess_repeat_fraction`
        subtracts the repeats that `K > M` forces.

    CN: 从结果本身计算的诊断量，可在不同规则间直接比较。`mean_load = K/M` 是平衡方案给每个模板顶点的负载；
        `excess_repeat_fraction` 扣除了 `K > M` 强制产生的重复。
    """
    assignment = np.asarray(pairing.assignment, dtype=np.int64)
    k = int(assignment.size)
    m = int(n_template)
    if m < 1:
        raise ValueError("n_template must be positive")
    if k < 1:
        raise ValueError("assignment is empty")
    if int(assignment.min()) < 0 or int(assignment.max()) >= m:
        raise ValueError(f"assignment indices must lie in [0, {m})")
    counts = np.bincount(assignment, minlength=m)
    unique = int((counts > 0).sum())
    max_load = int(counts.max())
    mean_load = float(k) / float(m)
    repeats = k - unique
    forced_repeats = max(0, k - m)
    excess = repeats - forced_repeats
    return PairingDiagnostics(
        rule=pairing.rule,
        template_coverage=float(unique / m),
        max_load=max_load,
        mean_load=mean_load,
        load_imbalance=float(max_load / mean_load) if mean_load > 0 else float("nan"),
        excess_repeat_fraction=float(excess / k),
        repeat_rate=float(repeats / k),
        unique_templates=unique,
        transport_cost=float(pairing.transport_cost),
    )
