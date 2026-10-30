"""
CN: **粗层到细层的传递** —— `Prolong`（Lemma 4）与 `Seed`（Lemma 5）。

    CN: `Prolong` 只做**对偶补全**：保留节点沿用粗层势，新节点取最小值。它给出一个对偶可行但对
        `S_ℓ` 上新节点未必紧的结果——见 Lemma 4 的备注，它只用于**排序**候选边，不构成可行解。

    CN: `Seed` 只影响起点 `F₀`。正确性只要求 `B_ℓ ⊆ F₀ ⊆ U_ℓ`（Lemma 5），排序规则只影响速度。

EN: **Coarse-to-fine transfer** -- `Prolong` (Lemma 4) and `Seed` (Lemma 5).

    `Prolong` performs **dual completion** only: retained nodes keep their coarse potentials, new nodes
    take the minimum. The result is dual feasible but not necessarily tight on the new nodes' support
    edges (see the remark after Lemma 4); it is used to **rank** candidate edges, not as a feasible point.

    `Seed` only affects the start set `F₀`. Correctness requires just `B_ℓ ⊆ F₀ ⊆ U_ℓ` (Lemma 5); the
    ranking rule affects speed only.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int

from ..core.edges import EdgeSet
from ..core.edges_advance import slack_toward_cost
from ..core.problem import InverseOTProblem
from .levels import Level


def prolong(problem: InverseOTProblem, coarse: Level, fine: Level,
            theta: Float[np.ndarray, "k"], f_coarse: Float[np.ndarray, "mc"],
            g_coarse: Float[np.ndarray, "nc"]) -> tuple[Float[np.ndarray, "m"], Float[np.ndarray, "n"]]:
    """
    CN: `f̃`：保留节点用粗层 `f`；新源点 `i` 取 `min_{j∈J_c}(c_ij − ḡ_j)`。
        `g̃`：保留节点用粗层 `g`；新目标点 `j` 取 `min_{i∈I_f}(c_ij − f̃_i)`（`I_f` 是**细层全部**源点）。
        `f` 先算、`g` 后用新的 `f`，所以 `g` 的补全是精确的（Lemma 4(b),(c)）。

        CN: `theta` 在补全中**不被使用**——两个方向的最小值都直接从代价得到。保留该参数是为了让签名与
        算法描述一致，并防止调用方误以为可以传粗层的 theta 来"近似"。
    EN: `f̃`: retained nodes keep the coarse `f`; a new source `i` takes `min_{j in J_c}(c_ij - g_coarse_j)`.
        `g̃`: retained nodes keep the coarse `g`; a new target `j` takes
        `min_{i in I_f}(c_ij - f̃_i)` over **all fine sources**. `f` is computed first, then `g` with the new
        `f`, so the `g` completion is exact (Lemma 4(b),(c)).

        `theta` is **not used** in the completion: both minima are taken directly from the costs. It is kept
        in the signature to match the algorithm description and to stop a caller from passing the coarse
        theta expecting an "approximation".
    """
    del theta  # CN: 补全只看代价，不看 theta。/ EN: the completion reads costs only.
    x = np.asarray(problem.features.source_points, dtype=np.float64)
    y = np.asarray(problem.features.target_points, dtype=np.float64)

    f_out: Float[np.ndarray, "m"] = np.empty(fine.n_source, dtype=np.float64)
    # CN: 粗层势按**粗层局部序号**给出，所以按位置写入。
    # EN: coarse potentials are indexed by the **coarse local** index, so assign by position.
    f_out[coarse.rows] = np.asarray(f_coarse, dtype=np.float64).reshape(-1)
    new_f = ~np.isin(fine.rows, coarse.rows)
    for idx in fine.rows[new_f]:
        cost = np.asarray(x[idx] * y[coarse.cols]).sum(axis=1)
        f_out[idx] = float(np.min(cost - np.asarray(g_coarse, dtype=np.float64).reshape(-1)))

    g_out: Float[np.ndarray, "n"] = np.empty(fine.n_target, dtype=np.float64)
    g_out[coarse.cols] = np.asarray(g_coarse, dtype=np.float64).reshape(-1)
    new_g = ~np.isin(fine.cols, coarse.cols)
    for idx in fine.cols[new_g]:
        cost = np.asarray(x[fine.rows] * y[idx]).sum(axis=1)
        g_out[idx] = float(np.min(cost - f_out[fine.rows]))
    return f_out, g_out


def seed(problem: InverseOTProblem, fine: Level, mandatory: EdgeSet, *,
         theta: Float[np.ndarray, "k"], source_dual: Float[np.ndarray, "m"],
         target_dual: Float[np.ndarray, "n"], kappa: int,
         carry_keys: Int[np.ndarray, "carry"] | None = None,
         n_target: int) -> tuple[EdgeSet, dict[str, int]]:
    """
    CN: `F₀ = B_ℓ ∪ A_row ∪ A_col ∪ A_carry`。
        `A_row`：每行**非支撑边**中 slack 最大的 κ 条（slack 越大越松，越可能成为违反边）。
        `A_col`：同理按列。`A_carry`：粗层最终边集（可选）。
    EN: `F₀ = B_ℓ ∪ A_row ∪ A_col ∪ A_carry`.
        `A_row`: the kappa largest-slack **non-support** edges per row (larger slack = looser = more likely
        to be violated). `A_col`: the same by column. `A_carry`: the coarse final edge set (optional).
    """
    nt = int(n_target)
    rows: Int[np.ndarray, "edges"] = np.repeat(np.arange(int(problem.n_source), dtype=np.int64), nt)
    cols: Int[np.ndarray, "edges"] = np.tile(np.arange(nt, dtype=np.int64), int(problem.n_source))
    keep = ~np.isin(rows * nt + cols, fine.support.keys)
    rows, cols = rows[keep], cols[keep]
    slack = slack_toward_cost(problem, theta, source_dual, target_dual, rows, cols)
    chosen: list[int] = []
    for group in (rows, cols):
        order = np.lexsort((slack, group))
        bounds = np.flatnonzero(np.diff(group[order])) + 1
        for chunk in np.split(order, bounds):
            chosen.extend(chunk[np.argsort(-slack[chunk])][: int(kappa)].tolist())
    picked = np.unique(np.asarray(chosen, dtype=np.int64))
    row_col = EdgeSet.from_arrays(np.ascontiguousarray(rows[picked]),
                                  np.ascontiguousarray(cols[picked]), n_target=nt)
    if carry_keys is not None and int(carry_keys.size) > 0:
        carry = EdgeSet.from_arrays((carry_keys // nt).astype(np.int64),
                                    (carry_keys % nt).astype(np.int64), n_target=nt)
    else:
        carry = EdgeSet.from_arrays(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                                    n_target=nt)
    seeded = mandatory.union(row_col).union(carry)
    return seeded, {"n_mandatory": int(mandatory.size), "n_row_col": int(row_col.size),
                    "n_carry": int(carry.size)}


__all__ = ["prolong", "seed"]
