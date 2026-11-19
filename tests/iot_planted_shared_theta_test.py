"""
EN: **Tests for the planted shared-`theta` sanity check** (`studies/synthetic/planted_shared_theta`).

    The study's claim is narrow: when the observed plans are exactly optimal at one planted cost, a `theta` fitted on
    some windows reproduces a held-out plan. Two things are pinned here, because both were measured and both are easy
    to undo by accident:

      1. the planted pipeline reproduces the held-out plan (`F_train` at machine zero, the predicted plan optimal
         under the planted cost), while the *parameter* is not recovered — the admissible set is a cone;
      2. a cost supported only on the same-zone column makes **any** diagonal-free plan optimal, so such a plan
         cannot distinguish a correct cost from that degenerate one. That is why the study offers
         `--same-zone-weight 0` and `--diagonal-offset`; with the diagonal free (offset 0, no same-zone weight) the
         planted plans keep ~60 % of their mass in place.

CN: **植入共享 `theta` 健全性检查的测试**（`studies/synthetic/planted_shared_theta`）。该研究的结论很窄：当观测计划恰好最优于
    一个植入代价时，用部分窗拟合的 `theta` 能复现留出计划。这里固定两件已实测、且容易被无意破坏的事：①植入流程能复现留出
    计划（`F_train` 在机器零、预测计划在植入代价下最优），而**参数**不被恢复（可行集是锥）；②仅支撑在同 zone 列上的代价使
    **任何**无对角质量的计划都最优，故这种计划无法区分正确代价与该退化代价 —— 这正是研究提供 `--same-zone-weight 0` 与
    `--diagonal-offset` 的原因（对角自由时，植入计划约 60% 的质量留在原地）。
"""

from __future__ import annotations

import numpy as np
import ot
import pytest

from iot.relaxed.multi_plan import distance_features, fit_shared_theta, regret_of
from iot_experiments.studies.synthetic.planted_shared_theta.run import (
    diagonal_mass,
    geometry,
    marginals,
    observed_plans,
    theta_star,
)


def test_planted_pipeline_reproduces_the_held_out_plan_but_not_the_parameter() -> None:
    """
    EN: the sanity check itself, on a tiny instance: fit on two windows, predict the third.

        `F_train` must be at machine zero and the predicted plan must be optimal under the planted cost; `theta` is
        **not** required to equal the planted one, because the plans only pin the cost down to a cone.

    CN: 健全性检查本身（极小实例）：两窗拟合，第三窗预测。`F_train` 必须在机器零，预测计划在植入代价下必须最优；
        **不**要求 `theta` 等于植入值，因为计划只把代价定到一个锥上。
    """
    n, windows, seed = 8, 3, 0
    offset = 0.52
    distances = [geometry(n, seed, w, fresh=False, diagonal_offset=offset) for w in range(windows)]
    planted = theta_star(0.0)
    pairs = marginals(n, windows, seed)
    plans = observed_plans(distances, planted, pairs, outside=False)

    fit = fit_shared_theta(np.stack(plans[:2]), tuple(distances[:2]), iterations=200)
    assert fit.objective < 1e-12, f"the planted data are exactly optimal, yet F={fit.objective!r}"

    features = distance_features(distances[2])
    truth = np.tensordot(features, planted, axes=([2], [0]))
    predicted = fit.predict(*pairs[2], features=features)
    optimum = float(ot.emd2(pairs[2][0], pairs[2][1], truth, numItermax=2_000_000))
    assert float(np.sum(truth * predicted)) == pytest.approx(optimum, rel=1e-9, abs=1e-12)
    # EN: and the parameter is not recovered: this is the identifiability statement, not a failure.
    # CN: 而参数并未被恢复：这是可辨识性陈述，不是失败。
    assert float(np.abs(fit.theta - planted).sum()) > 1e-3


def test_a_same_zone_only_cost_makes_any_diagonal_free_plan_optimal() -> None:
    """
    EN: the degenerate escape the first version of the planted test left open.

        A cost supported only on the same-zone column is zero on every off-diagonal pair, so a plan that never stays
        in place costs nothing and is optimal, whatever the data. The regret of such a plan is exactly zero, which is
        why "R = 0" alone is not evidence — the same finding as the constant-column and mask-column degeneracies in
        the taxi study.

    CN: 第一版植入测试留下的退化逃逸。代价只支撑在同 zone 列上时，所有非对角对的代价为 0，故任何不在原地停留的计划
        代价为 0、必然最优，与数据无关。其 regret 恰为 0，这就是"`R=0` 本身不是证据"的原因 —— 与出租车研究中常数列、
        掩码列的退化是同一类。
    """
    n, seed = 8, 1
    distance = geometry(n, seed, 0, fresh=False, diagonal_offset=0.0)
    features = distance_features(distance)
    same_zone_theta = np.array([0.0, 0.0, 0.0, 1.0])
    pairs = marginals(n, 1, seed)[0]
    # EN: a diagonal-free plan has to be built deliberately — under a *distance* cost staying put is free, so an
    #     unconstrained plan uses the diagonal heavily. Forbidding it takes a large diagonal cost.
    # CN: 无对角质量的计划必须刻意构造 —— 在**距离**代价下留在原地是免费的，无约束计划会大量使用对角；要禁止它需要一个
    #     很大的对角代价。
    forbidden = distance + 1e3 * np.eye(n)
    plan = np.asarray(ot.emd(pairs[0], pairs[1], forbidden, numItermax=2_000_000))
    assert diagonal_mass(plan) == pytest.approx(0.0)
    assert regret_of(same_zone_theta, features, plan) == pytest.approx(0.0, abs=1e-12)


def test_diagonal_offset_removes_the_free_diagonal() -> None:
    """
    EN: with the offset the planted plans keep mass in place only because that is optimal, not because staying is
        free; measured, the diagonal mass goes from 0 (same-zone-penalised) to ~0.6 (offset, no same-zone weight).

    CN: 有偏移时，植入计划留在原地是因为那样最优，而不是因为留在原地免费；实测对角质量从 0（惩罚同 zone）变到约 0.6
        （有偏移、无同 zone 权重）。
    """
    n, seed = 8, 2
    pairs = marginals(n, 1, seed)
    planted = theta_star(0.0)
    free = observed_plans([geometry(n, seed, 0, fresh=False, diagonal_offset=0.0)], planted, pairs, outside=False)
    priced = observed_plans([geometry(n, seed, 0, fresh=False, diagonal_offset=0.52)], planted, pairs, outside=False)
    assert diagonal_mass(free[0]) > 0.2
    assert diagonal_mass(priced[0]) > 0.1
