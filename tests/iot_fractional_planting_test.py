"""
CN: **分数 planting** 的测试：构造精确最优、|S| ∈ [n, 2n]、lowrank 被拒、slide_cap 生效、skipped 被报告。
EN: Tests for **fractional planting**: exact optimality, |S| in [n, 2n], low-rank rejected, slide cap
    enforced, skipped sources reported.
"""

from __future__ import annotations

import numpy as np
import pytest

from iot import dense_cost_matrix
from iot_experiments.harness.lib.fractional_planting import PROBLEM_TYPE, plant_fractional


def _pot_gap(planted) -> float:
    import ot

    problem = planted.problem
    cost = dense_cost_matrix(problem.features, planted.theta_star)
    a = np.ones(problem.n_source, dtype=np.float64)
    b = np.asarray(problem.plan.sum(axis=0)).reshape(-1)
    exact = ot.emd(a, b, cost, numItermax=4_000_000)
    plan_cost = float((cost * np.asarray(problem.plan.toarray())).sum())
    opt_cost = float((cost * np.asarray(exact)).sum())
    return plan_cost - opt_cost


@pytest.mark.parametrize("family,point_dimension", [("diagonal", 4), ("quadratic", 4)])
def test_fractional_plan_is_exactly_optimal(family: str, point_dimension: int) -> None:
    # CN: 方案必须对 c(theta*) 精确最优（POT 比较），且对偶可行、支撑上取等。
    # EN: the plan must be exactly optimal for c(theta*) (POT check), dual feasible and tight on support.
    planted = plant_fractional(family=family, n=300, point_dimension=point_dimension, seed=0,
                               split_fraction=1.0)
    assert planted.problem_type == PROBLEM_TYPE
    assert planted.mass_splitting is True
    assert abs(_pot_gap(planted)) < 1e-9
    assert planted.max_dual_violation <= 1e-9
    assert planted.max_support_residual <= 1e-9


def test_fractional_support_scales_with_split_fraction() -> None:
    # CN: split_fraction=0 => |S|=n（Monge）；=1 => |S| = 2n 减去 skipped。
    # EN: split_fraction=0 gives |S|=n (Monge); =1 gives |S| = 2n minus the skipped sources.
    monge = plant_fractional(family="quadratic", n=300, point_dimension=4, seed=0, split_fraction=0.0)
    assert monge.support_size == monge.n_eff
    assert monge.n_split == 0
    full = plant_fractional(family="quadratic", n=300, point_dimension=4, seed=0, split_fraction=1.0)
    # CN: 两个拆分源可能共享同一个搭档目标 → |S| <= n_eff + n_split（用 (row,col) 去重后）。
    # EN: two split sources can share a partner target, so |S| <= n_eff + n_split (deduplicated by (row,col)).
    assert full.support_size <= full.n_eff + full.n_split
    assert 1.0 < full.support_over_n <= 2.0


def test_fractional_slide_cap_is_enforced() -> None:
    # CN: 更小的滑动上限 => 更少的源被拆分、更多的源被跳过，且仍精确最优。
    # EN: a smaller slide cap splits fewer sources and skips more, while staying exactly optimal.
    tight = plant_fractional(family="quadratic", n=300, point_dimension=8, seed=0,
                             split_fraction=1.0, slide_cap=0.1)
    loose = plant_fractional(family="quadratic", n=300, point_dimension=8, seed=0,
                             split_fraction=1.0, slide_cap=4.0)
    assert tight.n_skipped > loose.n_skipped
    assert tight.n_split < loose.n_split
    assert abs(_pot_gap(tight)) < 1e-9
    assert abs(_pot_gap(loose)) < 1e-9


def test_fractional_rejects_lowrank() -> None:
    # CN: low-rank 是因子模型，没有 target_points，分数路径必须拒绝。
    # EN: low-rank is a factor model with no target_points; the fractional path must reject it.
    with pytest.raises(TypeError, match="fractional planting supports"):
        plant_fractional(family="lowrank", n=100, point_dimension=4, seed=0, split_fraction=0.5)
