"""
EN: tests for the NICP pairing decoders (`experiments.harness.nicp_pairing`).

    These validate the gate machinery on synthetic cost matrices where the answer is known, so that when the
    same functions are pointed at NSR's real cost matrices the numbers mean something. They do NOT test NSR.

    The important case is `test_argmin_collapses_where_transport_spreads`: it builds a cost matrix whose
    per-row argmin provably sends every target point to one template vertex, and asserts that the balanced
    transport plan spreads them. That is the claim the gate rests on.

    **Why the assertions are on `load_imbalance` rather than a collision count.** On the real problem
    `K >> M` (hundreds of thousands of scan points, 690 template vertices), so repeats are forced and a
    collision count is meaningless. `load_imbalance = max_load / (K/M)` is 1.0 for a balanced plan and `M`
    for a collapse, which is the distinction that matters.

CN: NICP 配对解码器的测试。用已知答案的合成代价矩阵验证门槛机制；**不**测试 NSR。断言使用
    `load_imbalance` 而不是碰撞计数，因为真实问题里 `K >> M`，重复是被迫的。
"""

from __future__ import annotations

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from experiments.harness.nicp_pairing import (
    DecodedPairing,
    cost_from_offsets,
    decode_argmin,
    decode_transport,
    diagnose,
)


def test_cost_from_offsets_is_the_squared_norm() -> None:
    """EN: `C[k,i] = ||offset||^2`, checked against an explicit loop. CN: 与显式循环对照。"""
    rng = np.random.default_rng(0)
    offsets = rng.normal(size=(5, 7, 3))
    cost = cost_from_offsets(offsets)
    expected = np.array([[float(np.dot(offsets[k, i], offsets[k, i])) for i in range(7)]
                         for k in range(5)])
    assert cost.shape == (5, 7)
    assert np.allclose(cost, expected, rtol=0, atol=1e-12)
    assert float(cost.min()) >= 0.0


def test_cost_from_offsets_rejects_malformed_offsets() -> None:
    """
    EN: malformed offsets must raise rather than be silently reshaped. Depending on the input, the shape is
        rejected either by the runtime type check (jaxtyping/beartype) or by the function's own validation;
        both are acceptable, silently reshaping is not.
    CN: 形状不对必须报错（由类型检查或函数自身校验拦下均可），绝不能静默 reshape。
    """
    for shape in ((3, 4), (3, 4, 2), (3, 4, 3, 1)):
        with pytest.raises((ValueError, BeartypeCallHintParamViolation)):
            cost_from_offsets(np.zeros(shape))


def test_argmin_recovers_a_permutation_when_the_cost_prefers_the_diagonal() -> None:
    """
    EN: with row minima on the diagonal, argmin recovers the identity exactly. The decoder must not itself
        introduce error, and a balanced plan must agree on a problem where no trade-off exists.
    CN: 行最小值在对角线上时，argmin 与运输方案都精确恢复恒等映射；解码器本身不能引入误差。
    """
    n = 6
    cost = np.ones((n, n)) - 2.0 * np.eye(n)
    for pairing in (decode_argmin(cost), decode_transport(cost)):
        assert np.array_equal(pairing.assignment, np.arange(n)), pairing.rule
        stats = diagnose(pairing, n_template=n)
        assert stats.template_coverage == 1.0
        assert stats.max_load == 1
        assert stats.load_imbalance == pytest.approx(1.0)
        assert stats.excess_repeat_fraction == pytest.approx(0.0)
        assert stats.repeat_rate == pytest.approx(0.0)


def test_argmin_collapses_where_transport_spreads() -> None:
    """
    EN: **the claim the gate rests on.**

        `k = 12` target points, `m = 4` template vertices, cost 0 in column 0 for every row and 5 elsewhere.
        The per-row argmin sends all 12 points to vertex 0: one vertex used, `max_load = 12`, imbalance `4`.
        A balanced plan must give each vertex `1/4` of the mass, so it uses all four and the heaviest load is
        `12/4 = 3`: imbalance `1`, no excess repeats.

    CN: **门槛所依赖的断言。** 第 0 列代价为 0、其余为 5，逐行 argmin 把 12 个点全送到顶点 0（负载失衡 4 倍）；
        平衡运输要求每顶点 1/4 质量，用满 4 个顶点、最大负载 3（失衡 1 倍、无多余重复）。
    """
    k, m = 12, 4
    cost = np.full((k, m), 5.0)
    cost[:, 0] = 0.0

    collapsed = diagnose(decode_argmin(cost), n_template=m)
    assert collapsed.unique_templates == 1
    assert collapsed.template_coverage == pytest.approx(1.0 / m)
    assert collapsed.max_load == k
    assert collapsed.load_imbalance == pytest.approx(float(m))
    # EN: repeats forced by K > M are (k - m) = 8; here there are 11, so 3 are excess over the minimum.
    # CN: K > M 强制的重复是 8 次；这里有 11 次，所以超出下限 3 次。
    assert collapsed.excess_repeat_fraction == pytest.approx((k - 1 - (k - m)) / k)

    spread = diagnose(decode_transport(cost), n_template=m)
    assert spread.unique_templates == m
    assert spread.template_coverage == pytest.approx(1.0)
    assert spread.max_load == k // m
    assert spread.load_imbalance == pytest.approx(1.0)
    assert spread.excess_repeat_fraction == pytest.approx(0.0)


def test_transport_is_perfectly_balanced_on_random_costs() -> None:
    """
    EN: with uniform marginals a balanced plan gives every template vertex the same load, so the heaviest load
        is `ceil(K/M)` and the imbalance is 1 whenever `K` divides by `M`. Asserted on random costs, so it is
        a property of the decoder and not of a constructed example.
    CN: 均匀边缘下平衡方案给每个模板顶点相同负载；在随机代价上断言，因此这是解码器的性质而非构造例的性质。
    """
    rng = np.random.default_rng(1)
    for k, m in ((20, 5), (30, 30), (7, 2), (100, 690)):
        cost = rng.random((k, m)) * 10.0
        stats = diagnose(decode_transport(cost), n_template=m)
        assert stats.template_coverage == pytest.approx(1.0 if k >= m else k / m)
        expected_max = int(np.ceil(k / m))
        assert stats.max_load == expected_max, f"k={k} m={m}: max_load={stats.max_load}"
        assert stats.excess_repeat_fraction == pytest.approx(0.0)


def test_both_rules_report_cost_in_the_same_units() -> None:
    """
    EN: regression for a units bug: `decode_argmin` summed the row minima while `decode_transport` reported a
        unit-mass `<P, C>`. The argmin then looked ~500x cheaper than a *constrained* minimum, which is
        impossible. Both must now be `<P, C>` for a plan of unit total mass, so transport can never be
        cheaper than argmin.
    CN: 单位 bug 的回归测试：argmin 曾对行最小求和、transport 报告单位质量的 `<P,C>`，使 argmin 看起来
        比受约束的最小值还便宜 500 倍。现在两者同单位，transport 不可能更便宜。
    """
    rng = np.random.default_rng(7)
    for _trial in range(4):
        cost = rng.random((40, 8)) * 5.0
        baseline = decode_argmin(cost)
        transported = decode_transport(cost)
        assert baseline.transport_cost == pytest.approx(float(cost.min(axis=1).mean()))
        assert transported.transport_cost >= baseline.transport_cost - 1e-9


def test_transport_cost_is_never_below_the_row_constrained_optimum() -> None:
    """
    EN: the per-row argmin minimises `<P, C>` under the row constraint alone, so adding the column constraint
        cannot **lower** the objective. This pins the inequality direction, which is easy to invert and would
        silently flip every conclusion drawn from the transport cost.
    CN: 逐行 argmin 是只带行约束时的最优，加入列约束不可能**降低**目标值。该断言钉住不等号方向。
    """
    rng = np.random.default_rng(2)
    for _trial in range(5):
        cost = rng.random((15, 5)) * 3.0
        transported = decode_transport(cost)
        row_optimum = float(cost.min(axis=1).sum()) / cost.shape[0]
        assert transported.transport_cost >= row_optimum - 1e-9


def test_marginals_are_validated() -> None:
    """EN: wrong-sized, negative or unbalanced marginals must raise. CN: 尺寸错、为负或不平衡的边缘必须报错。"""
    cost = np.zeros((4, 3))
    with pytest.raises(ValueError):
        decode_transport(cost, source_mass=np.full(3, 0.25))
    with pytest.raises(ValueError):
        decode_transport(cost, target_mass=np.full(2, 0.5))
    with pytest.raises(ValueError):
        decode_transport(cost, source_mass=np.full(4, 0.3), target_mass=np.full(3, 0.5))
    with pytest.raises(ValueError):
        decode_transport(cost, source_mass=np.array([-0.5, 0.5, 0.5, 0.5]),
                         target_mass=np.full(3, 1.0 / 3.0))


def test_non_finite_cost_is_rejected() -> None:
    """EN: a NaN in the cost must raise, never propagate into a plan. CN: 代价里的 NaN 必须报错。"""
    cost = np.zeros((3, 3))
    cost[1, 1] = np.nan
    with pytest.raises(ValueError):
        decode_argmin(cost)
    with pytest.raises(ValueError):
        decode_transport(cost)


def test_diagnose_rejects_an_out_of_range_assignment() -> None:
    """EN: an assignment pointing outside the template must raise. CN: 越界的索引必须报错。"""
    with pytest.raises(ValueError):
        diagnose(DecodedPairing(assignment=np.array([0, 5]), rule="hand", transport_cost=0.0),
                 n_template=3)


def test_diagnose_reports_partial_coverage() -> None:
    """EN: a hand-built assignment using part of the template reports the exact coverage. CN: 覆盖率必须精确。"""
    pairing = DecodedPairing(assignment=np.array([0, 0, 2, 2]), rule="hand", transport_cost=0.0)
    stats = diagnose(pairing, n_template=8)
    assert stats.unique_templates == 2
    assert stats.template_coverage == pytest.approx(0.25)
    assert stats.max_load == 2
    # EN: K=4, M=8, so no repeat is forced; both repeats are excess. CN: K=4 < M=8，重复全属多余。
    assert stats.excess_repeat_fraction == pytest.approx(0.5)
    assert stats.load_imbalance == pytest.approx(2.0 / (4.0 / 8.0))
