"""
EN: **Tests for `iot.external.metrics`** — the four scores a comparison paper reports, plus the IPF rescaling.

    Why these exist: a comparison table is the one place where a silent sign or normalisation error reads as a
    result. Every check below is a hand-computed value or a property that the metric must have (a perfect match
    scores 0 / 1 / 1; CPC is bounded and symmetric; IPF preserves the pattern and lands exactly on the marginals),
    so a wrong formula fails a test instead of inflating a table.

CN: **`iot.external.metrics` 的测试** —— 比较论文报告的四个指标，以及 IPF 重标定。比较表是最容易把符号或归一化错误读成
    结果的地方。下面每个检查要么是手算值，要么是指标必须满足的性质（完全一致得 0/1/1；CPC 有界且对称；IPF 保留模式并
    恰好落到给定边际），因此公式写错会挂测试，而不是把表格抬高。
"""

from __future__ import annotations

import numpy as np
import pytest

from iot.external.metrics import (
    FlowScores,
    align_to_marginals,
    cpc,
    flow_scores,
    pearson,
    relative_frobenius,
    rmse,
)


def test_rmse_matches_the_hand_computed_value() -> None:
    """EN: two entries off by 3 and 4 → sqrt((9+16)/4) = 2.5. CN: 两个元素相差 3 与 4 → 2.5。"""
    predicted = np.array([[0.0, 0.0], [3.0, 4.0]])
    observed = np.zeros((2, 2))
    assert rmse(predicted, observed) == pytest.approx(2.5)


def test_perfect_match_scores_zero_one_one() -> None:
    """EN: the direction of each metric, in one assertion. CN: 三个指标的方向，一次断言。"""
    plan = np.array([[0.4, 0.1], [0.2, 0.3]])
    assert rmse(plan, plan) == pytest.approx(0.0)
    assert pearson(plan, plan) == pytest.approx(1.0)
    assert cpc(plan, plan) == pytest.approx(1.0)


def test_cpc_is_bounded_and_symmetric() -> None:
    """
    EN: CPC ∈ [0, 1], symmetric, 0 when the supports are disjoint.
    CN: CPC 在 [0,1] 内、对称、支撑不交时为 0。
    """
    left = np.array([[1.0, 0.0], [0.0, 0.0]])
    right = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert cpc(left, right) == pytest.approx(0.0)
    assert cpc(right, left) == pytest.approx(0.0)
    half = np.array([[0.5, 0.0], [0.0, 0.0]]) + np.array([[0.0, 0.0], [0.0, 0.5]])
    assert cpc(half, left) == pytest.approx(2.0 * 0.5 / (1.0 + 1.0))
    assert cpc(left, half) == cpc(half, left)


def test_degenerate_inputs_return_nan_rather_than_a_number() -> None:
    """
    EN: a constant flow has no correlation; an all-zero pair has no CPC. Both are `nan`, never 0.
    CN: 常数流量没有相关性；全零对没有 CPC。两者都是 `nan`，绝不是 0。
    """
    constant = np.ones((2, 2))
    assert np.isnan(pearson(constant, constant))
    assert np.isnan(cpc(np.zeros((2, 2)), np.zeros((2, 2))))
    assert np.isnan(relative_frobenius(np.zeros((2, 2)), np.zeros((2, 2))))


def test_relative_frobenius_is_scale_aware() -> None:
    """EN: doubling the estimate doubles the error. CN: 估计翻倍，误差也翻倍。"""
    truth = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert relative_frobenius(truth, truth) == pytest.approx(0.0)
    assert relative_frobenius(2.0 * truth, truth) == pytest.approx(1.0)


def test_metrics_reject_shape_and_sign_errors() -> None:
    """EN: mismatched shapes and negative entries are errors, not clamped. CN: 形状不符与负值直接报错，不做截断。"""
    with pytest.raises(ValueError):
        rmse(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        cpc(np.array([[-1.0, 0.0], [0.0, 0.0]]), np.ones((2, 2)))
    with pytest.raises(ValueError):
        pearson(np.zeros((2, 2)), np.zeros((3, 3)))


def test_align_to_marginals_lands_on_the_marginals_and_keeps_the_cross_ratios() -> None:
    """
    EN: IPF's contract: exact row/column sums, and the cross-ratios unchanged — it is a rank-one rescaling
        `diag(r) @ plan @ diag(c)`, so per-row ratios are *not* invariant (the column factors differ).

        The first version of this test asserted per-row invariance and failed (2.0 became 2.30); the test was
        wrong, not the code, which is why the invariant stated here is the one IPF actually has.
    CN: IPF 的契约：行/列和精确，交叉比不变 —— 它是秩一缩放 `diag(r) @ plan @ diag(c)`，故**每行内部**的比例不
        不变（列因子逐列不同）。本测试第一版断言了行内比例不变并失败（2.0 变 2.30）：错的是测试而不是代码，因此这里
        写的 IPF 真正具有的不变量。
    """
    plan = np.array([[1.0, 2.0], [3.0, 4.0]])
    rows = np.array([2.0, 1.0])
    cols = np.array([1.0, 2.0])
    aligned = align_to_marginals(plan, rows, cols)
    np.testing.assert_allclose(aligned.sum(axis=1), rows, atol=1e-10)
    np.testing.assert_allclose(aligned.sum(axis=0), cols, atol=1e-10)
    # EN: cross-ratio of any 2x2 submatrix is preserved. CN: 任一 2x2 子阵的交叉比不变。
    expected = (plan[0, 0] * plan[1, 1]) / (plan[0, 1] * plan[1, 0])
    got = (aligned[0, 0] * aligned[1, 1]) / (aligned[0, 1] * aligned[1, 0])
    assert got == pytest.approx(expected, rel=1e-10)
    assert float(aligned.min()) > 0.0


def test_align_to_marginals_refuses_impossible_inputs() -> None:
    """
    EN: incompatible totals, or a positive mass on an empty row, are errors — a silent NaN would poison the table.
    CN: 总量不相容，或正质量落在空行上，都是错误 —— 静默 NaN 会污染表格。
    """
    plan = np.ones((2, 2))
    with pytest.raises(ValueError):
        align_to_marginals(plan, np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    empty_row = np.array([[1.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        align_to_marginals(empty_row, np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        align_to_marginals(-plan, np.array([1.0, 1.0]), np.array([1.0, 1.0]))


def test_flow_scores_bundles_the_three_table_columns() -> None:
    """EN: one call, three fields, JSON-ready. CN: 一次调用三个字段，可直接写 JSON。"""
    plan = np.array([[0.4, 0.1], [0.2, 0.3]])
    scores = flow_scores(plan, plan)
    assert isinstance(scores, FlowScores)
    table = scores.to_dict()
    assert table["rmse"] == pytest.approx(0.0)
    assert table["correlation"] == pytest.approx(1.0)
    assert table["cpc"] == pytest.approx(1.0)
