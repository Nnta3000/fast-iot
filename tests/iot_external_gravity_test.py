"""
EN: **Tests for the gravity baseline.**

    What is tested, and what is deliberately *not*: the gravity model has four parameters and its row/column
    covariates are themselves functions of the data, so on finite flow data the fitted `gamma` need not equal the
    exponent that generated it. Asserting exact recovery would therefore test an assumption, not the code. The
    properties tested are the ones the baseline is used for: the distance exponent must track the generating one
    (monotone, and in the right range), the fit must be a genuine least-squares solution of the stated design, and a
    prediction must land on the marginals it is given, stay positive, and correlate with the observed pattern.

CN: **引力基线的测试。** 明确**不**测的东西：引力模型有四个参数，且它的行/列协变量本身是数据的函数，因此在有限流量数据上
    拟合出的 `gamma` 不必等于生成它的指数；断言精确恢复等于在测一个假设而不是代码。所测的是该基线被使用的性质：距离指数
    必须跟随生成指数（单调且在合理范围）、拟合必须是所述设计矩阵的最小二乘解、预测必须落在给定 marginal 上、处处为正、
    且与观测模式相关。
"""

from __future__ import annotations

import numpy as np
import pytest

from iot.external.methods.gravity import fit_gravity


def _synthetic(n: int = 12, gamma: float = 1.7, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """EN: one window of flows from a power-law gravity law. CN: 幂律引力律生成的一窗流量。"""
    rng = np.random.default_rng(seed)
    coordinates = rng.random((n, 2))
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    np.fill_diagonal(distance, max(float(np.min(distance[distance > 0.0])), 1e-3))
    source = rng.random(n) + 0.5
    target = rng.random(n) + 0.5
    flow = source[:, None] * target[None, :] * distance ** (-gamma)
    return flow[None, :, :], distance


def test_gravity_fit_tracks_the_distance_exponent() -> None:
    """
    EN: a steeper law must give a larger fitted `gamma`. The values are checked to be in the neighbourhood of the
        generating exponent, not equal to it (see the module docstring).
    CN: 更陡的规律必须给出更大的拟合 `gamma`；只要求落在生成指数的附近，而不要求相等（见模块文档）。
    """
    fitted = []
    for gamma in (0.5, 1.5, 3.0):
        flow, distance = _synthetic(gamma=gamma)
        fit = fit_gravity(flow, distance)
        assert np.isfinite([fit.intercept, fit.alpha, fit.beta, fit.gamma]).all()
        fitted.append(fit.gamma)
        assert fit.gamma > 0.0, "a decaying law must fit a positive exponent"
        assert fit.gamma == pytest.approx(gamma, rel=0.35)
    assert fitted[0] < fitted[1] < fitted[2]


def test_gravity_fit_is_the_least_squares_solution_of_its_design() -> None:
    """
    EN: reconstruct the design matrix the module says it uses and solve it independently; the coefficients must
        agree. This is the check that catches a swapped column or a sign error, which a range test would not.
    CN: 独立重建模块声称的设计矩阵并独立求解；系数必须一致。这条能抓到换列或符号错误，范围检查抓不到。
    """
    flow, distance = _synthetic(gamma=1.2)
    fit = fit_gravity(flow, distance)
    rows = flow.sum(axis=2)[0]
    cols = flow.sum(axis=1)[0]
    observed = (flow[0] > 0.0) & (distance > 0.0)
    i, j = np.nonzero(observed)
    design = np.column_stack([np.ones(i.size), np.log(rows[i]), np.log(cols[j]), np.log(distance[i, j])])
    coefficients, *_ = np.linalg.lstsq(design, np.log(flow[0][i, j]), rcond=None)
    np.testing.assert_allclose(
        [fit.intercept, fit.alpha, fit.beta, -fit.gamma], coefficients, rtol=1e-12, atol=1e-12)


def test_gravity_prediction_matches_the_requested_marginals_and_is_positive() -> None:
    """
    EN: the prediction is in absolute units, so it must land on the marginals it was given (IPF) and stay positive,
        including on pairs that had no observed flow.
    CN: 预测是绝对量纲，必须落在给定 marginal 上（IPF）且处处为正，包括没有观测流量的对。
    """
    flow, distance = _synthetic()
    fit = fit_gravity(flow, distance)
    rows = np.full(distance.shape[0], 2.0)
    cols = np.full(distance.shape[1], 2.0)
    prediction = fit.predict(rows, cols, distance)
    np.testing.assert_allclose(prediction.sum(axis=1), rows, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(prediction.sum(axis=0), cols, rtol=1e-9, atol=1e-9)
    assert float(prediction.min()) > 0.0
    # EN: and the predicted pattern is the observed one, not a uniform matrix.
    # CN: 并且预测的是观测模式，不是均匀矩阵。
    observed = flow[0][flow[0] > 0.0]
    predicted = prediction[flow[0] > 0.0]
    correlation = np.corrcoef(observed, predicted)[0, 1]
    assert correlation > 0.9


def test_gravity_rejects_an_underdetermined_fit() -> None:
    """EN: two usable entries cannot identify four parameters. CN: 两个可用元素无法辨识四个参数。"""
    flow = np.zeros((1, 4, 4))
    flow[0, 0, 1] = 2.0
    flow[0, 2, 3] = 1.0
    distance = np.ones((4, 4))
    distance[0, 1] = 2.0
    distance[2, 3] = 3.0
    with pytest.raises(ValueError):
        fit_gravity(flow, distance)
