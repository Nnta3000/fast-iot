"""
CN: iot 测试的共享小实例 fixture；参数与既有测试的原构造完全一致，替换的是纯重复。
EN: Shared small-instance fixtures for iot tests; parameters match the existing tests'
    original constructions exactly, so only straight duplicates are replaced.
"""

from __future__ import annotations

import numpy as np
import pytest

from iot.grounded import (
    GroundedInstance,
    make_bilinear_psd_grounded_instance,
    make_diagonal_grounded_instance,
)


@pytest.fixture(scope="session")
def simplex_instance() -> GroundedInstance:
    """
    CN: 小 simplex 真值实例（n=12, K=3, seed=3），与 iot_grounded_test 的构造相同。
    EN: Small simplex grounded instance (n=12, K=3, seed=3), identical to the construction
        in iot_grounded_test.
    """
    rng = np.random.default_rng(3)
    return make_diagonal_grounded_instance(n_source=12, n_target=12, n_features=3, rng=rng)


@pytest.fixture(scope="session")
def psd_instance() -> GroundedInstance:
    """
    CN: 小 PSD 真值实例（n=8, d=2, seed=3），与 iot_sdp_test 的构造相同。
    EN: Small PSD grounded instance (n=8, d=2, seed=3), identical to the construction in
        iot_sdp_test.
    """
    rng = np.random.default_rng(3)
    return make_bilinear_psd_grounded_instance(
        n_source=8, n_target=8, point_dimension=2, rng=rng
    )
