from __future__ import annotations

import numpy as np
import pytest

from iot import (
    BilinearPSDFeatures,
    BilinearPSDTheta,
    InverseOTError,
    LinearMasterSolver,
    SdpMasterSolver,
    is_optimal_plan,
    make_bilinear_psd_grounded_instance,
    mandatory_edges,
    solve_inverse_ot,
)
from iot.grounded import GroundedInstance


def _grid(m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.repeat(np.arange(m, dtype=np.int64), n),
        np.tile(np.arange(n, dtype=np.int64), m),
    )


def test_bilinear_psd_features_are_linear_in_theta() -> None:
    rng = np.random.default_rng(0)
    features = BilinearPSDFeatures(rng.normal(size=(5, 3)), rng.normal(size=(4, 3)))
    theta = BilinearPSDTheta(3).sample(rng)
    rows, cols = _grid(features.n_source, features.n_target)
    phi = features.features(rows, cols)
    np.testing.assert_allclose(features.costs(theta, rows, cols), phi @ theta, atol=1e-12)


def test_bilinear_psd_theta_sample_is_psd_trace_one() -> None:
    rng = np.random.default_rng(1)
    theta_set = BilinearPSDTheta(3)
    theta = theta_set.sample(rng)
    assert theta_set.contains(theta)
    matrix = theta.reshape(3, 3)
    assert abs(float(np.trace(matrix)) - 1.0) < 1e-12
    assert float(np.linalg.eigvalsh(matrix).min()) > -1e-12
    assert not theta_set.contains(np.ones(9))


def test_lp_master_rejects_non_linear_theta() -> None:
    rng = np.random.default_rng(2)
    instance = make_bilinear_psd_grounded_instance(
        n_source=6, n_target=6, point_dimension=2, rng=rng
    )
    mandatory, _ = mandatory_edges(instance.problem)
    with pytest.raises(InverseOTError, match="LinearThetaSet"):
        LinearMasterSolver().solve(instance.problem, mandatory, tie_break="lex")


def test_sdp_loop_converges_and_certifies_optimality(psd_instance: GroundedInstance) -> None:
    pytest.importorskip("cvxpy")
    instance = psd_instance
    result = solve_inverse_ot(
        instance.problem,
        solver=SdpMasterSolver(),
        tie_break="lex",
        max_iterations=200,
        prune=False,
    )
    assert result.converged
    assert result.is_optimal is True
    assert instance.problem.theta_set.contains(result.theta, tol=1e-5)
    assert is_optimal_plan(instance.problem, result.theta).is_optimal
