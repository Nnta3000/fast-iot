"""
CN: `iot.hello_scan` 的等价性与回退测试：快速（HELLO torch）扫描必须与 NumPy 扫描给出相同的
    停机判定、相同的候选集合与相同的 max violation；非内积族必须回退到 CPU 扫描。
EN: Equivalence and fallback tests for `iot.hello_scan`: the fast (HELLO torch) scan must agree
    with the NumPy scan on the stop decision, the candidate set and the max violation; a
    non-inner-product family must fall back to the CPU scan.
"""

from __future__ import annotations

import numpy as np
import pytest

from iot import (
    DiagonalWeightFeatures,
    LinearMasterSolver,
    fast_scan_supported,
    make_diagonal_grounded_instance,
    mandatory_edges,
    scan_violations,
    scan_violations_fast,
    solve_inverse_ot,
)
from iot_experiments.benchmark.lowrank import LowRankFeatures


def _master(n: int = 40, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(int(seed))
    instance = make_diagonal_grounded_instance(
        n_source=int(n), n_target=int(n), n_features=int(k), rng=rng
    )
    edges, _partition = mandatory_edges(instance.problem)
    master = LinearMasterSolver().solve(instance.problem, edges, tie_break="none")
    return instance.problem, edges, master


def _pairs(scan) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in zip(scan.rows.tolist(), scan.cols.tolist())}


def test_fast_scan_supported_families() -> None:
    assert fast_scan_supported(DiagonalWeightFeatures(np.zeros((3, 2)), np.zeros((3, 2))))
    assert not fast_scan_supported(LowRankFeatures(np.zeros((3, 2)), np.zeros((3, 2))))


def test_fast_scan_matches_numpy_scan() -> None:
    problem, edges, master = _master()
    reference = scan_violations(problem, master, excluded=edges, tol=1e-9, gamma=1)
    fast = scan_violations_fast(problem, master, excluded=edges, tol=1e-9, gamma=1, device="cpu")
    assert fast.has_violations == reference.has_violations
    assert _pairs(fast) == _pairs(reference)
    assert abs(fast.max_violation - reference.max_violation) < 1e-6


def test_fast_scan_falls_back_for_non_inner_product_family() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(20, 3))
    target = rng.normal(size=(20, 3))
    features = LowRankFeatures(rng.normal(size=(20, 2)), rng.normal(size=(20, 2)))
    from iot import InverseOTProblem, SimplexTheta, dense_cost_matrix

    theta_set = SimplexTheta(2)
    theta_star = theta_set.sample(rng)
    import ot
    from scipy import sparse

    cost = dense_cost_matrix(features, theta_star)
    plan = ot.emd(np.full(20, 1 / 20), np.full(20, 1 / 20), cost, numItermax=2_000_000)
    problem = InverseOTProblem(features, theta_set, sparse.csr_matrix(plan))
    edges, _ = mandatory_edges(problem)
    master = LinearMasterSolver().solve(problem, edges, tie_break="none")
    reference = scan_violations(problem, master, excluded=edges, tol=1e-9, gamma=1)
    fast = scan_violations_fast(problem, master, excluded=edges, tol=1e-9, gamma=1, device="cpu")
    assert _pairs(fast) == _pairs(reference)
    assert fast.max_violation == reference.max_violation


def test_loop_hello_scan_matches_numpy_backend() -> None:
    rng = np.random.default_rng(7)
    instance = make_diagonal_grounded_instance(n_source=40, n_target=40, n_features=3, rng=rng)
    reference = solve_inverse_ot(
        instance.problem, tie_break="lex", scan_backend="numpy", max_iterations=200, validate=False
    )
    fast = solve_inverse_ot(
        instance.problem, tie_break="lex", scan_backend="hello", scan_device="cpu",
        max_iterations=200, validate=False,
    )
    assert reference.converged and fast.converged
    assert fast.n_iterations == reference.n_iterations
    assert abs(fast.final_max_violation - reference.final_max_violation) < 1e-9
    np.testing.assert_allclose(fast.theta, reference.theta, atol=1e-9)


def test_loop_rejects_unknown_scan_backend() -> None:
    rng = np.random.default_rng(0)
    instance = make_diagonal_grounded_instance(n_source=10, n_target=10, n_features=2, rng=rng)
    with pytest.raises(ValueError):
        solve_inverse_ot(instance.problem, scan_backend="nope", validate=False)
