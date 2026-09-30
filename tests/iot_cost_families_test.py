"""
CN: 成本族可解性测试：哪些代价族现在可解、哪些不可解。
EN: Cost-family solvability tests: which cost families are solvable now and which are not.

CN: LP 族（diagonal / L1 / quadratic / signed）应可解且证书成立；bilinear-PSD（SDP）只在小 n
    可解（n=8 lex），n=16 的 lex 步骤数值失败——这个边界在测试里固定下来。
EN: LP families (diagonal / L1 / quadratic / signed) must solve with a valid certificate; the
    bilinear-PSD (SDP) path is solvable only at tiny n (n=8 lex) and its n=16 lex step fails
    numerically — this boundary is pinned here.
"""

from __future__ import annotations

import numpy as np
import ot
import pytest
from scipy import sparse

from iot import (
    DenseFeatureTensor,
    InverseOTError,
    InverseOTProblem,
    SdpMasterSolver,
    SimplexTheta,
    WarmStartMasterSolver,
    dense_cost_matrix,
    is_optimal_plan,
    make_bilinear_psd_grounded_instance,
    solve_inverse_ot,
)


def _columns(kind: str, source: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    d = source.shape[1]
    if kind == "diagonal":
        return [(source[:, k][:, None] - target[:, k][None, :]) ** 2 for k in range(d)]
    if kind == "l1":
        return [np.abs(source[:, k][:, None] - target[:, k][None, :]) for k in range(d)]
    if kind == "quadratic":
        return [
            (source[:, a][:, None] - target[:, a][None, :])
            * (source[:, b][:, None] - target[:, b][None, :])
            for a in range(d)
            for b in range(a, d)
        ]
    if kind == "bilinear":
        return [
            source[:, a][:, None] * target[:, b][None, :] for a in range(d) for b in range(d)
        ]
    raise ValueError(kind)


def _grounded_lp_problem(kind: str, *, n: int, d: int, seed: int, signed: bool) -> InverseOTProblem:
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n, d))
    target = rng.normal(size=(n, d))
    base = np.stack(_columns(kind, source, target), axis=2)
    tensor = np.concatenate([base, -base], axis=2) if signed else base
    theta_set = SimplexTheta(tensor.shape[2])
    if signed:
        coefficients = rng.normal(size=base.shape[2])
        coefficients = coefficients / np.sum(np.abs(coefficients))
        theta_star = np.concatenate([np.maximum(coefficients, 0.0), np.maximum(-coefficients, 0.0)])
    else:
        theta_star = theta_set.sample(rng)
    features = DenseFeatureTensor(tensor)
    cost = dense_cost_matrix(features, theta_star)
    plan = ot.emd(np.full(n, 1.0 / n), np.full(n, 1.0 / n), cost, numItermax=2_000_000)
    return InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))


@pytest.mark.parametrize(
    ("kind", "signed", "k"),
    [("diagonal", False, 4), ("l1", False, 4), ("quadratic", False, 10),
     ("quadratic", True, 20), ("bilinear", True, 32)],
)
def test_lp_cost_families_are_solvable(kind: str, signed: bool, k: int) -> None:
    problem = _grounded_lp_problem(kind, n=16, d=4, seed=5, signed=signed)
    assert int(problem.n_features) == k
    result = solve_inverse_ot(
        problem,
        solver=WarmStartMasterSolver(epsilon=1e-4, random_seed=0, purge=True),
        tie_break="none",
        max_iterations=200,
        validate=True,
    )
    assert result.converged
    assert result.is_optimal is True
    assert is_optimal_plan(problem, result.theta).is_optimal


def test_bilinear_psd_is_solvable_only_at_tiny_n() -> None:
    rng = np.random.default_rng(0)
    instance = make_bilinear_psd_grounded_instance(
        n_source=6, n_target=6, point_dimension=2, rng=rng
    )
    result = solve_inverse_ot(
        instance.problem, solver=SdpMasterSolver(), tie_break="lex", max_iterations=60
    )
    assert result.converged
    assert result.is_optimal is True


def test_bilinear_psd_lex_fails_at_n16() -> None:
    # CN: 固定这个边界：n=16 的 SDP lex 数值上不可行（infeasible_inaccurate）。
    # EN: Pin the boundary: the n=16 SDP lex step is numerically infeasible.
    rng = np.random.default_rng(0)
    instance = make_bilinear_psd_grounded_instance(
        n_source=16, n_target=16, point_dimension=2, rng=rng
    )
    with pytest.raises(InverseOTError):
        solve_inverse_ot(
            instance.problem, solver=SdpMasterSolver(), tie_break="lex", max_iterations=60
        )
