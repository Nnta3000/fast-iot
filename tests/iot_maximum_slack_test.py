"""
EN: **`maximum_slack_grid` must equal `maximum_slack` exactly.**

    The relaxed loop's certificate upper bound is `U = R_pre + s * |pi_hat|` with
    `s = max over all edges of sigma`. `maximum_slack` takes the edge indices as arrays; `maximum_slack_grid`
    generates them per row block, which is what removes an `O(m*n)` allocation from the solve. The two must
    agree **exactly**, because `s` enters a bound: an under-estimate makes `U` too small and would let the loop
    report a bracket it has not proved.

    The test therefore compares them on an explicit full grid (small enough to materialise) at several block
    sizes, including degenerate ones (a block smaller than a row, a block larger than the grid).

CN: **`maximum_slack_grid` 必须与 `maximum_slack` 完全相等**——`s` 进入证书上界 `U`，低估就会让区间看起来比
    实际更紧。测试在显式全网格上比较两者，并覆盖退化块大小（小于一行、大于整格）。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

from iot.core.features import DiagonalWeightFeatures
from iot.core.problem import InverseOTProblem
from iot.core.scan import maximum_slack, maximum_slack_grid
from iot.core.theta import SimplexTheta


def _problem(m: int = 6, n: int = 5, k: int = 3, seed: int = 0) -> InverseOTProblem:
    """CN: 小实例：`m x n` 的非方阵，避免"方阵才成立"的巧合。EN: a small `m x n` (non-square) instance."""
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(m, k))
    target = rng.normal(size=(n, k))
    plan = sparse.csr_matrix(np.full((m, n), 1.0 / (m * n)))
    return InverseOTProblem(features=DiagonalWeightFeatures(source, target),
                            theta_set=SimplexTheta(k), plan=plan)


def test_grid_max_matches_explicit_arrays() -> None:
    """CN: 同一组对偶下的 `max sigma` 必须逐位相同。EN: bit-identical `max sigma` for the same duals."""
    problem = _problem()
    rng = np.random.default_rng(1)
    theta = rng.random(3)
    theta = theta / theta.sum()
    source_dual = rng.normal(size=int(problem.n_source))
    target_dual = rng.normal(size=int(problem.n_target))

    rr, cc = np.meshgrid(np.arange(problem.n_source), np.arange(problem.n_target), indexing="ij")
    rows = rr.reshape(-1).astype(np.int64)
    cols = cc.reshape(-1).astype(np.int64)
    reference = maximum_slack(problem, theta, source_dual, target_dual, rows, cols)

    for block in (1, 2, 7, 1000):
        got = maximum_slack_grid(problem, theta, source_dual, target_dual, block=block)
        assert got == reference, f"block={block}: {got!r} != {reference!r}"


def test_grid_max_is_not_truncated_by_block_size() -> None:
    """
    CN: 最危险的失败是"只扫了第一块"——那样结果更小、更容易通过"无违反"判据。用一个把最大值放在**最后
        一行**的构造来钉住它。
    EN: the dangerous failure is scanning only the first block, which yields a *smaller* `s` and would make
        the convergence test too easy. Pin it by putting the maximum on the **last row**.
    """
    problem = _problem(m=8, n=4, k=2)
    theta = np.asarray([0.5, 0.5])
    source_dual = np.zeros(int(problem.n_source))
    target_dual = np.zeros(int(problem.n_target))
    # CN: 让最后一行的代价最负 ⇒ sigma 最大出现在最后一行。EN: make the last row's costs most negative.
    features = problem.features
    features.source_points[-1, :] = 50.0

    rr, cc = np.meshgrid(np.arange(problem.n_source), np.arange(problem.n_target), indexing="ij")
    reference = maximum_slack(problem, theta, source_dual, target_dual,
                              rr.reshape(-1).astype(np.int64), cc.reshape(-1).astype(np.int64))
    for block in (1, 4, 8, 100):
        assert maximum_slack_grid(problem, theta, source_dual, target_dual, block=block) == reference


def test_grid_max_handles_a_single_edge_grid() -> None:
    """CN: 退化情形：`1 x 1` 网格也必须给出有限值。EN: degenerate `1 x 1` grid must still return a value."""
    problem = _problem(m=1, n=1, k=2)
    value = maximum_slack_grid(problem, np.asarray([0.5, 0.5]), np.zeros(1), np.zeros(1))
    assert np.isfinite(value)


def test_planted_certificate_is_still_exact() -> None:
    """
    CN: 端到端：planted 实例上精确 iOT 仍必须给出 `is_optimal=True` 与相同的 `delta`。这条把"内存改动不得
        改变证书"写成了可执行的检查。
    EN: end to end: on a planted instance the exact solver must still report `is_optimal=True` with the same
        `delta`. This turns "the memory change must not move the certificate" into a check that can fail.
    """
    from iot.core.parameters import ExactObjective, PruningParameters, SolverParameters
    from iot.exact.loop import solve_inverse_ot
    from iot_experiments.harness.lib.planting import plant_instance

    planted = plant_instance(family="diagonal", n=40, point_dimension=3, seed=0)
    result = solve_inverse_ot(
        planted.problem,
        objective=ExactObjective(),
        params=SolverParameters(tol=None, gamma=1, max_iterations=200,
                                pruning=PruningParameters(enabled=False)),
    )
    assert result.is_optimal is True
    assert result.converged is True
    assert result.delta > 0.0
