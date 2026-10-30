"""
CN: **合并的回归测试** —— 精确循环与松弛循环共用 `core/scan.py` 之后，必须保持两点：
      1. 精确循环的结果与合并前**逐位相同**（见 `iot_experiments/tools/loop_baseline.py` 的基线）；
      2. 三个扫描后端（numpy / torch / hello）在**两个**循环里给出同一条结论。
    这个测试锁住第 2 点，并保证 `maximum_slack` 与逐元素定义一致。

EN: **Regression test for the merge** -- now that the exact and relaxed loops share `core/scan.py`, two
    things must hold:
      1. the exact loop's numbers are **bit-identical** to before the merge (see the baseline in
         `iot_experiments/tools/loop_baseline.py`);
      2. the three scan backends (numpy / torch / hello) agree in **both** loops.
    This test pins point 2 and checks `maximum_slack` against its elementwise definition.
"""

from __future__ import annotations

import numpy as np
import ot
import pytest
import scipy.sparse as sparse
from jaxtyping import Float, Int

from iot import DiagonalWeightFeatures, InverseOTProblem
from iot.core.scan import maximum_slack, scan_for_violations
from iot.core.theta import SimplexTheta
from iot.exact.master import MasterSolution
from iot.exact.loop import solve_inverse_ot
from iot.relaxed.loop_regret import solve_inverse_ot_regret
from iot.relaxed.master_regret import RegretMasterSolver


def _identifiable_problem(m: int = 12, n: int = 12, d: int = 3, seed: int = 0) -> InverseOTProblem:
    """
    CN: 由已知 θ* 生成代价、π̂ 为其最优方案的实例，因此精确循环可行（δ* ≥ 0）。
    EN: costs from a known theta* with pi_hat its optimal plan, so the exact loop is feasible.
    """
    rng = np.random.default_rng(seed)
    x: Float[np.ndarray, "m d"] = np.ascontiguousarray(rng.normal(size=(m, d)))
    y: Float[np.ndarray, "n d"] = np.ascontiguousarray(rng.normal(size=(n, d)))
    theta_star: Float[np.ndarray, "d"] = np.abs(rng.normal(size=d))
    theta_star = theta_star / theta_star.sum()
    features = DiagonalWeightFeatures(x, y)
    rows: Int[np.ndarray, "m n"] = np.repeat(np.arange(m), n)
    cols: Int[np.ndarray, "m n"] = np.tile(np.arange(n), m)
    cost: Float[np.ndarray, "m n"] = np.asarray(features.costs(theta_star, rows, cols),
                                                 dtype=np.float64).reshape(m, n)
    plan: Float[np.ndarray, "m n"] = np.asarray(
        ot.emd(np.full(m, 1 / m), np.full(n, 1 / n), cost, numItermax=10_000_000), dtype=np.float64)
    return InverseOTProblem(features=features, theta_set=SimplexTheta(dimension=d),
                            plan=sparse.csr_matrix(plan))


def test_maximum_slack_matches_elementwise() -> None:
    """CN: `maximum_slack` 的分块结果必须等于一次性逐元素计算。EN: blocked result equals the flat one."""
    problem = _identifiable_problem()
    m, n, d = 12, 12, 3
    rng = np.random.default_rng(1)
    theta: Float[np.ndarray, "k"] = np.abs(rng.normal(size=d))
    theta = theta / theta.sum()
    f: Float[np.ndarray, "m"] = rng.normal(size=m)
    g: Float[np.ndarray, "n"] = rng.normal(size=n)
    rows: Int[np.ndarray, "m n"] = np.repeat(np.arange(m), n)
    cols: Int[np.ndarray, "m n"] = np.tile(np.arange(n), m)
    cost: Float[np.ndarray, "m n"] = np.asarray(
        problem.features.costs(theta, rows, cols), dtype=np.float64)
    flat = float((f[rows] + g[cols] - cost).max())
    for block in (1, 7, 1_000_000):
        assert maximum_slack(problem, theta, f, g, rows, cols, block=block) == pytest.approx(flat)


def test_exact_loop_is_unchanged_by_the_shared_scan() -> None:
    """
    CN: 精确循环的 δ*、轮数、|F| 与合并前的基线一致（m=12,n=12,d=3,seed=0 → δ*=0.0449795084449…）。
    EN: the exact loop's delta*, rounds and |F| match the pre-merge baseline.
    """
    problem = _identifiable_problem()
    result = solve_inverse_ot(problem, tol=1e-9, gamma=1, max_iterations=50, prune=False)
    assert result.converged
    assert bool(result.is_optimal) is True
    assert float(result.delta) == pytest.approx(0.044979508448159775, rel=1e-12)
    assert int(len(result.iterations)) == 4


def test_relaxed_scan_backends_agree() -> None:
    """
    CN: 松弛循环在 numpy / torch / hello 下结论一致；合并前它**只能**走 numpy（GPU 扫描不可达）。
    EN: the relaxed loop agrees across numpy / torch / hello; before the merge it could only use NumPy.
    """
    problem = _identifiable_problem()
    outcomes = []
    for backend in ("numpy", "torch", "hello"):
        result = solve_inverse_ot_regret(
            problem, tol=None, gamma=1, max_iterations=50,
            master=RegretMasterSolver(backend="highs", tie_break=False,
                                      params={"time_sec_limit": 120.0}),
            eps_r_rel=1e-6, prune=False, check_pot=False, selection="none",
            scan_backend=backend)
        outcomes.append((round(float(result.lower), 12), round(float(result.upper), 12),
                         int(result.n_iterations), bool(result.converged)))
    assert len(set(outcomes)) == 1, outcomes
    assert outcomes[0][3] is True


def test_scan_dispatch_rejects_an_unknown_backend() -> None:
    """CN: 非法后端必须报错，绝不静默回退。EN: an unknown backend raises, never falls back silently."""
    problem = _identifiable_problem()
    rng = np.random.default_rng(0)
    theta: Float[np.ndarray, "k"] = np.full(3, 1.0 / 3.0)
    dual = MasterSolution(delta=0.0, theta=theta,
                          source_dual=rng.normal(size=12), target_dual=rng.normal(size=12),
                          status="optimal")
    from iot.core.edges import EdgeSet  # noqa: PLC0415

    with pytest.raises(ValueError, match="scan_backend"):
        scan_for_violations(problem, dual, excluded=EdgeSet.from_arrays(
            np.array([0]), np.array([0]), n_target=12), tol=1e-9, gamma=1,
            scan_backend="cuda-but-typo")


def test_advance_edges_is_the_algebra_both_loops_use() -> None:
    """
    CN: `advance_edges` 必须等于 `mandatory ∪ pruned ∪ V`，且 mandatory 永不被丢掉。
    EN: `advance_edges` must equal `mandatory ∪ pruned ∪ V`, and mandatory must always survive.
    """
    from iot.core.edges import EdgeSet  # noqa: PLC0415
    from iot.core.edges_advance import advance_edges  # noqa: PLC0415

    mandatory = EdgeSet.from_arrays(np.array([0, 1]), np.array([0, 1]), n_target=4)
    pruned = EdgeSet.from_arrays(np.array([2]), np.array([3]), n_target=4)
    candidates = EdgeSet.from_arrays(np.array([3]), np.array([0]), n_target=4)
    merged, size = advance_edges(mandatory, pruned, candidates)
    assert set(merged.keys.tolist()) == set(mandatory.keys.tolist()) | set(pruned.keys.tolist()) | \
        set(candidates.keys.tolist())
    assert int(size) == int(merged.size) == 4
    # CN: pruning everything else away must still leave mandatory.
    # EN: pruning every optional edge away still leaves mandatory.
    kept, _ = advance_edges(mandatory, EdgeSet.from_arrays(np.zeros(0, np.int64),
                                                           np.zeros(0, np.int64), n_target=4),
                            EdgeSet.from_arrays(np.zeros(0, np.int64), np.zeros(0, np.int64), n_target=4))
    assert set(kept.keys.tolist()) == set(mandatory.keys.tolist())


def test_slack_conventions_are_opposite() -> None:
    """
    CN: `sigma = -slack`；`keep_by_tightness` 用 `slack <= tau`，等价于 `sigma >= -tau`。
    EN: `sigma = -slack`; `keep_by_tightness` uses `slack <= tau`, equivalent to `sigma >= -tau`.
    """
    from iot.core.edges import EdgeSet  # noqa: PLC0415
    from iot.core.edges_advance import (  # noqa: PLC0415
        keep_by_tightness, slack_toward_cost, slack_toward_duals)

    problem = _identifiable_problem()
    rng = np.random.default_rng(3)
    theta: Float[np.ndarray, "k"] = np.full(3, 1.0 / 3.0)
    f: Float[np.ndarray, "m"] = rng.normal(size=12)
    g: Float[np.ndarray, "n"] = rng.normal(size=12)
    rows: Int[np.ndarray, "edges"] = np.arange(12)
    cols: Int[np.ndarray, "edges"] = np.arange(12)
    slack = slack_toward_cost(problem, theta, f, g, rows, cols)
    sigma = slack_toward_duals(problem, theta, f, g, rows, cols)
    assert np.allclose(sigma, -slack)
    tau = 0.01
    edges = EdgeSet.from_arrays(rows, cols, n_target=12)
    by_slack = set(keep_by_tightness(edges, slack, tau_prune=tau).keys.tolist())
    by_sigma = set(edges.select_mask(sigma >= -tau).keys.tolist())
    assert by_slack == by_sigma


def test_hysteresis_pruner_needs_two_consecutive_rounds() -> None:
    """
    CN: 第一轮无论多松都不剪；第二轮才剪掉连续两轮都松的那条。
    EN: nothing is pruned on round 1 however loose; round 2 drops only the twice-loose edge.
    """
    from iot.core.edges import EdgeSet  # noqa: PLC0415
    from iot.core.pruning import HysteresisPruner  # noqa: PLC0415

    edges = EdgeSet.from_arrays(np.array([0, 1, 2]), np.array([0, 1, 2]), n_target=3)
    slack_r1: Float[np.ndarray, "edges"] = np.array([1.0, 1.0, 0.0])   # 0 and 1 loose
    pruner = HysteresisPruner.fresh(tau_prune=0.5)
    kept1, pruner = pruner.step(edges, slack_r1)
    assert int(kept1.size) == 3, "round 1 must prune nothing (no history)"
    slack_r2: Float[np.ndarray, "edges"] = np.array([1.0, 0.0, 0.0])   # only 0 loose twice
    kept2, pruner = pruner.step(edges, slack_r2)
    # CN: 键是 `i * n_target + j`，所以边 (1,1)、(2,2) 的键是 4、8。
    # EN: keys are `i * n_target + j`, so edges (1,1) and (2,2) have keys 4 and 8.
    assert set(kept2.keys.tolist()) == {4, 8}, kept2.keys


def test_hysteresis_matches_by_identity_when_the_population_changes() -> None:
    """
    CN: 边集在轮间变化时按**身份**匹配：消失的键不会残留，新出现的键不会误剪。
    EN: with a changing edge set, matching is by identity: vanished keys do not linger and new keys are
        never pruned on their first appearance.
    """
    from iot.core.edges import EdgeSet  # noqa: PLC0415
    from iot.core.pruning import HysteresisPruner  # noqa: PLC0415

    pruner = HysteresisPruner.fresh(tau_prune=0.5)
    first = EdgeSet.from_arrays(np.array([0, 1, 2]), np.array([0, 1, 2]), n_target=4)
    _, pruner = pruner.step(first, np.array([1.0, 1.0, 1.0]))
    # CN: 第二轮只剩 1 和 2，并且新出现 3；1 连续两轮松 ⇒ 剪；2 第二轮不松 ⇒ 留；3 是新的 ⇒ 留。
    # EN: round 2 has only 1, 2 and a new 3; 1 was loose twice => dropped; 2 is not loose now => kept;
    #     3 is new => kept.
    second = EdgeSet.from_arrays(np.array([1, 2, 3]), np.array([1, 2, 3]), n_target=4)
    kept, _ = pruner.step(second, np.array([1.0, 0.0, 1.0]))
    # CN: n_target=4 ⇒ 边 (2,2) 的键是 10，(3,3) 是 15。
    # EN: n_target=4, so edge (2,2) is key 10 and (3,3) is 15.
    assert set(kept.keys.tolist()) == {10, 15}, kept.keys


def test_exact_loop_prune_default_is_off_for_exactness() -> None:
    """CN: 精确循环的 `prune` 默认必须是 False。EN: the exact loop's `prune` default must be False."""
    import inspect  # noqa: PLC0415

    assert inspect.signature(solve_inverse_ot).parameters["prune"].default is False
