"""
CN: `iot.cupdlpx_master` 的测试。核心断言与具体求解器无关：适配器重排出的 LP 必须和 HiGHS
    master 有相同的最优值（这验证了 `loadData` 约定：等式行在前、`A x <= rhs` 在后、
    `max delta`）。只有当 CuPDLPx 扩展可 import 时才真正调用它。
EN: Tests for `iot.cupdlpx_master`. The core assertion is solver-agnostic: the LP the adapter
    reshapes must have the same optimum as the HiGHS master (this checks the `loadData`
    convention: equality rows first, then `A x <= rhs`, and `max delta`). The CuPDLPx extension
    is only actually called when it is importable.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from iot import (
    CuPdlpxMasterSolver,
    InverseOTError,
    LinearMasterSolver,
    build_cupdlpx_lp,
    make_diagonal_grounded_instance,
    mandatory_edges,
)


def _instance(n: int = 12, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(int(seed))
    return make_diagonal_grounded_instance(
        n_source=int(n), n_target=int(n), n_features=int(k), rng=rng
    )


def _pycupdlpx_available() -> bool:
    try:
        from hello_ot._native import pycupdlpx  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        import pycupdlpx  # noqa: F401

        return True
    except ImportError:
        return False


def test_cupdlpx_lp_reshapes_the_master_correctly() -> None:
    instance = _instance()
    edges, _partition = mandatory_edges(instance.problem)
    lp = build_cupdlpx_lp(instance.problem, edges)
    theta_eq_rows = int(instance.problem.theta_set.linear_constraints().a_eq.shape[0])
    in_support = np.isin(edges.keys, instance.problem.support_keys())
    # equality rows = support equalities + theta linear equalities + gauge g_0 = 0
    assert lp.n_eqs == int(in_support.sum()) + theta_eq_rows + 1
    # inequality rows = off-support edges
    assert lp.n_constraints == lp.n_eqs + int((~in_support).sum())
    assert lp.n_variables == 1 + int(instance.problem.n_features) + 2 * int(instance.problem.n_source)
    # objective is min -delta, delta >= 0 with no upper bound
    assert lp.c[int(lp.layout.delta)] == -1.0
    assert lp.lb[int(lp.layout.delta)] == 0.0
    assert np.isinf(lp.ub[int(lp.layout.delta)])


def test_cupdlpx_lp_has_the_same_optimum_as_the_highs_master() -> None:
    instance = _instance()
    edges, _partition = mandatory_edges(instance.problem)
    lp = build_cupdlpx_lp(instance.problem, edges)
    n_eq = int(lp.n_eqs)
    result = linprog(
        lp.c,
        A_ub=lp.a_csc[n_eq:],
        b_ub=lp.rhs[n_eq:],
        A_eq=lp.a_csc[:n_eq],
        b_eq=lp.rhs[:n_eq],
        bounds=list(zip(lp.lb, lp.ub)),
        method="highs",
    )
    assert result.success, result.message
    delta_adapter = -float(result.fun)
    reference = LinearMasterSolver().solve(instance.problem, edges, tie_break="none")
    assert abs(delta_adapter - reference.delta) < 1e-7
    theta = np.asarray(result.x)[
        int(lp.layout.theta_start) : int(lp.layout.theta_start) + int(instance.problem.n_features)
    ]
    assert instance.problem.theta_set.contains(theta, tol=1e-7)


def test_cupdlpx_master_rejects_lex_tie_break() -> None:
    instance = _instance()
    edges, _partition = mandatory_edges(instance.problem)
    with pytest.raises(InverseOTError):
        CuPdlpxMasterSolver().solve(instance.problem, edges, tie_break="lex")


def test_cupdlpx_master_raises_without_the_extension() -> None:
    if _pycupdlpx_available():
        pytest.skip("pycupdlpx is available; the missing-extension path does not apply")
    instance = _instance()
    edges, _partition = mandatory_edges(instance.problem)
    with pytest.raises(InverseOTError, match="CuPDLPx"):
        CuPdlpxMasterSolver().solve(instance.problem, edges, tie_break="none")


def test_cupdlpx_master_matches_highs_when_available() -> None:
    if not _pycupdlpx_available():
        pytest.skip("pycupdlpx extension not built; cannot call CuPDLPx here")
    instance = _instance()
    edges, _partition = mandatory_edges(instance.problem)
    solution = CuPdlpxMasterSolver().solve(instance.problem, edges, tie_break="none")
    reference = LinearMasterSolver().solve(instance.problem, edges, tie_break="none")
    assert solution.theta.size == int(instance.problem.n_features)
    assert instance.problem.theta_set.contains(solution.theta, tol=1e-6)
    # first-order tolerance: agree with the exact master to ~1e-4
    assert abs(solution.delta - reference.delta) < 1e-4
