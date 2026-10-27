"""
CN: `noisy.noisy_plan` 的不变量测试：确定性、计数语义、边际一致、种子改变抽样。
EN: Invariant tests for `noisy.noisy_plan`: determinism, count semantics, marginal
    consistency, and that a different seed changes the draw.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.benchmarks.noisy import noisy_plan  # noqa: E402


def _pi_star() -> np.ndarray:
    rng = np.random.default_rng(0)
    plan = rng.random((6, 7))
    return np.ascontiguousarray(plan / plan.sum())


def test_noisy_plan_is_deterministic() -> None:
    pi_star = _pi_star()
    first = noisy_plan(pi_star, 1000, 3)
    second = noisy_plan(pi_star, 1000, 3)
    np.testing.assert_array_equal(first.plan, second.plan)
    np.testing.assert_array_equal(first.source_mass, second.source_mass)
    np.testing.assert_array_equal(first.target_mass, second.target_mass)


def test_noisy_plan_is_counts_over_m_with_own_marginals() -> None:
    pi_star = _pi_star()
    M = 1000
    result = noisy_plan(pi_star, M, 1)
    counts = result.plan * float(M)
    assert np.allclose(counts, np.round(counts))
    assert float(counts.sum()) == float(M)
    np.testing.assert_allclose(result.source_mass, result.plan.sum(axis=1))
    np.testing.assert_allclose(result.target_mass, result.plan.sum(axis=0))


def test_noisy_plan_seed_changes_the_draw() -> None:
    pi_star = _pi_star()
    first = noisy_plan(pi_star, 1000, 1)
    second = noisy_plan(pi_star, 1000, 2)
    assert not np.array_equal(first.plan, second.plan)
