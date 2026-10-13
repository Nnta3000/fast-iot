"""
CN: `iot_experiments.core` 的不变量测试：边际轴、RunSpec 校验、以及一次微小端到端运行。
EN: Invariant tests for `iot_experiments.core`: the marginal axis, RunSpec validation, and one
    tiny end-to-end run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.core import (  # noqa: E402
    FamilySpec,
    MarginalSpec,
    RunSpec,
    marginal_spec,
    run_spec,
)


def test_uniform_marginal_is_flat_and_normalised() -> None:
    weights = marginal_spec("uniform").weights(16)
    np.testing.assert_allclose(weights, np.full(16, 1.0 / 16.0))


def test_gaussian_marginal_is_peaked_and_normalised() -> None:
    weights = marginal_spec("gaussian").weights(400)
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    assert float(weights.max()) > 1.0 / 400.0
    assert float(weights[200]) > float(weights[10])


def test_parameterised_gaussian_name() -> None:
    spec = marginal_spec("gaussian:0.5:0.03")
    assert spec.kind == "gaussian"
    assert spec.center == 0.5
    assert spec.width == 0.03


def test_invalid_marginal_rejected() -> None:
    with pytest.raises(ValueError, match="unknown marginal"):
        marginal_spec("dirichlet")


def test_family_feature_counts() -> None:
    assert FamilySpec("diagonal", point_dimension=8).n_features() == 8
    assert FamilySpec("quadratic", point_dimension=4).n_features() == 10
    assert FamilySpec("lowrank", lowrank_k=8).n_features() == 8
    assert FamilySpec("bilinear-psd", point_dimension=2).n_features() == 4
    assert FamilySpec("bilinear-psd", point_dimension=2).is_sdp


def test_run_spec_validation_and_run_id() -> None:
    spec = RunSpec(family="quadratic", marginal="gmm", m=10, n=10, point_dimension=4, seed=3)
    assert "quadratic" in spec.run_id()
    assert "seed3" in spec.run_id()
    with pytest.raises(ValueError, match="m and n"):
        RunSpec(family="diagonal", marginal="uniform", m=1, n=10)
    with pytest.raises(ValueError, match="solver"):
        RunSpec(family="diagonal", marginal="uniform", m=10, n=10, solver="bogus")


def test_tiny_end_to_end_run_records_verdict() -> None:
    spec = RunSpec(
        family="diagonal", marginal="gmm", m=40, n=40, point_dimension=3, seed=0, validate=True
    )
    record = run_spec(spec)
    assert record.n_features == 3
    assert record.support_size > 0
    assert record.certified
    assert record.converged
    assert record.is_optimal is True
    assert record.outcome == "converged"
    assert record.solve_seconds >= 0.0
