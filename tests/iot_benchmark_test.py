"""
CN: 基准模块的不变量测试：生成器与 exact 侧逐字节一致、真值证书、噪声方案语义、离线评估的
    参考行与归一化、checkpoint 往返、runner 的 theta/epsilon 不变量。
EN: Invariant tests for the benchmark module: byte-identical generator vs the exact side,
    ground-truth certificate, noisy-plan semantics, offline reference rows and normalisation,
    checkpoint roundtrip, and the runner's theta/epsilon invariants.

CN: 这些测试只断言不变量，不断言某一方法优于另一方法。
EN: These tests assert invariants only, never that one method beats another.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.benchmarks.checkpoint import read_run, write_run  # noqa: E402
from iot_experiments.benchmarks.evaluate import (  # noqa: E402
    reference_scores,
    score_theta,
)
from iot_experiments.benchmarks.instances import (  # noqa: E402
    build_clean_instance,
    build_instance,
    build_noisy_instance,
)
from iot_experiments.benchmarks.runner import (  # noqa: E402
    ExactConfig,
    GradientConfig,
    fit_theta_adamw,
    run_exact,
)


def test_generator_matches_common_build_problem_for_seeds_0_to_9() -> None:
    from iot_experiments.legacy.scratch.common import build_problem

    for seed in range(10):
        ours = build_instance(n=50, k=8, seed=seed, num_itermax=2_000_000)
        theirs, theta = build_problem(n=50, k=8, seed=seed, num_itermax=2_000_000)
        np.testing.assert_array_equal(ours.features.source_points, theirs.features.source_points)
        np.testing.assert_array_equal(ours.features.target_points, theirs.features.target_points)
        np.testing.assert_array_equal(ours.theta_star, theta)
        np.testing.assert_allclose(ours.observed_plan, theirs.plan.toarray(), atol=0.0, rtol=0.0)


def test_ground_truth_plan_certifies_and_clean_observed_equals_true() -> None:
    instance = build_clean_instance(n=40, k=8, seed=1)
    assert instance.certificate.certified
    assert not instance.is_noisy
    np.testing.assert_allclose(instance.observed_plan, instance.true_plan)
    np.testing.assert_allclose(instance.observed_source_mass, 1.0 / instance.n_source)


def test_noisy_plan_is_counts_over_m_with_empirical_marginals() -> None:
    n, seed, multiplier = 60, 3, 20
    instance = build_noisy_instance(n=n, k=8, seed=seed, noise_multiplier=multiplier)
    assert instance.is_noisy
    assert instance.noise_pairs == multiplier * n
    counts = np.rint(instance.observed_plan * float(instance.noise_pairs))
    assert np.allclose(counts, np.round(counts))
    assert counts.sum() == float(instance.noise_pairs)
    np.testing.assert_allclose(
        instance.observed_source_mass, instance.observed_plan.sum(axis=1)
    )
    np.testing.assert_allclose(
        instance.observed_target_mass, instance.observed_plan.sum(axis=0)
    )
    assert instance.certificate.certified


def test_noisy_lowest_level_records_zero_marginal_repair() -> None:
    instance = build_noisy_instance(n=200, k=8, seed=0, noise_multiplier=5)
    assert instance.zero_rows_dropped > 0
    assert instance.zero_cols_dropped > 0
    assert instance.n_source == 200 - instance.zero_rows_dropped
    assert np.all(instance.observed_source_mass > 0.0)
    assert np.all(instance.observed_target_mass > 0.0)


def test_evaluation_references_and_normalisation() -> None:
    instance = build_clean_instance(n=40, k=8, seed=2)
    references = reference_scores(instance)
    # theta* reference: R == 0 (and G == 0 on a clean plan).
    assert abs(references.theta_star.R) < 1e-9
    assert abs(references.theta_star.G) < 1e-9
    # Uniform reference: both metrics nonnegative; normalisation divides by cost std.
    assert references.uniform.G >= 0.0
    assert references.uniform.R >= 0.0
    score = score_theta(instance, np.full(instance.n_features, 1.0 / instance.n_features))
    assert np.isclose(score.G * score.cost_std, score.G_raw)
    assert np.isclose(score.R * score.cost_std, score.R_raw)
    assert score.G == pytest.approx(references.uniform.G)
    assert score.R == pytest.approx(references.uniform.R)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    instance = build_clean_instance(n=20, k=8, seed=0)
    run = fit_theta_adamw(instance, GradientConfig(num_outer=3, sinkhorn_iterations=100))
    path = tmp_path / "run.json"
    write_run(path, run)
    reloaded = read_run(path)
    assert reloaded.to_dict() == run.to_dict()


def test_gradient_runner_invariants() -> None:
    instance = build_clean_instance(n=20, k=8, seed=0)
    run = fit_theta_adamw(instance, GradientConfig(num_outer=3, sinkhorn_iterations=100))
    assert len(run.checkpoints) >= 10
    assert run.epsilon_floor_reached
    assert np.isfinite(run.final_sinkhorn_marginal_error)
    assert np.isfinite(run.peak_rss_mb) and run.peak_rss_mb > 0.0
    elapsed = [item.elapsed_seconds for item in run.checkpoints]
    assert all(later >= earlier for earlier, later in zip(elapsed, elapsed[1:]))
    theta = np.asarray(run.terminal_theta)
    assert instance.theta_set.contains(theta, tol=1e-9)
    # The uniform reference must be a valid score to compare against.
    assert np.isfinite(reference_scores(instance).uniform.R)


def test_exact_arm_runs_and_certifies_on_clean_small() -> None:
    instance = build_clean_instance(n=20, k=8, seed=0)
    run = run_exact(instance, ExactConfig())
    assert run.method == "exact-constraint-generation"
    assert len(run.checkpoints) >= 1
    score = score_theta(instance, np.asarray(run.terminal_theta))
    assert score.G <= 1e-6
    assert score.R <= 1e-6


def test_unsupported_noise_multiplier_raises() -> None:
    with pytest.raises(ValueError):
        build_noisy_instance(n=10, k=8, seed=0, noise_multiplier=7)
