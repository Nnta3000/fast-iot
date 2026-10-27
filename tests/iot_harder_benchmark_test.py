"""
CN: `hard_instances` / `harder_benchmark` 的不变量测试：各族真值可被认证、三条臂能运行、
    以及带符号族在 uniform 点平凡可容许（文档化的性质）。
EN: Invariant tests for `hard_instances` / `harder_benchmark`: each family's ground truth is
    certified, all three arms run, and signed families are trivially admissible at the uniform
    point (a documented property).
"""

from __future__ import annotations

import numpy as np

from iot_experiments.benchmarks.hard_instances import (
    FAMILIES,
    LP_FAMILIES,
    build_hard_instance,
    score_hard,
    uniform_theta,
)
from iot_experiments.benchmarks.harder_benchmark import (
    run_families_arm,
    run_noisy_arm,
    run_psd_arm,
)


def test_every_family_ground_truth_is_optimal() -> None:
    for family in FAMILIES:
        instance = build_hard_instance(30, family=family, seed=0, point_dimension=3, lowrank_k=5)
        score = score_hard(instance, instance.theta_star)
        assert abs(float(score["l0"])) < 1e-8, family
        if family != "bilinear-psd":
            assert score["is_optimal"] is True, family


def test_signed_families_are_trivially_admissible() -> None:
    # CN: 带符号族的特征银行对称，故 0 在凸包内 -> uniform theta 已经可容许（L0 = 0）。
    # EN: Signed banks are symmetric, so 0 is in the convex hull and the uniform theta is
    #     already admissible (L0 = 0); only max-margin/identifiability is nontrivial there.
    for family in ("quadratic-signed", "bilinear-signed"):
        instance = build_hard_instance(30, family=family, seed=0, point_dimension=3)
        l0_uniform = float(score_hard(instance, uniform_theta(instance), check_optimality=False)["l0"])
        assert abs(l0_uniform) < 1e-9, (family, l0_uniform)


def test_families_arm_runs_and_converges_on_easy_families() -> None:
    rows = run_families_arm(
        families=["diagonal", "quadratic"], sizes=[20], seeds=[0],
        point_dimension=3, lowrank_k=5,
        methods=["exact", "cutting-plane"], max_iterations=100,
        subgradient_lr=0.5, subgradient_rule="inv_sqrt", entropic_outer=5,
    )
    assert len(rows) == 4
    for row in rows:
        assert row.converged, (row.family, row.method, row.error)
        assert row.l0 is not None and row.l0 < 1e-7


def test_psd_arm_smoke() -> None:
    rows = run_psd_arm(
        sizes=[8], seeds=[0], point_dimension=2, methods=["subgradient"],
        max_iterations=200, subgradient_lr=0.5, subgradient_rule="inv_sqrt",
        entropic_outer=5,
    )
    assert len(rows) == 1
    assert rows[0].l0 is not None
    assert rows[0].is_optimal is True  # subgradient reaches the admissible set at n=8


def test_noisy_off_support_makes_exact_infeasible() -> None:
    rows = run_noisy_arm(
        n=40, k=3, seeds=[0], mix_values=[0.1],
        methods=["exact", "subgradient"], max_iterations=100,
        subgradient_lr=0.5, subgradient_rule="inv_sqrt", entropic_outer=5,
    )
    by_method = {row.method: row for row in rows}
    assert by_method["exact"].error is not None  # dense support => no admissible theta
    sub = by_method["subgradient"]
    assert sub.R is not None
    assert sub.R < float(sub.extra["uniform_R"])  # beats the uniform reference
