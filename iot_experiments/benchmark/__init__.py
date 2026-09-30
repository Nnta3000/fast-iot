"""
CN: 逆 OT 方法的共享基准：确定性实例、噪声方案构造、离线评估与梯度 runner。
EN: Shared benchmark for inverse-OT methods: deterministic instances, noisy-plan
    construction, offline evaluation, and the gradient runner.

CN: 本包只新增文件；它调用 `iot` 的公开 API（含他方维护的 `solve_inverse_ot` 与
    `dual_optimality_certificate`），但从不修改它们。
EN: This package only adds files; it calls the public `iot` API (including the
    other-agent-maintained `solve_inverse_ot` and `dual_optimality_certificate`) but never
    edits it.
"""

from __future__ import annotations

from .checkpoint import (
    InstanceParams,
    RunCheckpoint,
    ThetaCheckpoint,
    read_run,
    write_run,
)
from .evaluate import (
    ReferenceScores,
    ThetaScore,
    evaluate_checkpoints,
    observed_plan_gap,
    reference_scores,
    score_theta,
    true_cost_regret,
)
from .instances import (
    DEFAULT_K,
    DEFAULT_NUM_ITERMAX,
    NOISE_MULTIPLIERS,
    BenchmarkError,
    BenchmarkInstance,
    build_clean_instance,
    build_instance,
    build_noisy_instance,
)
from .noisy import NOISY_SEED_OFFSET, NoisyPlan, noisy_plan

__all__ = [
    "BenchmarkError",
    "BenchmarkInstance",
    "DEFAULT_K",
    "DEFAULT_NUM_ITERMAX",
    "InstanceParams",
    "NOISE_MULTIPLIERS",
    "NOISY_SEED_OFFSET",
    "NoisyPlan",
    "ReferenceScores",
    "RunCheckpoint",
    "ThetaCheckpoint",
    "ThetaScore",
    "build_clean_instance",
    "build_instance",
    "build_noisy_instance",
    "evaluate_checkpoints",
    "noisy_plan",
    "observed_plan_gap",
    "read_run",
    "reference_scores",
    "score_theta",
    "true_cost_regret",
    "write_run",
]
