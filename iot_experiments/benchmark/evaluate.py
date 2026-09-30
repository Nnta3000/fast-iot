"""
CN: 离线评估：从保存的 theta checkpoint 计算两个无量纲指标 G 与 R，以及 uniform / theta*
    两个参考解。此模块永不进入计时区（POT 求解代价高）。
EN: Offline evaluation: given saved theta checkpoints, compute the two dimensionless metrics
    G and R plus the uniform and theta* reference solutions. This module never runs inside a
    timed section (POT solves are expensive).

定义 / Definitions
------------------
CN: `sigma_star = std(C_theta* 的全部元素)`；全部量除以 `sigma_star` 使其无量纲。
- `G = (<C_theta_hat, pi_hat> - min_P <C_theta_hat, P>) / sigma_star`，P 用 pi_hat 自身的边际。
- `R = (<C_theta*, pi(theta_hat)> - <C_theta*, pi*>) / sigma_star`，用真值边际；pi(theta_hat)
  是 theta_hat 诱导的 OT 方案，pi* 是同一边际下的真值最优方案。
EN: `sigma_star = std of all entries of C_theta*`; everything is divided by it to stay
    dimensionless.
- `G = (<C_theta_hat, pi_hat> - min_P <C_theta_hat, P>) / sigma_star`, P over pi_hat's own
  marginals.
- `R = (<C_theta*, pi(theta_hat)> - <C_theta*, pi*>) / sigma_star`, over the true marginals;
  pi(theta_hat) is the plan induced by theta_hat and pi* the true optimal plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ot

from iot import is_optimal_plan
from iot.features import dense_cost_matrix

from .checkpoint import RunCheckpoint, read_run
from .instances import DEFAULT_NUM_ITERMAX, BenchmarkInstance, build_instance


@dataclass(frozen=True)
class ThetaScore:
    """
    CN: 一个 theta 的评估结果；`G`/`R` 已除以 sigma_star，`*_raw` 保留原值。
    EN: The evaluation of one theta; `G`/`R` are divided by sigma_star and `*_raw` keeps the
        unscaled values.
    """

    G: float
    R: float
    G_raw: float
    R_raw: float
    cost_std: float
    is_optimal: bool | None


@dataclass(frozen=True)
class ReferenceScores:
    """
    CN: 两个参考解的分数：uniform（无信息）与 theta*（真值）。
    EN: Scores of the two references: uniform (no information) and theta* (ground truth).
    """

    uniform: ThetaScore
    theta_star: ThetaScore


@dataclass(frozen=True)
class CheckpointScore:
    """
    CN: checkpoint 序列中的一个时间点的分数。
    EN: The score at one timestamp in a checkpoint sequence.
    """

    iteration: int
    elapsed_seconds: float
    score: ThetaScore


def _cost(instance: BenchmarkInstance, theta: np.ndarray) -> np.ndarray:
    """CN: 在实例特征上的稠密代价矩阵。EN: Dense cost matrix on the instance features."""
    return dense_cost_matrix(instance.features, np.asarray(theta, dtype=np.float64))


def observed_plan_gap(instance: BenchmarkInstance, theta: np.ndarray) -> float:
    """
    CN: 原始（未归一）G：用 pi_hat 自身边际求 min_P <C_theta, P>。
    EN: Raw (unscaled) G: `min_P <C_theta, P>` over pi_hat's own marginals.
    """
    cost = _cost(instance, theta)
    optimal = ot.emd(
        instance.observed_source_mass, instance.observed_target_mass, cost,
        numItermax=int(DEFAULT_NUM_ITERMAX),
    )
    return float(np.sum(cost * instance.observed_plan) - np.sum(cost * optimal))


def true_cost_regret(instance: BenchmarkInstance, theta: np.ndarray) -> float:
    """
    CN: 原始（未归一）R：真值代价下，theta 诱导方案相对真值方案的多余代价。
    EN: Raw (unscaled) R: excess true cost of the plan induced by theta over the true plan.
    """
    cost_theta = _cost(instance, theta)
    induced = ot.emd(
        instance.true_source_mass, instance.true_target_mass, cost_theta,
        numItermax=int(DEFAULT_NUM_ITERMAX),
    )
    cost_true = _cost(instance, instance.theta_star)
    return float(np.sum(cost_true * induced) - np.sum(cost_true * instance.true_plan))


def score_theta(
    instance: BenchmarkInstance,
    theta: np.ndarray,
    *,
    check_optimality: bool = False,
) -> ThetaScore:
    """
    CN: 计算归一化 G 与 R；可选调用固定的 `is_optimal_plan` 检查器。
    EN: Compute normalised G and R; optionally call the fixed `is_optimal_plan` checker.
    """
    value = np.asarray(theta, dtype=np.float64).reshape(-1)
    cost_theta = _cost(instance, value)
    cost_true = _cost(instance, instance.theta_star)
    cost_std = float(np.std(cost_true))
    if not np.isfinite(cost_std) or cost_std <= 0.0:
        raise ValueError("std of C_theta* is not positive; cannot normalise")

    observed_optimal = ot.emd(
        instance.observed_source_mass, instance.observed_target_mass, cost_theta,
        numItermax=int(DEFAULT_NUM_ITERMAX),
    )
    gap_raw = float(np.sum(cost_theta * instance.observed_plan) - np.sum(cost_theta * observed_optimal))

    induced = ot.emd(
        instance.true_source_mass, instance.true_target_mass, cost_theta,
        numItermax=int(DEFAULT_NUM_ITERMAX),
    )
    regret_raw = float(
        np.sum(cost_true * induced) - np.sum(cost_true * instance.true_plan)
    )

    is_optimal: bool | None = None
    if check_optimality:
        is_optimal = bool(is_optimal_plan(instance.observed_problem(), value).is_optimal)
    return ThetaScore(
        G=float(gap_raw) / cost_std,
        R=float(regret_raw) / cost_std,
        G_raw=float(gap_raw),
        R_raw=float(regret_raw),
        cost_std=cost_std,
        is_optimal=is_optimal,
    )


def reference_scores(
    instance: BenchmarkInstance, *, check_optimality: bool = False
) -> ReferenceScores:
    """
    CN: 计算 uniform 与 theta* 两个参考行的分数。
    EN: Compute the uniform and theta* reference rows.
    """
    uniform = np.full(int(instance.n_features), 1.0 / float(instance.n_features))
    return ReferenceScores(
        uniform=score_theta(instance, uniform, check_optimality=check_optimality),
        theta_star=score_theta(instance, instance.theta_star, check_optimality=check_optimality),
    )


def evaluate_checkpoints(
    instance: BenchmarkInstance, run: RunCheckpoint, *, check_optimality: bool = False
) -> list[CheckpointScore]:
    """
    CN: 对 run 中所有 theta 快照离线评分。
    EN: Score every theta checkpoint of a run offline.
    """
    return [
        CheckpointScore(
            iteration=int(item.iteration),
            elapsed_seconds=float(item.elapsed_seconds),
            score=score_theta(
                instance, np.asarray(item.theta, dtype=np.float64),
                check_optimality=check_optimality,
            ),
        )
        for item in run.checkpoints
    ]


def instance_from_params(run: RunCheckpoint) -> BenchmarkInstance:
    """
    CN: 由 checkpoint 参数确定性地重建同一实例。
    EN: Deterministically rebuild the identical instance from the checkpoint params.
    """
    params = run.params
    return build_instance(
        n=int(params.n),
        k=int(params.n_features),
        seed=int(params.seed),
        mixing_strength=float(params.mixing_strength),
        num_itermax=int(params.num_itermax),
        noise_multiplier=params.noise_multiplier,
    )


def score_run_file(
    path: Path, *, check_optimality: bool = False
) -> tuple[RunCheckpoint, BenchmarkInstance, list[CheckpointScore]]:
    """
    CN: 读取一个 run checkpoint 文件，重建实例并评分（离线，不进入计时）。
    EN: Read one run checkpoint file, rebuild the instance, and score it (offline, untimed).
    """
    run = read_run(Path(path))
    instance = instance_from_params(run)
    return run, instance, evaluate_checkpoints(instance, run, check_optimality=check_optimality)


__all__ = [
    "CheckpointScore",
    "ReferenceScores",
    "ThetaScore",
    "evaluate_checkpoints",
    "instance_from_params",
    "observed_plan_gap",
    "reference_scores",
    "score_run_file",
    "score_theta",
    "true_cost_regret",
]
