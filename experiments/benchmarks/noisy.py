"""
CN: 噪声方案的唯一构造器：从真值方案 pi* 独立同分布地抽取 M 个 (i, j) 索引对，返回经验方案
    `pi_hat = counts / M` 及其自身边际。exact 侧与梯度侧共用本文件。
EN: The single constructor for noisy plans: draw M i.i.d. (i, j) index pairs from the true plan
    pi*, and return the empirical plan `pi_hat = counts / M` together with its own marginals.
    Both the exact side and the gradient side use this file.

CN: 随机源固定为 `default_rng(10_000 + seed)`，因此相同 `(pi_star, M, seed)` 在任何机器上给出
    逐位相同的输出（同一个 numpy 版本内）。
EN: The RNG is pinned to `default_rng(10_000 + seed)`, so identical `(pi_star, M, seed)` gives
    bit-identical output (within one numpy version).

CN: 返回的边际可能含 0（例如 M = 5n 时会有空行/空列）；本函数不做任何平滑或丢弃，调用方自行
    决定如何在正边际子问题上构造有效的逆 OT 问题。
EN: The returned marginals may contain zeros (e.g. empty rows/columns at M = 5n). This function
    performs no smoothing and drops nothing; the caller decides how to build a valid inverse-OT
    problem on the positive-marginal sub-problem.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

NOISY_SEED_OFFSET: int = 10_000


class NoisyPlan(NamedTuple):
    """
    CN: 噪声方案及其边际；`NamedTuple` 同时支持属性访问与元组解包，便于跨模块调用。
    EN: A noisy plan and its marginals; a `NamedTuple` supports both attribute access and tuple
        unpacking, which keeps the cross-module call simple.
    """

    plan: np.ndarray
    source_mass: np.ndarray
    target_mass: np.ndarray


def noisy_plan(pi_star: np.ndarray, M: int, seed: int) -> NoisyPlan:
    """
    CN: 从 `pi_star.ravel()` 抽 M 个 iid 索引对，返回 `counts / M` 与其行列和。
    EN: Draw M i.i.d. index pairs from `pi_star.ravel()` and return `counts / M` with its
        row/column sums.

    CN: `pi_star` 会被归一化为概率分布（真值方案的行列和通常已是均匀质量）。
    EN: `pi_star` is normalised to a probability distribution (true plans usually already have
        uniform mass).
    """
    plan_star = np.asarray(pi_star, dtype=np.float64)
    if plan_star.ndim != 2:
        raise ValueError("pi_star must be a 2D array")
    if int(M) < 1:
        raise ValueError("M must be >= 1")
    if not np.all(np.isfinite(plan_star)) or np.any(plan_star < 0.0):
        raise ValueError("pi_star must be finite and nonnegative")
    flat = plan_star.reshape(-1)
    total = float(flat.sum())
    if total <= 0.0:
        raise ValueError("pi_star must have positive total mass")
    probability = flat / total

    rng = np.random.default_rng(NOISY_SEED_OFFSET + int(seed))
    indices = rng.choice(probability.size, size=int(M), p=probability)
    counts = np.bincount(indices, minlength=probability.size).astype(np.float64)
    pi_hat = np.ascontiguousarray((counts / float(M)).reshape(plan_star.shape), dtype=np.float64)
    return NoisyPlan(
        plan=pi_hat,
        source_mass=np.ascontiguousarray(pi_hat.sum(axis=1), dtype=np.float64),
        target_mass=np.ascontiguousarray(pi_hat.sum(axis=0), dtype=np.float64),
    )


__all__ = ["NOISY_SEED_OFFSET", "NoisyPlan", "noisy_plan"]
