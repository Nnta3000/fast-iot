"""
EN: **Shared machinery for the coupling-headroom study.**

    Reuses the v2 flow-matching study rather than copying it: the data generators, `VelocityMLP`, the sliced-W2
    metric with its floor, trajectory straightness, mode diagnostics and the record utilities. Adds the pieces this
    study needs: an EMA model, exact/Sinkhorn pair samplers, the model-aligned error matrix, a teacher bank, and one
    evaluation function used by both stages.

CN: **耦合 headroom 研究的共享机制。** 复用 v2 flow-matching 研究而非复制：数据生成器、`VelocityMLP`、带下限的切片 W2、
轨迹直线度、模态诊断与记录工具。新增本研究的部件：EMA 模型、精确/Sinkhorn 配对采样、模型对齐误差矩阵、teacher bank，
以及两阶段共用的评估函数。
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Generic, TypeVar, cast

import numpy as np
import ot
import torch
from torch import nn

from experiments.studies.flow_matching.data import DIMENSIONS, dataset
from experiments.studies.flow_matching.run import (
    EVAL_POINTS,
    VelocityMLP,
    _mode_metrics,
    _rollout,
    _sliced_w2,
)
from experiments.studies.flow_matching.seeds import seed_int

T = TypeVar("T", bound=nn.Module)

__all__ = [
    "BATCH",
    "EMA",
    "MAC_REG",
    "TARGETS",
    "evaluate",
    "floor",
    "mac_error",
    "pot_pairs",
    "sample_rows",
    "sinkhorn_pairs",
    "target_dim",
    "teacher_bank",
]

BATCH = 256
MAC_REG = 0.5
TARGETS = ("eight_gaussians", "two_moons", "mixture_d8")
SEEDS = {"eight_gaussians": 10, "two_moons": 10, "mixture_d8": 5}
NFE_GRID = (1, 2, 5, 10, 50)


class EMA(Generic[T]):
    """EN: exponential moving average of a model's parameters, used by `mac_full` for its coupling. CN: 参数的指数滑动平均，供 `mac_full` 求耦合。"""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.model = cast(T, copy.deepcopy(model))
        self.model.load_state_dict(model.state_dict())
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """EN: `ema ← decay·ema + (1−decay)·model`, in place. CN: 原地 `ema ← decay·ema + (1−decay)·model`。"""
        for ema_parameter, parameter in zip(self.model.parameters(), model.parameters(), strict=True):
            ema_parameter.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)


def target_dim(name: str) -> int:
    """EN: data dimension of a toy target. CN: 玩具目标的数据维度。"""
    return int(DIMENSIONS[name])


def floor(name: str) -> float:
    """EN: sliced-W2 floor, two fresh true samples at the evaluation size. CN: 切片 W2 下限，两个新的真实样本。"""
    first = dataset(name, EVAL_POINTS, 40001)
    second = dataset(name, EVAL_POINTS, 40002)
    return _sliced_w2(first, second, projections=512, seed=1)


def sample_rows(plan: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """EN: `count` pairs from a (near-)doubly-stochastic plan, one column per chosen row. CN: 从（近）双随机计划中取 `count` 对，每行一列。"""
    rows = np.asarray(rng.choice(plan.shape[0], size=count, replace=False) if count < plan.shape[0]
                      else rng.permutation(plan.shape[0]), dtype=np.int64)
    columns = np.empty(count, dtype=np.int64)
    for position, row in enumerate(rows):
        weights = np.asarray(plan[row], dtype=np.float64)
        total = weights.sum()
        probabilities = weights / total if total > 0 else np.full(weights.size, 1.0 / weights.size)
        columns[position] = int(rng.choice(weights.size, p=probabilities))
    return rows, columns


def pot_pairs(cost: np.ndarray, rng: np.random.Generator, count: int = BATCH) -> tuple[np.ndarray, np.ndarray]:
    """EN: exact POT plan on a square cost matrix, then `count` pairs. CN: 方代价矩阵上的精确 POT 计划，再取 `count` 对。"""
    side = int(cost.shape[0])
    mass = np.full(side, 1.0 / side, dtype=np.float64)
    plan = np.asarray(ot.emd(mass, mass, cost, numItermax=2_000_000))
    return sample_rows(plan, count, rng)


def sinkhorn_pairs(cost: np.ndarray, rng: np.random.Generator, *, reg: float = MAC_REG,
                   count: int = BATCH) -> tuple[np.ndarray, np.ndarray]:
    """EN: entropic plan (`reg` in the paper's units) on a square cost matrix, then `count` pairs. CN: 方代价矩阵上的熵正则计划（`reg` 用论文单位），再取 `count` 对。"""
    side = int(cost.shape[0])
    mass = np.full(side, 1.0 / side, dtype=np.float64)
    plan = np.asarray(ot.sinkhorn(mass, mass, cost, reg=float(reg), numItermax=200, stopThr=1e-6))
    return sample_rows(plan, count, rng)


@torch.no_grad()
def mac_error(model: VelocityMLP, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
    """
    EN: the model-aligned coupling cost of a pool, averaged over `t ∈ {0,1}`.

        At `t = 0`, `x_t = x0_i` and the target is `x1_j - x0_i`, so the error is `||(v0_i + x0_i) - x1_j||²`.
        At `t = 1`, `x_t = x1_j` and the target is `x1_j - x0_i`, so the error is `||(v1_j - x1_j) + x0_i||²`.
        Both are squared distances between transformed points, so the matrix costs `O(B²)` and only two forward
        passes, not `O(B²)` passes.

    CN: 池的模型对齐耦合代价，在 `t∈{0,1}` 上取平均。`t=0` 时误差为 `||(v0_i+x0_i)-x1_j||²`；`t=1` 时为
        `||(v1_j-x1_j)+x0_i||²`。两者都是变换后点之间的平方距离，故矩阵只需 `O(B²)` 与两次前向，而不是 `O(B²)` 次前向。
    """
    zero = torch.zeros(x0.shape[0], 1, device=x0.device)
    one = torch.ones(x0.shape[0], 1, device=x0.device)
    left = model(x0, zero) + x0
    right = model(x1, one) - x1
    first = ((left[:, None, :] - x1[None, :, :]) ** 2).sum(-1)
    second = ((x0[:, None, :] + right[None, :, :]) ** 2).sum(-1)
    return (0.5 * (first + second)).cpu().numpy()


@torch.no_grad()
def teacher_bank(model: VelocityMLP, *, size: int, steps: int, seed: int, device: str
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    """EN: `size` fresh noise points and the teacher's `steps`-Euler endpoints; returns noise, endpoints, seconds. CN: `size` 个新噪声点与 teacher 的 `steps` 步 Euler 终点；返回噪声、终点、耗时。"""
    generator = torch.Generator(device=device).manual_seed(seed_int("teacher-bank", seed))
    noise = torch.randn(size, model.dim, generator=generator, device=device)
    endpoints = noise.clone()
    started = time.perf_counter()
    for step in range(int(steps)):
        t = torch.full((size, 1), step / float(steps), device=device)
        endpoints = endpoints + model(endpoints, t) / float(steps)
    return noise.cpu().numpy(), endpoints.cpu().numpy(), float(time.perf_counter() - started)


def evaluate(model: VelocityMLP, *, name: str, seed: int, device: str) -> list[dict[str, float]]:
    """EN: sliced W2, straightness and mode diagnostics at the Euler grid. CN: Euler 网格上的切片 W2、直线度与模态诊断。"""
    rows: list[dict[str, float]] = []
    fresh = dataset(name, EVAL_POINTS, 30001 + seed)
    value_floor = floor(name)
    samples50, straightness = _rollout(model, nfe=50, sampler="euler", device=device, seed=seed + 1)
    weight_l1, mean_error, covariance_error = _mode_metrics(samples50, model.dim)
    for nfe in NFE_GRID:
        samples, _ = _rollout(model, nfe=nfe, sampler="euler", device=device, seed=seed + 1)
        rows.append({
            "sw2": _sliced_w2(samples, fresh, projections=512, seed=seed + 2),
            "floor_sw2": value_floor,
            "straightness": straightness,
            "mode_weight_l1": weight_l1,
            "mode_mean_error": mean_error,
            "mode_cov_error": covariance_error,
            "nfe": float(nfe),
        })
    return rows


def checkpoint_path(root: Path, name: str, arm: str, seed: int, steps: int) -> Path:
    """EN: one checkpoint per run. CN: 每次运行一个检查点。"""
    return root / f"{name}_{arm}_s{seed}_t{steps}.pt"
