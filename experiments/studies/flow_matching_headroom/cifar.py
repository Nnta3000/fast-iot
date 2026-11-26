"""
EN: **Stage 1: CIFAR-10 coupling headroom (server script), patched v2.**

    Unconditional CIFAR-10, small DiT, CFM loss, one coupling per run: `independent`, `w2_mb` (exact OT on a batch),
    `mac_full` (EMA model-error cost + entropic OT), `teacher_routed_ot`, `reflow_ref`. FID: 10 000 generated images
    against a 40 000-image training reference, with two floors (held-out 10 000 train images, the test set).

    Changes from the version at 8fde319 (each one is a bug or a cost, not a change of design):

      1. `mac_full` Sinkhorn. The old `ot.sinkhorn(reg=0.5)` on the raw cost underflowed (`exp(-C/0.5)` with C ~ 10^3),
         returned an all-zero plan, and `sample_rows` fell back to uniform columns: the arm was independent coupling.
         Now: cost centred (the plan is invariant to row/column constants), `eps = mac_reg_rel * std(centred cost)`,
         log-domain iterations, and a hard error if the plan is not finite or not normalised. `--mac-reg-rel 0.1`
         reproduces the toy setting (d = 2, reg 0.5 is 0.09 of the centred-cost std there). Plan sharpness and the
         agreement with exact EMD are logged and recorded (`mac_top1`, `mac_emd_agreement`).
      2. `mac_full` warm-up (one epoch, independent pairs) and an EMA with bias correction
         (`decay_t = min(decay, (1+n)/(10+n))`), so the coupling model is never "96 % initial weights".
         Coupling EMA 0.999 (as in Stage 0); evaluation EMA 0.9999.
      3. Cost matrices on the GPU, `||a-b||^2 = |a|^2 + |b|^2 - 2ab^T` in float64; only the 128 x 128 table goes to the
         CPU for `ot.emd`. No 3-D broadcast anywhere.
      4. Data stream on the GPU (random flip, permutation per epoch): no DataLoader, no PIL, no worker restarts.
         Generators are per run and per purpose, so arms at the same seed see the same data order, noise and times.
      5. FID: reference and floor statistics computed once (`--prepare-fid`, cached, atomic write), features from
         tensors (uint8-quantised), no PNG files. FID and straightness are measured on the evaluation EMA.
      6. Timers: per-run `train_seconds` (synchronised, FID excluded), separate `fid_seconds`; `--profile-steps K`
         prints a per-stage breakdown (data / couple / step / ema) and exits.
      7. TF32 on by default (`--no-tf32` to disable), optional `--bf16` for the training step.
      8. `teacher_routed_ot`: the cost is now the 128 x 128 matrix `||T(x0_i) - x1_j||^2` (it was a vector and ignored
         the endpoints); the bank is built in chunks, stored in float16 on the GPU, from the teacher's EMA.
      9. Records are rewritten after every run; a final checkpoint is saved; `--resume` re-evaluates from it.

CN: **Stage 1：CIFAR-10 耦合 headroom（服务器脚本），修补版 v2。** 相对 8fde319 的修改均为缺陷或开销修复，不改变设计：
    (1) `mac_full` 的 Sinkhorn 原先下溢、返回全零计划、退化为独立耦合，现改为中心化代价 + 相对正则 + 对数域 + 失败即报错；
    (2) 一 epoch 独立热身与带偏差校正的 EMA；(3) GPU 上用 Gram 恒等式、float64 算代价，只把 128×128 表送到 CPU；
    (4) GPU 常驻数据流，取代 DataLoader；(5) FID 参考统计只算一次并缓存，直接用张量；(6) 修正计时并加 `--profile-steps`；
    (7) 默认 TF32；(8) 修复 `teacher_routed_ot`；(9) 逐次写记录、存检查点、`--resume`。

    python -m experiments.studies.flow_matching_headroom.cifar --prepare-fid
    python -m experiments.studies.flow_matching_headroom.cifar --smoke --arms independent w2_mb mac_full
    python -m experiments.studies.flow_matching_headroom.cifar --profile-steps 100 --arms independent w2_mb mac_full
    python -m experiments.studies.flow_matching_headroom.cifar --steps 50000 --batch 128 --arms mac_full --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import ot
import torch
from torch import nn

from experiments.harness.records import RunLog, RunRecord, write_records
from experiments.studies.flow_matching.seeds import seed_int

OUT = Path("experiments/out/flow_matching_headroom")
DATA = Path("data/cifar10")
FID_STATS = OUT / "cifar_fid_reference.npz"
BATCH = 128
LR = 2e-4
ARMS = ("independent", "w2_mb", "mac_full", "teacher_routed_ot", "reflow_ref")
TEACHER_ARMS = ("teacher_routed_ot", "reflow_ref")
COUPLED = ("w2_mb", "mac_full", "teacher_routed_ot", "reflow_ref")
FID_COUNTS = (40000, 10000, 10000)


def _commit() -> str:
    """EN: short git commit, recorded in every row. CN: 短 git commit，写入每行。"""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _sync(device: str) -> None:
    """EN: wait for queued GPU work (no-op on CPU). CN: 等待已排队的 GPU 工作（CPU 上无操作）。"""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# --------------------------------------------------------------------------------------------------------------------
# model (unchanged)
# --------------------------------------------------------------------------------------------------------------------


def _time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """EN: sinusoidal time embedding. CN: 正弦时间嵌入。"""
    half = dim // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    arguments = t[:, None] * frequencies[None, :]
    return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)


class Block(nn.Module):
    """EN: a DiT block with adaLN modulation. CN: 带 adaLN 调制的 DiT 块。"""

    def __init__(self, dim: int, heads: int, mlp: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))
        self.modulation = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """EN: one block. CN: 一个块。"""
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, dim=1)
        hidden = self.norm1(tokens) * (1.0 + scale1[:, None]) + shift1[:, None]
        hidden = self.attn(hidden, hidden, hidden, need_weights=False)[0]
        tokens = tokens + gate1[:, None] * hidden
        hidden = self.norm2(tokens) * (1.0 + scale2[:, None]) + shift2[:, None]
        return tokens + gate2[:, None] * self.mlp(hidden)


class DiT(nn.Module):
    """EN: a small unconditional DiT for 32×32 images. CN: 面向 32×32 图像的小型无条件 DiT。"""

    def __init__(self, *, dim: int = 256, depth: int = 6, heads: int = 4, patch: int = 4,
                 channels: int = 3) -> None:
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.patch = int(patch)
        self.channels = int(channels)
        self.embed = nn.Conv2d(channels, dim, patch, patch)
        self.position = nn.Parameter(torch.zeros(1, (32 // patch) ** 2, dim))
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, 4 * dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(dim, 2 * dim)
        self.head = nn.Linear(dim, patch * patch * channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """EN: velocity at `(x, t)`; `t` shape `(batch, 1, 1, 1)`. CN: `(x,t)` 处的速度；`t` 形状 `(batch,1,1,1)`。"""
        tokens = self.embed(x).flatten(2).transpose(1, 2) + self.position
        condition = self.time(_time_embedding(t.reshape(-1), self.dim))
        for block in self.blocks:
            tokens = block(tokens, condition)
        shift, scale = self.modulation(condition).chunk(2, dim=1)
        tokens = self.norm(tokens) * (1.0 + scale[:, None]) + shift[:, None]
        patches = self.head(tokens)
        side = 32 // self.patch
        return patches.reshape(-1, side, side, self.patch, self.patch, self.channels).permute(
            0, 5, 1, 3, 2, 4).reshape(-1, self.channels, 32, 32)


# --------------------------------------------------------------------------------------------------------------------
# data, randomness, EMA, timers
# --------------------------------------------------------------------------------------------------------------------


def _load_uint8() -> tuple[np.ndarray, np.ndarray]:
    """EN: CIFAR-10 train and test as uint8 `(N, 32, 32, 3)`. CN: CIFAR-10 训练/测试集，uint8 `(N,32,32,3)`。"""
    from torchvision import datasets  # imported here: only the download needs torchvision

    train = datasets.CIFAR10(root=str(DATA), train=True, download=True)
    test = datasets.CIFAR10(root=str(DATA), train=False, download=True)
    return np.ascontiguousarray(train.data), np.ascontiguousarray(test.data)


def _nchw(images: np.ndarray) -> torch.Tensor:
    """EN: uint8 `(N,H,W,C)` array to an NCHW tensor view. CN: uint8 `(N,H,W,C)` 数组转 NCHW 张量视图。"""
    return torch.from_numpy(images).permute(0, 3, 1, 2)


class _Stream:
    """EN: GPU-resident CIFAR-10 batches in [-1, 1]: a permutation per epoch (`drop_last`), random horizontal flip.
    CN: GPU 常驻的 `[-1,1]` CIFAR-10 批：每 epoch 一次置换（丢弃末批），随机水平翻转。"""

    def __init__(self, images: np.ndarray, batch: int, key: str, device: str) -> None:
        self.images = _nchw(images).to(device).contiguous()
        self.count = int(self.images.shape[0])
        self.batch = int(batch)
        self.device = device
        self.per_epoch = self.count // self.batch
        self.generator = torch.Generator(device=device).manual_seed(seed_int("cifar-stream", key))
        self.order: torch.Tensor | None = None
        self.cursor = 0

    def next(self) -> torch.Tensor:
        """EN: the next batch. CN: 下一批。"""
        if self.order is None or self.cursor >= self.per_epoch:
            self.order = torch.randperm(self.count, generator=self.generator, device=self.device)
            self.cursor = 0
        index = self.order[self.cursor * self.batch:(self.cursor + 1) * self.batch]
        self.cursor += 1
        x = self.images[index].float().div_(127.5).sub_(1.0)
        flip = torch.rand(self.batch, generator=self.generator, device=self.device) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(3), x)


class _Rng:
    """EN: one generator per purpose, so arms at the same seed share noise, times and the pair stream layout.
    CN: 每种用途一个生成器，使同一种子下各臂共享噪声与时间。"""

    def __init__(self, key: str, device: str) -> None:
        self.device = device
        self.noise = torch.Generator(device=device).manual_seed(seed_int("cifar-noise", key))
        self.time = torch.Generator(device=device).manual_seed(seed_int("cifar-time", key))
        self.pair = torch.Generator(device=device).manual_seed(seed_int("cifar-pair", key))

    def randn_like(self, x: torch.Tensor) -> torch.Tensor:
        """EN: standard normal noise shaped like `x`. CN: 与 `x` 同形的标准正态噪声。"""
        return torch.randn(x.shape, generator=self.noise, device=x.device, dtype=x.dtype)

    def times(self, count: int) -> torch.Tensor:
        """EN: `count` uniform times, shape `(count,1,1,1)`. CN: `count` 个均匀时间，形状 `(count,1,1,1)`。"""
        return torch.rand(count, 1, 1, 1, generator=self.time, device=self.device)


class _Average:
    """EN: EMA of a model's parameters with bias correction, `foreach` update.
    CN: 带偏差校正的参数 EMA，使用 `foreach` 更新。"""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = float(decay)
        self.updates = 0
        self._source = list(model.parameters())
        self._target = list(self.model.parameters())

    @torch.no_grad()
    def update(self) -> None:
        """EN: `ema <- d*ema + (1-d)*model`, `d = min(decay, (1+n)/(10+n))`. CN: 同上，`d = min(decay,(1+n)/(10+n))`。"""
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        torch._foreach_mul_(self._target, decay)
        torch._foreach_add_(self._target, self._source, alpha=1.0 - decay)


class _Timer:
    """EN: synchronised per-stage timers, active only under `--profile-steps`. CN: 同步的分阶段计时，仅 `--profile-steps` 时启用。"""

    def __init__(self, enabled: bool, device: str) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.totals: dict[str, float] = defaultdict(float)

    @contextlib.contextmanager
    def stage(self, name: str):
        """EN: time a block. CN: 给代码块计时。"""
        if not self.enabled:
            yield
            return
        _sync(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.totals[name] += time.perf_counter() - started


# --------------------------------------------------------------------------------------------------------------------
# couplings
# --------------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def _gram(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """EN: squared distances between flattened rows, float64, via the Gram identity (O(B^2 D) as one matmul).
    CN: 展平行之间的平方距离，float64，用 Gram 恒等式（一次矩阵乘）。"""
    a = a.flatten(1).double()
    b = b.flatten(1).double()
    return ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T).clamp_min_(0.0)


@torch.no_grad()
def _mac_cost(model: DiT, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """EN: model-aligned cost averaged over `t in {0,1}`: `||(v0_i+x0_i) - x1_j||^2` and `||x0_i + (v1_j-x1_j)||^2`.
    CN: 在 `t∈{0,1}` 上平均的模型对齐代价。"""
    zero = torch.zeros(x0.shape[0], 1, 1, 1, device=x0.device)
    one = torch.ones(x0.shape[0], 1, 1, 1, device=x0.device)
    left = model(x0, zero) + x0
    right = model(x1, one) - x1
    return 0.5 * (_gram(left, x1) + _gram(x0, -right))


@torch.no_grad()
def _emd_pairs(cost: torch.Tensor) -> torch.Tensor:
    """EN: exact OT (uniform marginals) as an assignment: `cols[i]` is the partner of row `i`.
    CN: 精确 OT（均匀边际）化为指派：`cols[i]` 是第 `i` 行的伙伴。"""
    side = int(cost.shape[0])
    mass = np.full(side, 1.0 / side, dtype=np.float64)
    plan = np.asarray(ot.emd(mass, mass, cost.cpu().numpy(), numItermax=2_000_000))
    columns = plan.argmax(axis=1)
    if np.unique(columns).size != side:
        raise RuntimeError("ot.emd did not return a permutation (iteration limit?)")
    return torch.from_numpy(columns).to(cost.device)


@torch.no_grad()
def _sinkhorn_log(cost: torch.Tensor, rel_reg: float, iters: int, tol: float = 1e-3, check: int = 20
                  ) -> tuple[torch.Tensor, float, int]:
    """
    EN: entropic OT, uniform marginals, log domain, scale-free regularisation.

        The plan is invariant to adding row/column constants to the cost, so the cost is centred first and
        `eps = rel_reg * std(centred cost)`. Returns `(plan, row-marginal L1 error, iterations)` and raises if the plan
        is not finite or not normalised (the old code silently fell back to uniform columns here).

    CN: 熵正则 OT，均匀边际，对数域，无量纲正则。代价加行/列常数不改变计划，故先中心化，`eps = rel_reg * std`。
        计划非有限或未归一时抛错（旧代码在此静默退化为均匀列）。
    """
    side = int(cost.shape[0])
    centred = cost - cost.mean(1, keepdim=True) - cost.mean(0, keepdim=True) + cost.mean()
    eps = float(rel_reg) * float(centred.std().clamp_min(1e-12))
    kernel = -centred / eps
    log_mass = -math.log(side)
    u = torch.zeros(side, dtype=cost.dtype, device=cost.device)
    v = torch.zeros(side, dtype=cost.dtype, device=cost.device)
    iteration = 0
    for iteration in range(1, int(iters) + 1):
        u = log_mass - torch.logsumexp(kernel + v[None, :], dim=1)
        v = log_mass - torch.logsumexp(kernel + u[:, None], dim=0)
        if iteration % check == 0:
            plan = torch.exp(kernel + u[:, None] + v[None, :])
            if float((plan.sum(1) - 1.0 / side).abs().sum()) < tol:
                break
    plan = torch.exp(kernel + u[:, None] + v[None, :])
    error = float((plan.sum(1) - 1.0 / side).abs().sum())
    total = float(plan.sum())
    if not (math.isfinite(error) and math.isfinite(total) and abs(total - 1.0) < 1e-2):
        raise FloatingPointError(f"Sinkhorn plan invalid: sum {total}, error {error}, eps {eps}")
    return plan, error, iteration


def _couple(arm: str, x0: torch.Tensor, x1: torch.Tensor, *, ema: _Average | None,
            bank: tuple[torch.Tensor, torch.Tensor] | None, rng: _Rng, args: argparse.Namespace,
            diagnose: bool, stats: dict[str, list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """EN: re-pair one batch for a coupled arm. CN: 为带耦合的臂重新配对一批。"""
    if arm == "w2_mb":
        return x0, x1[_emd_pairs(_gram(x0, x1))]
    if arm == "mac_full":
        assert ema is not None
        cost = _mac_cost(ema.model, x0, x1)
        plan, error, iterations = _sinkhorn_log(cost, args.mac_reg_rel, args.sinkhorn_iters)
        columns = torch.multinomial(plan.float(), 1, generator=rng.pair).squeeze(1)
        if diagnose:
            side = float(plan.shape[0])
            exact = _emd_pairs(cost)
            stats["top1"].append(float((plan.max(1).values * side).mean()))
            stats["emd_agreement"].append(float((plan.argmax(1) == exact).double().mean()))
            stats["sinkhorn_error"].append(error)
            stats["sinkhorn_iters"].append(float(iterations))
            stats["pair_distinct"].append(float(torch.unique(columns).numel()) / side)
        return x0, x1[columns]
    assert bank is not None
    index = torch.randperm(bank[0].shape[0], generator=rng.pair, device=x0.device)[: x0.shape[0]]
    noise, ends = bank[0][index].float(), bank[1][index].float()
    if arm == "teacher_routed_ot":
        return noise, x1[_emd_pairs(_gram(ends, x1))]
    return noise, ends  # reflow_ref


# --------------------------------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------------------------------


def _step(model: nn.Module, optimiser: torch.optim.Optimizer, x0: torch.Tensor, x1: torch.Tensor, rng: _Rng,
          bf16: bool) -> None:
    """EN: one CFM step. CN: 一步 CFM。"""
    t = rng.times(x0.shape[0])
    x_t = (1.0 - t) * x0 + t * x1
    with torch.autocast(device_type=x0.device.type, dtype=torch.bfloat16, enabled=bool(bf16)):
        prediction = model(x_t, t)
    loss = ((prediction.float() - (x1 - x0)) ** 2).mean()
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()


def _loop(arm: str, model: nn.Module, optimiser: torch.optim.Optimizer, ema_eval: _Average,
          ema_couple: _Average | None, stream: _Stream, rng: _Rng, *, steps: int, warmup: int,
          bank: tuple[torch.Tensor, torch.Tensor] | None, args: argparse.Namespace, device: str, log: RunLog,
          label: str, timer: _Timer) -> tuple[float, dict[str, list[float]]]:
    """EN: the training loop shared by every arm; returns coupling seconds and coupling diagnostics.
    CN: 各臂共用的训练循环；返回耦合耗时与耦合诊断。"""
    stats: dict[str, list[float]] = defaultdict(list)
    coupling_seconds = 0.0
    log_every = max(1, int(steps) // 10)
    started_loop = time.perf_counter()
    for step in range(int(steps)):
        with timer.stage("data"):
            x1 = stream.next()
            x0 = rng.randn_like(x1)
        if arm in COUPLED and not (arm == "mac_full" and step < warmup):
            _sync(device)
            started = time.perf_counter()
            with timer.stage("couple"):
                x0, x1 = _couple(arm, x0, x1, ema=ema_couple, bank=bank, rng=rng, args=args,
                                 diagnose=(step + 1) % log_every == 0, stats=stats)
            _sync(device)
            coupling_seconds += time.perf_counter() - started
        with timer.stage("step"):
            _step(model, optimiser, x0, x1, rng, args.bf16)
        with timer.stage("ema"):
            ema_eval.update()
            if ema_couple is not None:
                ema_couple.update()
        if (step + 1) % log_every == 0:
            extra = ""
            if stats.get("top1"):
                extra = (f" | mac top1 {stats['top1'][-1]:.2f} emd-agree {stats['emd_agreement'][-1]:.2f} "
                         f"iters {stats['sinkhorn_iters'][-1]:.0f}")
            log.line(f"[{label}] step {step + 1}/{steps} {time.perf_counter() - started_loop:.0f}s{extra}")
    return coupling_seconds, stats


@torch.no_grad()
def _teacher_bank(teacher: nn.Module, *, size: int, steps: int, device: str, key: str, chunk: int = 5000
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """EN: noise bank and the teacher's Euler endpoints, built in chunks, stored in float16 on the device.
    CN: 噪声 bank 与 teacher 的 Euler 终点，分块构建，以 float16 存于设备。"""
    noise = torch.empty(size, 3, 32, 32, dtype=torch.float16, device=device)
    ends = torch.empty_like(noise)
    generator = torch.Generator(device=device).manual_seed(seed_int("cifar-bank", key))
    for start in range(0, size, chunk):
        count = min(chunk, size - start)
        z = torch.randn(count, 3, 32, 32, generator=generator, device=device)
        x = z.clone()
        for step in range(int(steps)):
            t = torch.full((count, 1, 1, 1), step / float(steps), device=device)
            x = x + teacher(x, t) / float(steps)
        noise[start:start + count] = z.half()
        ends[start:start + count] = x.half()
    return noise, ends


@torch.no_grad()
def _sample(model: nn.Module, *, count: int, nfe: int, device: str, seed: int, chunk: int = 1000) -> torch.Tensor:
    """EN: Euler rollout from fresh noise, in chunks. CN: 从新噪声分块做 Euler 积分。"""
    generator = torch.Generator(device=device).manual_seed(int(seed))
    pieces = []
    for start in range(0, count, chunk):
        n = min(chunk, count - start)
        x = torch.randn(n, 3, 32, 32, generator=generator, device=device)
        for step in range(int(nfe)):
            t = torch.full((n, 1, 1, 1), step / float(nfe), device=device)
            x = x + model(x, t) / float(nfe)
        pieces.append(x)
    return torch.cat(pieces)


@torch.no_grad()
def _straightness(model: nn.Module, *, count: int, device: str, seed: int) -> float:
    """EN: displacement / path length over a 32-step Euler rollout. CN: 32 步 Euler 轨迹的位移/路径长度。"""
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(count, 3, 32, 32, generator=generator, device=device)
    start = x.clone()
    length = torch.zeros(count, device=device)
    for step in range(32):
        t = torch.full((count, 1, 1, 1), step / 32.0, device=device)
        update = model(x, t) / 32.0
        x = x + update
        length += update.flatten(1).norm(dim=1)
    return float(((x - start).flatten(1).norm(dim=1) / (length + 1e-12)).mean())


# --------------------------------------------------------------------------------------------------------------------
# FID
# --------------------------------------------------------------------------------------------------------------------


class _Inception:
    """EN: pytorch-fid Inception features from uint8 NCHW tensors (same network and resizing as the PNG pipeline).
    CN: 来自 uint8 NCHW 张量的 pytorch-fid Inception 特征（网络与缩放与 PNG 流程一致）。"""

    def __init__(self, device: str) -> None:
        from pytorch_fid.inception import InceptionV3  # type: ignore[import-not-found]

        self.device = device
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    @torch.no_grad()
    def statistics(self, images: torch.Tensor, batch: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """EN: mean and covariance of the 2048-d features. CN: 2048 维特征的均值与协方差。"""
        features = []
        for start in range(0, int(images.shape[0]), batch):
            x = images[start:start + batch].to(self.device).float() / 255.0
            features.append(self.net(x)[0].squeeze(3).squeeze(2).double().cpu())
        matrix = torch.cat(features).numpy()
        return matrix.mean(axis=0), np.cov(matrix, rowvar=False)


def _frechet(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> float:
    """EN: Fréchet distance between two feature Gaussians. CN: 两个特征高斯之间的 Fréchet 距离。"""
    from pytorch_fid.fid_score import calculate_frechet_distance  # type: ignore[import-not-found]

    return float(calculate_frechet_distance(first[0], first[1], second[0], second[1]))


def _fid_reference(device: str, log: RunLog, inception: _Inception | None = None
                   ) -> dict[str, np.ndarray | float]:
    """
    EN: the 40k-train reference statistics and the two floors, computed once and cached (atomic write).

    CN: 40k 训练参考统计与两个下限，只算一次并缓存（原子写入）。
    """
    if FID_STATS.exists():
        cached = np.load(FID_STATS)
        if tuple(int(v) for v in cached["counts"]) != FID_COUNTS:
            raise RuntimeError(f"{FID_STATS} was built with other subset sizes; delete it")
        return {"mu": cached["mu"], "sigma": cached["sigma"],
                "floor_real": float(cached["floor_real"]), "floor_test": float(cached["floor_test"])}
    started = time.perf_counter()
    train, test = _load_uint8()
    inception = inception or _Inception(device)
    reference = inception.statistics(_nchw(train[:FID_COUNTS[0]]))
    held_out = inception.statistics(_nchw(train[FID_COUNTS[0]:FID_COUNTS[0] + FID_COUNTS[1]]))
    test_stats = inception.statistics(_nchw(test[:FID_COUNTS[2]]))
    floor_real, floor_test = _frechet(held_out, reference), _frechet(test_stats, reference)
    FID_STATS.parent.mkdir(parents=True, exist_ok=True)
    temporary = FID_STATS.parent / f"{FID_STATS.stem}.{os.getpid()}.tmp.npz"
    np.savez(temporary, mu=reference[0], sigma=reference[1], floor_real=floor_real, floor_test=floor_test,
             counts=np.asarray(FID_COUNTS))
    os.replace(temporary, FID_STATS)
    log.line(f"[fid] reference built in {time.perf_counter() - started:.0f}s: floor real {floor_real:.2f}, "
             f"floor test {floor_test:.2f}")
    return {"mu": reference[0], "sigma": reference[1], "floor_real": floor_real, "floor_test": floor_test}


def _quantise(images: torch.Tensor) -> torch.Tensor:
    """EN: `[-1,1]` floats to uint8 (what a saved PNG would hold). CN: `[-1,1]` 浮点转 uint8（即 PNG 所存）。"""
    return ((images.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


# --------------------------------------------------------------------------------------------------------------------
# one run and the entry point
# --------------------------------------------------------------------------------------------------------------------


def _run_one(arm: str, seed: int, *, args: argparse.Namespace, images: np.ndarray, device: str, log: RunLog,
             steps: int, out_dir: Path, timer: _Timer) -> tuple[nn.Module, dict[str, float]]:
    """EN: train one (arm, seed); returns the evaluation EMA and the cost/diagnostic metrics.
    CN: 训练一个 (arm, seed)；返回评估用 EMA 与代价/诊断指标。"""
    label = f"{arm} s{seed}"
    checkpoint = out_dir / f"{arm}_s{seed}_t{steps}.pt"
    torch.manual_seed(seed_int("cifar", seed))  # same initialisation for every arm at this seed
    model = DiT(dim=args.dim, depth=args.depth).to(device)
    ema_eval = _Average(model, args.ema_eval)
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device)
        ema_eval.model.load_state_dict(saved["ema_eval"])
        log.line(f"[{label}] resumed from {checkpoint}")
        return ema_eval.model, saved["metrics"]
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    ema_couple = _Average(model, args.ema_couple) if arm == "mac_full" else None
    stream = _Stream(images, args.batch, str(seed), device)
    rng = _Rng(str(seed), device)
    warmup = args.warmup if args.warmup is not None else stream.per_epoch
    warmup = min(int(warmup), max(1, int(steps) // 2))
    bank = None
    teacher_seconds = bank_seconds = 0.0
    if arm in TEACHER_ARMS:
        torch.manual_seed(seed_int("cifar-teacher", seed))
        teacher = DiT(dim=args.dim, depth=args.depth).to(device)
        teacher_ema = _Average(teacher, args.ema_eval)
        teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=LR)
        _sync(device)
        started = time.perf_counter()
        _loop("independent", teacher, teacher_opt, teacher_ema, None, _Stream(images, args.batch, f"teacher-{seed}", device),
              _Rng(f"teacher-{seed}", device), steps=steps, warmup=0, bank=None, args=args, device=device, log=log,
              label=f"{label} teacher", timer=timer)
        _sync(device)
        teacher_seconds = time.perf_counter() - started
        size = 64 if args.smoke else args.bank_factor * len(images)
        started = time.perf_counter()
        bank = _teacher_bank(teacher_ema.model, size=size, steps=args.teacher_steps, device=device, key=str(seed))
        _sync(device)
        bank_seconds = time.perf_counter() - started
        log.line(f"[bank] {label}: {size} points, {args.teacher_steps} Euler, {bank_seconds:.1f}s, "
                 f"teacher {teacher_seconds:.1f}s")
        del teacher, teacher_opt, teacher_ema
    _sync(device)
    started = time.perf_counter()
    coupling_seconds, stats = _loop(arm, model, optimiser, ema_eval, ema_couple, stream, rng, steps=steps,
                                    warmup=warmup, bank=bank, args=args, device=device, log=log, label=label,
                                    timer=timer)
    _sync(device)
    train_seconds = time.perf_counter() - started
    metrics = {"train_seconds": train_seconds, "coupling_seconds": coupling_seconds,
               "teacher_seconds": teacher_seconds, "bank_seconds": bank_seconds,
               "extra_forward_evals": 2.0 if arm == "mac_full" else 0.0, "warmup_steps": float(warmup),
               **{f"mac_{key}": float(np.mean(values)) for key, values in stats.items() if values}}
    if not timer.enabled and not args.smoke:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ema_eval": ema_eval.model.state_dict(), "metrics": metrics}, checkpoint)
    return ema_eval.model, metrics


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="Stage 1 CIFAR-10 coupling headroom (patched v2)")
    parser.add_argument("--arms", nargs="+", default=["independent", "w2_mb", "mac_full"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--bank-factor", type=int, default=10)
    parser.add_argument("--teacher-steps", type=int, default=20)
    parser.add_argument("--fid-samples", type=int, default=10000)
    parser.add_argument("--nfe", type=int, nargs="+", default=[2, 4, 8, 32])
    parser.add_argument("--mac-reg-rel", type=float, default=0.1,
                        help="Sinkhorn eps as a fraction of the centred-cost std (0.1 matches the toy reg 0.5 at d=2)")
    parser.add_argument("--sinkhorn-iters", type=int, default=300)
    parser.add_argument("--ema-couple", type=float, default=0.999)
    parser.add_argument("--ema-eval", type=float, default=0.9999)
    parser.add_argument("--warmup", type=int, default=None, help="independent warm-up steps for mac_full (default: 1 epoch)")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="bf16 autocast for the training step only")
    parser.add_argument("--profile-steps", type=int, default=0, help="time K steps per arm with synchronised stages, then exit")
    parser.add_argument("--prepare-fid", action="store_true", help="build the FID reference cache and exit")
    parser.add_argument("--resume", action="store_true", help="re-evaluate from a saved final checkpoint if present")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag", default="cifar")
    args = parser.parse_args(argv)
    if args.smoke:
        args.steps, args.batch, args.dim, args.depth = 4, 8, 32, 1
        args.seeds, args.fid_samples, args.nfe = [0], 0, [1]
    for arm in args.arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
    out_dir = OUT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "run.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.prepare_fid:
        _fid_reference(device, log)
        log.close()
        return 0
    profile = args.profile_steps > 0
    evaluate = not (args.smoke or profile)
    steps = args.profile_steps if profile else args.steps
    commit = _commit()
    images, _ = _load_uint8()
    inception = _Inception(device) if evaluate else None
    reference = _fid_reference(device, log, inception) if evaluate else None
    records: list[RunRecord] = []
    started_all = time.perf_counter()

    def config(arm: str, seed: int, nfe: int) -> dict[str, object]:
        return {"stage": "cifar", "arm": arm, "seed": seed, "steps": args.steps, "batch": args.batch,
                "dim": args.dim, "depth": args.depth, "nfe": nfe, "mac_reg_rel": args.mac_reg_rel,
                "sinkhorn_iters": args.sinkhorn_iters, "ema_couple": args.ema_couple, "ema_eval": args.ema_eval,
                "warmup": args.warmup, "bank_factor": args.bank_factor, "teacher_steps": args.teacher_steps,
                "tf32": not args.no_tf32, "bf16": args.bf16, "eval_weights": "ema", "device": device,
                "commit": commit}

    for arm in args.arms:
        for seed in args.seeds:
            timer = _Timer(profile, device)
            model, metrics = _run_one(arm, seed, args=args, images=images, device=device, log=log, steps=steps,
                                      out_dir=out_dir, timer=timer)
            if profile:
                total = sum(timer.totals.values()) or 1.0
                log.line(f"[profile] {arm}: " + ", ".join(
                    f"{name} {1e3 * seconds / steps:.1f} ms/step ({100 * seconds / total:.0f}%)"
                    for name, seconds in timer.totals.items()) + f" | total {1e3 * total / steps:.1f} ms/step")
                continue
            straightness = _straightness(model, count=min(256, max(8, args.batch)), device=device, seed=seed + 1)
            metrics = {**metrics, "straightness": straightness}
            if evaluate:
                assert inception is not None and reference is not None
                floors = {"fid_floor_real": float(reference["floor_real"]),
                          "fid_floor_test": float(reference["floor_test"])}
                reference_stats = (reference["mu"], reference["sigma"])
                fid_started = time.perf_counter()
                for nfe in args.nfe:
                    generated = _sample(model, count=args.fid_samples, nfe=nfe, device=device, seed=seed + 2)
                    value = _frechet(inception.statistics(_quantise(generated)), reference_stats)
                    log.line(f"[fid] {arm} s{seed} NFE {nfe}: {value:.2f}")
                    records.append(RunRecord(
                        study="flow_matching_headroom_cifar", geometry="cifar10", arm=arm, k=3, m=args.batch,
                        draw=seed, metrics={**metrics, **floors, "fid": value}, config=config(arm, seed, nfe),
                        extra={}))
                metrics = {**metrics, **floors, "fid_seconds": time.perf_counter() - fid_started}
            records.append(RunRecord(
                study="flow_matching_headroom_cifar", geometry="cifar10", arm=arm, k=3, m=args.batch, draw=seed,
                metrics=metrics, config=config(arm, seed, 0), extra={}))
            write_records(out_dir / "records.json", records,
                          provenance={"question": "is there coupling headroom beyond squared-Euclidean OT on CIFAR-10?",
                                      "stage": "cifar", "arms": args.arms, "seeds": args.seeds,
                                      "fid": "pytorch-fid Inception, uint8-quantised tensors, 10k generated vs 40k train "
                                             "reference; floors: 10k held-out real, test; evaluation EMA",
                                      "mac": f"centred cost, log-domain Sinkhorn, eps = {args.mac_reg_rel} x std, "
                                             f"EMA {args.ema_couple} with bias correction, 1-epoch warm-up",
                                      "baselines": "none; this is a headroom screen, not a SOTA comparison",
                                      "commit": commit})
    log.line(f"cifar done in {time.perf_counter() - started_all:.1f}s; {len(records)} records")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
