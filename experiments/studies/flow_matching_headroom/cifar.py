"""
EN: **Stage 1: CIFAR-10 coupling headroom (server script).**

    Unconditional CIFAR-10 (conditioning is deliberately absent; it interacts badly with a global coupling). A small
    DiT is trained with the same CFM loss and one of: `independent`, `w2_mb` (exact POT on a batch of noise and data),
    `mac_full` (EMA prediction error + Sinkhorn, reg 0.5), `teacher_routed_ot` (independent teacher, noise bank with
    teacher endpoints, POT against real images) or `reflow_ref` (synthetic `(x0, T(x0))` pairs). FID is measured on
    10 000 generated samples against a 40 000-image training reference, with two floors: a held-out 10 000-image real
    subset and the CIFAR-10 test set.

    This script is written for a server GPU. `--smoke` runs a few steps and skips FID, so the time per step can be
    measured before the full run. It does not install anything: run `pip install pytorch-fid` first (see `server.sh`).

CN: **Stage 1：CIFAR-10 耦合 headroom（服务器脚本）。** 无条件 CIFAR-10（刻意不加条件；条件与全局耦合相性差）。小 DiT 用同一
    CFM 损失，耦合取 `independent`、`w2_mb`（一批噪声与数据的精确 POT）、`mac_full`（EMA 预测误差 + Sinkhorn，reg 0.5）、
    `teacher_routed_ot`（独立 teacher、带 teacher 终点的噪声 bank、对真实图像做 POT）或 `reflow_ref`（合成 `(x0,T(x0))` 对）。
    FID 用 10 000 个生成样本对 40 000 张训练参考，并报两个下限：留出的 10 000 张真实子集与 CIFAR-10 测试集。脚本面向服务器
    GPU；`--smoke` 只跑几步且跳过 FID，便于先测每步耗时。它不安装任何东西：先 `pip install pytorch-fid`（见 `server.sh`）。

    python -m experiments.studies.flow_matching_headroom.cifar --smoke
    python -m experiments.studies.flow_matching_headroom.cifar --steps 50000 --batch 128 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from experiments.harness.records import RunLog, RunRecord, write_records
from experiments.studies.flow_matching.seeds import seed_int
from experiments.studies.flow_matching_headroom.common import EMA, MAC_REG, pot_pairs, sinkhorn_pairs

OUT = Path("experiments/out/flow_matching_headroom")
DATA = Path("data/cifar10")
IMAGE_DIR = OUT / "cifar_images"
BATCH = 128
LR = 2e-4


def _commit() -> str:
    """EN: short git commit, recorded in every row. CN: 短 git commit，写入每行。"""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


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


def _loader(batch: int, *, train: bool) -> DataLoader:
    """EN: CIFAR-10 loader in [-1, 1]. CN: `[-1,1]` 的 CIFAR-10 载入器。"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda image: image),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    data = datasets.CIFAR10(root=str(DATA), train=train, download=True, transform=transform)
    return DataLoader(data, batch_size=int(batch), shuffle=train, num_workers=4, drop_last=train, pin_memory=True)


def _step(model: DiT, optimiser: torch.optim.Optimizer, x0: torch.Tensor, x1: torch.Tensor) -> None:
    """EN: one CFM step. CN: 一步 CFM。"""
    t = torch.rand(x0.shape[0], 1, 1, 1, device=x0.device)
    x_t = (1.0 - t) * x0 + t * x1
    loss = ((model(x_t, t) - (x1 - x0)) ** 2).mean()
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()


@torch.no_grad()
def _mac_error(model: DiT, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
    """EN: model-aligned error on flattened images, averaged over `t ∈ {0,1}`. CN: 展平图像上的模型对齐误差，在 `t∈{0,1}` 上平均。"""
    batch = x0.shape[0]
    zero = torch.zeros(batch, 1, 1, 1, device=x0.device)
    one = torch.ones(batch, 1, 1, 1, device=x0.device)
    flat0, flat1 = x0.flatten(1), x1.flatten(1)
    left = model(x0, zero).flatten(1) + flat0
    right = model(x1, one).flatten(1) - flat1
    first = ((left[:, None, :] - flat1[None, :, :]) ** 2).sum(-1)
    second = ((flat0[:, None, :] + right[None, :, :]) ** 2).sum(-1)
    return (0.5 * (first + second)).cpu().numpy()


def _pot_pairs_images(x0: torch.Tensor, x1: torch.Tensor, rng: np.random.Generator
                      ) -> tuple[np.ndarray, np.ndarray]:
    """EN: exact POT on the squared distance between flattened images. CN: 展平图像平方距离上的精确 POT。"""
    flat0 = x0.flatten(1).cpu().numpy().astype(np.float64)
    flat1 = x1.flatten(1).cpu().numpy().astype(np.float64)
    cost = ((flat0[:, None, :] - flat1[None, :, :]) ** 2).sum(-1)
    return pot_pairs(cost, rng, count=flat0.shape[0])


def _teacher(model: DiT, data: DataLoader, *, steps: int, device: str) -> DiT:
    """EN: an independent-coupling teacher. CN: 独立耦合的 teacher。"""
    teacher = DiT(dim=model.dim, depth=model.depth).to(device)
    optimiser = torch.optim.AdamW(teacher.parameters(), lr=LR)
    batches = iter(data)
    for step in range(int(steps)):
        try:
            x1, _ = next(batches)
        except StopIteration:
            batches = iter(data)
            x1, _ = next(batches)
        x1 = x1.to(device)
        _step(teacher, optimiser, torch.randn_like(x1), x1)
    return teacher


@torch.no_grad()
def _teacher_bank(model: DiT, *, size: int, steps: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """EN: noise bank and teacher endpoints (float16, to keep the 500k bank small). CN: 噪声 bank 与 teacher 终点（float16，压缩 500k bank）。"""
    generator = torch.Generator(device=device).manual_seed(seed_int("cifar-bank"))
    noise = torch.randn(size, 3, 32, 32, generator=generator, device=device)
    endpoints = noise.clone()
    for step in range(int(steps)):
        t = torch.full((size, 1, 1, 1), step / float(steps), device=device)
        endpoints = endpoints + model(endpoints, t) / float(steps)
    return noise.cpu().numpy(), endpoints.half().cpu().numpy()


@torch.no_grad()
def _sample(model: DiT, *, count: int, nfe: int, device: str, seed: int) -> torch.Tensor:
    """EN: Euler rollout from fresh noise. CN: 从新噪声的 Euler 积分。"""
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(count, 3, 32, 32, generator=generator, device=device)
    for step in range(int(nfe)):
        t = torch.full((count, 1, 1, 1), step / float(nfe), device=device)
        x = x + model(x, t) / float(nfe)
    return x


@torch.no_grad()
def _straightness(model: DiT, *, count: int, device: str, seed: int) -> float:
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


def _save_images(images: torch.Tensor, folder: Path) -> None:
    """EN: save a batch of `[-1,1]` images as PNGs for FID. CN: 把 `[-1,1]` 图像保存为 PNG 供 FID 使用。"""
    folder.mkdir(parents=True, exist_ok=True)
    array = ((images.clamp(-1, 1) + 1.0) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    for index, item in enumerate(array):
        Image.fromarray(item).save(folder / f"{index:05d}.png")


def _prepare_fid_reference() -> tuple[Path, Path, Path]:
    """EN: 40k train reference, 10k held-out real floor, 10k test floor as PNG folders. CN: 40k 训练参考、10k 留出真实下限、10k 测试下限的 PNG 目录。"""
    train = datasets.CIFAR10(root=str(DATA), train=True, download=True, transform=transforms.ToTensor())
    test = datasets.CIFAR10(root=str(DATA), train=False, download=True, transform=transforms.ToTensor())
    train_ref = IMAGE_DIR / "train_40k"
    train_floor = IMAGE_DIR / "train_10k"
    test_dir = IMAGE_DIR / "test_10k"
    if not (train_ref.exists() and train_floor.exists() and test_dir.exists()):
        for name, subset in (("train_40k", Subset(train, range(40000))),
                             ("train_10k", Subset(train, range(40000, 50000))),
                             ("test_10k", Subset(test, range(10000)))):
            folder = IMAGE_DIR / name
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(len(subset)):
                image, _ = subset[index]  # type: ignore[misc]
                Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
                    folder / f"{index:05d}.png")
    return train_ref, train_floor, test_dir


def _fid(first: Path, second: Path, device: str) -> float:
    """EN: FID between two PNG folders (pytorch-fid). CN: 两个 PNG 目录之间的 FID（pytorch-fid）。"""
    from pytorch_fid.fid_score import calculate_fid_given_paths  # type: ignore[import-not-found]
    return float(calculate_fid_given_paths([str(first), str(second)], batch_size=64,
                                           device=device, dims=2048, num_workers=0))


ARMS = ("independent", "w2_mb", "mac_full", "teacher_routed_ot", "reflow_ref")


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="Stage 1 CIFAR-10 coupling headroom")
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag", default="cifar")
    args = parser.parse_args(argv)
    if args.smoke:
        args.steps, args.batch, args.dim, args.depth = 4, 8, 32, 1
        args.seeds, args.arms, args.fid_samples, args.nfe = [0], ["independent"], 0, [1]
    out_dir = OUT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "run.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    commit = _commit()
    records: list[RunRecord] = []
    started = time.perf_counter()
    train_loader = _loader(args.batch, train=True)
    for arm in args.arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        for seed in args.seeds:
            torch.manual_seed(seed_int("cifar", arm, seed))
            model = DiT(dim=args.dim, depth=args.depth).to(device)
            optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
            ema = EMA(model, decay=0.9999)
            rng = np.random.default_rng(seed_int("cifar-pairs", arm, seed))
            teacher = None
            bank_seconds = teacher_seconds = 0.0
            if arm in ("teacher_routed_ot", "reflow_ref"):
                started_teacher = time.perf_counter()
                teacher = _teacher(model, train_loader, steps=args.steps, device=device)
                teacher_seconds = time.perf_counter() - started_teacher
                if args.smoke:
                    bank_size = 64
                else:
                    bank_size = args.bank_factor * 50000
                started_bank = time.perf_counter()
                noise, endpoints = _teacher_bank(teacher, size=bank_size, steps=args.teacher_steps, device=device)
                bank_seconds = time.perf_counter() - started_bank
                log.line(f"[bank] {arm} s{seed}: {bank_size} points, {args.teacher_steps} Euler, "
                         f"{bank_seconds:.1f}s, teacher {teacher_seconds:.1f}s")
            coupling_seconds = 0.0
            extra_forward_evals = 2.0 if arm == "mac_full" else 0.0
            batches = iter(train_loader)
            for step in range(int(args.steps)):
                try:
                    x1, _ = next(batches)
                except StopIteration:
                    batches = iter(train_loader)
                    x1, _ = next(batches)
                x1 = x1.to(device)
                x0 = torch.randn_like(x1)
                started_coupling = time.perf_counter()
                if arm == "independent":
                    pass
                elif arm == "w2_mb":
                    rows, columns = _pot_pairs_images(x0, x1, rng)
                    x0, x1 = x0[rows], x1[columns]
                elif arm == "mac_full":
                    error = _mac_error(ema.model, x0, x1)
                    rows, columns = sinkhorn_pairs(error, rng, reg=MAC_REG, count=x0.shape[0])
                    x0, x1 = x0[rows], x1[columns]
                elif arm == "teacher_routed_ot":
                    bank_index = rng.choice(len(noise), size=x0.shape[0], replace=False)
                    data_index = rng.choice(len(x1), size=x0.shape[0], replace=False)
                    x0 = torch.as_tensor(noise[bank_index], dtype=torch.float32, device=device)
                    x1 = x1[data_index]
                    cost = ((x0.flatten(1) - x1.flatten(1)) ** 2).sum(-1).cpu().numpy()
                    rows, columns = pot_pairs(cost, rng, count=x0.shape[0])
                    x0, x1 = x0[rows], x1[columns]
                elif arm == "reflow_ref":
                    bank_index = rng.choice(len(noise), size=x0.shape[0], replace=False)
                    x0 = torch.as_tensor(noise[bank_index], dtype=torch.float32, device=device)
                    x1 = torch.as_tensor(endpoints[bank_index], dtype=torch.float16, device=device).float()
                coupling_seconds += time.perf_counter() - started_coupling
                _step(model, optimiser, x0, x1)
                ema.update(model)
                if (step + 1) % max(1, args.steps // 10) == 0:
                    log.line(f"[{arm} s{seed}] step {step + 1}/{args.steps} "
                             f"{time.perf_counter() - started:.0f}s")
            train_seconds = time.perf_counter() - started
            straightness = _straightness(model, count=min(256, max(8, args.batch)),
                                         device=device, seed=seed + 1)
            metrics = {"train_seconds": train_seconds, "coupling_seconds": coupling_seconds,
                       "teacher_seconds": teacher_seconds, "bank_seconds": bank_seconds,
                       "extra_forward_evals": extra_forward_evals, "straightness": straightness}
            if not args.smoke:
                train_ref, train_floor, test_dir = _prepare_fid_reference()
                floors = {"fid_floor_real": _fid(train_floor, train_ref, device),
                          "fid_floor_test": _fid(test_dir, train_ref, device)}
                for nfe in args.nfe:
                    images = _sample(model, count=args.fid_samples, nfe=nfe, device=device, seed=seed + 2)
                    folder = IMAGE_DIR / f"gen_{arm}_s{seed}_nfe{nfe}"
                    _save_images(images, folder)
                    value = _fid(folder, train_ref, device)
                    log.line(f"[fid] {arm} s{seed} NFE {nfe}: {value:.2f}")
                    records.append(RunRecord(
                        study="flow_matching_headroom_cifar", geometry="cifar10", arm=arm, k=3, m=args.batch,
                        draw=seed, metrics={**metrics, **floors, "fid": value},
                        config={"stage": "cifar", "arm": arm, "seed": seed, "steps": args.steps,
                                "batch": args.batch, "dim": args.dim, "depth": args.depth, "nfe": nfe,
                                "mac_reg": MAC_REG, "bank_factor": args.bank_factor,
                                "teacher_steps": args.teacher_steps, "device": device, "commit": commit},
                        extra={}))
                records.append(RunRecord(
                    study="flow_matching_headroom_cifar", geometry="cifar10", arm=arm, k=3, m=args.batch,
                    draw=seed, metrics={**metrics, **floors},
                    config={"stage": "cifar", "arm": arm, "seed": seed, "steps": args.steps,
                            "batch": args.batch, "dim": args.dim, "depth": args.depth, "nfe": 0,
                            "mac_reg": MAC_REG, "bank_factor": args.bank_factor,
                            "teacher_steps": args.teacher_steps, "device": device, "commit": commit},
                    extra={}))
            else:
                records.append(RunRecord(
                    study="flow_matching_headroom_cifar", geometry="cifar10", arm=arm, k=3, m=args.batch,
                    draw=seed, metrics=metrics,
                    config={"stage": "cifar", "arm": arm, "seed": seed, "steps": args.steps,
                            "batch": args.batch, "dim": args.dim, "depth": args.depth, "nfe": 0,
                            "mac_reg": MAC_REG, "bank_factor": args.bank_factor,
                            "teacher_steps": args.teacher_steps, "device": device, "commit": commit},
                    extra={}))
    write_records(out_dir / "records.json", records,
                  provenance={"question": "is there coupling headroom beyond squared-Euclidean OT on CIFAR-10?",
                              "stage": "cifar", "arms": args.arms, "seeds": args.seeds,
                              "fid": "pytorch-fid, 10k generated vs 40k train reference; floors: 10k held-out real, test",
                              "baselines": "none; this is a headroom screen, not a SOTA comparison",
                              "commit": commit})
    log.line(f"cifar done in {time.perf_counter() - started:.1f}s; {len(records)} records")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
