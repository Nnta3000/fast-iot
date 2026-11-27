"""
EN: Stage 1 sample grids from the existing 50k checkpoints. 64 samples per arm, the same noise seed for every arm,
    Euler NFE 4 and 32, plus an 8x8 grid of real CIFAR-10 images. No training and no FID; this is a visual check.

CN: 由已有 50k checkpoint 生成 Stage 1 样本网格。每臂 64 张、所有臂共用同一噪声种子，Euler NFE 4/32，另有
    8x8 真实 CIFAR-10 图像网格。不训练、不算 FID；只作直观检查。

    python -m experiments.studies.flow_matching_headroom.make_grids \
        --root experiments/out/flow_matching_headroom \
        --out experiments/out/flow_matching_headroom/cifar_grids
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.studies.flow_matching_headroom.cifar import DiT, _load_uint8, _quantise, _sample

ARMS = ("independent", "w2_mb", "mac_full")
TAGS = {"independent": "cifar_independent", "w2_mb": "cifar_w2_mb", "mac_full": "cifar_mac_full"}


def _checkpoint(root: Path, arm: str, seed: int) -> Path:
    """EN: the checkpoint path for one arm/seed. CN: 一个 arm/seed 的 checkpoint 路径。"""
    return root / TAGS[arm] / f"{arm}_s{seed}_t50000.pt"


def _tile(images: list[np.ndarray], scale: int = 4) -> np.ndarray:
    """EN: 8x8 uint8 tiles, nearest-neighbour upscaled for visibility. CN: 8x8 uint8 瓦片，最近邻放大。"""
    assert len(images) == 64
    rows = [np.concatenate(images[start:start + 8], axis=1) for start in range(0, 64, 8)]
    grid = np.concatenate(rows, axis=0)
    return np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)


def _sample_grid(model: DiT, *, nfe: int, seed: int, count: int, device: str) -> np.ndarray:
    """EN: sample `count` images with one model, return an 8x8 uint8 grid. CN: 用单模型采样 count 张，返回 8x8 网格。"""
    with torch.no_grad():
        generated = _sample(model, count=count, nfe=nfe, device=device, seed=seed)
    images = [image.permute(1, 2, 0).cpu().numpy() for image in _quantise(generated)]
    return _tile(images)


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="sample grids from 50k CIFAR checkpoints")
    parser.add_argument("--root", type=Path, default=Path("experiments/out/flow_matching_headroom"))
    parser.add_argument("--out", type=Path, default=Path("experiments/out/flow_matching_headroom/cifar_grids"))
    parser.add_argument("--seed", type=int, default=0, help="the single checkpoint seed shown in each arm")
    parser.add_argument("--noise-seed", type=int, default=0, help="the same noise seed for every arm")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--nfe", type=int, nargs="+", default=[4, 32])
    args = parser.parse_args(argv)
    if args.count != 64:
        raise ValueError("the grid is 8x8; keep --count 64")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    for nfe in args.nfe:
        for arm in ARMS:
            path = _checkpoint(args.root, arm, args.seed)
            if not path.exists():
                raise FileNotFoundError(path)
            saved = torch.load(path, map_location=device, weights_only=False)
            model = DiT(dim=256, depth=6).to(device)
            model.load_state_dict(saved["ema_eval"])
            model.eval()
            grid = _sample_grid(model, nfe=nfe, seed=args.noise_seed, count=args.count, device=device)
            Image.fromarray(grid).save(args.out / f"{arm}_nfe{nfe}_noiseseed{args.noise_seed}.png")
            print(f"saved {arm} NFE {nfe} -> {args.out / f'{arm}_nfe{nfe}_noiseseed{args.noise_seed}.png'}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    train, _ = _load_uint8()
    real = [train[index] for index in range(args.count)]
    Image.fromarray(_tile(real)).save(args.out / "real_8x8.png")
    print(f"saved real CIFAR -> {args.out / 'real_8x8.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
