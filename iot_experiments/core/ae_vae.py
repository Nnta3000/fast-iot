"""
CN: **小型 VAE 编码器**（与 DPM 项目的 AE 同族：对角高斯后验、`embed_dim=4`、取后验均值作为 latent），
    在本地 1000 张 T1 与 1000 张 T2 切片上**分别**训练（各自重建自己的模态，**不使用任何配对信息**）。

    为什么这样做：DPM 项目报告的 0.438 Recall@1 是在 **AE latent + 固定余弦代价、完全无标签** 的条件下
    得到的；我们的块特征 + 学到的 theta 只有 0.311。要判断逆 OT 在更好的表示上是否还有价值，
    必须先有可比的 latent。此脚本提供该 latent，**不改变**任何既有结果。

    设计（保持简单、可复现）：
      - 输入：192×192 灰度，归一化到 [-1, 1]，下采样到 128×128。
      - 编码器：4 个 stride-2 卷积块 (32,64,128,256) → 8×8×256 → 1×1 卷积到 2*embed_dim。
      - 后验：对角高斯 (mu, logvar)；`latent = mu`（与 DPM 的 `latent_mode="mode"` 一致）。
      - 解码器：对称转置卷积 + 1×1 卷积回单通道，`tanh` 输出。
      - 损失：MSE 重建 + `kl_weight * KL(q||N(0,I))`（beta-VAE 形式，KL 权重在前 20% epoch 线性升温）。
      - 优化器 Adam，余弦退火学习率。

EN: **Small VAE encoder** (same family as the DPM project's AE: diagonal-Gaussian posterior, `embed_dim=4`,
    posterior mean as the latent), trained **separately** on the 1000 local T1 and 1000 T2 slices -- each
    reconstructs only its own modality, **no pairing information is used anywhere**.

    Why: the DPM report's 0.438 Recall@1 comes from **AE latents + a fixed cosine cost, fully unsupervised**;
    our block features + learned theta reach 0.311. To test whether inverse OT still adds value on a better
    representation we first need a comparable latent. This script supplies it and **changes no existing
    result**.

    Design: 192x192 grayscale -> [-1, 1] -> 128x128; four stride-2 conv blocks (32,64,128,256) -> 8x8x256 ->
    1x1 conv to 2*embed_dim; diagonal Gaussian posterior, `latent = mu`; symmetric transposed-conv decoder
    with `tanh`; loss = MSE + `kl_weight * KL` (linear KL warm-up over the first 20% of epochs); Adam with
    cosine annealing.

用法 / EN: usage
    python -u -m iot_experiments.core.ae_vae --modality T1 --epochs 60
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DATA_ROOT = Path("/home/agent/Documents/dpm-inverse-ot/data/Task_002_MRI_T1_T2")
IMG = 192


def load_gray(path: Path, size: int = IMG) -> np.ndarray:
    """CN: 用 ImageMagick 读灰度 PNG 并缩放到 `size`。EN: read a grayscale PNG via ImageMagick."""
    raw = subprocess.run(
        ["magick", str(path), "-colorspace", "Gray", "-resize", f"{size}x{size}!",
         "-depth", "8", "gray:-"],
        check=True, capture_output=True,
    ).stdout
    return (np.frombuffer(raw, dtype=np.uint8).reshape(size, size).astype(np.float32) / 127.5) - 1.0


def build_vae(embed_dim: int, channels: tuple[int, ...] = (32, 64, 128, 256)):
    """CN: 构造 (VAE, encode, decode)。EN: build (VAE, encode, decode)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class VAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            enc: list[nn.Module] = []
            cin = 1
            for cout in channels:
                enc += [nn.Conv2d(cin, cout, 4, 2, 1), nn.GroupNorm(8, cout), nn.SiLU()]
                cin = cout
            self.enc = nn.Sequential(*enc)
            self.quant = nn.Conv2d(channels[-1], 2 * int(embed_dim), 1)
            dec: list[nn.Module] = [nn.Conv2d(int(embed_dim), channels[-1], 3, 1, 1), nn.SiLU()]
            # CN: 编码器有 `len(channels)` 次 stride-2，因此解码需要**同样次数**的转置卷积把尺寸翻倍回来。
            #     最后一层从最小通道 (channels[0]) 回到 1 通道。
            # EN: the encoder performs `len(channels)` stride-2 steps, so the decoder needs the same number of
            #     transposed convs to restore the size; the final layer goes from `channels[0]` back to 1.
            ladder = list(channels[::-1]) + [channels[0]]
            for cprev, cnext in zip(ladder, ladder[1:]):
                dec += [nn.ConvTranspose2d(cprev, cnext, 4, 2, 1), nn.GroupNorm(8, cnext), nn.SiLU()]
            dec += [nn.Conv2d(channels[0], 1, 3, 1, 1)]
            self.dec = nn.Sequential(*dec)

        def encode(self, x):
            h = self.enc(x)
            m = self.quant(h)
            mu, logvar = torch.chunk(m, 2, dim=1)
            return mu, logvar

        def decode(self, z):
            return torch.tanh(self.dec(z))

        def forward(self, x):
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return self.decode(z), mu, logvar

    def encode(model, x):
        mu, _ = model.encode(x)
        return mu

    def decode(model, z):
        return model.decode(z)

    return VAE(), encode, decode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="small VAE encoder on one modality")
    parser.add_argument("--modality", default="T1", choices=("T1", "T2"))
    parser.add_argument("--embed-dim", type=int, default=4)
    parser.add_argument("--downsample", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "ae")
    args = parser.parse_args(argv)

    import torch

    torch.manual_seed(int(args.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = sorted((_DATA_ROOT / "Train" / args.modality).glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))
    t0 = time.perf_counter()
    data = np.stack([load_gray(p) for p in files])[:, None]  # (N,1,192,192)
    print(f"[vae] {args.modality}: {data.shape} decoded in {time.perf_counter() - t0:.1f}s", flush=True)
    x_all = torch.from_numpy(data).to(device)
    if args.downsample != IMG:
        x_all = torch.nn.functional.interpolate(x_all, size=int(args.downsample), mode="bilinear",
                                                align_corners=False)
    n = int(x_all.shape[0])

    model, _, _ = build_vae(int(args.embed_dim))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.epochs))
    warm = max(1, int(0.2 * int(args.epochs)))

    print(f"{'epoch':>5} {'loss':>9} {'recon':>9} {'kl':>9} {'lr':>8} {'s':>6}", flush=True)
    history: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = rec_tot = kl_tot = 0.0
        t0 = time.perf_counter()
        for start in range(0, n, int(args.batch)):
            idx = perm[start:start + int(args.batch)]
            xb = x_all[idx]
            recon, mu, logvar = model(xb)
            rec = torch.mean((recon - xb) ** 2)
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=(1, 2, 3)))
            beta = float(args.kl_weight) * min(1.0, (epoch + 1) / warm)
            loss = rec + beta * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            k = xb.shape[0]
            tot += float(loss) * k
            rec_tot += float(rec) * k
            kl_tot += float(kl) * k
        sched.step()
        row = {"epoch": epoch + 1, "loss": tot / n, "recon": rec_tot / n, "kl": kl_tot / n,
               "lr": float(sched.get_last_lr()[0]), "s": time.perf_counter() - t0}
        history.append(row)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{row['epoch']:>5} {row['loss']:>9.5f} {row['recon']:>9.5f} {row['kl']:>9.5f} "
                  f"{row['lr']:>8.2e} {row['s']:>6.1f}", flush=True)

    model.eval()
    lat: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, 64):
            mu, _ = model.encode(x_all[start:start + 64])
            lat.append(mu.flatten(1).cpu().numpy())
    latents = np.concatenate(lat, axis=0).astype(np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"latents_{args.modality}.npz",
                        latents=latents,
                        ids=np.array([int(p.stem.split("_")[1]) for p in files]))
    torch.save(model.state_dict(), args.out / f"vae_{args.modality}.pt")
    (args.out / f"vae_{args.modality}.json").write_text(
        json.dumps({"modality": args.modality, "embed_dim": int(args.embed_dim),
                    "latent_dim": int(latents.shape[1]), "downsample": int(args.downsample),
                    "epochs": int(args.epochs), "batch": int(args.batch), "lr": float(args.lr),
                    "kl_weight": float(args.kl_weight), "seed": int(args.seed),
                    "history": history}, indent=2) + "\n", encoding="utf-8")
    print(f"[vae] {args.modality}: latent {latents.shape} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
