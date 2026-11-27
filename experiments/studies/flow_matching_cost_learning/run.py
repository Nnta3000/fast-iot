"""
EN: **Does a different ground cost improve OT-CFM on toys?** A first screen with fixed feature costs.

    The coupling is exact OT on a fresh 256-point noise/data pool, as in `w2_mb`. Only the ground cost changes:
    raw squared Euclidean (`l2`), a fixed random-Fourier RBF feature cost (`rbf`), and a PCA-whitened cost (`pca`).
    The flow model, loss, optimiser and 10k-step budget are identical to the v2 toy study. This is a cost screen,
    not a learned cost; a learned cost is only worth trying if a fixed feature cost moves the few-step result.

CN: **不同的 ground cost 能否改进玩具上的 OT-CFM？** 固定特征代价的第一次筛选。耦合仍是 256 点池上的精确 OT
    （同 `w2_mb`），只替换 ground cost：原始平方欧氏（`l2`）、固定 random-Fourier RBF 特征代价（`rbf`）、
    PCA 白化代价（`pca`）。模型、损失、优化器与 10k 步预算与 v2 玩具研究一致。这里不是学习代价；只有在固定
    特征代价能改变 few-step 结果时，学习代价才值得尝试。

    python -m experiments.studies.flow_matching_cost_learning.run --targets eight_gaussians two_moons
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import ot
import torch

from experiments.harness.records import RunLog, RunRecord, write_records
from experiments.studies.flow_matching.data import dataset
from experiments.studies.flow_matching.run import BATCH, LR, VelocityMLP
from experiments.studies.flow_matching.seeds import seed as seed_from
from experiments.studies.flow_matching.seeds import seed_int
from experiments.studies.flow_matching_headroom.common import evaluate

OUT = Path("experiments/out/flow_matching_cost_learning")
TRAIN_POINTS = 8192
STEPS = 10000
COSTS = ("l2", "rbf", "pca")


def _commit() -> str:
    """EN: short git commit. CN: 短 git commit。"""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """EN: squared Euclidean distances between rows, float64. CN: 行间平方欧氏距离，float64。"""
    a = a.double()
    b = b.double()
    return ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T).clamp_min_(0.0)


class _Features:
    """EN: a fixed feature map for a ground cost. CN: ground cost 的固定特征映射。"""

    def __init__(self, kind: str, data: np.ndarray, device: str, *, weights: tuple[float, ...] | None = None,
                 scale: float = 1.0) -> None:
        self.kind = kind
        self.device = device
        self.mean: torch.Tensor | None = None
        self.matrix: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None
        if kind == "l2":
            return
        if kind == "diag":
            if weights is None or len(weights) != int(data.shape[1]):
                raise ValueError(f"diag weights must have length {data.shape[1]}")
            self.weights = torch.as_tensor(np.sqrt(np.asarray(weights, dtype=np.float64)),
                                           dtype=torch.float32, device=device)
            return
        if kind == "rbf":
            dimension = int(data.shape[1])
            rng = np.random.default_rng(seed_from("cost-rbf", dimension, 0))
            sample = data[: min(512, len(data))]
            median = float(np.median(np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)))
            gamma = 1.0 / max(median, 1e-6)
            self.matrix = torch.as_tensor(rng.normal(scale=gamma * float(scale), size=(64, dimension)),
                                          dtype=torch.float32, device=device)
            self.phase = torch.as_tensor(rng.uniform(0.0, 2.0 * np.pi, size=64),
                                         dtype=torch.float32, device=device)
            return
        if kind == "pca":
            mean = data.mean(0)
            covariance = np.cov(data, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            whitening = eigenvectors / np.sqrt(eigenvalues)[None, :]
            self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
            self.matrix = torch.as_tensor(whitening, dtype=torch.float32, device=device)
            return
        raise ValueError(f"unknown cost: {kind!r}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """EN: map points to the cost feature space. CN: 将点映射到代价特征空间。"""
        if self.kind == "l2":
            return x
        if self.kind == "diag":
            assert self.weights is not None
            return x * self.weights
        if self.kind == "rbf":
            assert self.matrix is not None
            return np.sqrt(2.0 / 64.0) * torch.cos(x @ self.matrix.T + self.phase)
        assert self.mean is not None and self.matrix is not None
        return (x - self.mean) @ self.matrix


def _step(model: VelocityMLP, optimiser: torch.optim.Optimizer, x0: torch.Tensor, x1: torch.Tensor) -> None:
    """EN: one CFM step. CN: 一步 CFM。"""
    t = torch.rand(x0.shape[0], 1, device=x0.device)
    x_t = (1.0 - t) * x0 + t * x1
    loss = ((model(x_t, t) - (x1 - x0)) ** 2).mean()
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()


def _ot_columns(cost: torch.Tensor) -> torch.Tensor:
    """EN: exact OT on a square pool, returned as one column per row. CN: 方池上的精确 OT，返回每行的列。"""
    side = int(cost.shape[0])
    mass = np.full(side, 1.0 / side, dtype=np.float64)
    plan = np.asarray(ot.emd(mass, mass, cost.cpu().numpy(), numItermax=2_000_000))
    columns = plan.argmax(axis=1)
    if np.unique(columns).size != side:
        raise RuntimeError("ot.emd did not return a permutation")
    return torch.from_numpy(columns).to(cost.device)


def train(name: str, seed: int, *, cost_name: str, steps: int, device: str, log: RunLog,
          features: _Features | None = None) -> tuple[VelocityMLP, dict[str, float]]:
    """EN: train one flow with a fixed cost and return the model and its coupling cost. CN: 用固定代价训练一个 flow，返回模型与耦合开销。"""
    data = dataset(name, TRAIN_POINTS, 10001)
    torch.manual_seed(seed_int(name, "cost", seed))
    model = VelocityMLP(int(data.shape[1])).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    features = features or _Features(cost_name, data, device)
    rng = np.random.default_rng(seed_from(name, "cost", seed))
    coupling_seconds = 0.0
    started = time.perf_counter()
    for _ in range(int(steps)):
        x0 = torch.as_tensor(rng.normal(size=(BATCH, data.shape[1])), dtype=torch.float32, device=device)
        x1 = torch.as_tensor(data[rng.integers(0, len(data), size=BATCH)], dtype=torch.float32, device=device)
        coupling_started = time.perf_counter()
        cost = _sqdist(features(x0), features(x1))
        columns = _ot_columns(cost)
        coupling_seconds += time.perf_counter() - coupling_started
        _step(model, optimiser, x0, x1[columns])
    train_seconds = time.perf_counter() - started
    log.line(f"[{name}] {cost_name} s{seed}: train {train_seconds:.1f}s coupling {coupling_seconds:.1f}s")
    return model, {"train_seconds": train_seconds, "coupling_seconds": coupling_seconds}


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="fixed ground-cost screen for toy OT-CFM")
    parser.add_argument("--targets", nargs="+", default=["eight_gaussians", "two_moons"])
    parser.add_argument("--costs", nargs="+", default=list(COSTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--tag", default="cost_screen")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.steps, args.seeds, args.costs = 100, [0], ["l2", "rbf", "pca"]
    out_dir = OUT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "run.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []
    for name in args.targets:
        for cost_name in args.costs:
            for seed in args.seeds:
                model, info = train(name, seed, cost_name=cost_name, steps=args.steps, device=device, log=log)
                for row in evaluate(model, name=name, seed=seed, device=device):
                    records.append(RunRecord(
                        study="flow_matching_cost_learning", geometry=name, arm=f"cost_{cost_name}",
                        k=int(model.dim), m=BATCH, draw=seed,
                        metrics={**row, **info},
                        config={"stage": "cost_screen", "dataset": name, "cost": cost_name, "seed": seed,
                                "steps": args.steps, "batch": BATCH, "sampler": "euler",
                                "nfe": int(row["nfe"]), "device": device, "commit": _commit()},
                        extra={}))
    write_records(out_dir / "records.json", records,
                  provenance={"question": "does a fixed alternative ground cost improve toy OT-CFM?",
                              "coupling": "exact OT on a fresh 256-point pool, as w2_mb",
                              "costs": args.costs, "seeds": args.seeds, "steps": args.steps,
                              "note": "fixed feature costs only; no learned cost", "commit": _commit()})
    log.line(f"cost screen done: {len(records)} records")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
