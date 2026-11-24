"""
EN: **Does the coupling pool size matter for flow matching? A controlled study, v2.**

    Fixed: training batch 256, an MLP velocity field, the standard CFM loss. Varied: the coupling only. Arms are
    `independent`, `fixed_random` (fixed noise + fixed permutation, the control for `global_ot`), `ot_pool` at
    B = 256 (and B = 1024 on the 2-D targets only), and `global_ot` (one plan over the whole training set).

    v2 changes requested by the review: the primary metric is the **sliced 2-Wasserstein** (512 projections, 4 096
    samples), with the energy distance as a second, and a **fresh 4 096-point true sample** as the reference; the
    exact 1 024-point W2 is kept only as a bridge to v1, because it is biased below the finite-sample "floor". NFE is
    reported at Euler 1/2/5/10/50 and RK4 4/8/20/48, and the **straightness of the learned trajectories**
    (displacement / path length, 1 = straight) is measured. A d = 8 and a d = 32 Gaussian mixture are added, with
    mode weights, per-mode mean and covariance errors.

    This is scalable exact forward OT with HELLO as the eventual solver; not an inverse-OT experiment.

CN: **耦合池规模对 flow matching 是否有影响？受控研究 v2。** 固定训练批 256、MLP 速度场、标准 CFM 损失；只变耦合。
    按评审意见：主指标改为**切片 2-Wasserstein**（512 个投影、4 096 个样本），能量距离为第二指标，参照为**新的 4 096 点
    真实样本**；精确 1 024 点 W2 只保留为与 v1 的桥接（它有低于有限样本下限的偏差）。NFE 报 Euler 1/2/5/10/50 与
    RK4 4/8/20/48，并测**学到的轨迹的直线度**（位移/路径长度，1 为直线）。新增 d=8 与 d=32 高斯混合，报模态权重、逐模态
    均值与协方差误差。这是可扩展精确前向 OT（HELLO 为未来求解器），不是 inverse-OT 实验。

    python -m experiments.studies.flow_matching.run --mode precompute --workers 4
    python -m experiments.studies.flow_matching.run --mode train
    python -m experiments.studies.flow_matching.run --mode evaluate
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import ot
import torch
from torch import nn

from experiments.harness.records import RunLog, RunRecord, write_records
from experiments.studies.flow_matching.couplings import (
    PairStore,
    build_pairs,
    load_store,
    make_pairs,
    pair_path,
    precheck,
)
from experiments.studies.flow_matching.data import DATASETS, dataset, mode_centres
from experiments.studies.flow_matching.seeds import seed as seed_from
from experiments.studies.flow_matching.seeds import seed_int

OUT = Path("experiments/out/flow_matching")
PAIRS = OUT / "pairs"
TRAIN_POINTS = 8192
EVAL_POINTS = 4096
EXACT_POINTS = 1024
BATCH = 256
HIDDEN = 128
LR = 1e-3
EULER_NFE = (1, 2, 5, 10, 50)
RK4_NFE = (4, 8, 20, 48)
CHEAP_ARMS = (("independent", 0), ("fixed_random", 0), ("ot_pool", 256), ("global_ot", 0))
POOL_ARMS = (("ot_pool", 1024),)
SEEDS = {"eight_gaussians": 10, "two_moons": 10, "mixture_d8": 5, "mixture_d32": 5}
TWO_D = ("eight_gaussians", "two_moons")


def _arm_name(arm: str, pool: int) -> str:
    """EN: the record/arm label. CN: 记录里的臂名。"""
    return f"{arm}_B{pool}" if arm == "ot_pool" else arm


def _checkpoint(root: Path, dataset_name: str, arm: str, pool: int, seed: int, steps: int) -> Path:
    """EN: one checkpoint per run. CN: 每次运行一个检查点。"""
    return root / f"{dataset_name}_{_arm_name(arm, pool)}_s{seed}_t{steps}.pt"


def _precompute_task(payload: tuple[str, str, int, int, int]) -> str:
    """EN: one precompute job in its own process. CN: 独立进程中的一个预计算任务。"""
    dataset_name, arm, pool, seed, steps = payload
    data = dataset(dataset_name, TRAIN_POINTS, 10001)
    path = build_pairs(dataset=dataset_name, arm=arm, pool=pool, seed=seed, steps=steps,
                       batch=BATCH, data=data, out_root=PAIRS)
    return f"{dataset_name} {arm} B{pool} s{seed} -> {path.name}"


class VelocityMLP(nn.Module):
    """EN: `v_t(x)` on `dim`-dimensional points with four Fourier time features. CN: `dim` 维点上、带四个 Fourier 时间特征的 `v_t(x)`。"""

    def __init__(self, dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim + 1 + 8, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, self.dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """EN: velocity at `(x, t)`, `t` in `[0,1]`. CN: `(x,t)` 处的速度，`t∈[0,1]`。"""
        frequencies = 2.0 * torch.pi * torch.arange(1, 5, device=x.device, dtype=x.dtype)
        phases = t * frequencies[None, :]
        features = torch.cat([t, torch.sin(phases), torch.cos(phases)], dim=1)
        return self.net(torch.cat([x, features], dim=1))


def _train(dataset_name: str, arm: str, pool: int, seed: int, steps: int, data: np.ndarray,
           store: PairStore, device: str, checkpoint: Path) -> tuple[VelocityMLP, float]:
    """EN: train one flow for `steps` CFM steps and save its checkpoint. CN: 训练一个 flow `steps` 步并保存检查点。"""
    torch.manual_seed(seed_int(dataset_name, arm, pool, seed, steps, "model"))
    dim = int(data.shape[1])
    model = VelocityMLP(dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed_from(dataset_name, arm, pool, seed, steps, "train"))
    started = time.perf_counter()
    for step in range(int(steps)):
        if arm == "independent":
            x0 = rng.normal(size=(BATCH, dim))
            x1 = data[rng.integers(0, len(data), size=BATCH)]
        else:
            x0, x1 = make_pairs(store, step, data, rng)
        x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
        x1_t = torch.as_tensor(x1, dtype=torch.float32, device=device)
        t = torch.rand(BATCH, 1, device=device)
        x_t = (1.0 - t) * x0_t + t * x1_t
        loss = ((model(x_t, t) - (x1_t - x0_t)) ** 2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "dim": dim}, checkpoint)
    return model, float(time.perf_counter() - started)


@torch.no_grad()
def _rollout(model: VelocityMLP, *, nfe: int, sampler: str, device: str, seed: int,
             points: int = EVAL_POINTS) -> tuple[np.ndarray, float]:
    """EN: integrate from fresh noise; returns the endpoint and the trajectory straightness (displacement / path). CN: 从新噪声积分；返回终点与轨迹直线度（位移/路径）。"""
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(points, model.dim, generator=generator, device=device)
    start = x.clone()
    length = torch.zeros(points, device=device)
    if sampler == "euler":
        for step in range(int(nfe)):
            t = torch.full((points, 1), step / float(nfe), device=device)
            update = model(x, t) / float(nfe)
            x = x + update
            length = length + update.norm(dim=1)
    elif sampler == "rk4":
        if int(nfe) % 4 != 0:
            raise ValueError("RK4 needs a multiple of four function evaluations")
        count = int(nfe) // 4
        dt = 1.0 / float(count)
        for step in range(count):
            t0 = step * dt
            t = torch.full((points, 1), t0, device=device)
            mid = torch.full((points, 1), t0 + 0.5 * dt, device=device)
            end = torch.full((points, 1), t0 + dt, device=device)
            k1 = model(x, t)
            k2 = model(x + 0.5 * dt * k1, mid)
            k3 = model(x + 0.5 * dt * k2, mid)
            k4 = model(x + dt * k3, end)
            update = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            x = x + update
            length = length + update.norm(dim=1)
    else:
        raise ValueError(f"unknown sampler: {sampler!r}")
    displacement = (x - start).norm(dim=1)
    ratio = float((displacement / (length + 1e-12)).mean())
    return x.cpu().numpy(), ratio


def _sliced_w2(first: np.ndarray, second: np.ndarray, *, projections: int, seed: int) -> float:
    """EN: sliced 2-Wasserstein, mean over random 1-D projections. CN: 切片 2-Wasserstein，在随机一维投影上取均值。"""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(int(projections), first.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    first_proj = np.sort(first @ directions.T, axis=0)
    second_proj = np.sort(second @ directions.T, axis=0)
    return float(np.sqrt(np.mean((first_proj - second_proj) ** 2, axis=0)).mean())


def _energy(first: np.ndarray, second: np.ndarray) -> float:
    """EN: energy distance between two samples. CN: 两个样本之间的能量距离。"""

    def _mean_norm(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).mean())

    return float(2.0 * _mean_norm(first, second) - _mean_norm(first, first) - _mean_norm(second, second))


def _wasserstein2(first: np.ndarray, second: np.ndarray) -> float:
    """EN: exact 2-Wasserstein (squared Euclidean, uniform masses), kept as a bridge to v1. CN: 精确 2-Wasserstein（平方欧氏、均匀质量），仅作与 v1 的桥接。"""
    n, m = len(first), len(second)
    difference = first[:, None, :] - second[None, :, :]
    cost = np.einsum("bcd,bcd->bc", difference, difference)
    value = ot.emd2(np.full(n, 1.0 / n), np.full(m, 1.0 / m), cost, numItermax=5_000_000)
    return float(np.sqrt(value))


def _mode_metrics(samples: np.ndarray, dim: int) -> tuple[float, float, float]:
    """EN: mode-weight L1, mean per-mode centre error and mean covariance Frobenius error against `σ²I`. CN: 模态权重 L1、逐模态中心误差均值、相对 `σ²I` 的协方差 Frobenius 误差均值。"""
    centres = mode_centres(dim)
    distance = np.sqrt(((samples[:, None, :] - centres[None, :, :]) ** 2).sum(-1))
    labels = distance.argmin(axis=1)
    weights = np.bincount(labels, minlength=len(centres)) / len(samples)
    weight_l1 = float(np.abs(weights - 1.0 / len(centres)).sum())
    means = np.stack([samples[labels == k].mean(axis=0) if np.any(labels == k) else np.full(dim, np.nan)
                      for k in range(len(centres))])
    mean_error = float(np.nanmean(np.linalg.norm(means - centres, axis=1)))
    covariances = np.stack([np.cov(samples[labels == k].T) if np.sum(labels == k) > dim else np.full((dim, dim), np.nan)
                            for k in range(len(centres))])
    covariance_error = float(np.nanmean(np.linalg.norm(covariances - 0.01 * np.eye(dim), axis=(1, 2))))
    return weight_l1, mean_error, covariance_error


def _precompute(workers: int, steps_long: int, steps_short: int, targets: list[str]) -> None:
    """EN: build every missing pair file in parallel. CN: 并行构建所有缺失的配对文件。"""
    payloads = []
    for dataset_name in targets:
        for arm, pool in CHEAP_ARMS:
            for seed in range(SEEDS[dataset_name]):
                payloads.append((dataset_name, arm, pool, seed, steps_long))
        if dataset_name in TWO_D:
            for arm, pool in POOL_ARMS:
                for seed in range(3):
                    payloads.append((dataset_name, arm, pool, seed, steps_short))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        for message in pool.map(_precompute_task, payloads):
            print(message, flush=True)
    print(f"{len(payloads)} jobs in {time.perf_counter() - started:.1f}s", flush=True)


def _precheck_records(log: RunLog, targets: list[str]) -> list[RunRecord]:
    """EN: run the coupling diagnostic on each target. CN: 在每个目标上运行耦合诊断。"""
    rows: list[RunRecord] = []
    for dataset_name in targets:
        data = dataset(dataset_name, TRAIN_POINTS, 10001)
        for item in precheck(dataset=dataset_name, data=data, pools=(256, 1024), draws=20):
            log.line(f"[precheck] {item.dataset} B={item.pool}: differing={item.differing:.3f} "
                     f"msd {item.msd_local:.3f} vs {item.msd_global:.3f} gap={item.cost_gap:+.1%}")
            rows.append(RunRecord(
                study="flow_matching_precheck", geometry=item.dataset, arm=f"pool{item.pool}",
                k=int(data.shape[1]), m=TRAIN_POINTS, draw=0,
                metrics={"differing": item.differing, "msd_local": item.msd_local,
                         "msd_global": item.msd_global, "cost_gap": item.cost_gap},
                config={"mode": "precheck", "pool": item.pool, "draws": 20}, extra={}))
    return rows


def _train_records(log: RunLog, *, root: Path, steps_long: int, steps_short: int, device: str,
                   targets: list[str]) -> list[RunRecord]:
    """EN: train every run, save its checkpoint, and record the timing. CN: 训练每个运行、保存检查点并记录耗时。"""
    records: list[RunRecord] = []
    for dataset_name in targets:
        data = dataset(dataset_name, TRAIN_POINTS, 10001)
        runs = [(arm, pool, seed, steps) for arm, pool in CHEAP_ARMS
                for seed in range(SEEDS[dataset_name]) for steps in (steps_long, steps_short)]
        if dataset_name in TWO_D:
            runs += [(arm, pool, seed, steps_short) for arm, pool in POOL_ARMS for seed in range(3)]
        for arm, pool, seed, steps in runs:
            if arm == "independent":
                store = PairStore(arm=arm, seed=seed, dataset=dataset_name, pool=0, batch=BATCH,
                                  dim=int(data.shape[1]), plan_seconds=0.0,
                                  rows=np.zeros((0, 0)), cols=np.zeros((0, 0)),
                                  fixed_noise=None, permutation=None)
            else:
                store_steps = steps_long if (arm, pool) in CHEAP_ARMS else steps
                path = pair_path(PAIRS, dataset_name, arm, pool, seed, store_steps)
                store = load_store(dataset=dataset_name, arm=arm, pool=pool, seed=seed,
                                   steps=store_steps, batch=BATCH, dim=int(data.shape[1]), path=path)
            checkpoint = _checkpoint(root, dataset_name, arm, pool, seed, steps)
            _, train_seconds = _train(dataset_name, arm, pool, seed, steps, data, store, device, checkpoint)
            records.append(RunRecord(
                study="flow_matching", geometry=dataset_name, arm=_arm_name(arm, pool),
                k=int(data.shape[1]), m=BATCH, draw=seed,
                metrics={"train_seconds": train_seconds, "plan_seconds": float(store.plan_seconds),
                         "coupling_share": float(store.plan_seconds / max(1e-9, store.plan_seconds + train_seconds))},
                config={"steps": steps, "pool": pool, "arm": arm, "seed": seed, "dataset": dataset_name,
                        "batch": BATCH, "hidden": HIDDEN, "lr": LR},
                extra={"checkpoint": str(checkpoint)}))
            log.line(f"[train] {dataset_name} {_arm_name(arm, pool)} s{seed} t{steps}: "
                     f"{train_seconds:.1f}s train + {store.plan_seconds:.1f}s plan")
    return records


def _evaluate_records(log: RunLog, *, root: Path, train: list[RunRecord], steps_long: int,
                      device: str) -> list[RunRecord]:
    """EN: reload every checkpoint and evaluate the NFE and straightness grid on fresh samples. CN: 重新载入每个检查点，在新样本上评估 NFE 与直线度网格。"""
    floor_by_dataset: dict[str, tuple[float, float, float]] = {}
    for dataset_name in {record.geometry for record in train}:
        first = dataset(dataset_name, EVAL_POINTS, 40001)
        second = dataset(dataset_name, EVAL_POINTS, 40002)
        exact_a = first[:EXACT_POINTS]
        exact_b = second[:EXACT_POINTS]
        floor_by_dataset[dataset_name] = (
            _sliced_w2(first, second, projections=512, seed=1),
            _energy(first[:2048], second[:2048]),
            _wasserstein2(exact_a, exact_b) if dataset_name in TWO_D else -1.0,
        )
    records: list[RunRecord] = []
    for entry in train:
        config = entry.config
        dataset_name = str(config["dataset"])
        arm = str(config["arm"])
        pool, seed, steps = int(str(config["pool"])), int(str(config["seed"])), int(str(config["steps"]))
        dim = int(entry.k)
        checkpoint = _checkpoint(root, dataset_name, arm, pool, seed, steps)
        payload = torch.load(checkpoint, map_location=device)
        model = VelocityMLP(int(payload["dim"])).to(device)
        model.load_state_dict(payload["state"])
        model.eval()
        held = dataset(dataset_name, EXACT_POINTS, 20001)
        fresh = dataset(dataset_name, EVAL_POINTS, 30001 + seed)
        floor_sw2, floor_energy, floor_w2 = floor_by_dataset[dataset_name]
        # EN: straightness once, from the 50-step Euler rollout. CN: 直线度只用 50 步 Euler 轨迹测一次。
        samples50, ratio = _rollout(model, nfe=50, sampler="euler", device=device, seed=seed + 1)
        weight_l1, mean_error, covariance_error = _mode_metrics(samples50, dim)
        sw2_at_ten = float("nan")
        for sampler, grid in (("euler", EULER_NFE), ("rk4", RK4_NFE)):
            for nfe in grid:
                samples, _ = _rollout(model, nfe=nfe, sampler=sampler, device=device, seed=seed + 1)
                metrics = {
                    "sw2": _sliced_w2(samples, fresh, projections=512, seed=seed + 2),
                    "floor_sw2": floor_sw2,
                    "straightness": ratio,
                }
                if sampler == "euler" and nfe == 10:
                    sw2_at_ten = metrics["sw2"]
                if sampler == "euler" and nfe in (1, 5, 50):
                    metrics["energy"] = _energy(samples[:1024], fresh[:1024])
                    metrics["floor_energy"] = floor_energy
                    metrics["mode_weight_l1"] = weight_l1
                    metrics["mode_mean_error"] = mean_error
                    metrics["mode_cov_error"] = covariance_error
                    if dataset_name in TWO_D:
                        metrics["w2_exact"] = _wasserstein2(samples[:EXACT_POINTS], held)
                        metrics["floor_w2"] = floor_w2
                records.append(RunRecord(
                    study="flow_matching", geometry=dataset_name, arm=_arm_name(arm, pool),
                    k=dim, m=BATCH, draw=seed, metrics=metrics,
                    config={"steps": steps, "pool": pool, "arm": arm, "seed": seed, "dataset": dataset_name,
                            "sampler": sampler, "nfe": nfe},
                    extra={}))
        log.line(f"[eval] {dataset_name} {_arm_name(arm, pool)} s{seed} t{steps}: "
                 f"sw2@euler10={sw2_at_ten:.4f} straightness={ratio:.3f}")
    return records


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="flow matching with coupling pools (v2)")
    parser.add_argument("--mode", required=True, choices=("precheck", "precompute", "train", "evaluate"))
    parser.add_argument("--steps-long", type=int, default=10000)
    parser.add_argument("--steps-short", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--targets", nargs="+", default=list(DATASETS))
    parser.add_argument("--tag", default="coupling_v2")
    args = parser.parse_args(argv)
    targets = [target for target in args.targets if target in DATASETS]
    out_dir = OUT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "run.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    if args.mode == "precheck":
        records = _precheck_records(log, targets)
        write_records(out_dir / "precheck.json", records,
                      provenance={"question": "how far is a pool OT pairing from the global pairing?",
                                  "targets": targets, "pools": [256, 1024], "draws": 20})
    elif args.mode == "precompute":
        _precompute(int(args.workers), int(args.steps_long), int(args.steps_short), targets)
        records = []
    elif args.mode == "train":
        records = _train_records(log, root=out_dir, steps_long=int(args.steps_long),
                                 steps_short=int(args.steps_short), device=device, targets=targets)
        write_records(out_dir / "train.json", records,
                      provenance={"question": "train the coupling arms", "targets": targets,
                                  "batch": BATCH, "steps": [int(args.steps_short), int(args.steps_long)],
                                  "solver": "POT exact EMD (CPU); HELLO is not used on this machine"})
    else:
        train_path = out_dir / "train.json"
        train_records = json.loads(train_path.read_text())["records"]
        entries = [RunRecord(study=r["study"], geometry=r["geometry"], arm=r["arm"], k=r["k"], m=r["m"],
                             draw=r["draw"], metrics=r["metrics"], config=r["config"]) for r in train_records]
        records = _evaluate_records(log, root=out_dir, train=entries,
                                    steps_long=int(args.steps_long), device=device)
        write_records(out_dir / "eval.json", records,
                      provenance={"question": "sample quality and straightness at the NFE grid, v2 metric",
                                  "primary_metric": "sliced W2, 512 projections, 4096 generated vs a fresh 4096 true sample",
                                  "secondary": "energy distance (2048), exact W2 (1024) only on 2-D",
                                  "floor": "fresh true sample vs fresh true sample; exact W2 floor at 1024",
                                  "targets": targets})
    log.line(f"{args.mode} done in {time.perf_counter() - started:.1f}s; {len(records)} records")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
