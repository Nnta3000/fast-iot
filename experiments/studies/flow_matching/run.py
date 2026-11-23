"""
EN: **Does the coupling pool size matter for flow matching? A controlled 2-D study.**

    Fixed: training batch 256, an MLP velocity field, the standard CFM loss. Varied: the coupling only. The four
    arms are `independent`, `fixed_random` (fixed noise + fixed random permutation, the control for `global_ot`),
    `ot_pool` (exact OT on a fresh pool of `B` noise and `B` data points every step), and `global_ot` (one exact OT
    plan between the fixed noise set and the whole training set, computed once).

    The diagnostic (`--mode precheck`) compares the pool pairing with the global pairing on the same noise points: the
    fraction of different partners, the mean squared displacement and the cost gap. The training sweep
    (`--mode precompute` then `--mode train`) reports exact W2 at a grid of Euler and RK4 budgets, against the
    finite-sample W2 floor. Plans are precomputed because they do not depend on the network.

    This is a study of **scalable exact forward OT** with HELLO as the eventual solver; it is not an inverse-OT
    experiment and is not evidence for the relaxed-iOT method.

CN: **耦合池规模对 flow matching 是否有影响？受控 2-D 研究。** 固定训练批 256、MLP 速度场、标准 CFM 损失；只变耦合。
    四条臂见英文。诊断（`--mode precheck`）在同一批噪声点上比较池配对与全局配对：伙伴不同比例、均方位移、代价差。
    训练 sweep（先 `--mode precompute` 再 `--mode train`）报告精确 W2、Euler/RK4 预算网格与有限样本 W2 下限。计划与网络
    无关，故预计算。这是**可扩展精确前向 OT**（HELLO 为未来求解器）的研究，不是 inverse-OT 实验，也不是 relaxed-iOT 方法的证据。

    python -m experiments.studies.flow_matching.run --mode precheck
    python -m experiments.studies.flow_matching.run --mode precompute --workers 8
    python -m experiments.studies.flow_matching.run --mode train
"""

from __future__ import annotations

import argparse
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
EVAL_POINTS = 1024
BATCH = 256
HIDDEN = 128
LR = 1e-3
EULER_NFE = (1, 2, 5, 10, 50)
RK4_NFE = (4, 8, 20, 48)
CHEAP_ARMS = (("independent", 0), ("fixed_random", 0), ("ot_pool", 256), ("global_ot", 0))
POOL_ARMS = (("ot_pool", 1024),)


def _arm_name(arm: str, pool: int) -> str:
    """EN: the record/arm label. CN: 记录里的臂名。"""
    return f"{arm}_B{pool}" if arm == "ot_pool" else arm


def _precompute_task(payload: tuple[str, str, int, int, int]) -> str:
    """EN: one precompute job in its own process. CN: 独立进程中的一个预计算任务。"""
    dataset_name, arm, pool, seed, steps = payload
    data = dataset(dataset_name, TRAIN_POINTS, 10001)
    path = build_pairs(dataset=dataset_name, arm=arm, pool=pool, seed=seed, steps=steps,
                       batch=BATCH, data=data, out_root=PAIRS)
    return f"{dataset_name} {arm} B{pool} s{seed} -> {path.name}"


class VelocityMLP(nn.Module):
    """EN: `v_t(x)` on 2-D points with four Fourier time features. CN: 2-D 点上、带四个 Fourier 时间特征的 `v_t(x)`。"""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1 + 8, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """EN: velocity at `(x, t)`, `t` in `[0,1]`. CN: `(x,t)` 处的速度，`t∈[0,1]`。"""
        frequencies = 2.0 * torch.pi * torch.arange(1, 5, device=x.device, dtype=x.dtype)
        phases = t * frequencies[None, :]
        features = torch.cat([t, torch.sin(phases), torch.cos(phases)], dim=1)
        return self.net(torch.cat([x, features], dim=1))


def _train(dataset_name: str, arm: str, pool: int, seed: int, steps: int, data: np.ndarray,
           store: PairStore, device: str) -> tuple[VelocityMLP, float]:
    """EN: train one flow for `steps` CFM steps on precomputed or independent pairs. CN: 在预计算或独立配对上一个 flow 训练 `steps` 步。"""
    torch.manual_seed(seed_int(dataset_name, arm, pool, seed, steps, "model"))
    model = VelocityMLP().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed_from(dataset_name, arm, pool, seed, steps, "train"))
    started = time.perf_counter()
    for step in range(int(steps)):
        if arm == "independent":
            x0 = rng.normal(size=(BATCH, 2))
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
    return model, float(time.perf_counter() - started)


@torch.no_grad()
def _sample(model: VelocityMLP, *, nfe: int, sampler: str, device: str, seed: int) -> np.ndarray:
    """EN: integrate the flow from fresh noise; `nfe` counts function evaluations. CN: 从新噪声积分 flow；`nfe` 为函数求值次数。"""
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(EVAL_POINTS, 2, generator=generator, device=device)
    if sampler == "euler":
        for step in range(int(nfe)):
            t = torch.full((EVAL_POINTS, 1), step / float(nfe), device=device)
            x = x + model(x, t) / float(nfe)
        return x.cpu().numpy()
    if sampler == "rk4":
        if int(nfe) % 4 != 0:
            raise ValueError("RK4 needs a multiple of four function evaluations")
        count = int(nfe) // 4
        dt = 1.0 / float(count)
        for step in range(count):
            t0 = step * dt
            t = torch.full((EVAL_POINTS, 1), t0, device=device)
            mid = torch.full((EVAL_POINTS, 1), t0 + 0.5 * dt, device=device)
            end = torch.full((EVAL_POINTS, 1), t0 + dt, device=device)
            k1 = model(x, t)
            k2 = model(x + 0.5 * dt * k1, mid)
            k3 = model(x + 0.5 * dt * k2, mid)
            k4 = model(x + dt * k3, end)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x.cpu().numpy()
    raise ValueError(f"unknown sampler: {sampler!r}")


def _wasserstein2(first: np.ndarray, second: np.ndarray) -> float:
    """EN: exact 2-Wasserstein for squared Euclidean cost, uniform masses. CN: 平方欧氏代价、均匀质量下的精确 2-Wasserstein。"""
    n, m = len(first), len(second)
    difference = first[:, None, :] - second[None, :, :]
    cost = np.einsum("bcd,bcd->bc", difference, difference)
    value = ot.emd2(np.full(n, 1.0 / n), np.full(m, 1.0 / m), cost, numItermax=5_000_000)
    return float(np.sqrt(value))


def _coverage(samples: np.ndarray) -> tuple[float, float]:
    """EN: fraction within `3σ` of the nearest mode, and fraction of modes holding at least 1 % of the samples. CN: 距离最近模态 `3σ` 内的比例，以及至少占 1 % 样本的模态比例。"""
    centres = mode_centres()
    difference = samples[:, None, :] - centres[None, :, :]
    distance = np.sqrt(np.einsum("bcd,bcd->bc", difference, difference))
    nearest = distance.argmin(axis=1)
    within = float(np.mean(distance.min(axis=1) <= 0.3))
    counts = np.bincount(nearest, minlength=len(centres)) / len(samples)
    return within, float(np.mean(counts >= 0.01))


def _floor(dataset_name: str, held: np.ndarray, draws: int = 3) -> float:
    """EN: the finite-sample W2 floor: fresh true samples against the held-out set. CN: 有限样本 W2 下限：新的真实样本对留出集。"""
    values = [_wasserstein2(dataset(dataset_name, len(held), 30000 + draw), held) for draw in range(int(draws))]
    return float(np.mean(values))


def _precompute(workers: int, steps_long: int, steps_short: int) -> None:
    """EN: build every pair file in parallel; prints one line per job. CN: 并行构建全部配对文件；每个任务打印一行。"""
    payloads = []
    for dataset_name in DATASETS:
        for arm, pool in CHEAP_ARMS:
            for seed in range(10):
                payloads.append((dataset_name, arm, pool, seed, steps_long))
        for arm, pool in POOL_ARMS:
            for seed in range(3):
                payloads.append((dataset_name, arm, pool, seed, steps_short))
    # global/fixed arms cost nothing per step; the plan cost is in the first pass
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        for message in pool.map(_precompute_task, payloads):
            print(message, flush=True)
    print(f"{len(payloads)} jobs in {time.perf_counter() - started:.1f}s", flush=True)


def _precheck_records(log: RunLog) -> list[RunRecord]:
    """EN: run the diagnostic on both targets and write it as records. CN: 在两个目标上运行诊断并写成记录。"""
    rows: list[RunRecord] = []
    for dataset_name in DATASETS:
        data = dataset(dataset_name, TRAIN_POINTS, 10001)
        for item in precheck(dataset=dataset_name, data=data, pools=(256, 1024), draws=20):
            log.line(f"[precheck] {item.dataset} B={item.pool}: differing={item.differing:.3f} "
                     f"msd {item.msd_local:.3f} vs {item.msd_global:.3f} gap={item.cost_gap:+.1%}")
            rows.append(RunRecord(
                study="flow_matching_precheck", geometry=item.dataset, arm=f"pool{item.pool}",
                k=2, m=TRAIN_POINTS, draw=0,
                metrics={"differing": item.differing, "msd_local": item.msd_local,
                         "msd_global": item.msd_global, "cost_gap": item.cost_gap},
                config={"mode": "precheck", "pool": item.pool, "draws": 20, "target": "global_ot"},
                extra={}))
    return rows


def _train_records(log: RunLog, *, steps_long: int, steps_short: int, device: str) -> list[RunRecord]:
    """EN: train and evaluate every (dataset, arm, steps, seed) run and write one row per NFE. CN: 训练并评估每个运行，按 NFE 写一行。"""
    records: list[RunRecord] = []
    for dataset_name in DATASETS:
        data = dataset(dataset_name, TRAIN_POINTS, 10001)
        held = dataset(dataset_name, EVAL_POINTS, 20001)
        floor = _floor(dataset_name, held)
        log.line(f"[{dataset_name}] W2 floor (fresh vs held-out) = {floor:.4f}")
        plans = [
            *[(arm, pool, seed, steps) for arm, pool in CHEAP_ARMS for seed in range(10)
              for steps in (steps_long, steps_short)],
            *[(arm, pool, seed, steps_short) for arm, pool in POOL_ARMS for seed in range(3)],
        ]
        for arm, pool, seed, steps in plans:
            if arm == "independent":
                store = PairStore(arm=arm, seed=seed, dataset=dataset_name, pool=0, batch=BATCH,
                                  plan_seconds=0.0, rows=np.zeros((0, 0)), cols=np.zeros((0, 0)),
                                  fixed_noise=None, permutation=None)
            else:
                store_steps = steps_long if (arm, pool) in CHEAP_ARMS else steps
                path = pair_path(PAIRS, dataset_name, arm, pool, seed, store_steps)
                store = load_store(dataset=dataset_name, arm=arm, pool=pool, seed=seed,
                                   steps=store_steps, batch=BATCH, path=path)
            model, train_seconds = _train(dataset_name, arm, pool, seed, steps, data, store, device)
            share = store.plan_seconds / max(1e-9, store.plan_seconds + train_seconds) if store.plan_seconds else 0.0
            w2_at: list[float] = []
            for sampler, grid in (("euler", EULER_NFE), ("rk4", RK4_NFE)):
                for nfe in grid:
                    samples = _sample(model, nfe=nfe, sampler=sampler, device=device, seed=seed + 1)
                    value = _wasserstein2(samples, held)
                    w2_at.append(value)
                    within, modes = _coverage(samples) if dataset_name == "eight_gaussians" else (float("nan"), float("nan"))
                    records.append(RunRecord(
                        study="flow_matching", geometry=dataset_name, arm=_arm_name(arm, pool),
                        k=2, m=BATCH, draw=seed,
                        metrics={"w2": value, "floor_w2": floor, "coverage": within, "modes_covered": modes,
                                 "train_seconds": train_seconds, "coupling_share": share},
                        config={"steps": steps, "pool": pool, "sampler": sampler, "nfe": nfe,
                                "batch": BATCH, "hidden": HIDDEN, "lr": LR,
                                "arm": arm, "seed": seed, "dataset": dataset_name},
                        extra={"plan_seconds": store.plan_seconds}))
            log.line(f"[{dataset_name}] {_arm_name(arm, pool)} s{seed} t{steps}: "
                     f"train {train_seconds:.1f}s plan {store.plan_seconds:.1f}s "
                     f"w2@euler50={w2_at[-1]:.4f}")
    return records


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="flow matching with coupling pools")
    parser.add_argument("--mode", required=True, choices=("precheck", "precompute", "train"))
    parser.add_argument("--steps-long", type=int, default=10000)
    parser.add_argument("--steps-short", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default="coupling_v1")
    args = parser.parse_args(argv)
    out_dir = OUT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "run.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    if args.mode == "precheck":
        records = _precheck_records(log)
        write_records(out_dir / "precheck.json", records,
                      provenance={"question": "how far is a pool OT pairing from the global pairing, on the same noise?",
                                  "target": "the study's own 2-D targets; not a Gaussian proxy",
                                  "pools": [256, 1024], "draws": 20})
    elif args.mode == "precompute":
        _precompute(int(args.workers), int(args.steps_long), int(args.steps_short))
        records = []
    else:
        records = _train_records(log, steps_long=int(args.steps_long), steps_short=int(args.steps_short),
                                 device=device)
        write_records(out_dir / "records.json", records,
                      provenance={"question": "does the coupling pool size change flow-matching sample quality?",
                                  "family": "2-D eight-Gaussians and two-moons; MLP velocity; CFM loss",
                                  "arms": "independent, fixed_random (control), ot_pool B=256/1024, global_ot",
                                  "batch": BATCH, "steps": [int(args.steps_short), int(args.steps_long)],
                                  "samplers": {"euler": list(EULER_NFE), "rk4": list(RK4_NFE)},
                                  "floor": "exact W2 between fresh true samples and the held-out set",
                                  "solver": "POT exact EMD (CPU); HELLO is not used on this machine"})
    log.line(f"{args.mode} done in {time.perf_counter() - started:.1f}s; {len(records)} records")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
