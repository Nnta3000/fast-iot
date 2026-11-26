"""
EN: **Stage 0: coupling headroom on the toy targets.**

    Trains the three new arms (`mac_full`, `teacher_routed_ot`, `reflow_ref`) on 8-Gaussians, two-moons and the
    d = 8 mixture, and evaluates them with the v2 metric. The `independent`, `w2_mb` (B = 256) and `global_ot`
    baselines are read from the v2 records, paired by seed, and are not retrained.

    `mac_full` warms up for 32 independent steps, then uses the current EMA model's prediction error on a fresh
    256-pool, Sinkhorn with `reg = 0.5`. `teacher_routed_ot` trains an independent teacher, builds a bank of 10x the
    data size with a 20-step Euler endpoint, and couples noise to **real** images with POT on `||T(x0)-x1||²`.
    `reflow_ref` trains on `(x0, T(x0))` synthetic pairs and is a reference only.

CN: **Stage 0：玩具目标上的耦合 headroom。** 训练三条新臂（`mac_full`、`teacher_routed_ot`、`reflow_ref`），用 v2 指标评估。
    `independent`、`w2_mb`（B=256）与 `global_ot` 基线直接读 v2 记录并按种子配对，不重训。`mac_full` 先独立热身 32 步，再用
    当前 EMA 模型在 256 池上的预测误差做 Sinkhorn（`reg=0.5`）。`teacher_routed_ot` 训练独立 teacher，建 10×数据量的 bank
    与 20 步 Euler 终点，用 `||T(x0)-x1||²` 的 POT 把噪声配到**真实**图像。`reflow_ref` 用 `(x0,T(x0))` 合成对，仅作参考。

    python -m experiments.studies.flow_matching_headroom.toys --smoke
    python -m experiments.studies.flow_matching_headroom.toys --steps 10000
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from experiments.harness.records import RunLog, RunRecord, write_records
from experiments.studies.flow_matching.data import DATASETS, dataset
from experiments.studies.flow_matching.run import VelocityMLP
from experiments.studies.flow_matching.seeds import seed as seed_from
from experiments.studies.flow_matching.seeds import seed_int
from experiments.studies.flow_matching_headroom.common import (
    BATCH,
    MAC_REG,
    SEEDS,
    TARGETS,
    EMA,
    checkpoint_path,
    evaluate,
    mac_error,
    pot_pairs,
    sinkhorn_pairs,
    target_dim,
    teacher_bank,
)

OUT = Path("experiments/out/flow_matching_headroom")
TEACHER_STEPS = 20
BANK_FACTOR = 10
WARMUP = 32


def _commit() -> str:
    """EN: short git commit of the working tree, recorded in every row. CN: 工作树的短 git commit，写入每行。"""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _step(model: VelocityMLP, optimiser: torch.optim.Optimizer, x0: torch.Tensor, x1: torch.Tensor) -> None:
    """EN: one CFM step. CN: 一步 CFM。"""
    t = torch.rand(x0.shape[0], 1, device=x0.device)
    x_t = (1.0 - t) * x0 + t * x1
    loss = ((model(x_t, t) - (x1 - x0)) ** 2).mean()
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()


def _independent(model: VelocityMLP, optimiser, data: np.ndarray, *, steps: int, seed: int, device: str,
                 rng: np.random.Generator) -> None:
    """EN: `steps` independent-coupling steps. CN: `steps` 步独立耦合。"""
    for _ in range(int(steps)):
        x0 = torch.as_tensor(rng.normal(size=(BATCH, data.shape[1])), dtype=torch.float32, device=device)
        x1 = torch.as_tensor(data[rng.integers(0, len(data), size=BATCH)], dtype=torch.float32, device=device)
        _step(model, optimiser, x0, x1)


def train_mac(name: str, seed: int, *, steps: int, device: str, log: RunLog) -> tuple[VelocityMLP, dict[str, float]]:
    """EN: `mac_full`: one epoch of independent warm-up, then the EMA model's error + Sinkhorn. CN: `mac_full`：一 epoch 独立热身，再用 EMA 误差 + Sinkhorn。"""
    data = dataset(name, 8192, 10001)
    torch.manual_seed(seed_int(name, "mac", seed))
    model = VelocityMLP(target_dim(name)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.999)
    rng = np.random.default_rng(seed_from(name, "mac", seed))
    warmup = min(WARMUP, max(1, int(steps) // 10))
    _independent(model, optimiser, data, steps=warmup, seed=seed, device=device, rng=rng)
    coupling_seconds = 0.0
    for step in range(warmup, int(steps)):
        x0_np = rng.normal(size=(BATCH, data.shape[1]))
        x1_np = data[rng.integers(0, len(data), size=BATCH)]
        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=device)
        x1 = torch.as_tensor(x1_np, dtype=torch.float32, device=device)
        started = time.perf_counter()
        error = mac_error(ema.model, x0, x1)
        rows, columns = sinkhorn_pairs(error, rng, reg=MAC_REG, count=BATCH)
        coupling_seconds += time.perf_counter() - started
        _step(model, optimiser, x0[rows], x1[columns])
        ema.update(model)
    return model, {"coupling_seconds": coupling_seconds, "extra_forward_evals": 2.0,
                   "teacher_seconds": 0.0, "bank_seconds": 0.0, "bank_size": 0.0}


def train_teacher(name: str, seed: int, *, steps: int, device: str, log: RunLog
                  ) -> tuple[VelocityMLP, dict[str, float]]:
    """EN: `teacher_routed_ot`: independent teacher, noise bank with teacher endpoints, POT on `||T(x0)-x1||²` to real data. CN: `teacher_routed_ot`：独立 teacher、带 teacher 终点的噪声 bank、对真实数据用 `||T(x0)-x1||²` 做 POT。"""
    data = dataset(name, 8192, 10001)
    torch.manual_seed(seed_int(name, "teacher", seed))
    teacher = VelocityMLP(target_dim(name)).to(device)
    teacher_opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed_from(name, "teacher", seed))
    started = time.perf_counter()
    _independent(teacher, teacher_opt, data, steps=int(steps), seed=seed, device=device, rng=rng)
    teacher_seconds = time.perf_counter() - started
    bank_size = BANK_FACTOR * len(data)
    noise, endpoints, bank_seconds = teacher_bank(teacher, size=bank_size, steps=TEACHER_STEPS, seed=seed,
                                                  device=device)
    log.line(f"[teacher] {name} s{seed}: teacher {teacher_seconds:.1f}s bank {bank_seconds:.1f}s "
             f"({bank_size} points, {TEACHER_STEPS} Euler)")
    torch.manual_seed(seed_int(name, "student", seed))
    student = VelocityMLP(target_dim(name)).to(device)
    student_opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    coupling_seconds = 0.0
    for _ in range(int(steps)):
        bank_index = rng.choice(bank_size, size=BATCH, replace=False)
        data_index = rng.choice(len(data), size=BATCH, replace=False)
        cost = ((endpoints[bank_index][:, None, :] - data[data_index][None, :, :]) ** 2).sum(-1)
        started = time.perf_counter()
        rows, columns = pot_pairs(cost, rng, count=BATCH)
        coupling_seconds += time.perf_counter() - started
        x0 = torch.as_tensor(noise[bank_index][rows], dtype=torch.float32, device=device)
        x1 = torch.as_tensor(data[data_index][columns], dtype=torch.float32, device=device)
        _step(student, student_opt, x0, x1)
    return student, {"coupling_seconds": coupling_seconds, "extra_forward_evals": 0.0,
                     "teacher_seconds": teacher_seconds, "bank_seconds": bank_seconds,
                     "bank_size": float(bank_size)}


def train_reflow(name: str, seed: int, *, steps: int, device: str, log: RunLog
                 ) -> tuple[VelocityMLP, dict[str, float]]:
    """EN: `reflow_ref`: independent teacher, then CFM on `(x0, T(x0))`; no OT. CN: `reflow_ref`：独立 teacher，再在 `(x0,T(x0))` 上做 CFM；无 OT。"""
    data = dataset(name, 8192, 10001)
    torch.manual_seed(seed_int(name, "reflow", seed))
    teacher = VelocityMLP(target_dim(name)).to(device)
    teacher_opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed_from(name, "reflow", seed))
    started = time.perf_counter()
    _independent(teacher, teacher_opt, data, steps=int(steps), seed=seed, device=device, rng=rng)
    teacher_seconds = time.perf_counter() - started
    bank_size = BANK_FACTOR * len(data)
    noise, endpoints, bank_seconds = teacher_bank(teacher, size=bank_size, steps=TEACHER_STEPS, seed=seed,
                                                  device=device)
    torch.manual_seed(seed_int(name, "reflow-student", seed))
    student = VelocityMLP(target_dim(name)).to(device)
    student_opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    for _ in range(int(steps)):
        index = rng.choice(bank_size, size=BATCH, replace=False)
        x0 = torch.as_tensor(noise[index], dtype=torch.float32, device=device)
        x1 = torch.as_tensor(endpoints[index], dtype=torch.float32, device=device)
        _step(student, student_opt, x0, x1)
    return student, {"coupling_seconds": 0.0, "extra_forward_evals": 0.0,
                     "teacher_seconds": teacher_seconds, "bank_seconds": bank_seconds,
                     "bank_size": float(bank_size)}


ARMS = {"mac_full": train_mac, "teacher_routed_ot": train_teacher, "reflow_ref": train_reflow}


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="Stage 0 coupling headroom")
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag", default="stage0")
    args = parser.parse_args(argv)
    if args.smoke:
        args.steps = 20
        args.seeds = [0]
        args.targets = ["eight_gaussians"]
    # EN: the default stays the pre-registered TARGETS; explicit d=32 is the exploratory extension.
    # CN: 默认仍是预注册的 TARGETS；显式 d=32 是探索性扩展。
    targets = [target for target in args.targets if target in DATASETS]
    arms = [arm for arm in args.arms if arm in ARMS]
    out_dir = OUT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "run.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    commit = _commit()
    records: list[RunRecord] = []
    started = time.perf_counter()
    for name in targets:
        seeds = args.seeds if args.seeds is not None else list(range(SEEDS[name]))
        for arm in arms:
            for seed in seeds:
                run_started = time.perf_counter()
                model, info = ARMS[arm](name, seed, steps=int(args.steps), device=device, log=log)
                train_seconds = time.perf_counter() - run_started
                checkpoint_path(out_dir, name, arm, seed, int(args.steps)).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state": model.state_dict(), "dim": model.dim},
                           checkpoint_path(out_dir, name, arm, seed, int(args.steps)))
                for row in evaluate(model, name=name, seed=seed, device=device):
                    records.append(RunRecord(
                        study="flow_matching_headroom", geometry=name, arm=arm, k=model.dim, m=BATCH, draw=seed,
                        metrics={**row, "train_seconds": train_seconds,
                                 "coupling_seconds": info["coupling_seconds"],
                                 "teacher_seconds": info["teacher_seconds"],
                                 "bank_seconds": info["bank_seconds"],
                                 "extra_forward_evals": info["extra_forward_evals"]},
                        config={"stage": "toys", "dataset": name, "arm": arm, "seed": seed,
                                "steps": int(args.steps), "batch": BATCH, "mac_reg": MAC_REG,
                                "warmup": WARMUP, "teacher_steps": TEACHER_STEPS,
                                "bank_size": info["bank_size"], "sampler": "euler",
                                "nfe": int(row["nfe"]), "device": device, "commit": commit},
                        extra={}))
                log.line(f"[{name}] {arm} s{seed}: train {train_seconds:.1f}s "
                         f"coupling {info['coupling_seconds']:.1f}s teacher {info['teacher_seconds']:.1f}s "
                         f"bank {info['bank_seconds']:.1f}s")
    write_records(out_dir / "records.json", records,
                  provenance={"question": "is there coupling headroom beyond squared-Euclidean OT?",
                              "stage": "toys", "targets": targets, "arms": arms, "steps": int(args.steps),
                              "metric": "sliced W2 (512 projections, 4096 vs fresh 4096)",
                              "baselines": "v2 records, paired by seed",
                              "mac": f"Sinkhorn reg={MAC_REG}, EMA 0.999, {WARMUP}-step warm-up",
                              "teacher": f"independent teacher, bank {BANK_FACTOR}x, {TEACHER_STEPS}-step Euler, POT",
                              "commit": commit})
    log.line(f"stage0 done in {time.perf_counter() - started:.1f}s; {len(records)} records")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
