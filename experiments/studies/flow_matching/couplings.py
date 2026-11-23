"""
EN: **Couplings for flow matching, and the diagnostic that decides what the sweep should show.**

    A coupling is a plan between a noise pool and a data pool. Four arms are implemented:

      `independent`   fresh noise and an i.i.d. data partner every step (no plan);
      `fixed_random`  a fixed set of noise points and a fixed random permutation (the control for `global_ot`);
      `ot_pool`       exact OT (POT) on a fresh pool of `B` noise and `B` data points every step;
      `global_ot`     one exact OT plan between the fixed noise set and the whole training set, computed once.

    Pairs are precomputed because they do not depend on the network: each file stores, per step, the row (noise
    index) and column (data index) of the `batch` training pairs. The pools themselves are regenerated from
    `(seed, step)`, so the store stays small.

    The pre-check measures, on the same noise points, the local pool pairing against the global pairing: the fraction
    of different partners, the mean squared displacement, and the cost gap. It is a diagnostic, not a stop rule.

CN: **flow matching 的耦合，以及决定 sweep 该显示什么的诊断。** 耦合是噪声池与数据池之间的计划。四条臂见英文。
    配对可预计算（与网络无关）：每个文件按步存 `batch` 个训练对的行（噪声下标）与列（数据下标）；池本身由
    `(seed, step)` 重建，故存储很小。预检查在同一批噪声点上比较局部池配对与全局配对：伙伴不同比例、均方位移、代价差。
    它是诊断，不是停止规则。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import ot
import scipy.sparse as sparse
from jaxtyping import Float, Int

from .seeds import seed as seed_from

__all__ = [
    "ARMS",
    "CHECKS",
    "PrecheckRow",
    "PairStore",
    "build_pairs",
    "make_pairs",
    "pair_path",
    "precheck",
    "solve_plan",
]

ARMS: tuple[str, ...] = ("independent", "fixed_random", "ot_pool", "global_ot")
CHECKS: tuple[str, ...] = ("eight_gaussians", "two_moons")
TRAIN_POINTS: int = 8192


def solve_plan(source: Float[np.ndarray, "b d"], target: Float[np.ndarray, "b d"]
               ) -> sparse.csr_matrix:
    """EN: exact OT with uniform marginals and squared Euclidean cost, returned sparse. CN: 均匀边际、平方欧氏代价的精确 OT，返回稀疏计划。"""
    n = int(source.shape[0])
    mass = np.full(n, 1.0 / n, dtype=np.float64)
    difference = source[:, None, :] - target[None, :, :]
    cost = np.einsum("bcd,bcd->bc", difference, difference)
    plan = ot.emd(mass, mass, cost, numItermax=5_000_000)
    return sparse.csr_matrix(np.asarray(plan))


def _sample_columns(plan: sparse.csr_matrix, rows: Int[np.ndarray, "batch"],
                    rng: np.random.Generator) -> Int[np.ndarray, "batch"]:
    """EN: one column per row, drawn from that row's conditional of the plan. CN: 每行按计划的条件分布抽一列。"""
    indptr = np.asarray(plan.indptr, dtype=np.int64)
    indices = np.asarray(plan.indices, dtype=np.int64)
    values = np.asarray(plan.data, dtype=np.float64)
    columns = np.empty(rows.size, dtype=np.int32)
    for position, row in enumerate(rows):
        low, high = int(indptr[row]), int(indptr[row + 1])
        weights = values[low:high]
        columns[position] = indices[low + int(rng.choice(high - low, p=weights / weights.sum()))]
    return columns


def _sample_rows(plan: sparse.csr_matrix, count: int, rng: np.random.Generator) -> Int[np.ndarray, "batch"]:
    """EN: `count` distinct rows (all rows when `count` equals the side). CN: `count` 个不同行（等于边长时取全部）。"""
    side = int(np.asarray(plan.shape).reshape(-1)[0])
    if count >= side:
        return np.asarray(rng.permutation(side), dtype=np.int32)
    return np.asarray(rng.choice(side, size=count, replace=False), dtype=np.int32)


def pair_path(root: Path, dataset: str, arm: str, pool: int, seed: int, steps: int) -> Path:
    """EN: one `.npz` per (dataset, arm, pool, seed, steps). CN: 每个 (dataset, arm, pool, seed, steps) 一个 `.npz`。"""
    return root / f"{dataset}_{arm}_B{pool}_s{seed}_t{steps}.npz"


def load_store(*, dataset: str, arm: str, pool: int, seed: int, steps: int, batch: int,
               path: Path) -> "PairStore":
    """EN: load a precomputed pair file and reconstruct the fixed noise/permutation it indexes. CN: 载入预计算配对文件并重建其索引的固定噪声/置换。"""
    payload = np.load(path)
    fixed_noise = None
    permutation = None
    if arm in ("global_ot", "fixed_random"):
        rng = np.random.default_rng(seed_from(str(dataset), "pairs", "global_ot", int(pool), int(seed)))
        fixed_noise = rng.normal(size=(TRAIN_POINTS, 2))
        if arm == "fixed_random":
            permutation = np.asarray(rng.permutation(TRAIN_POINTS), dtype=np.int64)
    return PairStore(arm=arm, seed=int(seed), dataset=str(dataset), pool=int(pool), batch=int(batch),
                     plan_seconds=float(payload["seconds"]) if "seconds" in payload else 0.0,
                     rows=np.asarray(payload["rows"], dtype=np.int64),
                     cols=np.asarray(payload["cols"], dtype=np.int64),
                     fixed_noise=fixed_noise, permutation=permutation)


def build_pairs(*, dataset: str, arm: str, pool: int, seed: int, steps: int, batch: int,
                data: Float[np.ndarray, "m d"], out_root: Path) -> Path:
    """
    EN: precompute the `(row, column)` pairs of one arm and seed.

        `ot_pool` solves a fresh `pool × pool` OT problem every step; `global_ot` solves one `m × m` problem and
        draws rows from it; `fixed_random` stores the sampled rows of a fixed permutation; `independent` stores
        nothing (the training loop draws fresh noise).

    CN: 预计算一条臂、一个种子的 `(行, 列)` 配对。`ot_pool` 每步解一个新的 `pool × pool` OT；`global_ot` 解一次
        `m × m` 并从其中抽行；`fixed_random` 存固定置换上抽到的行；`independent` 不存（训练循环现抽噪声）。
    """
    out_root.mkdir(parents=True, exist_ok=True)
    path = pair_path(out_root, dataset, arm, pool, seed, steps)
    if path.exists():
        return path
    # EN: `fixed_random` must share the *same* fixed noise set as `global_ot`; only the pairing differs.
    # CN: `fixed_random` 必须与 `global_ot` 共享**同一**固定噪声集；只有配对不同。
    rng = np.random.default_rng(seed_from(str(dataset), "pairs", "global_ot" if arm == "fixed_random" else str(arm),
                                          int(pool), int(seed)))
    if arm == "global_ot":
        fixed_noise = rng.normal(size=(TRAIN_POINTS, 2))
        plan = solve_plan(fixed_noise, data)
    elif arm == "ot_pool":
        plan = None
        fixed_noise = None
    elif arm == "fixed_random":
        plan = None
        fixed_noise = rng.normal(size=(TRAIN_POINTS, 2))
    elif arm == "independent":
        np.savez_compressed(path, rows=np.zeros((0, 0), dtype=np.int32),
                            cols=np.zeros((0, 0), dtype=np.int32), seconds=np.asarray(0.0))
        return path
    else:
        raise ValueError(f"unknown arm: {arm!r}")

    started = time.perf_counter()
    rows = np.empty((steps, batch), dtype=np.int32)
    cols = np.empty((steps, batch), dtype=np.int32)
    for step in range(int(steps)):
        step_rng = np.random.default_rng(seed_from(str(dataset), str(arm), int(pool), int(seed), int(step)))
        if arm == "global_ot":
            assert plan is not None
            picked = _sample_rows(plan, batch, step_rng)
            rows[step] = picked
            cols[step] = _sample_columns(plan, picked, step_rng)
        elif arm == "fixed_random":
            rows[step] = np.asarray(step_rng.choice(TRAIN_POINTS, size=batch, replace=False), dtype=np.int32)
        elif arm == "ot_pool":
            noise = step_rng.normal(size=(pool, 2))
            take = step_rng.choice(len(data), size=pool, replace=False)
            step_plan = solve_plan(noise, data[take])
            picked = _sample_rows(step_plan, batch, step_rng)
            rows[step] = picked
            cols[step] = take[_sample_columns(step_plan, picked, step_rng)]
    np.savez_compressed(path, rows=rows, cols=cols, seconds=np.asarray(time.perf_counter() - started))
    return path


@dataclass(frozen=True)
class PairStore:
    """EN: the precomputed pairs of one run, with the fixed objects they index into. CN: 一次运行的预计算配对及其索引的固定对象。"""

    arm: str
    seed: int
    dataset: str
    pool: int
    rows: Int[np.ndarray, "steps batch"]
    cols: Int[np.ndarray, "steps batch"]
    batch: int
    plan_seconds: float
    fixed_noise: Float[np.ndarray, "m 2"] | None
    permutation: Int[np.ndarray, "m"] | None


def make_pairs(store: PairStore, step: int, data: Float[np.ndarray, "m d"],
               rng: np.random.Generator) -> tuple[Float[np.ndarray, "batch d"], Float[np.ndarray, "batch d"]]:
    """EN: the training pair `(x0, x1)` of one step for a precomputed or independent arm. CN: 一步的训练对 `(x0, x1)`。"""
    batch = int(store.batch)
    if store.arm == "independent":
        noise = rng.normal(size=(batch, 2))
        return noise, data[rng.integers(0, len(data), size=batch)]
    rows, cols = store.rows[step], store.cols[step]
    if store.arm == "ot_pool":
        step_rng = np.random.default_rng(seed_from(str(store.dataset), "ot_pool", int(store.pool), int(store.seed), int(step)))
        noise_pool = step_rng.normal(size=(store.pool, 2))
        return noise_pool[rows], data[cols]
    if store.arm == "fixed_random":
        assert store.fixed_noise is not None and store.permutation is not None
        return store.fixed_noise[rows], data[store.permutation[rows]]
    if store.arm == "global_ot":
        assert store.fixed_noise is not None
        return store.fixed_noise[rows], data[cols]
    raise ValueError(f"unknown arm: {store.arm!r}")


@dataclass(frozen=True)
class PrecheckRow:
    """EN: one diagnostic row: pool size, differing partners, both displacements and the relative cost gap. CN: 一行诊断：池规模、伙伴不同比例、两种位移与相对代价差。"""

    dataset: str
    pool: int
    differing: float
    msd_local: float
    msd_global: float
    cost_gap: float


def precheck(*, dataset: str, data: Float[np.ndarray, "m d"], pools: Iterable[int] = (256, 1024),
             draws: int = 20, seed: int = 0) -> list[PrecheckRow]:
    """
    EN: compare a fresh pool plan with the global plan **on the same noise points**.

        For each draw: pick `pool` noise points and `pool` data points, solve the pool OT, and pair every noise point
        with one partner; take the global partner of the same noise points from the global plan. `differing` is the
        fraction of different data partners, `msd_*` the mean squared displacement of each pairing, and `cost_gap` the
        relative excess of the local pairing over the global one. Measured on the target itself, not on a Gaussian
        proxy.

    CN: 在**同一批噪声点**上比较新池计划与全局计划。每次抽 `pool` 个噪声点与 `pool` 个数据点，解池 OT 并给每个噪声点配一个
        伙伴；同一批噪声点的全局伙伴取自全局计划。`differing` 是伙伴不同的比例，`msd_*` 是两种配对的均方位移，
        `cost_gap` 是局部配对相对全局配对的相对超出。直接在目标上测量，不用高斯代理。
    """
    rng = np.random.default_rng(seed_from(str(dataset), "precheck", int(seed)))
    fixed_noise = rng.normal(size=(TRAIN_POINTS, 2))
    global_plan = solve_plan(fixed_noise, data)
    rows: list[PrecheckRow] = []
    for pool in pools:
        differing, local, global_msd = [], [], []
        for draw in range(int(draws)):
            draw_rng = np.random.default_rng(seed_from(str(dataset), "precheck", int(pool), int(draw), int(seed)))
            noise_index = draw_rng.choice(TRAIN_POINTS, size=int(pool), replace=False)
            data_index = draw_rng.choice(len(data), size=int(pool), replace=False)
            noise = fixed_noise[noise_index]
            step_plan = solve_plan(noise, data[data_index])
            picked = _sample_rows(step_plan, int(pool), draw_rng)
            local_partner = data_index[_sample_columns(step_plan, picked, draw_rng)]
            global_partner = _sample_columns(global_plan, noise_index[picked], draw_rng)
            difference_local = noise[picked] - data[local_partner]
            difference_global = noise[picked] - data[global_partner]
            local.append(float(np.mean(np.einsum("bd,bd->b", difference_local, difference_local))))
            global_msd.append(float(np.mean(np.einsum("bd,bd->b", difference_global, difference_global))))
            differing.append(float(np.mean(local_partner != global_partner)))
        rows.append(PrecheckRow(
            dataset=dataset, pool=int(pool), differing=float(np.mean(differing)),
            msd_local=float(np.mean(local)), msd_global=float(np.mean(global_msd)),
            cost_gap=float((np.mean(local) - np.mean(global_msd)) / np.mean(global_msd)),
        ))
    return rows
