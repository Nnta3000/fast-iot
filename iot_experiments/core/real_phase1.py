"""
CN: **Phase 1：在真实 T1/T2 数据上学 `theta`，并做"实时"与"打乱"两种方案的对照。**
    一个实例、本地 GPU、串行。

EN: **Phase 1 on real T1/T2 data: learn `theta` and run the real-vs-shuffled control.** One instance at a
    time, local GPU, serial.

CN: 三种方案，同一个特征、同一个求解器、同一套读数：
      `real`  —— 真实 EMD 耦合（native T1→T2 配对）。
      `shuffled` — 把 target 行随机置换后重算 EMD（**同一组点、同一边缘质量、同一边数**，只破坏配对语义）。
      `identity` — 直接按文件名的 native 配对（单位置换），不经 EMD。
    对照 `real` 与 `shuffled` 才能回答："认证到底带不带信息"。

    Phase 1 的预测性读数（held-out R@1）在此**不**计算：本驱动只做认证/难度/可辨识性；
    预测实验需要 held-out 划分，见后续驱动。

EN: Three plans, same features, same solver, same read-outs:
      `real`   -- exact EMD coupling of the native T1->T2 pairs.
      `shuffled` -- randomly permute target rows then re-solve EMD (same points, same marginals, same edge
                    count; only the pairing semantics are destroyed).
      `identity` -- the native name-based pairing (identity permutation), no EMD.
    Only `real` vs `shuffled` answers "does the certificate carry information".

    Predictive read-outs (held-out R@1) are **not** computed here: this driver covers certification,
    difficulty and identifiability only.

用法 / EN: usage
    python -m iot_experiments.core.real_phase1 --n 1000 --grid 8 --plans real shuffled identity
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import ot
from scipy import sparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import (  # noqa: E402
    DiagonalWeightFeatures,
    InverseOTProblem,
    SimplexTheta,
    is_optimal_plan,
    solve_inverse_ot,
)
from iot.cupdlpx_master import CuPdlpxMasterSolver  # noqa: E402

_DATA_ROOT = Path("/home/agent/Documents/dpm-inverse-ot/data/Task_002_MRI_T1_T2/Train")
_IMG = 192


def load_gray(path: Path) -> np.ndarray:
    """
    CN: 用 ImageMagick 读 8-bit 灰度 PNG 为 `_IMG x _IMG` float64 数组。
    EN: Read an 8-bit grayscale PNG into an `_IMG x _IMG` float64 array via ImageMagick.
    """
    raw = subprocess.run(
        ["magick", str(path), "-colorspace", "Gray", "-depth", "8", "gray:-"],
        check=True, capture_output=True,
    ).stdout
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != _IMG * _IMG:
        raise ValueError(f"{path}: expected {_IMG * _IMG} bytes, got {arr.size}")
    return arr.reshape(_IMG, _IMG).astype(np.float64) / 255.0


def block_features(image: np.ndarray, grid: int) -> np.ndarray:
    """CN: `grid x grid` 块均值特征。EN: `grid x grid` block-mean intensity features."""
    step = image.shape[0] // int(grid)
    return image.reshape(int(grid), step, int(grid), step).mean(axis=(1, 3)).reshape(-1)


def cost_matrix(features, theta: np.ndarray) -> np.ndarray:
    """
    CN: 完整代价矩阵（仅用于读数）。EN: Full cost matrix (read-out only).
    """
    m, n = int(features.n_source), int(features.n_target)
    r, c = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    return np.asarray(features.costs(np.asarray(theta, dtype=np.float64), r.reshape(-1), c.reshape(-1)),
                      dtype=np.float64).reshape(m, n)


def build_plan(name: str, cost: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 返回 (方案, 方案里每个源对应的目标) 供 R@1/cost 读数。
        `shuffled`：先对 target 的**行**做随机置换，再在该置换后的点云上求 EMD；
        于是得到的耦合仍是同一点集的合法 EMD，但"配对语义"被破坏。
    EN: Returns (plan, per-source target) for R@1/cost read-outs. `shuffled` permutes the target rows,
        then solves EMD on the permuted cloud, so the coupling is still a valid EMD of the same point set
        while the pairing semantics are destroyed.
    """
    m = cost.shape[0]
    a = np.ones(m) / m
    b = np.ones(cost.shape[1]) / cost.shape[1]
    if name == "identity":
        plan = np.zeros_like(cost)
        plan[np.arange(m), np.arange(m)] = 1.0 / m
        return plan, np.arange(m)
    if name == "real":
        plan = ot.emd(a, b, cost, numItermax=10_000_000)
        return plan, np.argmax(plan, axis=1)
    if name == "independent":
        # CN: 真正破坏配对的对照：对**每一行**独立随机指派一个目标（仍是置换）。
        # EN: the control that genuinely breaks the pairing: assign every row an independent random target
        #     (still a permutation).
        targets = rng.permutation(m)
        plan = np.zeros_like(cost)
        plan[np.arange(m), targets] = 1.0 / m
        return plan, targets
    if name == "shuffled":
        # CN: 注意：置换 target **坐标**是渐近同构的，EMD 会选出同一个置换，因此这不是一个有效对照。
        #     真正的互信息对照见 `independent`。
        # EN: caution -- permuting target *coordinates* is asymptotically isomorphic and EMD returns the same
        #     assignment, so this is not a valid control. Use `independent` for the information test.
        perm = rng.permutation(m)
        plan_perm = ot.emd(a, b, cost[:, perm], numItermax=10_000_000)
        plan = np.zeros_like(plan_perm)
        plan[:, perm] = plan_perm
        return plan, np.argmax(plan, axis=1)
    raise ValueError(f"unknown plan {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 on real T1/T2 data")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--plans", nargs="+", default=["real", "shuffled", "identity"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scan", default="torch", choices=("numpy", "hello", "torch"))
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--master-time-limit", type=float, default=120.0)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "core" / "out" / "real_phase1")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[: int(args.n)]
    x_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    y_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    pooled = np.concatenate([x_raw, y_raw], axis=0)
    mu = pooled.mean(axis=0, keepdims=True)
    sd = pooled.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    x = (x_raw - mu) / sd
    y = (y_raw - mu) / sd
    m = int(x.shape[0])
    print(f"[phase1] n={m} grid={args.grid} K={x.shape[1]} plans={args.plans}", flush=True)

    features = DiagonalWeightFeatures(x, y)
    theta_set = SimplexTheta(features.n_features)
    r_idx, c_idx = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    flat_r, flat_c = r_idx.reshape(-1), c_idx.reshape(-1)
    unit = np.full(features.n_features, 1.0 / features.n_features)
    cost = np.asarray(features.costs(unit, flat_r, flat_c), dtype=np.float64).reshape(m, m)
    rng = np.random.default_rng(int(args.seed))

    records: list[dict[str, object]] = []
    print(f"{'plan':>9} {'nnz':>6} {'T':>4} {'delta':>10} {'maxviol':>9} {'|F|max':>7} {'F/(m+n)':>8} "
          f"{'is_opt':>7} {'relgap':>9} {'R@1':>6} {'wall':>6}")
    for plan_name in args.plans:
        plan, targets = build_plan(str(plan_name), cost, rng)
        nnz = int((plan > 0).sum())
        problem = InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))
        master = CuPdlpxMasterSolver(params={"time_sec_limit": float(args.master_time_limit)},
                                     epsilon=1e-4)
        t0 = time.perf_counter()
        result = solve_inverse_ot(
            problem, solver=master, tie_break="regularized", max_iterations=int(args.max_iterations),
            scan_backend=str(args.scan), validate=True, prune=False, random_seed=int(args.seed),
        )
        wall = time.perf_counter() - t0
        fmax = max((r.support_size for r in result.iterations), default=result.support_size)
        check = is_optimal_plan(problem, np.asarray(result.theta, dtype=np.float64))
        # CN: 检索 R@1 = 按**代价**取 argmin 命中 native；不是 argmax(plan)（那衡量耦合本身，与代价检索力无关）。
        # EN: retrieval R@1 = cost argmin hits the native partner; NOT argmax(plan), which measures the
        #     coupling and says nothing about the cost.
        cost_at_theta = cost_matrix(features, np.asarray(result.theta, dtype=np.float64))
        recall = float(np.mean(np.argmin(cost_at_theta, axis=1) == np.arange(m)))
        rec = {
            "plan": str(plan_name), "n": m, "nnz": nnz, "T": int(result.n_iterations),
            "converged": bool(result.converged), "delta": float(result.delta),
            "delta_over_median_cost": float(result.delta) / float(np.median(cost)),
            "maxviol": float(result.final_max_violation), "F_max": int(fmax),
            "F_over_m_plus_n": float(fmax) / float(2 * m),
            "is_optimal": bool(check.is_optimal), "relative_gap": float(check.relative_gap),
            "support_recall": recall, "wall": float(wall),
            "theta": np.asarray(result.theta, dtype=np.float64).tolist(),
            "iterations": [
                {"i": int(it.index), "delta": float(it.delta), "viol": int(it.violation_count),
                 "added": int(it.added), "F": int(it.support_size)} for it in result.iterations
            ],
        }
        records.append(rec)
        print(f"{rec['plan']:>9} {nnz:>6} {rec['T']:>4} {rec['delta']:>10.3e} {rec['maxviol']:>9.2e} "
              f"{rec['F_max']:>7} {rec['F_over_m_plus_n']:>8.2f} {str(rec['is_optimal']):>7} "
              f"{rec['relative_gap']:>9.1e} {rec['support_recall']:>6.3f} {rec['wall']:>5.0f}s",
              flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"n": m, "grid": int(args.grid), "seed": int(args.seed),
               "data_root": str(_DATA_ROOT), "records": records}
    (args.out / "real_phase1.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[phase1] wrote {args.out / 'real_phase1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
