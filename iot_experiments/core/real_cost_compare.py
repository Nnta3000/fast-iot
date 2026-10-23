"""
CN: **真实数据上的"表达力更强的代价"对照** —— 在**同一组** 64 维块特征上，比较
      (1) `diagonal`（L2²）与 `quadratic`（交叉项）的**未学习**得分 R@1；
      (2) 同一族下**逆 OT 学到**的 theta 的 R@1。
    并同时给出"方案 ArgMax 命中率"（这才是 Phase 1 里被我误标为 R@1 的量）。

EN: **More expressive cost on real data** -- on the **same** 64-dim block features, compare
      (1) the untrained (fixed) scores' R@1 for `diagonal` (L2²) vs `quadratic` (cross terms);
      (2) the R@1 of the theta **fitted by inverse OT** under the same family.
    Also reports the plan-argmax hit rate (the quantity I previously mislabelled as R@1 in Phase 1).

CN: R@1 定义（正确版）：对每个源，按**代价** `c_ij(theta)` 排序所有目标，看 native 目标是否排第一。
    这**不是** "EMD 方案的 argmax 是否等于 native"——后者是耦合本身的性质，与代价的检索能力无关。

EN: R@1 (correct definition): for each source, rank all targets by the **cost** `c_ij(theta)` and check
    whether the native target is rank 1. This is **not** "is argmax(EMD plan) the native target", which is a
    property of the coupling and says nothing about the cost's retrieval power.

用法 / EN: usage
    python -m iot_experiments.core.real_cost_compare --n 1000 --grid 8
"""

from __future__ import annotations

import argparse
import json
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
    FeatureMap,
    InverseOTProblem,
    QuadraticFeatures,
    SimplexTheta,
    is_optimal_plan,
    solve_inverse_ot,
)
from iot.backends.cupdlpx_master import CuPdlpxMasterSolver  # noqa: E402

from .real_phase1 import _DATA_ROOT, block_features, load_gray  # noqa: E402


def cost_matrix(features: FeatureMap, theta: np.ndarray) -> np.ndarray:
    """CN: 完整代价矩阵（仅用于读数，n=1000 时 1e6 项可接受）。EN: full cost matrix (read-out only)."""
    m, n = int(features.n_source), int(features.n_target)
    r, c = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    return np.asarray(features.costs(theta, r.reshape(-1), c.reshape(-1)),
                      dtype=np.float64).reshape(m, n)


def r_at_1(cost: np.ndarray) -> float:
    """
    CN: 检索 R@1：每行按**代价**取 argmin，命中 native 置换的比例。
    EN: retrieval R@1: per-row cost argmin vs the native permutation.
    """
    m = cost.shape[0]
    return float(np.mean(np.argmin(cost, axis=1) == np.arange(m)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="diagonal vs quadratic on real block features")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--families", nargs="+", default=["diagonal", "quadratic"])
    parser.add_argument("--plan", default="real", choices=("real", "identity"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--master-time-limit", type=float, default=120.0)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "core" / "out" / "real_cost_compare")
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
    print(f"[cost-compare] n={m} grid={args.grid} families={args.families} plan={args.plan}", flush=True)

    records: list[dict[str, object]] = []
    print(f"{'family':>10} {'K':>6} {'cost':>10} {'R@1':>7} {'planhit':>8} {'T':>4} {'|F|':>6} "
          f"{'F/(m+n)':>8} {'is_opt':>7} {'relgap':>9} {'wall':>6}")
    for family in args.families:
        if family == "diagonal":
            features: FeatureMap = DiagonalWeightFeatures(x, y)
        elif family == "quadratic":
            features = QuadraticFeatures(x, y)
        else:
            raise ValueError(f"unknown family {family!r}")
        theta_set = SimplexTheta(features.n_features)
        unit = np.full(features.n_features, 1.0 / features.n_features)
        cost = cost_matrix(features, unit)

        # CN: 未学习的两个公平基准：均匀 theta（等价 L2² 的平均）与 L2 归一化后的均匀 theta。
        # EN: two untrained baselines: uniform theta (mean L2²) and uniform theta on L2-normalised points.
        train_base = {"uniform": r_at_1(cost)}

        a = np.ones(m) / m
        if args.plan == "real":
            plan = ot.emd(a, a, cost, numItermax=10_000_000)
        else:
            plan = np.zeros_like(cost)
            plan[np.arange(m), np.arange(m)] = 1.0 / m
        plan_hit = float(np.mean(np.argmax(plan, axis=1) == np.arange(m)))

        problem = InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))
        master = CuPdlpxMasterSolver(params={"time_sec_limit": float(args.master_time_limit)},
                                     epsilon=1e-4)
        t0 = time.perf_counter()
        result = solve_inverse_ot(
            problem, solver=master, tie_break="regularized", max_iterations=int(args.max_iterations),
            scan_backend="torch", validate=True, prune=False, random_seed=int(args.seed),
        )
        wall = time.perf_counter() - t0
        theta_fit = np.asarray(result.theta, dtype=np.float64)
        r_fit = r_at_1(cost_matrix(features, theta_fit))
        fmax = max((it.support_size for it in result.iterations), default=result.support_size)
        check = is_optimal_plan(problem, theta_fit)

        rec = {
            "family": family, "K": int(features.n_features), "n": m, "plan": args.plan,
            "r_at_1_uniform": train_base["uniform"], "r_at_1_fitted": r_fit,
            "plan_argmax_hit": plan_hit, "T": int(result.n_iterations),
            "F_max": int(fmax), "F_over_m_plus_n": float(fmax) / float(2 * m),
            "is_optimal": bool(check.is_optimal), "relative_gap": float(check.relative_gap),
            "delta": float(result.delta), "wall": float(wall),
            "theta_fitted": theta_fit.tolist(),
            "theta_uniform_times_K": (theta_fit * features.n_features).tolist(),
        }
        records.append(rec)
        print(f"{family:>10} {rec['K']:>6} {'uniform':>10} {rec['r_at_1_uniform']:>7.4f} "
              f"{plan_hit:>8.3f} {rec['T']:>4} {rec['F_max']:>6} {rec['F_over_m_plus_n']:>8.2f} "
              f"{str(rec['is_optimal']):>7} {rec['relative_gap']:>9.1e} {rec['wall']:>5.0f}s", flush=True)
        print(f"{'':>10} {'':>6} {'fitted':>10} {rec['r_at_1_fitted']:>7.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "real_cost_compare.json").write_text(
        json.dumps({"n": m, "grid": int(args.grid), "seed": int(args.seed), "records": records},
                   indent=2) + "\n", encoding="utf-8")
    print(f"[cost-compare] wrote {args.out / 'real_cost_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
