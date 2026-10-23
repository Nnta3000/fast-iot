"""
CN: **降低代价排序 vs 原始代价排序** —— 在真实 T1/T2 上，比较两种检索读数：
      (a) `cost`          ：按 `c_ij(theta)` 升序排序（普通检索）。
      (b) `reduced_cost`  ：按 `f_i + g_j - c_ij` 降序排序（对偶降低代价，与"最优性证书"同源）。
    读法来自 DPM 的 `runner.py`（`cost_ranks` / `reduced_cost_ranks`），定义与 `compute_native_ranks`
    完全一致，故与 sibling 的 0.394 可直接比较。

EN: **Reduced-cost ranking vs raw-cost ranking** on real T1/T2. Two retrieval read-outs:
      (a) `cost`         -- rank ascending by `c_ij(theta)` (ordinary retrieval).
      (b) `reduced_cost` -- rank descending by `f_i + g_j - c_ij` (the dual reduced cost, the same object the
                           optimality certificate uses).
    The definitions mirror DPM's `runner.py` (`cost_ranks` / `reduced_cost_ranks`) and `compute_native_ranks`,
    so the numbers are directly comparable to the sibling's 0.394.

用法 / EN: usage
    python -m iot_experiments.core.real_reduced_cost --n 1000 --grid 8
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

from iot import InverseOTProblem, SimplexTheta, solve_inverse_ot  # noqa: E402
from iot.backends.cupdlpx_master import CuPdlpxMasterSolver  # noqa: E402

from .real_phase1 import _DATA_ROOT, block_features, cost_matrix, load_gray  # noqa: E402


def native_ranks(score: np.ndarray, *, larger_is_better: bool) -> np.ndarray:
    """
    CN: 1-based native 排名，与 DPM 的 `compute_native_ranks` 同语义（stable argsort）。
    EN: 1-based native rank, same semantics as DPM's `compute_native_ranks` (stable argsort).
    """
    order = (np.argsort(-score, axis=1, kind="stable") if larger_is_better
             else np.argsort(score, axis=1, kind="stable"))
    native = np.arange(score.shape[0])
    return np.argmax(order == native[:, None], axis=1) + 1


def recall(ranks: np.ndarray) -> dict[str, float]:
    """CN: R@k。EN: R@k."""
    return {f"recall_at_{int(k)}": float(np.mean(ranks <= int(k))) for k in (1, 5, 10, 50)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="reduced-cost vs raw-cost ranking on real T1/T2")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--master-time-limit", type=float, default=120.0)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "core" / "out" / "real_reduced_cost")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[: int(args.n)]
    x_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    y_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    pooled = np.concatenate([x_raw, y_raw], axis=0)
    mu = pooled.mean(axis=0, keepdims=True)
    sd = np.where(pooled.std(axis=0, keepdims=True) > 1e-12, pooled.std(axis=0, keepdims=True), 1.0)
    x = (x_raw - mu) / sd
    y = (y_raw - mu) / sd
    m = int(x.shape[0])

    from iot import DiagonalWeightFeatures
    features = DiagonalWeightFeatures(x, y)
    theta_set = SimplexTheta(features.n_features)
    theta_uniform = np.full(features.n_features, 1.0 / features.n_features)
    cost_u = cost_matrix(features, theta_uniform)
    plan = ot.emd(np.ones(m) / m, np.ones(m) / m, cost_u, numItermax=10_000_000)

    master = CuPdlpxMasterSolver(params={"time_sec_limit": float(args.master_time_limit)}, epsilon=1e-4)
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))
    t0 = time.perf_counter()
    result = solve_inverse_ot(
        problem, solver=master, tie_break="regularized", max_iterations=int(args.max_iterations),
        scan_backend="torch", validate=True, prune=False, random_seed=int(args.seed),
    )
    wall = time.perf_counter() - t0
    theta = np.asarray(result.theta, dtype=np.float64)
    cost = cost_matrix(features, theta)

    f = np.asarray(result.source_dual, dtype=np.float64)
    g = np.asarray(result.target_dual, dtype=np.float64)
    reduced = f[:, None] + g[None, :] - cost

    rows = {
        "cost_uniform_theta": recall(native_ranks(cost_u, larger_is_better=False)),
        "cost_fitted_theta": recall(native_ranks(cost, larger_is_better=False)),
        "reduced_cost_fitted_theta": recall(native_ranks(reduced, larger_is_better=True)),
    }
    print(f"[reduced-cost] n={m} T={result.n_iterations} is_optimal={result.is_optimal} "
          f"rel_gap={result.relative_gap} wall={wall:.0f}s", flush=True)
    print(f"{'ranking':>28} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@50':>7}")
    for name, metrics in rows.items():
        print(f"{name:>28} {metrics['recall_at_1']:>7.4f} {metrics['recall_at_5']:>7.4f} "
              f"{metrics['recall_at_10']:>7.4f} {metrics['recall_at_50']:>7.4f}", flush=True)
    print(f"{'chance':>28} {1.0 / m:>7.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": m, "grid": int(args.grid), "seed": int(args.seed),
        "T": int(result.n_iterations), "delta": float(result.delta),
        "is_optimal": None if result.is_optimal is None else bool(result.is_optimal),
        "relative_gap": None if result.relative_gap is None else float(result.relative_gap),
        "wall": float(wall), "chance_r_at_1": 1.0 / m,
        "theta": theta.tolist(), "rows": rows,
    }
    (args.out / f"real_reduced_cost_n{m}.json").write_text(json.dumps(payload, indent=2) + "\n",
                                                           encoding="utf-8")
    print(f"[reduced-cost] wrote {args.out / f'real_reduced_cost_n{m}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
