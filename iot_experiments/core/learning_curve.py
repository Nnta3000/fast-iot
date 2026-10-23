"""
CN: **带标签置换方案的学习曲线** —— 只用 `m` 个**科目不相交**的标注配对拟合 `theta`，
    把 `theta` 冻结后去匹配**另一批科目**的 T1/T2，报告 R@1 / R@5 / 中位排名。

    协议（与讨论定稿一致）：
      - 观测方案：标注的 `m` 对给出 `(1/m)·I`（置换），**只是训练信号**。
      - 代价族：`diagonal`，`theta` 在**概率单纯形**上。
      - 拟合：`solve_inverse_ot`，只在这 `m` 条边组成的方案上。
      - **防泄漏**：特征标准化（以及可选的 PCA）**只在标注集上拟合**，再原样作用到 held-out。
      - 读数：held-out 上按 `c_theta` 排序，报告 R@1 / R@5 / 中位排名；基准是 `theta = uniform`。
      - 同时报告 in-sample 的 plan-hit 与 R@1（健全性检查，**不是**泛化证据）。

EN: **Learning curve from a labelled permutation plan** -- fit `theta` on `m` **subject-disjoint** labelled
    pairs, freeze it, and match a **different** set of subjects; report R@1 / R@5 / median rank.

    Protocol (as agreed):
      - Observed plan: the `m` labelled pairs give `(1/m)·I` (a permutation); it is **only** the training
        signal.
      - Cost family: `diagonal`, `theta` on the **probability simplex**.
      - Fit: `solve_inverse_ot` on that `m`-edge plan only.
      - **No leakage**: feature standardisation (and any PCA) is fitted on the **labelled set only**, then
        applied unchanged to held-out.
      - Read-out: rank held-out targets by `c_theta`; report R@1 / R@5 / median rank, with `theta = uniform`
        as the baseline. In-sample plan-hit and R@1 are reported as a sanity check, **not** as evidence of
        generalisation.

用法 / EN: usage
    python -m iot_experiments.core.learning_curve --sizes 100 300 500 --n 1000 --grid 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import DiagonalWeightFeatures, InverseOTProblem, SimplexTheta, solve_inverse_ot  # noqa: E402
from iot.backends.cupdlpx_master import CuPdlpxMasterSolver  # noqa: E402

from .real_phase1 import _DATA_ROOT, block_features, cost_matrix, load_gray  # noqa: E402


def ranking_scores(cost: np.ndarray) -> tuple[float, float, float]:
    """
    CN: 返回 (R@1, R@5, 中位排名)，其中 native 目标就是同下标（held-out 池内一一对应）。
    EN: Returns (R@1, R@5, median rank); the native target is the same index (a bijection inside the pool).
    """
    order = np.argsort(cost, axis=1, kind="stable")
    native = np.arange(cost.shape[0])
    ranks = np.argmax(order == native[:, None], axis=1) + 1
    return float(np.mean(ranks <= 1)), float(np.mean(ranks <= 5)), float(np.median(ranks))


def standardise(train: np.ndarray, others: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    CN: **只在标注集上**拟合均值/标准差，再作用到其它集合（防止泄漏）。
    EN: Fit mean/std on the **labelled set only**, then apply to the others (prevents leakage).
    """
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (train - mu) / sd, [(other - mu) / sd for other in others]


def fit_theta(x_lab: np.ndarray, y_lab: np.ndarray, *, seed: int, max_iterations: int,
              master_time_limit: float) -> tuple[np.ndarray, dict[str, object]]:
    """
    CN: 在 `m` 条边的置换方案上拟合 `theta`（单纯形）。方案只是训练信号。
    EN: Fit `theta` (simplex) on the `m`-edge permutation plan. The plan is only the training signal.
    """
    features = DiagonalWeightFeatures(x_lab, y_lab)
    m = int(features.n_source)
    theta_set = SimplexTheta(features.n_features)
    plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=plan)
    master = CuPdlpxMasterSolver(params={"time_sec_limit": float(master_time_limit)}, epsilon=1e-4)
    t0 = time.perf_counter()
    result = solve_inverse_ot(
        problem, solver=master, tie_break="regularized", max_iterations=int(max_iterations),
        scan_backend="torch", validate=True, prune=False, random_seed=int(seed),
    )
    info: dict[str, object] = {
        "T": int(result.n_iterations), "delta": float(result.delta),
        "is_optimal": None if result.is_optimal is None else bool(result.is_optimal),
        "relative_gap": None if result.relative_gap is None else float(result.relative_gap),
        "fit_wall": float(time.perf_counter() - t0),
    }
    return np.asarray(result.theta, dtype=np.float64), info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="learning curve from a labelled permutation plan")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300, 500])
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--master-time-limit", type=float, default=120.0)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "core" / "out" / "learning_curve")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[: int(args.n)]
    x_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    y_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    n = int(x_raw.shape[0])
    print(f"[learning-curve] n={n} grid={args.grid} sizes={args.sizes} seeds={args.seeds}", flush=True)

    records: list[dict[str, object]] = []
    header = (f"{'m':>5} {'seed':>4} {'T':>3} {'is_opt':>6} | {'in R@1':>7} {'planhit':>7} | "
              f"{'ho R@1':>7} {'ho R@5':>7} {'ho med':>7} | {'uni R@1':>8} {'uni R@5':>8} {'uni med':>7}")
    print(header, flush=True)
    for seed in args.seeds:
        for m in args.sizes:
            if m >= n:
                raise SystemExit(f"labelled size {m} must be < n={n} to leave a held-out pool")
            rng = np.random.default_rng(int(seed))
            order = rng.permutation(n)
            lab, held = order[: int(m)], order[int(m):]

            # CN: 特征标准化只用标注集的 (T1, T2) 拟合；held-out 用同一组 mu/sd。
            # EN: standardisation fitted on the labelled (T1, T2) only; held-out reuses the same mu/sd.
            lab_stack = np.concatenate([x_raw[lab], y_raw[lab]], axis=0)
            scaled_train, scaled_others = standardise(lab_stack, [x_raw[held], y_raw[held]])
            x_lab, y_lab = scaled_train[: lab.size], scaled_train[lab.size:]
            x_ho, y_ho = scaled_others[0], scaled_others[1]

            theta_fit, info = fit_theta(x_lab, y_lab, seed=int(seed),
                                        max_iterations=args.max_iterations,
                                        master_time_limit=args.master_time_limit)

            # CN: in-sample 健全性：用拟合 theta 求 EMD，看置换是否被选中。
            # EN: in-sample sanity: EMD under the fitted theta; does it select the permutation?
            in_features = DiagonalWeightFeatures(x_lab, y_lab)
            in_cost = cost_matrix(in_features, theta_fit)
            m_lab = int(lab.size)
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(in_cost)
            plan_hit = float(np.mean(cols == np.arange(m_lab)))
            in_r1, _, _ = ranking_scores(in_cost)

            # CN: held-out：冻结 theta，在 held-out 池内按代价排序。
            # EN: held-out: freeze theta and rank inside the held-out pool.
            ho_features = DiagonalWeightFeatures(x_ho, y_ho)
            ho_cost = cost_matrix(ho_features, theta_fit)
            ho_r1, ho_r5, ho_med = ranking_scores(ho_cost)
            uni = np.full(ho_features.n_features, 1.0 / ho_features.n_features)
            u_r1, u_r5, u_med = ranking_scores(cost_matrix(ho_features, uni))

            rec = {
                "m": int(m), "seed": int(seed), "n": n, "held": int(held.size),
                "in_sample_r_at_1": in_r1, "in_sample_plan_hit": plan_hit,
                "heldout_r_at_1": ho_r1, "heldout_r_at_5": ho_r5, "heldout_median_rank": ho_med,
                "uniform_r_at_1": u_r1, "uniform_r_at_5": u_r5, "uniform_median_rank": u_med,
                "theta": theta_fit.tolist(), **info,
            }
            records.append(rec)
            print(f"{m:>5} {rec['seed']:>4} {rec['T']:>3} {str(rec['is_optimal']):>6} | "
                  f"{in_r1:>7.4f} {plan_hit:>7.4f} | {ho_r1:>7.4f} {ho_r5:>7.4f} {ho_med:>7.1f} | "
                  f"{u_r1:>8.4f} {u_r5:>8.4f} {u_med:>7.1f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"learning_curve_n{n}_g{args.grid}.json"
    path.write_text(json.dumps({"n": n, "grid": int(args.grid), "records": records}, indent=2) + "\n",
                    encoding="utf-8")
    print(f"[learning-curve] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
