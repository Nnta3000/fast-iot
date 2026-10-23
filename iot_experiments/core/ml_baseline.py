"""
CN: **基础机器学习基线** —— 在同一设定下（16×16 掩码 + 标准化、只在标注集拟合、池固定 700）比较：
      `learned`（regret 求解器）、`LDA`、`logreg`、`diag-lda`（对角协方差 LDA，即"学习一个正权重"的经典做法）。
    两个监督基线都在**配对差** `d_i = x_i - y_i` 上拟合：LDA/logreg 用"真配对 vs 打乱配对"的两类标签。

    CN: **Baseline ML comparators** under the identical protocol (16x16 masked + standardised, fitted on the
        labelled set only, pool fixed at 700): `learned` (regret solver), `LDA`, `logreg`, and `diag-lda`
        (diagonal-covariance LDA, i.e. the classical "learn positive feature weights" estimator). Both
        supervised baselines are fitted on paired differences `d_i = x_i - y_i` with two classes
        ("true pairing" vs "scrambled pairing").

用法 / EN: usage
    python -u -m iot_experiments.core.ml_baseline --sizes 100 300 --draws 0 1 2
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

from iot import DiagonalWeightFeatures, InverseOTProblem, SimplexTheta  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402

from .real_phase1 import _DATA_ROOT, load_gray  # noqa: E402
from .regret_gate import feature_mask, standardise_labelled  # noqa: E402
from .regret_gate_matrix import block_features  # noqa: E402

POOL = 700


def emd_recovered(cost: np.ndarray, *, seed: int) -> float:
    """CN: 打乱列后的 EMD 命中率（用 argsort(perm) 跟踪真伙伴）。EN: EMD hit with tracked shuffle."""
    h = int(cost.shape[0])
    perm = np.random.default_rng(int(seed)).permutation(int(cost.shape[1]))
    plan = ot.emd(np.ones(h) / h, np.ones(h) / h, np.asarray(cost)[:, perm], numItermax=10_000_000)
    return float(np.mean(np.argmax(plan, axis=1) == np.argsort(perm)))


def supervised_pair_scores(xl: np.ndarray, yl: np.ndarray, *, n_neg: int = 4,
                           seed: int = 0) -> dict[str, np.ndarray]:
    """
    CN: 在配对差上拟合两个监督基线，返回**每对特征**的得分方向（用于构造评分）。
        - `logreg`：真配对（1）vs 打乱配对（0）的 L2 正则逻辑回归。
        - `diag-lda`：对角协方差 LDA 的判别方向 `(mu1-mu0)/var`，等价于经典的"学习正权重"。
    EN: Fit two supervised baselines on paired differences and return per-pair scoring directions.
    """
    d = int(xl.shape[1])
    rng = np.random.default_rng(int(seed))
    pos = np.asarray(xl) - np.asarray(yl)
    negs = []
    for _ in range(int(n_neg)):
        perm = rng.permutation(xl.shape[0])
        while np.any(perm == np.arange(xl.shape[0])):
            perm = rng.permutation(xl.shape[0])
        negs.append(np.asarray(xl) - np.asarray(yl)[perm])
    neg = np.concatenate(negs, axis=0)
    X = np.concatenate([pos, neg], axis=0)
    t = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])])

    # diag-lda
    mu1, mu0 = pos.mean(axis=0), neg.mean(axis=0)
    var = 0.5 * (pos.var(axis=0) + neg.var(axis=0))
    var = np.where(var > 1e-12, var, 1e-12)
    w_lda = (mu1 - mu0) / var

    # logreg (L2, full-batch GD)
    w = np.zeros(d)
    b = 0.0
    for _ in range(400):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - t
        w -= 0.5 * (X.T @ g / X.shape[0] + 1e-3 * w)
        b -= 0.5 * float(g.mean())
    return {"diag-lda": w_lda, "logreg": w}


def compute_cost(xp: np.ndarray, yp: np.ndarray, *, arm: str, w: np.ndarray | None,
                 theta: np.ndarray | None) -> np.ndarray:
    """
    CN: 构造池内代价。`learned` 用对角代价 `sum theta_k (x-y)^2`；监督基线的代价是
        `(x-y)^T W (x-y)` 或线性得分——为可比，统一转换为"越小越好"的代价：
          LDA/logreg 得分 `s = (x-y).w`，代价取 `-s`（单调，仅改变排序，EMD 只依赖序）。
    EN: Build the pool cost. `learned` uses the diagonal cost; the supervised baselines use a linear score
        `s = (x-y).w` and return `-s` (monotone; EMD only depends on the ordering).
    """
    if arm == "learned":
        feats = DiagonalWeightFeatures(xp, yp)
        h = int(xp.shape[0])
        g = np.meshgrid(np.arange(h), np.arange(h), indexing="ij")
        return np.asarray(feats.costs(theta, g[0].reshape(-1), g[1].reshape(-1)),
                          dtype=np.float64).reshape(h, h)
    diff = xp[:, None, :] - yp[None, :, :]
    return -(diff @ w)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="basic ML baselines on the pool task")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--eps-r-rel", type=float, default=1e-2)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "ml_baseline")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[:1000]
    X = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    Y = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    oracle = emd_recovered(np.where(np.eye(50) > 0, 0.0, 1.0), seed=0)
    print(f"# EMD oracle: {oracle:.4f} | grid={args.grid} pool_fixed={POOL}", flush=True)

    print(f"{'m':>4} {'draw':>4} {'arm':>10} {'recovered':>9} {'R@1':>7} {'uniq':>5}", flush=True)
    rows: list[dict[str, object]] = []
    for m in args.sizes:
        for draw in args.draws:
            order = np.random.default_rng(int(draw)).permutation(1000)
            lab = np.sort(order[: int(m)])
            pool_idx = np.sort(order[int(m): int(m) + POOL])
            mask = feature_mask(np.concatenate([X[lab], Y[lab]]))
            s, o = standardise_labelled(np.concatenate([X[lab], Y[lab]]),
                                        [X[pool_idx], Y[pool_idx]])
            xl, yl = s[: int(m)][:, mask], s[int(m):][:, mask]
            xp, yp = o[0][:, mask], o[1][:, mask]

            mm = int(xl.shape[0])
            feats = DiagonalWeightFeatures(xl, yl)
            plan = sparse.csr_matrix((np.full(mm, 1.0 / mm), (np.arange(mm), np.arange(mm))),
                                     shape=(mm, mm))
            problem = InverseOTProblem(features=feats,
                                       theta_set=SimplexTheta(int(feats.n_features)), plan=plan)
            t0 = time.perf_counter()
            res = solve_inverse_ot_regret(problem, prune=False, gamma=int(args.gamma),
                                          max_iterations=int(args.max_iterations),
                                          eps_r_rel=float(args.eps_r_rel), selection="none",
                                          check_pot=False)
            lp_s = time.perf_counter() - t0
            theta = np.asarray(res.theta, dtype=np.float64)
            dirs = supervised_pair_scores(xl, yl, seed=int(draw))

            for arm in ("learned", "diag-lda", "logreg"):
                if arm == "learned":
                    cost = compute_cost(xp, yp, arm=arm, w=None, theta=theta)
                else:
                    cost = compute_cost(xp, yp, arm=arm, w=dirs[arm], theta=None)
                h = int(xp.shape[0])
                rec = emd_recovered(cost, seed=int(draw))
                nearest = np.argmin(cost, axis=1)
                r1 = float(np.mean(nearest == np.arange(h)))
                uniq = int(np.unique(nearest).size)
                rows.append({"m": int(m), "draw": int(draw), "arm": arm, "recovered": rec,
                             "R@1": r1, "uniq_cols": uniq,
                             "LP_s": float(lp_s) if arm == "learned" else 0.0})
                print(f"{m:>4} {draw:>4} {arm:>10} {rec:>9.4f} {r1:>7.4f} {uniq:>5}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"ml_baseline_g{args.grid}.json").write_text(
        json.dumps({"oracle": oracle, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[ml] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
