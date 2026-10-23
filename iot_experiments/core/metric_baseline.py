"""
CN: **正确的机器学习基线：度量学习** —— 与我们的 theta 同族 `d(x,y) = sum_k theta_k (x_k - y_k)^2`，
    但用**判别式目标**拟合，而不是逆 OT。三个基线：

      `ratio`   —— 对角度量学习：最大化 `E_pos[d] / E_neg[d]`（在单纯形上指数参数化，梯度上升）。
                   **与我们的假设类完全相同**，只是目标不同 → 最锐利的对照。
      `nca-diag` —— 对角 NCA：`-log( sum_pos exp(-d) / sum_all exp(-d) )`，梯度上升。
      `lda-sq`  —— 在**平方特征** `Phi_ij = (x_i - y_j)^2` 上做 LDA，取 `w = (mu1-mu0)/var`，
                   截断到非负并归一化 → 经典监督权重，与对角族直接可比。

EN: **Correct ML baselines: metric learning** -- the same hypothesis class as our theta,
    `d(x,y) = sum_k theta_k (x_k - y_k)^2`, fitted with discriminative objectives instead of inverse OT:
      `ratio`   -- diagonal metric learning maximising `E_pos[d] / E_neg[d]` (exponential parameterisation on
                   the simplex, gradient ascent). **Identical hypothesis class, different objective.**
      `nca-diag` -- diagonal NCA on the same cost.
      `lda-sq`  -- LDA over the **squared** features `Phi_ij = (x_i - y_j)^2`, `w = (mu1-mu0)/var`, clipped to
                   be non-negative and normalised.

用法 / EN: usage
    python -u -m iot_experiments.core.metric_baseline --sizes 100 300 --draws 0 1 2
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
    """CN: 打乱列后的 EMD 命中（用 argsort(perm) 跟踪真伙伴）。EN: EMD hit with a tracked shuffle."""
    h = int(cost.shape[0])
    perm = np.random.default_rng(int(seed)).permutation(int(cost.shape[1]))
    plan = ot.emd(np.ones(h) / h, np.ones(h) / h, np.asarray(cost)[:, perm], numItermax=10_000_000)
    return float(np.mean(np.argmax(plan, axis=1) == np.argsort(perm)))


def diag_costs(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """CN: `d_ij = sum_k theta_k (x_ik - y_jk)^2`（分块）。EN: diagonal cost matrix."""
    return np.einsum("ijk,k->ij", (x[:, None, :] - y[None, :, :]) ** 2, np.asarray(theta))


def fit_ratio(xl: np.ndarray, yl: np.ndarray, *, seed: int = 0, steps: int = 600,
              lr: float = 0.05, n_neg: int = 4) -> np.ndarray:
    """
    CN: 对角度量学习：`max_theta E_pos[d] / E_neg[d]`。参数化 `theta = softmax(w)`，梯度上升。
        负样本按行置换采样（每行 4 个随机目标）。
    EN: Diagonal metric learning `max_theta E_pos[d] / E_neg[d]`, `theta = softmax(w)`, gradient ascent with
        row-permuted negatives (4 per row).
    """
    m, k = int(xl.shape[0]), int(xl.shape[1])
    rng = np.random.default_rng(int(seed))
    phi_pos = (xl - yl) ** 2
    negs = []
    for _ in range(int(n_neg)):
        perm = rng.permutation(m)
        negs.append((xl - yl[perm]) ** 2)
    phi_neg = np.concatenate(negs, axis=0)
    w = np.zeros(k)
    for _ in range(int(steps)):
        theta = np.exp(w - w.max())
        theta /= theta.sum()
        pos = phi_pos @ theta
        neg = phi_neg @ theta
        if neg.mean() <= 1e-30:
            break
        # d/dtheta (E_pos/E_neg) = E_pos' /E_neg - E_pos * E_neg'/E_neg^2
        ep, en = phi_pos.mean(0), phi_neg.mean(0)
        mp, mn = float(pos.mean()), float(neg.mean())
        grad = ep / mn - mp * en / (mn ** 2)
        gw = theta * (grad - float(grad @ theta))
        w -= float(lr) * (-gw)     # ascent
    theta = np.exp(w - w.max())
    return theta / theta.sum()


def fit_nca(xl: np.ndarray, yl: np.ndarray, *, seed: int = 0, steps: int = 400,
            lr: float = 0.05) -> np.ndarray:
    """
    CN: 对角 NCA：`-log( exp(-d_ii) / sum_j exp(-d_ij) )`（对角化 softmax），梯度上升。
    EN: Diagonal NCA: `-log( exp(-d_ii) / sum_j exp(-d_ij) )`, gradient ascent.
    """
    m, k = int(xl.shape[0]), int(xl.shape[1])
    phi = (xl[:, None, :] - yl[None, :, :]) ** 2
    w = np.zeros(k)
    idx = np.arange(m)
    for _ in range(int(steps)):
        theta = np.exp(w - w.max())
        theta /= theta.sum()
        c = np.einsum("ijk,k->ij", phi, theta)
        logits = -c
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        grad_cost = -p                                  # dL/dc_ij for -log p_ii
        grad_cost[idx, idx] += 1.0
        c_mean = c.mean() if c.mean() > 0 else 1.0
        gc = grad_cost / c_mean                       # scale-normalised
        grad = np.einsum("ij,ijk->k", gc, phi)
        gw = theta * (grad - float(grad @ theta))
        w -= float(lr) * (-gw)
    theta = np.exp(w - w.max())
    return theta / theta.sum()


def fit_lda_squared(xl: np.ndarray, yl: np.ndarray, *, seed: int = 0, n_neg: int = 4) -> np.ndarray:
    """
    CN: 在平方特征上做 LDA：正类 = 真配对 `(x_i-y_i)^2`，负类 = 打乱配对；`w = (mu1-mu0)/var`，
        截断为非负后归一化（对角族要求 `theta >= 0`）。
    EN: LDA over squared features (positive = true pairs, negative = scrambled); `w = (mu1-mu0)/var`,
        clipped to non-negative and normalised.
    """
    m = int(xl.shape[0])
    rng = np.random.default_rng(int(seed))
    pos = (xl - yl) ** 2
    negs = []
    for _ in range(int(n_neg)):
        perm = rng.permutation(m)
        negs.append((xl - yl[perm]) ** 2)
    neg = np.concatenate(negs, axis=0)
    mu1, mu0 = pos.mean(0), neg.mean(0)
    var = 0.5 * (pos.var(0) + neg.var(0))
    var = np.where(var > 1e-12, var, 1e-12)
    w = np.clip((mu1 - mu0) / var, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(w.size, 1.0 / w.size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="metric-learning baselines")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--eps-r-rel", type=float, default=1e-2)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "metric_baseline")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[:1000]
    X = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    Y = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    oracle = emd_recovered(np.where(np.eye(50) > 0, 0.0, 1.0), seed=0)
    print(f"# EMD oracle {oracle:.4f} | grid={args.grid} pool={POOL}", flush=True)

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

            t0 = time.perf_counter()
            arms = {
                "learned": np.asarray(res.theta, dtype=np.float64),
                "ratio": fit_ratio(xl, yl, seed=int(draw)),
                "nca-diag": fit_nca(xl, yl, seed=int(draw)),
                "lda-sq": fit_lda_squared(xl, yl, seed=int(draw)),
            }
            fit_s = time.perf_counter() - t0

            h = int(xp.shape[0])
            for arm, th in arms.items():
                cost = diag_costs(xp, yp, th)
                rec = emd_recovered(cost, seed=int(draw))
                nearest = np.argmin(cost, axis=1)
                r1 = float(np.mean(nearest == np.arange(h)))
                rows.append({"m": int(m), "draw": int(draw), "arm": arm, "recovered": rec,
                             "R@1": r1, "uniq_cols": int(np.unique(nearest).size),
                             "nz": int(np.count_nonzero(th > 1e-12)),
                             "fit_s": float(lp_s if arm == "learned" else fit_s)})
                print(f"{m:>4} {draw:>4} {arm:>10} {rec:>9.4f} {r1:>7.4f} "
                      f"{int(np.unique(nearest).size):>5}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"metric_baseline_g{args.grid}.json").write_text(
        json.dumps({"oracle": oracle, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[metric] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
