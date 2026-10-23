"""
CN: **配对打分基线（非 OT）** —— 在**平方特征** `phi_ij = (x_i - y_j)^2` 上做线性打分，
    对 = 真配对（1），错 = 打乱配对（0），minibatch SGD 训练。给出两种指派方式：

      `pair+argmax`  —— 每行取分数最优的候选（**无任何指派/OT**）。这是"直接配对、不学代价、不做 OT"的基线。
      `pair+emd`     —— 同一个分数，用精确 EMD 指派（与我们的 `learned` 同一指派，只换目标函数）。

    这样：
      `pair+argmax` vs `learned`  ⇒ 指派（assignment）是否重要；
      `pair+emd`    vs `learned`  ⇒ 在指派固定时，margin 目标 vs 二分类目标是否重要。

    数据量检查：同时报告训练集规模（每 m 只有 m 个正例）与训练/验证损失，以判断是否欠拟合或过拟合。

EN: **Pairwise scoring baseline (non-OT)** -- linear scoring on the **squared** pair features
    `phi_ij = (x_i - y_j)^2`, positives = true pairs, negatives = scrambled pairs, minibatch SGD. Two
    assignment rules: `pair+argmax` (per-row best, **no assignment/OT at all**) and `pair+emd` (same score,
    exact EMD). This separates "does the assignment matter" (vs learned) from "does the margin objective
    matter, holding the assignment fixed".

用法 / EN: usage
    python -u -m iot_experiments.core.pair_score --sizes 100 300 --draws 0 1 2
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
    """CN: 打乱列后的 EMD 命中（argsort(perm) 跟踪）。EN: EMD hit with tracked shuffle."""
    h = int(cost.shape[0])
    perm = np.random.default_rng(int(seed)).permutation(int(cost.shape[1]))
    plan = ot.emd(np.ones(h) / h, np.ones(h) / h, np.asarray(cost)[:, perm], numItermax=10_000_000)
    return float(np.mean(np.argmax(plan, axis=1) == np.argsort(perm)))


def pair_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CN: `phi_ij = (x_i - y_j)^2`，形状 (m, n, K)。EN: squared pair features, shape (m, n, K)."""
    return (np.asarray(x)[:, None, :] - np.asarray(y)[None, :, :]) ** 2


def build_training_set(xl: np.ndarray, yl: np.ndarray, *, negatives_per_row: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 训练集 = 每个真配对（正例）+ `negatives_per_row` 个行内随机错配对（负例）。
        这是"用 m 对标注学到能推广的打分"的全部监督信号——注意每 m 只有 m 个正例。
    EN: training set = each true pair (positive) + `negatives_per_row` random wrong pairs per row (negatives).
        This is the whole supervision available: only m positives for a labelled set of size m.
    """
    m = int(xl.shape[0])
    pos = pair_features(xl, yl)[np.arange(m), np.arange(m)]
    negs = []
    for _ in range(int(negatives_per_row)):
        cols = rng.permutation(m)
        negs.append(pair_features(xl, yl)[np.arange(m), cols])
    neg = np.concatenate(negs, axis=0)
    X = np.concatenate([pos, neg], axis=0)
    t = np.concatenate([np.ones(m), np.zeros(neg.shape[0])])
    return X, t


def fit_linear_scorer(x: np.ndarray, y: np.ndarray, *, epochs: int = 200, batch: int = 256,
                      lr: float = 0.05, l2: float = 1e-4, negatives_per_row: int = 4,
                      seed: int = 0) -> tuple[np.ndarray, dict[str, float]]:
    """
    CN: 线性打分 `s_ij = phi_ij . w + b`，逻辑损失，minibatch SGD（Adam 简化版：动量）。
        返回 (w, 诊断)。诊断含训练损失与满分（训练集上的准确率）。
    EN: linear scorer `s_ij = phi_ij . w + b` with logistic loss, minibatch SGD. Returns (w, diagnostics),
        including the training loss and accuracy.
    """
    rng = np.random.default_rng(int(seed))
    X, t = build_training_set(x, y, negatives_per_row=int(negatives_per_row), rng=rng)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Xs = (X - mu) / sd
    d = int(Xs.shape[1])
    w = np.zeros(d)
    b = 0.0
    m_w = np.zeros(d)
    v_w = np.zeros(d)
    m_b = v_b = 0.0
    n = int(Xs.shape[0])
    t_step = 0
    for _epoch in range(int(epochs)):
        order = rng.permutation(n)
        for start in range(0, n, int(batch)):
            idx = order[start:start + int(batch)]
            xb, tb = Xs[idx], t[idx]
            z = xb @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (p - tb) / xb.shape[0]
            grad_w = xb.T @ g + float(l2) * w
            grad_b = float(g.sum())
            t_step += 1
            m_w = 0.9 * m_w + 0.1 * grad_w
            v_w = 0.999 * v_w + 0.001 * grad_w ** 2
            m_b = 0.9 * m_b + 0.1 * grad_b
            v_b = 0.999 * v_b + 0.001 * grad_b ** 2
            corr1 = 1.0 - 0.9 ** t_step
            corr2 = 1.0 - 0.999 ** t_step
            w -= float(lr) * (m_w / corr1) / (np.sqrt(v_w / corr2) + 1e-8)
            b -= float(lr) * (m_b / corr1) / (np.sqrt(v_b / corr2) + 1e-8)
    z = Xs @ w + b
    loss = float(np.mean(np.logaddexp(0.0, z) - t * z))
    acc = float(np.mean((z > 0) == (t > 0)))
    # 把 w 映射回原始尺度：s = ((phi - mu)/sd) . w = phi . (w/sd) + const
    w_raw = w / sd.reshape(-1)
    return w_raw, {"train_n": float(n), "train_pos": float(x.shape[0]), "train_loss": loss,
                   "train_acc": acc}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pairwise scoring baseline (no OT)")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--eps-r-rel", type=float, default=1e-2)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "pair_score")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[:1000]
    X = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    Y = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    oracle = emd_recovered(np.where(np.eye(50) > 0, 0.0, 1.0), seed=0)
    print(f"# EMD oracle {oracle:.4f} | grid={args.grid} pool={POOL}", flush=True)

    print(f"{'m':>4} {'draw':>4} {'arm':>14} {'recovered':>9} {'R@1':>7} {'uniq':>5} "
          f"{'trainN':>7} {'trAcc':>6} {'trLoss':>7}", flush=True)
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
            res = solve_inverse_ot_regret(problem, prune=False, gamma=int(args.gamma),
                                          max_iterations=int(args.max_iterations),
                                          eps_r_rel=float(args.eps_r_rel), selection="none",
                                          check_pot=False)
            theta = np.asarray(res.theta, dtype=np.float64)

            t0 = time.perf_counter()
            w, diag = fit_linear_scorer(xl, yl, seed=int(draw))
            fit_s = time.perf_counter() - t0

            h = int(xp.shape[0])
            phi_pool = pair_features(xp, yp)
            diag_cost = np.einsum("ijk,k->ij", phi_pool, theta)
            pair_cost = np.einsum("ijk,k->ij", phi_pool, -w)   # lower = better

            for arm, cost in (("learned", diag_cost), ("pair+argmax", pair_cost),
                              ("pair+emd", pair_cost)):
                nearest = np.argmin(cost, axis=1)
                r1 = float(np.mean(nearest == np.arange(h)))
                rec = emd_recovered(cost, seed=int(draw))
                rows.append({"m": int(m), "draw": int(draw), "arm": arm, "recovered": rec,
                             "R@1": r1, "uniq_cols": int(np.unique(nearest).size),
                             **diag, "fit_s": float(fit_s) if arm != "learned" else 0.0})
                print(f"{m:>4} {draw:>4} {arm:>14} {rec:>9.4f} {r1:>7.4f} "
                      f"{int(np.unique(nearest).size):>5} {int(diag['train_n']):>7} "
                      f"{diag['train_acc']:>6.3f} {diag['train_loss']:>7.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"pair_score_g{args.grid}.json").write_text(
        json.dumps({"oracle": oracle, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[pair] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
