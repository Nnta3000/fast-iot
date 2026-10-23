"""
CN: **regret-gate 矩阵**：学到的距离 vs 非学习基线 vs 默认（uniform），含机会水平。
    每个 (feature 设置, m, seed) 用**新的划分**（split_seed = seed），held-out 永久固定 400。
    学习臂：`loop_regret`（stage 1，prune=False）；默认臂：uniform theta；对比臂：InfoNCE（有超时）。

EN: **Regret-gate matrix**: learned distance vs non-learned baseline vs default (uniform), with chance level.
    Every (feature setting, m, seed) uses a **fresh split** (split_seed = seed); held-out is always 400.
    Learned arm: `loop_regret` (stage 1, prune=False); default arm: uniform theta; baseline: InfoNCE
    (time-boxed).

用法 / EN: usage
    python -m iot_experiments.core.regret_gate_matrix --sizes 100 300 --seeds 0 1 2
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

from iot import DiagonalWeightFeatures, InverseOTProblem, SimplexTheta  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402

from .real_phase1 import _DATA_ROOT, load_gray  # noqa: E402
from .regret_gate import HELD_OUT, feature_mask, standardise_labelled  # noqa: E402
from .regret_gate_diag import emd_hit, emd_oracle_hit  # noqa: E402


def block_features(image: np.ndarray, grid: int) -> np.ndarray:
    """CN: `grid x grid` 块均值。EN: `grid x grid` block means."""
    step = image.shape[0] // int(grid)
    return image.reshape(int(grid), step, int(grid), step).mean(axis=(1, 3)).reshape(-1)


def rank_stats(cost: np.ndarray) -> tuple[float, float, float]:
    """CN: 对同一置换的目标算 (R@1, R@5, 中位排名)。EN: (R@1, R@5, median rank) vs the identity target."""
    order = np.argsort(cost, axis=1, kind="stable")
    m = cost.shape[0]
    ranks = np.argmax(order == np.arange(m)[:, None], axis=1) + 1
    return float(np.mean(ranks <= 1)), float(np.mean(ranks <= 5)), float(np.median(ranks))


def fit_infonce(x_lab: np.ndarray, y_lab: np.ndarray, *, budget_s: float, steps: int = 400,
                temperature: float = 0.1, lr: float = 0.05) -> np.ndarray | None:
    """
    CN: 非 OT 基线（时间盒）：`theta = softmax(w)`，代价按行均值归一化，**对称** InfoNCE（行与列两个方向），
        梯度用 log-sum-exp 计算（不减 exp 上限）。超出预算或发散则返回 None。
    EN: Non-OT baseline (time-boxed): `theta = softmax(w)`, cost normalised by its mean, **symmetric**
        InfoNCE (row and column directions), gradients via log-sum-exp (no overflow). Returns None if the
        budget is exhausted or the fit diverges.
    """
    m, k = int(x_lab.shape[0]), int(x_lab.shape[1])
    feats = DiagonalWeightFeatures(x_lab, y_lab)
    g = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    phi = np.asarray(feats.features(g[0].reshape(-1), g[1].reshape(-1)),
                     dtype=np.float64).reshape(m, m, k)
    w = np.zeros(k, dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(int(steps)):
        if time.perf_counter() - t0 > float(budget_s):
            return None
        theta = np.exp(w - w.max())
        theta /= theta.sum()
        cost = np.einsum("ijk,k->ij", phi, theta)
        scale = cost.mean()
        if not np.isfinite(scale) or scale <= 0:
            return None
        logits = -cost / (scale * float(temperature))
        # symmetric: rows match columns (identity both ways)
        row_max = logits.max(axis=1, keepdims=True)
        lse_r = row_max + np.log(np.exp(logits - row_max).sum(axis=1, keepdims=True))
        col_max = logits.max(axis=0, keepdims=True)
        lse_c = col_max + np.log(np.exp(logits - col_max).sum(axis=0, keepdims=True))
        p_r = np.exp(logits - lse_r)
        p_c = np.exp(logits - lse_c)
        grad_logit = (p_r + p_c.T) * 0.5
        grad_logit[np.arange(m), np.arange(m)] -= 1.0
        denom = float(temperature) * scale
        grad_theta = np.einsum("ij,ijk->k", grad_logit / denom, phi)
        # theta = softmax(w): dtheta/dw = diag(theta) - theta theta^T
        grad_w = theta * (grad_theta - float(grad_theta @ theta))
        w -= float(lr) * grad_w
        if not np.all(np.isfinite(w)):
            return None
    theta = np.exp(w - w.max())
    theta /= theta.sum()
    return theta if np.all(np.isfinite(theta)) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="regret-gate matrix")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--grids", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--eps-r-rel", type=float, default=1e-2)
    parser.add_argument("--infonce-budget", type=float, default=30.0)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "regret_gate")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[: int(args.n)]
    oracle = emd_oracle_hit(HELD_OUT, seed=0)
    print(f"# EMD oracle check (cost 0 on true pairs, columns shuffled, partner tracked): {oracle:.4f} "
          f"(must be 1.0)", flush=True)
    if abs(oracle - 1.0) > 1e-12:
        raise SystemExit("EMD scoring is wrong; refusing to report an EMD column")

    images = {g: (np.stack([load_gray(_DATA_ROOT / "T1" / p.name) for p in names]), None)
              for g in args.grids}
    records: list[dict[str, object]] = []
    header = (f"{'grid':>4} {'K':>4} {'m':>4} {'seed':>4} {'arm':>22} {'R@1':>7} {'R@5':>7} "
              f"{'median':>7} {'EMD':>7} {'ties':>5} {'LP_s':>6}")
    print(header, flush=True)

    for grid in args.grids:
        x_img, _ = images[grid]
        y_img = np.stack([load_gray(_DATA_ROOT / "T2" / p.name) for p in names])
        X = np.stack([block_features(im, grid) for im in x_img])
        Y = np.stack([block_features(im, grid) for im in y_img])
        n = int(X.shape[0])
        for m in args.sizes:
            for seed in args.seeds:
                order = np.random.default_rng(int(seed)).permutation(n)
                held = np.sort(order[:HELD_OUT])
                pool = order[HELD_OUT:]
                lab = np.sort(np.random.default_rng(1000 + int(seed)).choice(
                    pool, size=int(m), replace=False))
                mm = int(lab.size)
                mask = feature_mask(np.concatenate([X[lab], Y[lab]]))  # labelled only
                for std in (True, False):
                    lab_stack = np.concatenate([X[lab], Y[lab]])
                    ho_stack = np.concatenate([X[held], Y[held]])
                    if std:
                        s, o = standardise_labelled(lab_stack, [X[held], Y[held]])
                        lab_s, ho_s = s, np.concatenate(o, axis=0)
                    else:
                        lab_s, ho_s = lab_stack, ho_stack
                    xl, yl = lab_s[:mm][:, mask], lab_s[mm:][:, mask]
                    xh, yh = ho_s[:held.size][:, mask], ho_s[held.size:][:, mask]

                    feats = DiagonalWeightFeatures(xl, yl)
                    plan = sparse.csr_matrix(
                        (np.full(mm, 1.0 / mm), (np.arange(mm), np.arange(mm))), shape=(mm, mm))
                    problem = InverseOTProblem(features=feats,
                                               theta_set=SimplexTheta(int(feats.n_features)),
                                               plan=plan)
                    t0 = time.perf_counter()
                    res = solve_inverse_ot_regret(
                        problem, prune=False, gamma=int(args.gamma),
                        max_iterations=int(args.max_iterations), eps_r_rel=float(args.eps_r_rel),
                        selection="none", check_pot=False)
                    lp_s = time.perf_counter() - t0
                    theta_fit = np.asarray(res.theta, dtype=np.float64)
                    uni = np.full(int(feats.n_features), 1.0 / feats.n_features)
                    nce = fit_infonce(xl, yl, budget_s=float(args.infonce_budget))

                    ho_feats = DiagonalWeightFeatures(xh, yh)
                    h = int(held.size)
                    gg = np.meshgrid(np.arange(h), np.arange(h), indexing="ij")
                    base_cost = np.asarray(ho_feats.costs(uni, gg[0].reshape(-1), gg[1].reshape(-1)),
                                           dtype=np.float64).reshape(h, h)
                    for arm, th in (("learned(regret)", theta_fit), ("default(uniform)", uni),
                                    ("baseline(InfoNCE)", nce)):
                        if th is None:
                            print(f"{grid:>4} {int(feats.n_features):>4} {m:>4} {seed:>4} {arm:>22} "
                                  f"{'DROPPED (budget/diverged)':>40}", flush=True)
                            records.append({"grid": grid, "K": int(feats.n_features), "m": m,
                                            "seed": seed, "standardised": bool(std), "arm": arm,
                                            "dropped": True})
                            continue
                        cost = np.asarray(ho_feats.costs(th, gg[0].reshape(-1), gg[1].reshape(-1)),
                                          dtype=np.float64).reshape(h, h)
                        r1, r5, med = rank_stats(cost)
                        hit, _tf = emd_hit(cost, seed=int(seed))
                        ties = 1 - np.unique(np.round(cost, 15)).size / cost.size
                        rec = {"grid": grid, "K": int(feats.n_features), "m": m, "seed": seed,
                               "standardised": bool(std), "arm": arm, "R@1": r1, "R@5": r5,
                               "median": med, "EMD": float(hit), "ties": float(ties),
                               "LP_s": float(lp_s) if arm == "learned(regret)" else 0.0,
                               "theta_nonzero": int(np.count_nonzero(th > 1e-12)),
                               "converged": bool(res.converged) if arm == "learned(regret)" else True}
                        records.append(rec)
                        print(f"{grid:>4} {int(feats.n_features):>4} {m:>4} {seed:>4} "
                              f"{arm + ('' if std else ' unstd'):>22} {r1:>7.4f} {r5:>7.4f} "
                              f"{med:>7.1f} {hit:>7.4f} {ties:>5.3f} "
                              f"{(f'{lp_s:.1f}' if arm=='learned(regret)' else '-'):>6}", flush=True)
                    _ = base_cost

    print(f"\nchance: R@1={1.0/HELD_OUT:.4f}  R@5={5.0/HELD_OUT:.4f}  median={HELD_OUT/2:.1f}  "
          f"EMD={1.0/HELD_OUT:.4f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "regret_gate_matrix.json").write_text(
        json.dumps({"emd_oracle": oracle, "records": records}, indent=2) + "\n", encoding="utf-8")
    print(f"[matrix] wrote {args.out / 'regret_gate_matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
