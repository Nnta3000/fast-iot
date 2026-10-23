"""
CN: **全问题下的 R@1**：在 100 个标注对上拟合 theta，在**另外 900 个科目**上评估：
    每个未配对的 T1 与全部 1000 个 T2 排序，R@1 = 真实 T2 排第一的比例。
    与池实验的区别：候选集是全体 1000（含真实伙伴），训练/评估科目不相交。

    臂：learned theta（逆 OT）、uniform（默认 L2^2）、variance。
    另报告 R@5 / 中位排名。

EN: **R@1 on the full problem**: fit theta on 100 labelled pairs, evaluate on the **900 other subjects**;
    each unpaired T1 is ranked against all 1000 T2 slices; R@1 = fraction whose true T2 ranks first.
    Unlike the pool experiment the candidate set is all 1000 (true partner included) and train/eval
    subjects are disjoint.

    Arms: learned theta (inverse OT), uniform (plain L2^2), variance; plus R@5 / median rank.

用法 / EN: usage
    python -u -m iot_experiments.core.full_r1 --m 100 --draws 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import DiagonalWeightFeatures, InverseOTProblem, SimplexTheta  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402

from .real_phase1 import _DATA_ROOT, load_gray  # noqa: E402
from .regret_gate import feature_mask, standardise_labelled  # noqa: E402
from .regret_gate_matrix import block_features  # noqa: E402


def variance_theta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CN: `theta_k ∝ 1/Var(x_k-y_k)`。EN: variance-based weights."""
    v = np.var(np.asarray(x) - np.asarray(y), axis=0)
    pos = v[v > 0]
    v = np.where(v > 0, v, float(pos.min()) if pos.size else 1.0)
    w = 1.0 / v
    return w / w.sum()


def rank_metrics(cost: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """
    CN: `cost` 形状 (n_eval, n_cand)，`targets[i]` 是第 i 个 T1 的真实 T2 列。
    EN: `cost` is (n_eval, n_cand); `targets[i]` is the true T2 column for T1 i.
    """
    order = np.argsort(cost, axis=1, kind="stable")
    ranks = np.argmax(order == np.asarray(targets)[:, None], axis=1) + 1
    return {"R@1": float(np.mean(ranks <= 1)), "R@5": float(np.mean(ranks <= 5)),
            "median": float(np.median(ranks))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R@1 on the full problem")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--eps-r-rel", type=float, default=1e-2)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "full_r1")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[:1000]
    X = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    Y = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    print(f"# grid={args.grid} m={args.m} | eval = the other {1000 - args.m} subjects, "
          f"candidates = all 1000 T2 | chance R@1 = {1.0/1000:.4f}", flush=True)

    print(f"{'draw':>4} {'arm':>10} {'n_eval':>6} {'R@1':>8} {'R@5':>8} {'median':>7} {'K':>4}",
          flush=True)
    rows: list[dict[str, object]] = []
    for draw in args.draws:
        order = np.random.default_rng(int(draw)).permutation(1000)
        lab = order[: int(args.m)]
        ev = order[int(args.m):]
        mask = feature_mask(np.concatenate([X[lab], Y[lab]]))   # labelled only
        s, o = standardise_labelled(np.concatenate([X[lab], Y[lab]]), [X[ev], Y])
        xl, yl = s[: int(args.m)][:, mask], s[int(args.m):][:, mask]
        xe, all_y = o[0][:, mask], o[1][:, mask]
        targets = ev  # eval T1 i's true T2 is column ev[i] of the full 1000

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

        arms = {"learned": theta,
                "uniform": np.full(int(feats.n_features), 1.0 / feats.n_features),
                "variance": variance_theta(xl, yl)}
        cost_feats = DiagonalWeightFeatures(xe, all_y)
        n_eval, n_cand = int(xe.shape[0]), int(all_y.shape[0])
        g = np.meshgrid(np.arange(n_eval), np.arange(n_cand), indexing="ij")
        rr, cc = g[0].reshape(-1), g[1].reshape(-1)
        for arm, th in arms.items():
            cost = np.asarray(cost_feats.costs(th, rr, cc), dtype=np.float64).reshape(n_eval, n_cand)
            met = rank_metrics(cost, targets)
            rows.append({"draw": int(draw), "arm": arm, "n_eval": n_eval,
                         "K": int(feats.n_features), **met})
            print(f"{draw:>4} {arm:>10} {n_eval:>6} {met['R@1']:>8.4f} {met['R@5']:>8.4f} "
                  f"{met['median']:>7.1f} {int(feats.n_features):>4}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"full_r1_m{args.m}_g{args.grid}.json").write_text(
        json.dumps({"m": args.m, "grid": args.grid, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[full-r1] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
