"""
CN: **决定性对照：好表示（VAE latent）上，逆 OT 是否还加价值？**
    同一组 latent、同一划分、同一评估（100 标注拟合，其余 900 排序于 1000 候选），比较：
      `cosine+uniform`  —— 固定余弦（DPM 报告的设定：**无标签**）
      `l2+uniform`      —— 固定 L2²（无标签）
      `l2+learned`      —— 逆 OT 学到的 theta（100 个标注）
      `l2+variance`     —— 方差启发式

EN: **Decisive control: does inverse OT still add value on a good representation (VAE latents)?**
    Same latents, same split, same evaluation (fit on 100, rank the other 900 against all 1000):
      `cosine+uniform` -- fixed cosine (the DPM report's setting: **no labels**)
      `l2+uniform`     -- fixed L2^2 (no labels)
      `l2+learned`     -- theta learned by inverse OT from the 100 labelled pairs
      `l2+variance`    -- variance heuristic

用法 / EN: usage
    python -u -m iot_experiments.core.ae_compare --m 100 --draws 0 1 2
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

from .full_r1 import variance_theta  # noqa: E402


def load_latents(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """CN: 读 latent 与 id，按 id 排序。EN: load latents and ids, sorted by id."""
    d = np.load(path)
    order = np.argsort(d["ids"])
    return d["latents"][order].astype(np.float64), d["ids"][order]


def rank_metrics(cost: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """CN: R@1/R@5/中位排名。EN: R@1/R@5/median rank."""
    order = np.argsort(cost, axis=1, kind="stable")
    ranks = np.argmax(order == np.asarray(targets)[:, None], axis=1) + 1
    return {"R@1": float(np.mean(ranks <= 1)), "R@5": float(np.mean(ranks <= 5)),
            "median": float(np.median(ranks))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AE latent: cosine vs learned theta")
    parser.add_argument("--ae-dir", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "ae")
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--eps-r-rel", type=float, default=1e-2)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "ae_compare")
    args = parser.parse_args(argv)

    x, xid = load_latents(args.ae_dir / "latents_T1.npz")
    y, yid = load_latents(args.ae_dir / "latents_T2.npz")
    if not np.array_equal(xid, yid):
        raise SystemExit("T1/T2 latent ids differ")
    n = int(x.shape[0])
    print(f"# latent dim={x.shape[1]} n={n} m={args.m} | chance R@1={1.0/n:.4f}", flush=True)

    print(f"{'draw':>4} {'arm':>16} {'R@1':>8} {'R@5':>8} {'median':>7}", flush=True)
    rows: list[dict[str, object]] = []
    for draw in args.draws:
        order = np.random.default_rng(int(draw)).permutation(n)
        lab, ev = order[: int(args.m)], order[int(args.m):]
        # CN: 标准化只用标注集的合并分布（与块特征实验一致）。
        # EN: standardise using the labelled set's pooled distribution (consistent with the block runs).
        pooled = np.concatenate([x[lab], y[lab]], axis=0)
        mu = pooled.mean(axis=0, keepdims=True)
        sd = pooled.std(axis=0, keepdims=True)
        sd = np.where(sd > 1e-12, sd, 1.0)
        xl, yl = (x[lab] - mu) / sd, (y[lab] - mu) / sd
        xe, ya = (x[ev] - mu) / sd, (y - mu) / sd

        mm, k = int(xl.shape[0]), int(xl.shape[1])
        feats = DiagonalWeightFeatures(xl, yl)
        plan = sparse.csr_matrix((np.full(mm, 1.0 / mm), (np.arange(mm), np.arange(mm))),
                                 shape=(mm, mm))
        problem = InverseOTProblem(features=feats, theta_set=SimplexTheta(k), plan=plan)
        res = solve_inverse_ot_regret(problem, prune=False, gamma=int(args.gamma),
                                      max_iterations=int(args.max_iterations),
                                      eps_r_rel=float(args.eps_r_rel), selection="none",
                                      check_pot=False)
        theta = np.asarray(res.theta, dtype=np.float64)

        ne, nc = int(xe.shape[0]), int(ya.shape[0])
        diff = xe[:, None, :] - ya[None, :, :]
        l2 = (diff ** 2).sum(-1)
        xn = xe / (np.linalg.norm(xe, axis=1, keepdims=True) + 1e-12)
        yn = ya / (np.linalg.norm(ya, axis=1, keepdims=True) + 1e-12)
        cost_cos = 1.0 - xn @ yn.T
        g = np.meshgrid(np.arange(ne), np.arange(nc), indexing="ij")
        arms = {
            "cosine+uniform": cost_cos,
            "l2+uniform": l2,
            "l2+learned": np.einsum("ijk,k->ij", diff ** 2, theta),
            "l2+variance": np.einsum("ijk,k->ij", diff ** 2, variance_theta(xl, yl)),
        }
        for arm, cost in arms.items():
            met = rank_metrics(cost, ev)
            rows.append({"draw": int(draw), "arm": arm, **met})
            print(f"{draw:>4} {arm:>16} {met['R@1']:>8.4f} {met['R@5']:>8.4f} "
                  f"{met['median']:>7.1f}", flush=True)
        _ = g

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"ae_compare_m{args.m}.json").write_text(
        json.dumps({"m": args.m, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[ae-compare] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
