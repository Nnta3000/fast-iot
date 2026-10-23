"""
CN: **池内 EMD 配对实验**（无 AE、无 Bi-DPM）—— 用标注集拟合 theta，在"隐藏标签"的池里做精确 EMD 配对，
    评估恢复率、最近候选 R@1，以及按置信度（每行**未指派候选**中最小 reduced cost）的 precision@coverage。

    协议：
      - 标注集 `m ∈ {100,300}`，三次不同的抽取；特征 16×16 块均值，掩码 + 标准化，全部**只用标注集**拟合。
      - 池 = 其余配对（标签隐藏）；池内 T1 与池内 T2 做精确 EMD（候选列打乱、跟踪真伙伴、表头打印 oracle）。
      - 臂：learned（regret 求解器）、default（uniform）、variance（`theta_k ∝ 1/Var_labelled(x_k-y_k)`）。
      - 置信度：`reduced_ij = c_ij - (u_i + v_j)`（POT 对偶）；取每行**未被指派**候选中的最小值。
        按置信度升序（最可信在前）累积取前 25/50/75/100% 的行算 precision。

EN: **Pool EMD matching experiment** (no AE, no Bi-DPM) -- fit theta on the labelled set, match the
    label-hidden pool with exact EMD, and report recovery fraction, nearest-candidate R@1 and
    precision-at-coverage using each row's smallest reduced cost over non-assigned candidates.

    Protocol: labelled set `m in {100,300}`, three labelled draws; 16x16 block means, masked and standardised
    on the **labelled set only**; pool = remaining pairs (labels hidden); exact EMD between pool T1 and pool T2
    (columns shuffled, true partner tracked, oracle in the header). Arms: learned (regret solver), default
    (uniform), variance (`theta_k ∝ 1/Var_labelled(x_k-y_k)`). Confidence: `reduced_ij = c_ij - (u_i + v_j)`
    from POT duals; per row the minimum over non-assigned candidates; rows are sorted by ascending confidence
    (most trusted first) and precision is reported at 25/50/75/100% coverage.

用法 / EN: usage
    python -u -m iot_experiments.core.pool_emd --sizes 100 300 --draws 0 1 2
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

COVERAGE = (0.25, 0.50, 0.75, 1.00)


def variance_theta(x_lab: np.ndarray, y_lab: np.ndarray) -> np.ndarray:
    """
    CN: 平凡加权 `theta_k ∝ 1 / Var_labelled(x_k - y_k)`，归一化为和为 1；零方差坐标按最小正方差处理。
    EN: Trivial weighting `theta_k ∝ 1 / Var_labelled(x_k - y_k)`, normalised to sum 1; zero variances are
        floored at the smallest positive variance.
    """
    var = np.var(np.asarray(x_lab) - np.asarray(y_lab), axis=0)
    positive = var[var > 0.0]
    floor = float(positive.min()) if positive.size else 1.0
    var = np.where(var > 0.0, var, floor)
    w = 1.0 / var
    return w / w.sum()


def solve_assignment(cost: np.ndarray, *, seed: int) -> tuple[float, np.ndarray, np.ndarray]:
    """
    CN: 打乱候选列后求精确 EMD（方形、均匀边缘 ⇒ 置换）。返回
        (命中率, 打乱后的代价, 每行指派的列, 每行的最小未指派 reduced cost)。
    EN: Exact EMD after shuffling candidate columns (square, uniform marginals => a permutation). Returns
        (hit rate, assigned column per row, per-row minimum reduced cost over non-assigned candidates).
    """
    h = int(cost.shape[0])
    perm = np.random.default_rng(int(seed)).permutation(int(cost.shape[1]))
    shuffled = np.asarray(cost, dtype=np.float64)[:, perm]
    a = np.ones(h, dtype=np.float64) / h
    plan, log = ot.emd(a, a, shuffled, numItermax=10_000_000, log=True)
    assigned = np.argmax(plan, axis=1)
    true_cols = np.argsort(perm)
    hit = float(np.mean(assigned == true_cols))
    u = np.asarray(log["u"], dtype=np.float64)
    v = np.asarray(log["v"], dtype=np.float64)
    reduced = shuffled - (u[:, None] + v[None, :])
    rows = np.arange(h)
    off = reduced.copy()
    off[rows, assigned] = np.inf
    conf = off.min(axis=1)
    return hit, assigned, conf


def precision_at_coverage(hits: np.ndarray, conf: np.ndarray) -> dict[str, float]:
    """
    CN: 按置信度升序（最可信在前）排列，在覆盖率 25/50/75/100% 处报告命中率（precision）。
    EN: Order rows by ascending confidence (most trusted first) and report the hit rate (precision) at
        25/50/75/100% coverage.
    """
    order = np.argsort(conf, kind="stable")
    out: dict[str, float] = {}
    n = hits.size
    for c in COVERAGE:
        k = max(1, int(round(c * n)))
        out[f"P@{int(c * 100)}"] = float(np.mean(hits[order[:k]]))
    return out


def oracle_check(size: int, *, seed: int = 0) -> float:
    """CN: oracle：对角 0、其余 1，打乱列后命中必须为 1。EN: oracle diagonal-0/off-diagonal-1 hit must be 1."""
    h = int(size)
    cost = np.ones((h, h), dtype=np.float64)
    np.fill_diagonal(cost, 0.0)
    return solve_assignment(cost, seed=seed)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pool EMD matching with confidence")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--eps-r-rel", type=float, default=1e-2)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "pool_emd")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[:1000]
    oracle = oracle_check(1000, seed=0)
    print(f"# EMD oracle (1000x1000, diag 0 / off-diag 1): {oracle:.4f} (must be 1.0)", flush=True)
    if abs(oracle - 1.0) > 1e-12:
        raise SystemExit("EMD scoring is wrong; refusing to report")

    X = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    Y = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    print(f"# grid={args.grid} K_raw={X.shape[1]}", flush=True)

    header = (f"{'m':>4} {'draw':>4} {'arm':>18} {'pool':>5} {'recovered':>9} {'R@1':>8} "
              f"{'P@25':>7} {'P@50':>7} {'P@75':>7} {'P@100':>7} {'LP_s':>7}")
    print(header, flush=True)
    rows: list[dict[str, object]] = []
    for m in args.sizes:
        for draw in args.draws:
            order = np.random.default_rng(int(draw)).permutation(1000)
            lab = np.sort(order[: int(m)])
            pool_idx = np.sort(order[int(m):])
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
            arms = {
                "learned": np.asarray(res.theta, dtype=np.float64),
                "default(uniform)": np.full(int(feats.n_features), 1.0 / feats.n_features),
                "variance": variance_theta(xl, yl),
            }
            pool_feats = DiagonalWeightFeatures(xp, yp)
            h = int(xp.shape[0])
            gg = np.meshgrid(np.arange(h), np.arange(h), indexing="ij")
            for arm, th in arms.items():
                cost = np.asarray(pool_feats.costs(th, gg[0].reshape(-1), gg[1].reshape(-1)),
                                  dtype=np.float64).reshape(h, h)
                hit, assigned, conf = solve_assignment(cost, seed=int(draw))
                hits_vec = (assigned == np.arange(h)).astype(np.float64)
                pc = precision_at_coverage(hits_vec, conf)
                nearest = float(np.mean(np.argmin(cost, axis=1) == np.arange(h)))
                rec = {"m": int(m), "draw": int(draw), "arm": arm, "pool": h,
                       "recovered": float(hit), "R@1": nearest, **pc,
                       "LP_s": float(lp_s) if arm == "learned" else 0.0,
                       "converged": bool(res.converged), "K": int(feats.n_features)}
                rows.append(rec)
                print(f"{m:>4} {draw:>4} {arm:>18} {h:>5} {hit:>9.4f} {nearest:>8.4f} "
                      f"{pc['P@25']:>7.4f} {pc['P@50']:>7.4f} {pc['P@75']:>7.4f} {pc['P@100']:>7.4f} "
                      f"{(lp_s if arm == 'learned' else float('nan')):>7.1f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pool_emd.json").write_text(json.dumps({"oracle": oracle, "rows": rows}, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"[pool-emd] wrote {args.out / 'pool_emd.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
