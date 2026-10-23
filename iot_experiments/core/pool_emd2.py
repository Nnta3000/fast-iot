"""
CN: **池内 EMD 配对（修正版）** —— 修复了 precision 计分（错用打乱列索引），加入两项可信度：
      (a) 互最近邻一致（mutual NN agreement）；(b) 行/列裕度 `min(best alternative in row, ... in column)`。
    同时固定池大小（pool = 700），使 m 与池大小不再混杂；并统计**逐行最近候选**选中的不同列数（hubness）。

    协议：
      - 用 1000 对中的 `m ∈ {100,300}` 作标注集（3 次抽取），其余 **700** 对作为池，标签隐藏。
      - 16×16 块均值、掩码 + 标准化，全部只在标注集上拟合。
      - 池内 T1 与池内 T2 做精确 EMD（T1 侧也取 700 行，使方阵），候选列打乱、跟踪真伙伴。
      - 臂：learned / default(uniform) / variance。
      - `P@k` 按可信度**升序**（最可信在前）取前 k% 的平均命中；**断言 `P@100 == recovered`**。

EN: **Pool EMD matching (corrected)** -- fixes the precision scoring (it used shuffled column indices), adds two
    confidence measures: (a) mutual nearest-neighbour agreement and (b) the row/column margin to the best
    alternative. The pool is fixed at 700 so that pool size does not confound m; the number of distinct columns
    chosen by the per-row nearest-candidate rule is reported (hubness).

    Protocol: labelled set of `m in {100,300}` from the 1000 pairs (3 draws); the remaining **700** pairs form
    the pool with labels hidden; 16x16 block means, masked and standardised on the labelled set only; exact EMD
    on the pool (700 rows on the T1 side too, so the problem is square), candidate columns shuffled with the
    true partner tracked; arms learned / default / variance; `P@k` averages hits over the most trusted k% of
    rows, with **`P@100 == recovered` asserted**.

用法 / EN: usage
    python -u -m iot_experiments.core.pool_emd2 --sizes 100 300 --draws 0 1 2
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
COVERAGE = (0.25, 0.50, 0.75, 1.00)


def variance_theta(x_lab: np.ndarray, y_lab: np.ndarray) -> np.ndarray:
    """CN: `theta_k ∝ 1/Var_labelled(x_k-y_k)`，归一化。EN: `theta_k ∝ 1/Var(x-y)` on labelled pairs."""
    var = np.var(np.asarray(x_lab) - np.asarray(y_lab), axis=0)
    positive = var[var > 0.0]
    floor = float(positive.min()) if positive.size else 1.0
    var = np.where(var > 0.0, var, floor)
    w = 1.0 / var
    return w / w.sum()


def emd_tracked(cost: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 打乱候选列后求精确 EMD，返回 (每行的命中布尔, 每行指派的**打乱后**列号)。
        命中判定用 `argsort(perm)`——**不是** `arange`（这是上一版的计分错误）。
    EN: Exact EMD after shuffling candidate columns; returns (per-row hit mask, assigned **shuffled** column).
        Hits compare against `argsort(perm)`, **not** `arange` (the previous scoring bug).
    """
    h = int(cost.shape[0])
    perm = np.random.default_rng(int(seed)).permutation(int(cost.shape[1]))
    shuffled = np.asarray(cost, dtype=np.float64)[:, perm]
    a = np.ones(h, dtype=np.float64) / h
    plan = ot.emd(a, a, shuffled, numItermax=10_000_000)
    assigned = np.argmax(plan, axis=1)
    true_cols = np.argsort(perm)
    return (assigned == true_cols), assigned


def mutual_nn(cost: np.ndarray) -> np.ndarray:
    """
    CN: 互最近邻：行 argmin = j 且 列 argmin = i 才算一致（返回布尔）。不做列打乱（与索引无关）。
    EN: Mutual NN: row argmin == j and column argmin == i. Index-independent (no shuffling needed).
    """
    row_best = np.argmin(cost, axis=1)
    col_best = np.argmin(cost, axis=0)
    return col_best[row_best] == np.arange(cost.shape[0])


def margin_confidence(cost: np.ndarray) -> np.ndarray:
    """
    CN: 裕度可信度 = `min(行内最优与次优之差, 列内最优与次优之差)`；越大越可信。
    EN: Margin confidence = `min(row gap between best and 2nd best, same within the column)`; larger is
        more trustworthy.
    """
    row_sorted = np.sort(cost, axis=1)
    col_sorted = np.sort(cost, axis=0)
    row_gap = row_sorted[:, 1] - row_sorted[:, 0]
    col_gap = col_sorted[1, :] - col_sorted[0, :]
    return np.minimum(row_gap, col_gap)


def precision_at_coverage(hits: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """
    CN: `score` 越大越可信；按降序取前 25/50/75/100% 的平均命中。`P@100` 必须等于 `hits.mean()`。
    EN: `score` larger = more trustworthy; average hits over the top 25/50/75/100%. `P@100` must equal
        `hits.mean()`.
    """
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    out: dict[str, float] = {}
    for c in COVERAGE:
        k = max(1, int(round(c * hits.size)))
        out[f"P@{int(round(c * 100))}"] = float(np.mean(np.asarray(hits, dtype=np.float64)[order[:k]]))
    if abs(out["P@100"] - float(np.mean(hits))) > 1e-12:
        raise AssertionError("P@100 must equal the overall hit rate")
    return out


def oracle_check(size: int, *, seed: int = 0) -> float:
    """CN: oracle 命中必须为 1，且 P@100 相等。EN: oracle hit must be 1 and P@100 must match."""
    h = int(size)
    cost = np.ones((h, h), dtype=np.float64)
    np.fill_diagonal(cost, 0.0)
    hits, _ = emd_tracked(cost, seed=seed)
    pc = precision_at_coverage(hits, margin_confidence(cost))
    if abs(pc["P@100"] - hits.mean()) > 1e-12:
        raise AssertionError("oracle P@100 mismatch")
    return float(hits.mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pool EMD (corrected scoring, fixed pool)")
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
    oracle = oracle_check(POOL, seed=0)
    print(f"# EMD oracle ({POOL}x{POOL}, diag 0 / off-diag 1): {oracle:.4f}  (P@100==recovered asserted)",
          flush=True)
    if abs(oracle - 1.0) > 1e-12:
        raise SystemExit("EMD scoring is wrong; refusing to report")

    X = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    Y = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    print(f"# grid={args.grid} K_raw={X.shape[1]} pool_fixed={POOL}", flush=True)

    header = (f"{'m':>4} {'draw':>4} {'arm':>16} {'recovered':>9} {'R@1':>7} {'uniq_cols':>9} "
              f"{'MNN':>6} | {'mrg25':>6} {'mrg50':>6} {'mrg75':>6} {'mrg100':>6} | "
              f"{'mn25':>6} {'mn50':>6} {'mn75':>6} {'mn100':>6} {'LP_s':>6}")
    print(header, flush=True)
    rows: list[dict[str, object]] = []
    for m in args.sizes:
        for draw in args.draws:
            order = np.random.default_rng(int(draw)).permutation(1000)
            lab = np.sort(order[: int(m)])
            rest = order[int(m):]
            # CN: 池固定 700：池的 T2 取前 700 个剩余，T1 侧也取 700 行（方阵）；其余 1000-m-700 不参与。
            # EN: pool fixed at 700: the first 700 remaining pairs; 700 rows on the T1 side too (square).
            pool_idx = np.sort(rest[:POOL])
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
            arms = {"learned": np.asarray(res.theta, dtype=np.float64),
                    "default(uniform)": np.full(int(feats.n_features), 1.0 / feats.n_features),
                    "variance": variance_theta(xl, yl)}
            pool_feats = DiagonalWeightFeatures(xp, yp)
            h = int(xp.shape[0])
            gg = np.meshgrid(np.arange(h), np.arange(h), indexing="ij")
            for arm, th in arms.items():
                cost = np.asarray(pool_feats.costs(th, gg[0].reshape(-1), gg[1].reshape(-1)),
                                  dtype=np.float64).reshape(h, h)
                hits, assigned = emd_tracked(cost, seed=int(draw))
                nearest = np.argmin(cost, axis=1)
                r1 = float(np.mean(nearest == np.arange(h)))
                uniq = int(np.unique(nearest).size)
                mnn = mutual_nn(cost)
                pm = precision_at_coverage(hits, margin_confidence(cost))
                pn = precision_at_coverage(hits, mnn.astype(np.float64) * 1e6)
                rec = {"m": int(m), "draw": int(draw), "arm": arm, "pool": h,
                       "recovered": float(np.mean(hits)), "R@1": r1, "uniq_cols": uniq,
                       "MNN": float(mnn.mean()),
                       "P@100": pm["P@100"],
                       **{f"mrg{k}": v for k, v in pm.items()},
                       **{f"mn{k}": v for k, v in pn.items()},
                       "LP_s": float(lp_s) if arm == "learned" else 0.0,
                       "K": int(feats.n_features)}
                if abs(rec["recovered"] - rec["P@100"]) > 1e-12:
                    raise AssertionError("P@100 must equal recovered")
                rows.append(rec)
                print(f"{m:>4} {draw:>4} {arm:>16} {rec['recovered']:>9.4f} {r1:>7.4f} {uniq:>9d} "
                      f"{rec['MNN']:>6.3f} | {pm['P@25']:>6.3f} {pm['P@50']:>6.3f} {pm['P@75']:>6.3f} "
                      f"{pm['P@100']:>6.3f} | {pn['P@25']:>6.3f} {pn['P@50']:>6.3f} {pn['P@75']:>6.3f} "
                      f"{pn['P@100']:>6.3f} {(lp_s if arm == 'learned' else float('nan')):>6.1f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pool_emd2.json").write_text(json.dumps({"oracle": oracle, "pool": POOL, "rows": rows},
                                                        indent=2) + "\n", encoding="utf-8")
    print(f"[pool-emd2] wrote {args.out / 'pool_emd2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
