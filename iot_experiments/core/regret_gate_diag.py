"""
CN: **regret-gate 的单配置诊断驱动**（m=100, seed 0, diagonal）—— 实现四项修正后的读数：
    (1) 只用标注集决定特征掩码：零值比例 > 0.9 或 σ < 0.1·median σ 的特征丢弃；
    (2) 评估用**未居中**代价，并给出"参考集居中"（标注集均值作用到 held-out）作为第二列；
        **居中只发生在拟合内部**；
    (3) 每次 EMD 之前打乱 held-out 的候选列：跟踪真伙伴的新索引，按该索引计分。有真实信号的代价
        命中率不因列置换而下降；只有"按索引顺序"的假象才会掉到随机。
    (4) 守卫改为**标注集 tie fraction**：> 50% 则标记 θ 退化（不再用 RMS 守卫）。

EN: **Single-configuration regret-gate driver** (m=100, seed 0, diagonal) implementing the four fixes:
    (1) feature mask decided on the **labelled set only**: drop features with zero fraction > 0.9 or
        sigma < 0.1 x median sigma;
    (2) evaluation on the **uncentred** cost, with a second column for reference-set centring (labelled
        means applied to held-out); **centring happens inside the fit only**;
    (3) shuffle the held-out candidate columns before every EMD, track the true partner's new index and
        score against that index. A cost with real signal keeps its hit rate under a column permutation;
        only an index-order artefact falls to chance.
    (4) the guard is now the **labelled tie fraction**: flagged as degenerate if > 50% of entries tie.

用法 / EN: usage
    python -m iot_experiments.core.regret_gate_diag --m 100 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import ot
from scipy import sparse
from scipy.optimize import linear_sum_assignment

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import DiagonalWeightFeatures, InverseOTProblem, SimplexTheta  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402

from .real_phase1 import _DATA_ROOT, block_features, load_gray  # noqa: E402
from .regret_gate import (  # noqa: E402
    HELD_OUT,
    centring_means,
    double_centre,
    double_centre_with,
    feature_mask,
    standardise_labelled,
    tie_fraction,
)


def rank_stats(cost: np.ndarray, targets: np.ndarray) -> tuple[float, float, float]:
    """
    CN: 按代价行排序，与**给定**目标索引比较，返回 (R@1, R@5, 中位排名)。
    EN: Rank each row by cost and compare against the **given** target indices;
        returns (R@1, R@5, median rank).
    """
    order = np.argsort(cost, axis=1, kind="stable")
    ranks = np.argmax(order == targets[:, None], axis=1) + 1
    return float(np.mean(ranks <= 1)), float(np.mean(ranks <= 5)), float(np.median(ranks))


def emd_oracle_hit(h: int, *, seed: int = 0) -> float:
    """
    CN: 自检：代价在对角为 0、其余为 1 的 oracle 下，打乱列并跟踪真伙伴后 EMD 命中率必须为 1。
    EN: Self-check: with an oracle cost (0 on true pairs, 1 elsewhere), shuffling columns and tracking the
        true partner must yield an EMD hit rate of exactly 1.
    """
    cost = np.ones((int(h), int(h)), dtype=np.float64)
    np.fill_diagonal(cost, 0.0)
    hit, _ = emd_hit(cost, seed=int(seed))
    return float(hit)


def emd_hit(cost: np.ndarray, *, seed: int) -> tuple[float, int]:
    """
    CN: 打乱候选列后求 EMD（均匀边缘），跟踪真伙伴的新列索引并计命中率；同时返回列置换后的 tie fraction。
    EN: Shuffle candidate columns, solve EMD (uniform marginals), track the true partner's new column index
        and score the hit rate; also returns the tie fraction of the (permuted) cost matrix.
    """
    m = cost.shape[0]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(cost.shape[1])
    shuffled = cost[:, perm]
    # CN: 逆置换必须用 argsort（searchsorted 只对**有序**数组成立；perm 无序）。
    # EN: the inverse permutation must use argsort (searchsorted requires a sorted array; perm is not).
    true_cols = np.argsort(perm)
    plan = ot.emd(np.ones(m) / m, np.ones(cost.shape[1]) / cost.shape[1], shuffled,
                  numItermax=10_000_000)
    chosen = np.argmax(plan, axis=1)
    return float(np.mean(chosen == true_cols)), 1 - np.unique(np.round(shuffled, 15)).size / shuffled.size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="regret-gate diagnostics after the four fixes")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--eps-r-rel", type=float, default=1e-2)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "regret_gate")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[: int(args.n)]
    x_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    y_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    n = int(x_raw.shape[0])

    order = np.random.default_rng(int(args.split_seed)).permutation(n)
    held = np.sort(order[:HELD_OUT])
    pool = order[HELD_OUT:]
    lab = np.sort(np.random.default_rng(1000 + int(args.seed)).choice(
        pool, size=int(args.m), replace=False))
    m = int(lab.size)
    print(f"[diag] n={n} held={held.size} lab={m} split_seed={args.split_seed} seed={args.seed}", flush=True)

    # (1) feature mask from the labelled set only
    lab_stack_raw = np.concatenate([x_raw[lab], y_raw[lab]], axis=0)
    mask = feature_mask(lab_stack_raw)
    dropped = np.nonzero(~mask)[0]
    print(f"(1) dropped {dropped.size} of {mask.size} features: {dropped.tolist()}", flush=True)

    scaled, others = standardise_labelled(lab_stack_raw, [x_raw[held], y_raw[held]])
    x_lab_all, y_lab_all = scaled[:m], scaled[m:]
    x_ho_all, y_ho_all = others[0], others[1]
    # (2) centring is fit INSIDE the fit and reused for held-out; evaluation also reported uncentred.
    fit_feats = double_centre(DiagonalWeightFeatures(x_lab_all[:, mask], y_lab_all[:, mask]))
    means = centring_means(fit_feats)
    ho_uncentred = DiagonalWeightFeatures(x_ho_all[:, mask], y_ho_all[:, mask])
    ho_centred = double_centre_with(ho_uncentred, means)

    theta_set = SimplexTheta(int(fit_feats.n_features))
    plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
    problem = InverseOTProblem(features=fit_feats, theta_set=theta_set, plan=plan)
    result = solve_inverse_ot_regret(
        problem, prune=False, gamma=int(args.gamma), max_iterations=int(args.max_iterations),
        eps_r_rel=float(args.eps_r_rel), selection="none", check_pot=False)

    theta_fit = np.asarray(result.theta, dtype=np.float64)
    uni = np.full(fit_feats.n_features, 1.0 / fit_feats.n_features)
    r_idx, c_idx = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    labelled_cost = np.asarray(fit_feats.costs(theta_fit, r_idx.reshape(-1), c_idx.reshape(-1)),
                               dtype=np.float64).reshape(m, m)
    lab_ties = tie_fraction(labelled_cost)
    is_vertex = bool(np.count_nonzero(theta_fit > 1e-12) == 1)
    vertex_idx = int(np.argmax(theta_fit)) if is_vertex else -1
    print(f"(4) labelled C_theta tie fraction = {lab_ties:.4f}  => guard "
          f"{'FLAGGED degenerate (>50%)' if lab_ties > 0.5 else 'ok'}", flush=True)
    print(f"    theta: nonzero={int(np.count_nonzero(theta_fit > 1e-12))}/{theta_fit.size} "
          f"vertex={is_vertex}{' e_'+str(vertex_idx) if is_vertex else ''} "
          f"converged={result.converged} iters={result.n_iterations} R*={result.regret:.6e}", flush=True)

    h = int(held.size)
    targets = np.arange(h)
    rows = []
    for label, feats in (("uncentred", ho_uncentred), ("ref-centred", ho_centred)):
        for name, th in (("fitted", theta_fit), ("uniform", uni)):
            hh = np.meshgrid(np.arange(h), np.arange(h), indexing="ij")
            cost = np.asarray(feats.costs(th, hh[0].reshape(-1), hh[1].reshape(-1)),
                              dtype=np.float64).reshape(h, h)
            # (3) column-shuffled EMD, scored against the tracked index
            hit, _tf = emd_hit(cost, seed=int(args.seed))
            r1, r5, med = rank_stats(cost, targets)
            tf = tie_fraction(cost)
            rows.append({"centring": label, "theta": name, "R@1": r1, "R@5": r5, "median": med,
                         "EMD_hit": hit, "tie_frac": tf, "C_rms": float(np.sqrt(np.mean(cost ** 2)))})
            print(f"  held-out [{label:11s}] theta={name:8s}  R@1={r1:.4f} R@5={r5:.4f} "
                  f"med={med:6.1f}  EMD_hit={hit:.4f}  ties={tf:.4f}  C_rms={np.sqrt(np.mean(cost**2)):.3e}",
                  flush=True)

    print(f"  chance: R@1={1.0/h:.4f} R@5={5.0/h:.4f} median={h/2:.1f} EMD_hit={1.0/h:.4f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"n": n, "m": m, "seed": int(args.seed), "split_seed": int(args.split_seed),
               "dropped_features": dropped.tolist(), "labelled_tie_fraction": lab_ties,
               "theta_is_vertex": is_vertex, "theta_vertex_index": vertex_idx,
               "theta": theta_fit.tolist(), "converged": bool(result.converged),
               "iterations": int(result.n_iterations), "R_star": float(result.regret),
               "rows": rows}
    (args.out / f"regret_gate_diag_m{m}_seed{args.seed}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[diag] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
