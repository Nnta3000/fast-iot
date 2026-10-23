"""
CN: **regret-gate 审计与对照**（m=100, seed 0, diagonal）—— 基线面板 + 两项对照，一个表。
    (1) 掩码来源：声明掩码在**标注集**上计算；另在 600 池上重算并比较（冻结阈值）。
    (2) 基线面板：uniform 与 fitted 两种 theta，在
          features {64 std, 44 masked+std, 44 masked unstd} x cost {uncentred, TRANSDUCTIVE
          (held-out 自己双中心化), reference-centred}；
        reference-centring 按指定实现：每个 held-out 候选在**原始 feature 空间**上减去它对标注源的
        平均差（源对称地减去它对标注目标的平均差）——这是对角代价下**非平凡**的居中。
        另加余弦与原始 L2 两个非学习基线。
    (3) 对照：(a) 打乱标注对得到 π̂ 再拟合；(b) split_seed 1 与 2。

EN: **regret-gate audits and controls** (m=100, seed 0, diagonal) -- baseline panel plus two controls, one table.
    (1) Mask provenance: state that the mask is computed on the **labelled set**; recompute on the 600 pool
        and compare (thresholds frozen).
    (2) Baseline panel: uniform vs fitted theta over features {64 std, 44 masked+std, 44 masked unstd} x
        cost {uncentred, TRANSDUCTIVE (held-out double-centred on itself), reference-centred}, with
        reference-centring implemented as specified: each held-out candidate has its mean difference to the
        labelled sources subtracted (sources symmetrically vs labelled targets). Plus cosine and raw-L2 as
        non-learned baselines.
    (3) Controls: (a) shuffle the labelled pairing to build pi_hat, then fit and evaluate; (b) split_seed 1, 2.

用法 / EN: usage
    python -m iot_experiments.core.regret_gate_audit --m 100 --seed 0
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

from .real_phase1 import _DATA_ROOT, block_features, load_gray  # noqa: E402
from .regret_gate import HELD_OUT, feature_mask, standardise_labelled, tie_fraction  # noqa: E402
from .regret_gate_diag import emd_hit  # noqa: E402


def rank_stats(cost: np.ndarray, targets: np.ndarray) -> tuple[float, float, float]:
    """CN: (R@1, R@5, 中位排名) 对照给定目标索引。EN: (R@1, R@5, median rank) vs given targets."""
    order = np.argsort(cost, axis=1, kind="stable")
    ranks = np.argmax(order == targets[:, None], axis=1) + 1
    return float(np.mean(ranks <= 1)), float(np.mean(ranks <= 5)), float(np.median(ranks))


def ref_centre(x_lab: np.ndarray, y_lab: np.ndarray, x_ho: np.ndarray,
               y_ho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: **reference-centring**（对角代价下非平凡）：`x_i <- x_i - mean_j(y_lab_j)`，
        `y_j <- y_j - mean_i(x_lab_i)`。即在原始 feature 空间把 held-out 点平移到标注集的"平均对手"处，
        使对角代价的平方项不再被两模态的全局偏移主导。
    EN: **reference-centring** (non-trivial for a diagonal cost): `x_i <- x_i - mean_j(y_lab_j)` and
        `y_j <- y_j - mean_i(x_lab_i)`. Held-out points are shifted, in the raw feature space, by the labelled
        set's mean counterpart so the global modality offset no longer dominates the squared cost.
    """
    return x_ho - y_lab.mean(axis=0, keepdims=True), y_ho - x_lab.mean(axis=0, keepdims=True)


def fit_theta(x_lab: np.ndarray, y_lab: np.ndarray, plan_perm: np.ndarray, *, gamma: int,
              max_iterations: int, eps_r_rel: float):
    """CN: 在给定置换方案上拟合 theta。EN: fit theta on the given permutation plan."""
    m = int(x_lab.shape[0])
    feats = DiagonalWeightFeatures(x_lab, y_lab)
    plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), plan_perm)), shape=(m, m))
    problem = InverseOTProblem(features=feats, theta_set=SimplexTheta(int(feats.n_features)),
                               plan=plan)
    res = solve_inverse_ot_regret(problem, prune=False, gamma=int(gamma),
                                  max_iterations=int(max_iterations), eps_r_rel=float(eps_r_rel),
                                  selection="none", check_pot=False)
    return np.asarray(res.theta, dtype=np.float64), res


def fit_theta_infonce(x_lab: np.ndarray, y_lab: np.ndarray, *, temperature: float = 0.1,
                      epochs: int = 200, lr: float = 0.05) -> np.ndarray:
    """
    CN: 非 OT 基线：在**同一组**标注对上用 softmax/InfoNCE 拟合线性于 Phi 的 theta（单纯形参数化）。
        代价 `c_ij = sum_k theta_k (x_ik - y_jk)^2`，目标为 `-log softmax_j(-c_ij/tau)` 的对角项。
    EN: Non-OT baseline: fit theta linear in Phi on the **same** labelled pairs with a softmax/InfoNCE
        objective over the simplex, with the same diagonal cost `c_ij = sum_k theta_k (x_ik-y_jk)^2`.
    """
    m, k = int(x_lab.shape[0]), int(x_lab.shape[1])
    feats = DiagonalWeightFeatures(x_lab, y_lab)
    g = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    phi = np.asarray(feats.features(g[0].reshape(-1), g[1].reshape(-1)),
                     dtype=np.float64).reshape(m, m, k)
    w = np.full(k, 1.0 / k, dtype=np.float64)
    for _ in range(int(epochs)):
        theta = np.exp(w) / np.exp(w).sum()
        cost = np.einsum("ijk,k->ij", phi, theta)
        logits = -cost / float(temperature)
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        grad_logits = prob.copy()
        grad_logits[np.arange(m), np.arange(m)] -= 1.0
        grad_cost = grad_logits / float(temperature)          # d/d cost
        grad_theta = np.einsum("ij,ijk->k", grad_cost, phi)
        w -= float(lr) * (grad_theta - theta * (grad_theta @ theta))
    return np.exp(w) / np.exp(w).sum()


def evaluate(feats_ho: DiagonalWeightFeatures, theta: np.ndarray, *, seed: int) -> dict[str, float]:
    """CN: 在 held-out 上算 R@1/R@5/median/EMD-hit 与 tie fraction。EN: held-out metrics."""
    h = int(feats_ho.n_source)
    g = np.meshgrid(np.arange(h), np.arange(h), indexing="ij")
    cost = np.asarray(feats_ho.costs(theta, g[0].reshape(-1), g[1].reshape(-1)),
                      dtype=np.float64).reshape(h, h)
    r1, r5, med = rank_stats(cost, np.arange(h))
    hit, _ = emd_hit(cost, seed=int(seed))
    return {"R@1": r1, "R@5": r5, "median": med, "EMD_hit": hit, "ties": tie_fraction(cost)}


def unlearned_costs(x_ho: np.ndarray, y_ho: np.ndarray) -> dict[str, np.ndarray]:
    """CN: 非学习基线：原始 L2^2 与余弦。EN: non-learned baselines: raw L2^2 and cosine."""
    l2 = ((x_ho[:, None, :] - y_ho[None, :, :]) ** 2).sum(-1)
    xn = x_ho / (np.linalg.norm(x_ho, axis=1, keepdims=True) + 1e-12)
    yn = y_ho / (np.linalg.norm(y_ho, axis=1, keepdims=True) + 1e-12)
    return {"raw-L2": l2, "cosine": 1.0 - xn @ yn.T}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="regret-gate audits and controls")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0])
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

    rows: list[dict[str, object]] = []
    for split_seed in args.split_seeds:
        order = np.random.default_rng(int(split_seed)).permutation(n)
        held = np.sort(order[:HELD_OUT])
        pool = order[HELD_OUT:]
        lab = np.sort(np.random.default_rng(1000 + int(args.seed)).choice(
            pool, size=int(args.m), replace=False))
        m = int(lab.size)

        mask_lab = feature_mask(np.concatenate([x_raw[lab], y_raw[lab]]))
        mask_pool = feature_mask(np.concatenate([x_raw[pool], y_raw[pool]]))
        print(f"[audit] split_seed={split_seed}: mask(lab) == mask(pool) ? "
              f"{np.array_equal(mask_lab, mask_pool)}  kept={int(mask_lab.sum())}", flush=True)
        mask = mask_lab  # frozen, labelled-only

        scaled_lab, scaled_others = standardise_labelled(
            np.concatenate([x_raw[lab], y_raw[lab]]), [x_raw[held], y_raw[held]])
        scaled_ho = np.concatenate(scaled_others, axis=0)
        feature_sets = {
            "64 std": (scaled_lab[:, :], scaled_ho[:, :]),
            "44 masked+std": (scaled_lab[:, mask], scaled_ho[:, mask]),
            "44 masked unstd": (np.concatenate([x_raw[lab], y_raw[lab]])[:, mask] / 1.0,
                                np.concatenate([x_raw[held], y_raw[held]])[:, mask]),
        }

        for fname, (lab_stack, ho_stack) in feature_sets.items():
            x_lab, y_lab = lab_stack[:m], lab_stack[m:]
            x_ho, y_ho = ho_stack[:held.size], ho_stack[held.size:]
            theta_fit, res = fit_theta(x_lab, y_lab, np.arange(m), gamma=int(args.gamma),
                                       max_iterations=int(args.max_iterations),
                                       eps_r_rel=float(args.eps_r_rel))
            uni = np.full(int(x_lab.shape[1]), 1.0 / x_lab.shape[1])
            variant = {
                "uncentred": (DiagonalWeightFeatures(x_ho, y_ho),
                              DiagonalWeightFeatures(x_ho, y_ho)),
                "TRANSDUCTIVE": (DiagonalWeightFeatures(x_ho, y_ho),
                                 _self_centred(x_ho, y_ho)),
                "ref-centred": (DiagonalWeightFeatures(x_ho, y_ho),
                                DiagonalWeightFeatures(*ref_centre(x_lab, y_lab, x_ho, y_ho))),
            }
            for cname, (_fit_view, ho_view) in variant.items():
                for tname, th in (("uniform", uni), ("fitted", theta_fit)):
                    met = evaluate(ho_view, th, seed=int(split_seed) * 100 + int(args.seed))
                    rows.append({"split_seed": split_seed, "features": fname, "centring": cname,
                                 "theta": tname, **met,
                                 "theta_nonzero": int(np.count_nonzero(theta_fit > 1e-12)),
                                 "converged": bool(res.converged), "iters": int(res.n_iterations)})

        # (3a) shuffled-label control, on the frozen feature set
        lab_stack, ho_stack = feature_sets["44 masked+std"]
        x_lab, y_lab = lab_stack[:m], lab_stack[m:]
        perm = np.random.default_rng(4242).permutation(m)
        theta_shuf, res_shuf = fit_theta(x_lab, y_lab, perm, gamma=int(args.gamma),
                                         max_iterations=int(args.max_iterations),
                                         eps_r_rel=float(args.eps_r_rel))
        ho_view = DiagonalWeightFeatures(ho_stack[:held.size], ho_stack[held.size:])
        met = evaluate(ho_view, theta_shuf, seed=int(split_seed) * 100 + int(args.seed))
        # (3) do the reference-centred offsets agree with the transductive ones?
        xm_t = x_ho.mean(axis=0)
        ym_t = y_ho.mean(axis=0)
        xm_r = y_lab.mean(axis=0)
        ym_r = x_lab.mean(axis=0)
        corr_x = float(np.corrcoef(xm_t, xm_r)[0, 1]) if xm_t.std() > 0 and xm_r.std() > 0 else float("nan")
        corr_y = float(np.corrcoef(ym_t, ym_r)[0, 1]) if ym_t.std() > 0 and ym_r.std() > 0 else float("nan")
        print(f"[offset-corr] split={split_seed}: transductive-vs-reference offset correlation "
              f"x={corr_x:+.3f} y={corr_y:+.3f}", flush=True)
        rows.append({"split_seed": split_seed, "features": "44 masked+std", "centring": "uncentred",
                     "theta": "fitted(shuffled-labels)", **met,
                     "theta_nonzero": int(np.count_nonzero(theta_shuf > 1e-12)),
                     "converged": bool(res_shuf.converged), "iters": int(res_shuf.n_iterations)})

        # (4) non-OT baseline on the same features
        theta_nce = fit_theta_infonce(x_lab, y_lab)
        met = evaluate(DiagonalWeightFeatures(ho_stack[:held.size], ho_stack[held.size:]),
                       theta_nce, seed=int(split_seed) * 100 + int(args.seed))
        rows.append({"split_seed": split_seed, "features": "44 masked+std", "centring": "uncentred",
                     "theta": "fitted(InfoNCE)", **met,
                     "theta_nonzero": int(np.count_nonzero(theta_nce > 1e-12)),
                     "converged": True, "iters": 0})

        # non-learned baselines on raw blocks
        for bname, cost in unlearned_costs(x_raw[held], y_raw[held]).items():
            r1, r5, med = rank_stats(cost, np.arange(held.size))
            hit, _ = emd_hit(cost, seed=int(split_seed))
            rows.append({"split_seed": split_seed, "features": "raw 64", "centring": "—",
                         "theta": bname, "R@1": r1, "R@5": r5, "median": med,
                         "EMD_hit": hit, "ties": tie_fraction(cost),
                         "theta_nonzero": 0, "converged": True, "iters": 0})

    print(f"\n{'split':>5} {'features':>16} {'centring':>13} {'theta':>22} {'R@1':>7} {'R@5':>7} "
          f"{'median':>7} {'EMD':>7} {'ties':>6}", flush=True)
    for r in rows:
        print(f"{r['split_seed']:>5} {r['features']:>16} {r['centring']:>13} {r['theta']:>22} "
              f"{r['R@1']:>7.4f} {r['R@5']:>7.4f} {r['median']:>7.1f} {r['EMD_hit']:>7.4f} {r['ties']:>6.3f}",
              flush=True)
    print(f"chance: R@1={1/HELD_OUT:.4f} R@5={5/HELD_OUT:.4f} median={HELD_OUT/2:.1f} EMD={1/HELD_OUT:.4f}",
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"regret_gate_audit_m{args.m}_seed{args.seed}.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"[audit] wrote {args.out}")
    return 0


def _self_centred(x: np.ndarray, y: np.ndarray) -> DiagonalWeightFeatures:
    """CN: TRANSDUCTIVE：held-out 在自己身上双中心化。EN: TRANSDUCTIVE: held-out double-centred on itself."""
    f = DiagonalWeightFeatures(x, y)
    xm = f.source_points.mean(axis=0)
    ym = f.target_points.mean(axis=0)
    return DiagonalWeightFeatures(np.asarray(f.source_points) - xm, np.asarray(f.target_points) - ym)


if __name__ == "__main__":
    raise SystemExit(main())
