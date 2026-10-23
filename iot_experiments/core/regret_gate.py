"""
CN: **真实数据上的一次 regret-iOT 检查**（单一问题）：
    「在标注真配对上拟合的 theta，是否比 uniform theta 更好地检索 held-out 真配对？」

    协议：
      - 1000 个已配准 T1/T2 对；按科目划分：HELD-OUT 固定 400（永不参与拟合），LABELLED 从其余 600 中抽 m ∈ {100,300}。
      - 方案 `pi_hat = (1/m)·I`（**真置换**），不来自任何前向 EMD。
      - 特征卫生：标准化 + 双中心化，**只用 LABELLED 拟合**，再作用于 held-out。
      - 族：`diagonal`（K=64）；`bilinear`（PCA-16 的符号双线性，K=512）。
      - 求解：`loop_regret`，`prune=False`，只跑 stage 1。

EN: **One regret-iOT check on real data** (a single question):
    "does theta fitted on labelled true pairs retrieve held-out true partners better than uniform theta?"

    Protocol: 1000 registered T1/T2 pairs; subject split with a fixed 400-pair HELD-OUT (never used for
    fitting); LABELLED = m drawn from the remaining 600, m in {100, 300}; plan `pi_hat = (1/m)·I` (the true
    permutation, not a forward EMD); feature hygiene (standardise, then double-centre) fitted on LABELLED
    only; families `diagonal` (K=64) and `bilinear` on PCA-16 (K=512); solver `loop_regret`, `prune=False`,
    stage 1 only.

用法 / EN: usage
    python -m iot_experiments.core.regret_gate --sizes 100 300 --seeds 0 1
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

from iot import (  # noqa: E402
    DiagonalWeightFeatures,
    DenseFeatureTensor,
    InverseOTProblem,
    SimplexTheta,
)
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402

from .real_phase1 import _DATA_ROOT, block_features, cost_matrix, load_gray  # noqa: E402

HELD_OUT = 400


def standardise_labelled(train: np.ndarray, others: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    CN: **只在标注集上**拟合均值/标准差，再作用于其它集合。
    EN: Fit mean/std on the **labelled set only**, then apply to the others.
    """
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (train - mu) / sd, [(other - mu) / sd for other in others]


def feature_mask(raw_lab: np.ndarray, *, zero_frac_max: float = 0.9,
                 sigma_rel_min: float = 0.1) -> np.ndarray:
    """
    CN: 丢掉 (a) 零值比例 > `zero_frac_max` 的特征，(b) 标准差 < `sigma_rel_min` x 中位标准差的特征。
        **只用标注集**决定掩码，再作用于 held-out。
    EN: Drop features with (a) zero fraction > `zero_frac_max`, or (b) std < `sigma_rel_min` x median std.
        The mask is decided on the **labelled set only** and applied to held-out.
    """
    zero = np.mean(np.asarray(raw_lab) == 0.0, axis=0)
    sigma = np.asarray(raw_lab).std(axis=0)
    med = float(np.median(sigma[sigma > 0.0])) if np.any(sigma > 0.0) else 1.0
    keep = (zero <= float(zero_frac_max)) & (sigma >= float(sigma_rel_min) * med)
    return keep


def tie_fraction(cost: np.ndarray, *, rtol: float = 0.0) -> float:
    """
    CN: 代价矩阵中"与其它项取同一值"的比例（按 float64 精确比较）。1 - 唯一值数 / 总项数。
    EN: Fraction of entries sharing a value with another entry (exact float64 comparison):
        `1 - #unique / #entries`.
    """
    flat = np.asarray(cost, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(1.0 - np.unique(flat).size / flat.size)


def centring_means(features: DiagonalWeightFeatures) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 返回该点云的双中心化均值 `(xm, ym)`，**供另一集合复用**（防止 held-out 各自重新拟合）。
    EN: Return this cloud's double-centring means `(xm, ym)` so another set can reuse them instead of
        refitting its own.
    """
    xm = np.asarray(features.source_points, dtype=np.float64).mean(axis=0)
    ym = np.asarray(features.target_points, dtype=np.float64).mean(axis=0)
    return xm, ym


def double_centre_with(features: DiagonalWeightFeatures,
                       means: tuple[np.ndarray, np.ndarray]) -> DiagonalWeightFeatures:
    """
    CN: 用**给定**均值做双中心化：`x <- x - xm`，`y <- y - ym`。与 `double_centre` 的区别是不再重算均值，
        因此可以把标注集的居中作用原样应用到 held-out。
    EN: Double-centre with **given** means: `x <- x - xm`, `y <- y - ym`. Unlike `double_centre` it does not
        refit the means, so the labelled centring can be applied unchanged to held-out.
    """
    xm, ym = (np.asarray(means[0], dtype=np.float64), np.asarray(means[1], dtype=np.float64))
    x = np.asarray(features.source_points, dtype=np.float64) - xm
    y = np.asarray(features.target_points, dtype=np.float64) - ym
    return DiagonalWeightFeatures(x, y)


def double_centre(features: DiagonalWeightFeatures) -> DiagonalWeightFeatures:
    """
    CN: 特征双中心化：令 `sum_i Phi_ij = 0`、`sum_j Phi_ij = 0`（对每个特征坐标）。
        这保持 OT 方案不变（行列常数项不影响最优耦合），并移除"低方差特征被单纯形偏爱"的偏置。
    EN: Double-centre each feature over rows and columns so `sum_i Phi_ij = 0` and `sum_j Phi_ij = 0`.
        This leaves the OT plan unchanged (row/column constants do not affect the optimal coupling) and removes
        the simplex's bias toward low-variance features.
    """
    x = np.array(features.source_points, dtype=np.float64, copy=True)
    y = np.array(features.target_points, dtype=np.float64, copy=True)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    centre = 0.5 * (x.mean(axis=0, keepdims=True) + y.mean(axis=0, keepdims=True))
    x = x - centre
    y = y - centre
    return DiagonalWeightFeatures(x, y)


def ranking_stats(cost: np.ndarray) -> tuple[float, float, float]:
    """
    CN: 在**给定候选池**内按代价排序，返回 (R@1, R@5, 中位排名)。
    EN: Rank inside the given candidate pool; returns (R@1, R@5, median rank).
    """
    order = np.argsort(cost, axis=1, kind="stable")
    native = np.arange(cost.shape[0])
    ranks = np.argmax(order == native[:, None], axis=1) + 1
    return float(np.mean(ranks <= 1)), float(np.mean(ranks <= 5)), float(np.median(ranks))


def bilinear_signed_features(x_lab: np.ndarray, y_lab: np.ndarray, *,
                             pca_dim: int) -> DenseFeatureTensor:
    """
    CN: 在**标注集**上拟合 PCA（合并 T1/T2），取前 `pca_dim` 维，然后构造**符号**双线性特征：
        `Phi_ij[(a,b)] = x_ia * y_jb`（K = pca_dim^2）。行/列为 0 与负值均允许（单纯形 theta 下代价可正可负）。
    EN: Fit PCA on the **labelled set** (pooled T1/T2), take the first `pca_dim` components, then build
        **signed** bilinear features `Phi_ij[(a,b)] = x_ia * y_jb` (K = pca_dim^2).
    """
    pooled = np.concatenate([x_lab, y_lab], axis=0)
    mean = pooled.mean(axis=0, keepdims=True)
    centred = pooled - mean
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    basis = vt[: int(pca_dim)].T  # (d, p)
    px = (x_lab - mean) @ basis
    py = (y_lab - mean) @ basis
    m, n = px.shape[0], py.shape[0]
    p = px.shape[1]
    tensor = np.einsum("ia,jb->ijab", px, py).reshape(m, n, p * p)
    return DenseFeatureTensor(tensor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="one real-data regret-iOT check")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--families", nargs="+", default=["diagonal", "bilinear"])
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--eps-r-rel", type=float, default=1e-2)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "core" / "out" / "regret_gate")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[: int(args.n)]
    x_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    y_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    n = int(x_raw.shape[0])

    split_rng = np.random.default_rng(int(args.split_seed))
    order = split_rng.permutation(n)
    held = np.sort(order[:HELD_OUT])
    pool = order[HELD_OUT:]
    print(f"[regret-gate] n={n} held-out={held.size} labelled pool={pool.size} "
          f"split_seed={args.split_seed}", flush=True)
    chance = (1.0 / held.size, 5.0 / held.size, held.size / 2.0)

    records: list[dict[str, object]] = []
    header = (f"{'family':>9} {'m':>4} {'seed':>4} {'conv':>5} {'it':>3} {'relR fit':>9} {'relR uni':>9} | "
              f"{'R@1 fit':>8} {'R@1 uni':>8} | {'R@5 fit':>8} {'R@5 uni':>8} | {'med fit':>8} {'med uni':>8}")
    print(header, flush=True)
    for family in args.families:
        for m in args.sizes:
            if m > pool.size:
                raise SystemExit(f"labelled size {m} exceeds the pool ({pool.size})")
            for seed in args.seeds:
                draw_rng = np.random.default_rng(1000 + int(seed))
                lab = np.sort(draw_rng.choice(pool, size=int(m), replace=False))
                lab_stack = np.concatenate([x_raw[lab], y_raw[lab]], axis=0)
                scaled, others = standardise_labelled(lab_stack, [x_raw[held], y_raw[held]])
                x_lab, y_lab = scaled[: lab.size], scaled[lab.size:]
                x_ho, y_ho = others[0], others[1]

                if family == "diagonal":
                    features_lab = double_centre(DiagonalWeightFeatures(x_lab, y_lab))
                    features_ho = double_centre(DiagonalWeightFeatures(x_ho, y_ho))
                elif family == "bilinear":
                    features_lab = bilinear_signed_features(x_lab, y_lab, pca_dim=args.pca_dim)
                    features_ho = bilinear_signed_features(x_ho, y_ho, pca_dim=args.pca_dim)
                else:
                    raise ValueError(f"unknown family {family!r}")

                theta_set = SimplexTheta(features_lab.n_features)
                plan = sparse.csr_matrix(
                    (np.full(int(m), 1.0 / m), (np.arange(int(m)), np.arange(int(m)))),
                    shape=(int(m), int(m)))
                problem = InverseOTProblem(features=features_lab, theta_set=theta_set, plan=plan)

                t0 = time.perf_counter()
                result = solve_inverse_ot_regret(
                    problem, prune=False, gamma=int(args.gamma),
                    max_iterations=int(args.max_iterations), eps_r_rel=float(args.eps_r_rel),
                    selection="none", check_pot=False)
                wall = time.perf_counter() - t0

                theta_fit = np.asarray(result.theta, dtype=np.float64)
                uni = np.full(features_lab.n_features, 1.0 / features_lab.n_features)

                # CN: 相对 regret = R/(theta.phi_bar)，在**标注集**上。
                # EN: relative regret = R/(theta.phi_bar) on the **labelled** set.
                from iot.relaxed.master_regret import regret_objective
                obj = regret_objective(problem)
                phi_bar = obj[: features_lab.n_features]
                a = -obj[features_lab.n_features: features_lab.n_features + int(m)]
                b = -obj[features_lab.n_features + int(m):]

                def rel(reg: float, th: np.ndarray) -> float:
                    denom = float(phi_bar @ th)
                    return float(reg) / denom if abs(denom) > 1e-12 else float("nan")

                # CN: uniform 的 regret：用可行对偶 f_i = min_j c_ij、g = 0 直接算。
                r_idx, c_idx = np.meshgrid(np.arange(int(m)), np.arange(int(m)), indexing="ij")
                cost_uni = np.asarray(features_lab.costs(uni, r_idx.reshape(-1), c_idx.reshape(-1)),
                                      dtype=np.float64).reshape(int(m), int(m))
                f_u = cost_uni.min(axis=1)
                r_uni = float(phi_bar @ uni - a @ f_u - b @ np.zeros(int(m)))

                fit_r1, fit_r5, fit_med = ranking_stats(cost_matrix(features_ho, theta_fit))
                uni_r1, uni_r5, uni_med = ranking_stats(cost_matrix(features_ho, uni))
                vertex = int(np.argmax(theta_fit))
                is_vertex = bool(np.count_nonzero(theta_fit > 1e-12) == 1)

                rec = {
                    "family": family, "m": int(m), "seed": int(seed), "split_seed": int(args.split_seed),
                    "converged": bool(result.converged), "iterations": int(result.n_iterations),
                    "relregret_fit": rel(result.regret, theta_fit), "relregret_uniform": rel(r_uni, uni),
                    "R_star": float(result.regret), "R_uniform": r_uni,
                    "lower": float(result.lower), "upper": float(result.upper), "gap": float(result.gap),
                    "heldout_r1_fit": fit_r1, "heldout_r5_fit": fit_r5, "heldout_med_fit": fit_med,
                    "heldout_r1_uniform": uni_r1, "heldout_r5_uniform": uni_r5,
                    "heldout_med_uniform": uni_med,
                    "theta_is_vertex": is_vertex, "theta_argmax": vertex,
                    "theta_nonzero": int(np.count_nonzero(theta_fit > 1e-12)),
                    "K": int(features_lab.n_features), "wall": float(wall),
                }
                records.append(rec)
                flag = " NOT CONVERGED" if not result.converged else ""
                print(f"{family:>9} {m:>4} {seed:>4} {str(rec['converged']):>5} {rec['iterations']:>3} "
                      f"{rec['relregret_fit']:>9.3e} {rec['relregret_uniform']:>9.3e} | "
                      f"{fit_r1:>8.4f} {uni_r1:>8.4f} | {fit_r5:>8.4f} {uni_r5:>8.4f} | "
                      f"{fit_med:>8.1f} {uni_med:>8.1f}{flag}", flush=True)
                if is_vertex:
                    print(f"{'':>9} {'':>4} {'':>4}   theta is a simplex vertex (e_{vertex})", flush=True)

    print(f"\nchance: R@1={chance[0]:.4f}  R@5={chance[1]:.4f}  median rank={chance[2]:.1f}",
          flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"regret_gate_split{args.split_seed}.json"
    path.write_text(json.dumps({"n": n, "held_out": HELD_OUT, "chance": chance,
                                "records": records}, indent=2) + "\n", encoding="utf-8")
    print(f"[regret-gate] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
