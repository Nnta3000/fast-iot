"""
CN: **补跑** 16×16（标准化臂）缺失行 + 打乱标注对的对照。只跑 16×16、std、m ∈ {100,300}。
    无缓冲输出（`python -u`），每行即时打印。

EN: **Re-run** of the missing 16x16 rows (standardised arm) plus a scrambled-pairs control.
    16x16, std only, m in {100, 300}; unbuffered output, one line per arm as soon as it is measured.

用法 / EN: usage
    python -u -m iot_experiments.core.regret_gate_rerun --sizes 100 300 --seeds 1 2
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
from .regret_gate_matrix import block_features, rank_stats  # noqa: E402


def measure(feats_ho: DiagonalWeightFeatures, theta: np.ndarray, *, seed: int) -> dict[str, float]:
    """CN: held-out 读数。EN: held-out metrics."""
    h = int(feats_ho.n_source)
    g = np.meshgrid(np.arange(h), np.arange(h), indexing="ij")
    cost = np.asarray(feats_ho.costs(theta, g[0].reshape(-1), g[1].reshape(-1)),
                      dtype=np.float64).reshape(h, h)
    r1, r5, med = rank_stats(cost)
    hit, _ = emd_hit(cost, seed=seed)
    return {"R@1": r1, "R@5": r5, "median": med, "EMD": float(hit),
            "ties": 1 - np.unique(np.round(cost, 15)).size / cost.size}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="16x16 std re-run + scrambled control")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--scramble-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--eps-r-rel", type=float, default=1e-2)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "regret_gate")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[:1000]
    print(f"# EMD oracle: {emd_oracle_hit(HELD_OUT, seed=0):.4f} (must be 1.0)", flush=True)
    X = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    Y = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    print(f"# grid={args.grid} decoded: K_raw={X.shape[1]}", flush=True)

    records: list[dict[str, object]] = []
    plan_counter = 0
    for m in args.sizes:
        seeds = list(args.seeds) + list(args.scramble_seeds)
        for seed in seeds:
            scrambled = seed in args.scramble_seeds
            order = np.random.default_rng(int(seed)).permutation(1000)
            held = np.sort(order[:HELD_OUT])
            pool = order[HELD_OUT:]
            lab = np.sort(np.random.default_rng(1000 + int(seed)).choice(pool, size=int(m),
                                                                        replace=False))
            mm = int(lab.size)
            mask = feature_mask(np.concatenate([X[lab], Y[lab]]))
            s, o = standardise_labelled(np.concatenate([X[lab], Y[lab]]), [X[held], Y[held]])
            xl, yl = s[:mm][:, mask], s[mm:][:, mask]
            ho = np.concatenate(o, axis=0)
            feats = DiagonalWeightFeatures(xl, yl)
            # CN: 打乱对照：pi_hat 用打乱的标注对，评分的 held-out 仍是真实配对。
            # EN: scrambled control: pi_hat from a scrambled pairing; evaluation stays on true pairs.
            perm = (np.random.default_rng(777).permutation(mm) if scrambled else np.arange(mm))
            plan = sparse.csr_matrix((np.full(mm, 1.0 / mm), (np.arange(mm), perm)), shape=(mm, mm))
            problem = InverseOTProblem(features=feats,
                                       theta_set=SimplexTheta(int(feats.n_features)),
                                       plan=plan)
            t0 = time.perf_counter()
            res = solve_inverse_ot_regret(problem, prune=False, gamma=int(args.gamma),
                                          max_iterations=int(args.max_iterations),
                                          eps_r_rel=float(args.eps_r_rel), selection="none",
                                          check_pot=False)
            lp_s = time.perf_counter() - t0
            theta = np.asarray(res.theta, dtype=np.float64)
            uni = np.full(int(feats.n_features), 1.0 / feats.n_features)
            ho_feats = DiagonalWeightFeatures(ho[:HELD_OUT][:, mask], ho[HELD_OUT:][:, mask])
            for arm, th in (("learned(regret)", theta), ("default(uniform)", uni)):
                label = arm + (" [SCRAMBLED pi_hat]" if scrambled else "")
                plan_counter += 1
                met = measure(ho_feats, th, seed=int(seed))
                rec = {"grid": args.grid, "K": int(feats.n_features), "m": m, "seed": seed,
                       "scrambled": bool(scrambled), "arm": label, **met,
                       "LP_s": float(lp_s) if arm == "learned(regret)" else 0.0,
                       "theta_nonzero": int(np.count_nonzero(th > 1e-12)),
                       "converged": bool(res.converged) if arm == "learned(regret)" else True,
                       "R_star": float(res.regret) if arm == "learned(regret)" else 0.0}
                records.append(rec)
                print(f"grid={args.grid:>2} K={feats.n_features:>3} m={m:>3} seed={seed} "
                      f"{label:>32} R@1={met['R@1']:.4f} R@5={met['R@5']:.4f} "
                      f"med={met['median']:6.1f} EMD={met['EMD']:.4f} ties={met['ties']:.3f} "
                      f"LP={lp_s:.1f}s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "regret_gate_rerun_16.json"
    path.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    print(f"[rerun] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
