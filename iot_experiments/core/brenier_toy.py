"""
CN: **上游 Brenier 玩具问题的对比实验** —— 直接复用 `export_experiments/synthetic.py` 里已实现的两种
    Brenier 构造（论文使用的形式），把我们自己的约束生成循环跑在上面，得到可与论文对照的读数。

    两种上游构造：
      1. `brenier`（`make_brenier_problem`）：`source ~ N(0,I)`，`target = source + 0.1*tanh(source)`，
         再随机置换。由凸势梯度连接（论文的 toy 形式）。
      2. `correlated`（`make_correlated_brenier_problem`）：强相关潜变量 + 噪声，`target = scale*source`
         再置换；**已知最优 permutation**（`ground_truth_cols`）。

    CN: 为什么两者都能喂进我们的逆问题：两者的 `target` 都是 `source` **逐坐标**变换再置换，所以真值方案
        就是那条**置换**（每条边 (i -> π(i))，|S| = n），在 diagonal 代价 `Σ θ_k (x_k-y_k)²` 下最优。
        我们把该置换作为 `π̂`，把 `θ*` 设为 `1` 归一化到单纯形（恒等变换的最优 θ 是常数向量）。

    CN: 关键取读：迭代数 T、|F|/(m+n)、δ、`is_optimal`、`relative_gap`、墙钟时间。
        论文报告的对照量是 objective error / assignment recall / KKT residual。

EN: **Comparison on the upstream Brenier toy problems** -- reuse the two Brenier constructions already
    implemented in `export_experiments/synthetic.py` (the forms the paper uses) and run *our*
    constraint-generation loop on them, so the read-outs are directly comparable.

    Both constructions make `target` a **coordinate-wise** transform of `source`, then permute, so the
    ground-truth plan is that **permutation** (`|S| = n`), optimal for the diagonal cost when `θ` is the
    constant vector (normalised into the simplex).

用法 / EN: usage
    python -m iot_experiments.core.brenier_toy --n 400 1600 --scan hello --solver cupdlpx
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

from iot import DiagonalWeightFeatures, InverseOTProblem, SimplexTheta, solve_inverse_ot  # noqa: E402
from iot.cupdlpx_master import CuPdlpxMasterSolver  # noqa: E402
from iot.master import LinearMasterSolver  # noqa: E402
from export_experiments.synthetic import (  # noqa: E402
    make_brenier_problem,
    make_correlated_brenier_problem,
)


def build_permutation_problem(kind: str, *, n: int, d: int, seed: int) -> tuple[InverseOTProblem, np.ndarray, dict]:
    """
    CN: 由上游 Brenier 构造得到 (problem, theta_star, meta)。真值方案是那条置换。
    EN: Build (problem, theta_star, meta) from an upstream Brenier construction; truth is the permutation.
    """
    if kind == "brenier":
        source, target = make_brenier_problem(n=n, d=d, seed=seed)
        # CN: 置换未知（上游未返回），用 POT/HELLO 无关的方式：直接构造 i -> i 是否最优？
        #     `make_brenier_problem` 先加 0.1*tanh 再置换，但没有返回置换；因此我们**恢复**它：
        #     对每个源，其目标就是 perm 后的同一个点，可用最近邻在变换后的点集中匹配。
        meta: dict = {"kind": kind, "n": int(n), "d": int(d), "seed": int(seed)}
        transform = source + np.float32(0.1) * np.tanh(source)
        # CN: 用 (target == perm(transform)) 的关系恢复行对应：对每个变换后点找最近的 target。
        #     n 不大时用分块暴力最近邻，O(n^2 d)，仅用于 toy 规模。
        cols = _match_rows_to_targets(transform, target)
        plan_cols = cols
    elif kind == "correlated":
        source, target, ground_truth_cols, objective = make_correlated_brenier_problem(n=n, d=d, seed=seed)
        meta = {"kind": kind, "n": int(n), "d": int(d), "seed": int(seed), "objective": float(objective)}
        # CN: `ground_truth_cols[i]` = 与源 i 配对的目标下标。
        plan_cols = np.asarray(ground_truth_cols, dtype=np.int64)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    features = DiagonalWeightFeatures(source.astype(np.float64), target.astype(np.float64))
    theta_set = SimplexTheta(features.n_features)
    # CN: 恒等变换的最优 theta 是常数向量；归一化到单纯形（代价尺度不影响最优方案）。
    # EN: the identity-ish transform is optimal for a constant theta; normalise into the simplex.
    theta_star = np.full(features.n_features, 1.0 / float(features.n_features), dtype=np.float64)

    rows = np.arange(int(n), dtype=np.int64)
    vals = np.full(int(n), 1.0 / float(n), dtype=np.float64)
    plan = sparse.csr_matrix((vals, (rows, plan_cols)), shape=(int(n), int(n)))
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=plan)
    meta["support"] = int(problem.support_size)
    return problem, theta_star, meta


def _match_rows_to_targets(points: np.ndarray, targets: np.ndarray, *, block: int = 512) -> np.ndarray:
    """
    CN: 为每个变换后点找最近的目标点（上游 brenier 构造的置换恢复）。
        用分块 **matmul** 形式 `|a|² - 2a·b + |b|²`，避免 materialise `(block, n, d)` 张量：
        后者在 n=6400 时是 512×6400×8 的 3 维数组，慢且占内存，是本地跑小 n 尚可、服务器大 n 不可行的原因。
    EN: Nearest target for each transformed point (recovers the upstream permutation). Uses the blockwise
        **matmul** form `|a|^2 - 2a.b + |b|^2` instead of materialising a `(block, n, d)` tensor, which at
        n=6400 is 512x6400x8 and is why the old version only scaled to small n.
    """
    pts = np.asarray(points, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    tgt_sq = np.einsum("ij,ij->i", tgt, tgt)
    out = np.empty(pts.shape[0], dtype=np.int64)
    for start in range(0, pts.shape[0], int(block)):
        stop = min(pts.shape[0], start + int(block))
        block_pts = pts[start:stop]
        pts_sq = np.einsum("ij,ij->i", block_pts, block_pts)
        dist = pts_sq[:, None] - 2.0 * (block_pts @ tgt.T) + tgt_sq[None, :]
        out[start:stop] = np.argmin(dist, axis=1)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="upstream Brenier toy problems through our loop")
    parser.add_argument("--kinds", nargs="+", default=["brenier", "correlated"])
    parser.add_argument("--n", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--tag", default="", help="suffix for the output JSON (queued runs)")
    parser.add_argument("--scan", default="hello", choices=("numpy", "hello", "torch"))
    parser.add_argument("--solver", default="cupdlpx", choices=("highs", "cupdlpx"))
    parser.add_argument("--prune-strategy", default="none", choices=("none", "tolerance", "hysteresis"))
    parser.add_argument("--tau-prune", type=float, default=1e-12)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--master-time-limit", type=float, default=60.0)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "core" / "out" / "brenier_toy")
    args = parser.parse_args(argv)

    records: list[dict[str, object]] = []
    print(f"{'kind':10s} {'d':>2} {'n':>5} {'|S|':>5} {'T':>3} {'conv':>5} {'F/(m+n)':>8} "
          f"{'delta':>10} {'maxviol':>10} {'isopt':>6} {'relgap':>9} {'wall':>6}")
    for kind in args.kinds:
        for d in args.dims:
            for n in args.n:
                for seed in args.seeds:
                    problem, theta_star, meta = build_permutation_problem(
                        kind, n=int(n), d=int(d), seed=int(seed))
                    prune = str(args.prune_strategy) in ("tolerance", "hysteresis")
                    master = (CuPdlpxMasterSolver(params={"time_sec_limit": float(args.master_time_limit)})
                              if args.solver == "cupdlpx" else LinearMasterSolver(random_seed=int(seed)))
                    t0 = time.perf_counter()
                    try:
                        r = solve_inverse_ot(
                            problem, solver=master, tie_break="none",
                            max_iterations=int(args.max_iterations), scan_backend=str(args.scan),
                            validate=True, prune=prune,
                            prune_hysteresis=(str(args.prune_strategy) == "hysteresis"),
                            tau_prune=float(args.tau_prune), random_seed=int(seed),
                        )
                        Fmax = max((x.support_size for x in r.iterations), default=problem.support_size)
                        rec = {
                            **meta,
                            "support": int(problem.support_size),
                            "T": int(r.n_iterations), "converged": bool(r.converged),
                            "F_over_m_plus_n": float(Fmax) / float(2 * int(n)),
                            "delta": float(r.delta), "maxviol": float(r.final_max_violation),
                            "is_optimal": None if r.is_optimal is None else bool(r.is_optimal),
                            "relative_gap": None if r.relative_gap is None else float(r.relative_gap),
                            "wall": float(time.perf_counter() - t0),
                            "theta_star": np.asarray(theta_star).tolist(),
                            "theta_returned": np.asarray(r.theta).tolist(),
                        }
                        records.append(rec)
                        print(f"{kind:10s} {d:>2} {n:>5} {problem.support_size:>5} {rec['T']:>3} "
                              f"{str(rec['converged']):>5} {rec['F_over_m_plus_n']:>8.2f} "
                              f"{rec['delta']:>10.3e} {rec['maxviol']:>10.2e} "
                              f"{str(rec['is_optimal']):>6} "
                              f"{(rec['relative_gap'] if rec['relative_gap'] is not None else float('nan')):>9.1e} "
                              f"{rec['wall']:>5.0f}s", flush=True)
                    except Exception as error:  # noqa: BLE001
                        print(f"{kind:10s} {d:>2} {n:>5}  EXC {type(error).__name__}: {str(error)[:70]}",
                              flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f".{args.tag}" if args.tag else ""
    path = args.out / f"brenier_toy{suffix}.json"
    # CN: 记录 tol 与中位代价，便于事后把 delta 按代价尺度归一化（绝对 tol 会随尺度变严）。
    # EN: record tol and the median cost so delta can be normalised by cost scale afterwards
    #     (an absolute tol tightens with scale).
    summary = {
        "tol": 1e-9, "tau_prune": float(args.tau_prune), "solver": str(args.solver),
        "scan": str(args.scan), "max_iterations": int(args.max_iterations),
        "master_time_limit": float(args.master_time_limit),
        "prune_strategy": str(args.prune_strategy), "records": records,
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[brenier_toy] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
