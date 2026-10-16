"""
CN: **自然退化族的探针** —— 不用手造的常数块，而是用真实点云产生退化：
     (a) `duplicate`：抽 n 个点后复制一部分（重复点 → 多行代价完全相同 → argmin 大范围并列）；
     (b) `grid`：规则格点云（对称 → 大量等距对）；
     (c) `lattice1d`：d=1 的等距点列（最极端，d=1 时代价只依赖一个坐标）。
     在 d ∈ {1,2} 上测 **迭代轮数 T** 与 **|F|/(m+n)**。

EN: **Natural-degeneracy probe** -- instead of a hand-built constant block, produce degeneracy from real
    point clouds:
     (a) `duplicate`: draw n points then duplicate a fraction (repeated points => identical cost rows
         => wide argmin ties);
     (b) `grid`: a regular grid cloud (symmetry => many equidistant pairs);
     (c) `lattice1d`: an equispaced 1-D chain (most extreme; at d=1 the cost depends on one coordinate).
     Measure **T (iterations)** and **|F|/(m+n)** for d in {1,2}.

用法 / EN: usage
    python -m iot_experiments.core.degenerate_natural --n 1000 --family quadratic --dims 1 2
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

from iot import DiagonalWeightFeatures, InverseOTProblem, QuadraticFeatures, SimplexTheta  # noqa: E402


def _clouds(kind: str, n: int, d: int, rng: np.random.Generator,
            dup_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 生成 (source, target) 点云。`kind` 决定退化来源。
    EN: Build the (source, target) clouds; `kind` selects the degeneracy source.
    """
    if kind == "duplicate":
        # CN: 抽 n_unique 个点，复制一部分使恰好 n 个点（重复点 → 成本行并列）。
        # EN: draw n_unique points and duplicate a fraction, so exactly n points (repeated rows tie).
        n_unique = max(1, int(round(n * (1.0 - float(dup_fraction)))))
        base = rng.normal(size=(n_unique, d))
        reps = np.arange(n) % n_unique  # round-robin replication to reach n
        source = base[reps]
        target = rng.normal(size=(n, d))
    elif kind == "grid":
        # CN: 二维/一维规则格点（每边 sqrt(n) 个）。EN: regular grid cloud (~sqrt(n) per side).
        side = max(1, int(round(np.sqrt(n))))
        if d == 1:
            coords = np.linspace(0.0, 1.0, side)
            source = coords[reps1d(n, side)][:, None]
            target = coords[reps1d(n, side)][:, None]
        else:
            axis = np.linspace(0.0, 1.0, side)
            gx, gy = np.meshgrid(axis, axis, indexing="ij")
            grid = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
            idx = reps1d(n, grid.shape[0])
            source = grid[idx]
            target = grid[idx]
    elif kind == "lattice1d":
        coords = np.linspace(0.0, 1.0, n)
        source = coords[:, None]
        target = coords[:, None]
    else:
        raise ValueError(f"unknown kind {kind!r}")
    # CN: 避免源/目标完全相同（那会使 OT 退化为恒等且 maxviol 平凡）；给目标一个小的平移。
    # EN: avoid identical clouds (trivial identity OT); shift the target slightly.
    target = target + 1e-3
    return np.ascontiguousarray(source, dtype=np.float64), np.ascontiguousarray(target, dtype=np.float64)


def reps1d(n: int, m: int) -> np.ndarray:
    """CN: 把 m 个索引轮转复制到长度 n。EN: round-robin m indices out to length n."""
    return np.arange(n) % m


def _one(*, kind: str, family: str, n: int, d: int, seed: int, prune: str, tau_prune: float,
         tau_sep: float, gamma: int, scan: str, max_iterations: int) -> dict[str, object]:
    from iot import solve_inverse_ot
    from iot.cupdlpx_master import CuPdlpxMasterSolver

    rng = np.random.default_rng(int(seed))
    source, target = _clouds(kind, int(n), int(d), rng)
    if family == "diagonal":
        features = DiagonalWeightFeatures(source, target)
    elif family == "quadratic":
        features = QuadraticFeatures(source, target)
    else:
        raise ValueError(family)
    theta_set = SimplexTheta(features.n_features)
    theta_star = theta_set.sample(np.random.default_rng(int(seed) + 1))

    # CN: 用 POT 小规模解真值方案（n<=4000 → m·n<=1.6e7，可承受）。
    # EN: POT ground truth (n<=4000 => m*n<=1.6e7, affordable).
    import ot
    from iot import dense_cost_matrix
    cost = dense_cost_matrix(features, theta_star)
    a = np.ones(int(n), dtype=np.float64) / int(n)
    plan = ot.emd(a, a, cost, numItermax=4_000_000)
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=sparse.csr_matrix(plan))

    master = CuPdlpxMasterSolver(params={"time_sec_limit": 60.0})
    t0 = time.perf_counter()
    r = solve_inverse_ot(problem, solver=master, tie_break="none", max_iterations=int(max_iterations),
                         scan_backend=str(scan), validate=False, prune=(prune != "none"),
                         prune_hysteresis=(prune == "hysteresis"), tau_prune=float(tau_prune),
                         tol=float(tau_sep), gamma=int(gamma), random_seed=int(seed))
    wall = time.perf_counter() - t0
    Fmax = max((x.support_size for x in r.iterations), default=problem.support_size)
    return {
        "kind": kind, "family": family, "n": int(n), "d": int(d), "seed": int(seed),
        "prune": prune, "tau_prune": float(tau_prune),
        "n_eff": int(problem.n_source), "m_eff": int(problem.n_target),
        "support": int(problem.support_size), "n_support": int(problem.support_size) / (2 * int(n)),
        "T": int(r.n_iterations), "conv": bool(r.converged), "delta": float(r.delta),
        "maxviol": float(r.final_max_violation), "Fmax": int(Fmax),
        "F_over_N": float(Fmax) / (2.0 * int(n)), "wall": float(wall),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="natural-degeneracy probe")
    p.add_argument("--kinds", nargs="+", default=["duplicate", "grid", "lattice1d"])
    p.add_argument("--family", default="quadratic")
    p.add_argument("--n", type=int, nargs="+", default=[1000, 2000])
    p.add_argument("--dims", type=int, nargs="+", default=[2, 1])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--prune-strategy", default="none", choices=("none", "hysteresis"))
    p.add_argument("--tau-prune", type=float, default=1e-3)
    p.add_argument("--tau-sep", type=float, default=1e-9)
    p.add_argument("--gamma", type=int, default=1)
    p.add_argument("--scan", default="hello", choices=("hello", "torch", "numpy"))
    p.add_argument("--max-iterations", type=int, default=60)
    p.add_argument("--out", type=Path,
                   default=_REPO_ROOT / "iot_experiments" / "core" / "out" / "degenerate",
                   help="directory for the JSON record")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    print(f"{'kind':10s} {'fam':10s} {'d':>2} {'n':>6} {'m_eff':>6} {'|S|/(2n)':>9} "
          f"{'T':>3} {'conv':>5} {'F/(m+n)':>8} {'delta':>10} {'maxviol':>10} {'wall':>6}")
    for kind in args.kinds:
        for d in args.dims:
            for n in args.n:
                for seed in args.seeds:
                    try:
                        r = _one(kind=kind, family=str(args.family), n=int(n), d=int(d), seed=int(seed),
                                 prune=str(args.prune_strategy), tau_prune=float(args.tau_prune),
                                 tau_sep=float(args.tau_sep), gamma=int(args.gamma), scan=str(args.scan),
                                 max_iterations=int(args.max_iterations))
                        records.append(r)
                        print(f"{r['kind']:10s} {r['family']:10s} {r['d']:>2} {r['n']:>6} {r['m_eff']:>6} "
                              f"{r['n_support']:>9.2f} {r['T']:>3} {str(r['conv']):>5} {r['F_over_N']:>8.2f} "
                              f"{r['delta']:>10.3e} {r['maxviol']:>10.2e} {r['wall']:>5.0f}s", flush=True)
                    except Exception as error:  # noqa: BLE001
                        print(f"{kind:10s} {args.family:10s} {d:>2} {n:>6}  EXC {type(error).__name__}: "
                              f"{str(error)[:60]}", flush=True)
    (args.out / "degenerate_natural.json").write_text(
        json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    print(f"[degenerate_natural] wrote {args.out / 'degenerate_natural.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
