"""
CN: **退化（degenerate）实例构造** —— 使最优面上有 Θ(m·n) 条紧边，检验循环在这些实例上的行为。

构造：把源/目标分成两组，令**所有跨组代价恒等**（在同一 theta 下），组内代价严格更大。
于是最优面的维数很大：跨组子矩阵的任意行/列置换都最优，跨组块内的每条边都在某个最优对偶下取等
→ 紧边数可达 Θ(m·n)（本文件的 `block` 变体中，跨组块就是全部 m·n 条边）。

EN: **Degenerate instance construction** -- yield Theta(m*n) tight edges on the optimal face, to test
    how the loop behaves there.

    Construction: split sources/targets into two groups, make **every cross-group cost identical**
    (at the same theta) and the within-group costs strictly larger. The optimal face then has large
    dimension: any row/column permutation inside the cross block is optimal, so every cross edge is
    tight for some optimal dual => Theta(m*n) tight edges (the cross block IS all m*n edges in the
    `block` variant below).

用法 / EN: usage
    python -m iot_experiments.core.degenerate_instances --n 1000 --variant block
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import FeatureMap, InverseOTProblem, SimplexTheta  # noqa: E402


@dataclass(frozen=True)
class BlockConstantFeatures:
    """
    CN: 分组块常数代价。`c_ij(theta) = sum_k theta_k phi_k(i,j)`，其中特征
        `phi_0(i,j) = 1`（常数），`phi_1(i,j) = 1` 当 i,j 同组（惩罚），否则 0。
        取 theta=(1,0) → 全常数（**完全退化**：所有边都是紧的）。
        取 theta=(1,eps) → 组内贵 eps，跨组仍是常数块（跨组块 = 全部跨组边，仍 Θ(m·n) 紧）。
    EN: Grouped block-constant cost. `phi_0 = 1` (constant), `phi_1 = 1` iff same group (penalty).
        theta=(1,0) -> fully constant (maximally degenerate: every edge is tight).
        theta=(1,eps) -> within-group costs eps more; the cross block stays constant with Theta(m*n)
        tight edges.
    """

    group_of_source: np.ndarray
    group_of_target: np.ndarray

    def __post_init__(self) -> None:
        s = np.asarray(self.group_of_source, dtype=np.int64).reshape(-1)
        t = np.asarray(self.group_of_target, dtype=np.int64).reshape(-1)
        object.__setattr__(self, "group_of_source", np.ascontiguousarray(s))
        object.__setattr__(self, "group_of_target", np.ascontiguousarray(t))

    @property
    def n_source(self) -> int:
        return int(self.group_of_source.size)

    @property
    def n_target(self) -> int:
        return int(self.group_of_target.size)

    @property
    def n_features(self) -> int:
        return 2

    def features(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        r = np.asarray(rows, dtype=np.int64).reshape(-1)
        c = np.asarray(cols, dtype=np.int64).reshape(-1)
        same = (self.group_of_source[r] == self.group_of_target[c]).astype(np.float64)
        return np.ascontiguousarray(np.stack([np.ones_like(same), same], axis=1), dtype=np.float64)

    def costs(self, theta: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        value = np.asarray(theta, dtype=np.float64).reshape(-1)
        phi = self.features(rows, cols)
        return np.ascontiguousarray(phi @ value, dtype=np.float64)


def build_degenerate(*, n: int, variant: str, penalty: float = 1.0) -> tuple[InverseOTProblem, dict]:
    """
    CN: 构造退化实例。计划取「源 i → 目标 i」的置换，质量均匀。证书由构造保证：
        跨组块常数 ⇒ 任意双射在该块内都最优；这里给出的置换就是其中之一。
    EN: Build the degenerate instance. The plan is the identity permutation with uniform mass. The
        certificate is by construction: the cross block is constant, so any bijection inside it is
        optimal; the identity is one of them.
    """
    n = int(n)
    half = n // 2
    if variant == "block":
        # CN: 两组，前 half / 后 n-half。EN: two groups, first `half` / rest.
        gs = np.array([0] * half + [1] * (n - half), dtype=np.int64)
        gt = np.array([0] * half + [1] * (n - half), dtype=np.int64)
    elif variant == "half":
        # CN: 全部源在组 0，目标分两组（跨组 = 左半块 m×half）。EN: all sources group 0, targets split.
        gs = np.zeros(n, dtype=np.int64)
        gt = np.array([0] * half + [1] * (n - half), dtype=np.int64)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    features = BlockConstantFeatures(gs, gt)
    theta_set = SimplexTheta(2)
    # theta = (1, penalty) then normalised into the simplex: constant + within-group penalty.
    raw = np.array([1.0, float(penalty)], dtype=np.float64)
    theta_star = raw / raw.sum()

    # CN: 置换计划：i -> i（组内配对），质量均匀。跨组块常数使其最优。
    # EN: permutation plan i -> i (within-group pairing), uniform mass; the constant cross block makes
    #     it optimal.
    rows = np.arange(n, dtype=np.int64)
    cols = np.arange(n, dtype=np.int64)
    vals = np.full(n, 1.0 / n, dtype=np.float64)
    plan = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=plan)

    n_cross = int(np.sum(gs[:, None] != gt[None, :])) if variant == "block" else int(
        np.sum(gs[:, None] != gt[None, :])
    )
    meta = {
        "variant": variant, "n": n, "theta_star": theta_star.tolist(),
        "penalty": float(penalty), "n_cross_edges": n_cross,
        "cross_fraction": float(n_cross) / float(n * n),
    }
    return problem, meta


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="degenerate-instance probe")
    p.add_argument("--n", type=int, nargs="+", default=[200, 1000])
    p.add_argument("--variant", default="block", choices=("block", "half"))
    p.add_argument("--penalty", type=float, nargs="+", default=[1.0, 1e-3, 0.0])
    p.add_argument("--scan", default="hello", choices=("hello", "torch", "numpy"))
    p.add_argument("--solver", default="cupdlpx", choices=("cupdlpx", "highs"))
    p.add_argument("--prune-strategy", default="none", choices=("none", "tolerance", "hysteresis"))
    p.add_argument("--tau-prune", type=float, default=1e-12)
    p.add_argument("--tie-break", default="none", choices=("none", "regularized"))
    p.add_argument("--epsilon", type=float, default=1e-4)
    p.add_argument("--max-iterations", type=int, default=60)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from iot import solve_inverse_ot
    from iot.cupdlpx_master import CuPdlpxMasterSolver
    from iot.master import LinearMasterSolver

    for n in args.n:
        for penalty in args.penalty:
            problem, meta = build_degenerate(n=int(n), variant=str(args.variant), penalty=float(penalty))
            prune = str(args.prune_strategy) in ("tolerance", "hysteresis")
            if args.solver == "cupdlpx":
                master = CuPdlpxMasterSolver(params={"time_sec_limit": 30.0},
                                             epsilon=float(args.epsilon))
            else:
                master = LinearMasterSolver(random_seed=0)
            t0 = time.perf_counter()
            try:
                r = solve_inverse_ot(
                    problem, solver=master, tie_break=str(args.tie_break),
                    max_iterations=int(args.max_iterations), scan_backend=str(args.scan),
                    validate=False, prune=prune, prune_hysteresis=(str(args.prune_strategy) == "hysteresis"),
                    tau_prune=float(args.tau_prune), random_seed=0,
                )
                wall = time.perf_counter() - t0
                Fmax = max((x.support_size for x in r.iterations), default=r.support_size)
                print(f"n={n} pen={penalty:g} {args.variant}: conv={r.converged} it={r.n_iterations} "
                      f"delta={r.delta:.3e} maxviol={r.final_max_violation:+.3e} "
                      f"Fmax={Fmax} Fmax/(m+n)={Fmax/(2*n):.2f} |S|={problem.support_size} "
                      f"wall={wall:.0f}s | cross_frac={meta['cross_fraction']:.2f}", flush=True)
            except Exception as error:  # noqa: BLE001
                print(f"n={n} pen={penalty:g} {args.variant}: EXCEPTION {type(error).__name__}: "
                      f"{str(error)[:90]} | wall={time.perf_counter()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
