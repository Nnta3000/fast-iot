"""
CN: **Phase A —— 密集 LP 的诊断与增长曲线**。

    CN: 问题：一次 m=600 的 `fit_full_lp` 花了 178 s，而此前 m=300 只要 0.4 s。差 400 倍，必须先弄清是
        **建边**、**求解**、还是**求解器状态异常**（迭代上限/时间上限）。然后给出 m ∈ {100,200,400,600}
        的增长曲线。

    CN: 报告每次：建边时间、求解时间、求解器状态与迭代数、是否触到时间上限、`R'/R_uniform`。
        `fit_full_lp` 目前只回传 `build_s`/`solve_s`/`status`；这里直接调用底层 `RegretMasterSolver`，
        以便拿到 `iterations` 与 `message`。

EN: **Phase A -- dense-LP diagnosis and growth curve.**

    Problem: one `fit_full_lp` at m=600 took 178 s, while m=300 previously took 0.4 s. A 400x difference must
    be explained -- is it **edge construction**, **the solve**, or an **abnormal solver state** (iteration or
    time cap)? Then give the growth curve for m in {100,200,400,600}.

    Per fit: build time, solve time, solver status and iterations, whether the time cap was hit, and
    `R'/R_uniform`. `fit_full_lp` only returns `build_s`/`solve_s`/`status`, so this calls
    `RegretMasterSolver` directly to also obtain `iterations` and `message`.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import EdgeSet, InverseOTProblem  # noqa: E402
from iot.core.features import DiagonalWeightFeatures  # noqa: E402
from iot.relaxed.master_regret import RegretMasterSolver, regret_objective  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunLog, format_table  # noqa: E402
from iot_experiments.harness.solver import box_simplex  # noqa: E402

PROVENANCE: dict[str, object] = {
    "purpose": "explain the 178 s dense fit at m=600 and give the growth curve",
    "solver": "RegretMasterSolver(backend=cupdlpx) on every edge (m*m)",
    "config": "Theta = box ∩ simplex, box_relative 0.25 (the iOT default)",
}


@dataclass(frozen=True)
class DenseFit:
    """CN: 一次密集 LP 的读数。EN: one dense-LP reading."""

    m: int
    k: int
    build_s: float
    solve_s: float
    total_s: float
    status: str
    iterations: int
    message: str
    r_prime: float
    r_uniform: float
    r_ratio: float
    n_edges: int


def dense_fit(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, rel: float, solver: str,
              time_limit_s: float) -> DenseFit:
    """
    CN: 解一次**全边** LP，并回传求解器状态。CN: 这与 `fit_full_lp` 是同一个问题，只是多取诊断字段。
    EN: solve the **all-edges** LP once and return solver state. The same problem as `fit_full_lp`, with more
        diagnostics.
    """
    m, k = int(x.shape[0]), int(x.shape[1])
    t0 = time.perf_counter()
    plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
    problem = InverseOTProblem(features=DiagonalWeightFeatures(x, y),
                               theta_set=box_simplex(k, box_relative=float(rel)), plan=plan)
    r: Int[np.ndarray, "m m"]
    c: Int[np.ndarray, "m m"]
    r, c = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    edges = EdgeSet.from_arrays(r.reshape(-1), c.reshape(-1), n_target=m)
    build_s = time.perf_counter() - t0

    uni = np.full(k, 1.0 / k, dtype=np.float64)
    cost_l = np.asarray(problem.features.costs(uni, r.reshape(-1), c.reshape(-1)),
                        dtype=np.float64).reshape(m, m)
    f_uni = cost_l.min(axis=1)
    g_uni = np.zeros(m, dtype=np.float64)
    r_uniform = float(regret_objective(problem) @ np.concatenate([uni, f_uni, g_uni]))

    t0 = time.perf_counter()
    master = RegretMasterSolver(backend=str(solver), tie_break=False,
                                params={"time_sec_limit": float(time_limit_s)})
    status = "ok"
    iterations = -1
    message = ""
    r_prime = float("nan")
    try:
        sol = master.solve(problem, edges)
        r_prime = float(sol.regret)
        iterations = int(getattr(sol, "iterations", -1))
        message = str(getattr(sol, "message", ""))
    except Exception as error:  # noqa: BLE001 - reported, never substituted
        status = f"FAILED: {type(error).__name__}: {str(error)[:100]}"
    solve_s = time.perf_counter() - t0
    ratio = r_prime / r_uniform if r_uniform > 1e-12 else float("nan")
    return DenseFit(m=m, k=k, build_s=round(build_s, 3), solve_s=round(solve_s, 3),
                    total_s=round(build_s + solve_s, 3), status=status, iterations=iterations,
                    message=message[:60], r_prime=round(r_prime, 6) if np.isfinite(r_prime) else float("nan"),
                    r_uniform=round(r_uniform, 6), r_ratio=round(ratio, 4), n_edges=int(m * m))


@dataclass(frozen=True)
class PhaseAConfig(RunConfig):
    """CN: Phase A 的旋钮。EN: Phase A's knobs."""

    section = "synthrad_phase_a"
    study: str = "synthrad_phase_a"
    ms: tuple[int, ...] = (100, 200, 400, 600)
    k: int = 64
    rel: float = 0.25
    time_limit_s: float = 600.0

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本阶段特有的参数。EN: only this phase's flags."""
        parser.add_argument("--ms", type=int, nargs="+", default=[100, 200, 400, 600])
        parser.add_argument("--k", type=int, default=64)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(PhaseAConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = PhaseAConfig.from_namespace(args, parser)
    log = RunLog(config.run_dir / "run.log")
    log.line(f"config: ms={config.ms} k={config.k} rel={config.rel} time_limit={config.time_limit_s}s")

    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for m in config.ms:
        # CN: 用与 SynthRAD 拟合同形制的特征（随机但同维），A 阶段只关心**求解器**的规模行为。
        # EN: synthetic features of the same shape as the SynthRAD fits; Phase A is about solver scaling only.
        z = rng.normal(size=(m, int(config.k)))
        x = z / np.linalg.norm(z, axis=1, keepdims=True)
        y = (z + rng.normal(scale=0.1, size=(m, int(config.k))))
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        fit = dense_fit(x, y, rel=float(config.rel), solver=str(config.solver),
                        time_limit_s=float(config.time_limit_s))
        rows.append({"m": fit.m, "edges": fit.n_edges, "build_s": fit.build_s, "solve_s": fit.solve_s,
                     "total_s": fit.total_s, "status": fit.status[:20], "iter": fit.iterations,
                     "R_prime": fit.r_prime, "R_uni": fit.r_uniform, "R_ratio": fit.r_ratio})
        log.line(f"m={m:>5} edges={fit.n_edges:>9} build={fit.build_s:>6.2f}s solve={fit.solve_s:>8.2f}s "
                 f"status={fit.status[:20]} iters={fit.iterations} R'={fit.r_prime:.3e} "
                 f"R_uni={fit.r_uniform:.3e} ratio={fit.r_ratio:.4f} msg={fit.message!r}")
        if fit.status != "ok" or fit.solve_s >= 0.95 * float(config.time_limit_s):
            log.line(f"  NOTE: m={m} hit a solver limit (status={fit.status!r}, "
                     f"solve={fit.solve_s:.1f}s of {config.time_limit_s}s)")
    log.line("\n" + format_table(rows, ("m", "edges", "build_s", "solve_s", "total_s", "status", "iter",
                                        "R_ratio")))
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
